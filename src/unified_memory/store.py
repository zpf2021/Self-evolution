"""Control state, durable ingestion jobs and audit only. No semantic document store."""

from __future__ import annotations
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


class ControlStore:
    def __init__(self, root: Path):
        self.path = root / ".index" / "unified-control.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS flags(scope TEXT, id TEXT, inactive INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0, snooze_until REAL DEFAULT 0, PRIMARY KEY(scope,id));
              CREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY, scope TEXT, session TEXT,
                kind TEXT, time REAL, payload TEXT, event_key TEXT UNIQUE);
              CREATE INDEX IF NOT EXISTS audit_scope ON audit(scope,session,seq);
              CREATE TABLE IF NOT EXISTS checkpoints(scope TEXT,session TEXT,payload TEXT,
                PRIMARY KEY(scope,session));
              CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,scope TEXT,session TEXT,
                kind TEXT,payload TEXT,payload_hash TEXT,status TEXT DEFAULT 'pending',error TEXT,time REAL);
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def audit(self, scope, session, kind, payload, key=None):
        with self.db() as db:
            db.execute(
                "INSERT OR IGNORE INTO audit(scope,session,kind,time,payload,event_key) VALUES(?,?,?,?,?,?)",
                (
                    scope,
                    session,
                    kind,
                    time.time(),
                    json.dumps(payload, ensure_ascii=False),
                    key,
                ),
            )

    def events(self, scope, session=None, limit=100, kind=None):
        sql, args = "SELECT * FROM audit WHERE scope=?", [scope]
        if session is not None:
            sql += " AND session=?"
            args.append(session)
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        sql += " ORDER BY seq DESC LIMIT ?"
        args.append(max(1, min(int(limit), 1000)))
        with self.db() as db:
            return [
                {**dict(r), "payload": json.loads(r["payload"])}
                for r in db.execute(sql, args)
            ]

    def flags(self, scope):
        with self.db() as db:
            return {
                r["id"]: dict(r)
                for r in db.execute("SELECT * FROM flags WHERE scope=?", (scope,))
            }

    def flag(self, scope, ident, field, value):
        if field not in ("inactive", "pinned", "snooze_until"):
            raise ValueError("Invalid control field")
        with self.db() as db:
            db.execute(
                "INSERT OR IGNORE INTO flags(scope,id) VALUES(?,?)", (scope, ident)
            )
            db.execute(
                f"UPDATE flags SET {field}=? WHERE scope=? AND id=?",
                (value, scope, ident),
            )

    def checkpoint(self, scope, session, payload=None):
        with self.db() as db:
            if payload is not None:
                db.execute(
                    "INSERT OR REPLACE INTO checkpoints VALUES(?,?,?)",
                    (scope, session, json.dumps(payload, ensure_ascii=False)),
                )
            row = db.execute(
                "SELECT payload FROM checkpoints WHERE scope=? AND session=?",
                (scope, session),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def enqueue(self, scope, session, kind, payload, key):
        ident = digest([scope, session, kind, key])
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.db() as db:
            existing = db.execute(
                "SELECT payload_hash FROM jobs WHERE id=?", (ident,)
            ).fetchone()
            if existing and existing[0] != digest(payload):
                raise ValueError("Idempotency key reused for different payload")
            db.execute(
                "INSERT OR IGNORE INTO jobs(id,scope,session,kind,payload,payload_hash,time) VALUES(?,?,?,?,?,?,?)",
                (ident, scope, session, kind, encoded, digest(payload), time.time()),
            )
            row = db.execute(
                "SELECT id,status,error FROM jobs WHERE id=?", (ident,)
            ).fetchone()
        return dict(row)

    def recover(self):
        # An interrupted remote LLM extraction cannot safely be replayed blindly.
        with self.db() as db:
            db.execute(
                "UPDATE jobs SET status='uncertain',error='Worker interrupted; inspect EverOS before replay' WHERE status='processing'"
            )

    def claim(self):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs j WHERE status='pending' AND NOT EXISTS (SELECT 1 FROM jobs old WHERE old.scope=j.scope AND old.session=j.session AND old.time<j.time AND old.status IN ('processing','uncertain','failed')) ORDER BY time LIMIT 1"
            ).fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET status='processing' WHERE id=?", (row["id"],))
            return {**dict(row), "payload": json.loads(row["payload"])}

    def finish(self, ident, status, error=None):
        with self.db() as db:
            db.execute(
                "UPDATE jobs SET status=?,error=? WHERE id=?", (status, error, ident)
            )
            if status == "done":
                # Ingestion transport is not a second archive of long-term content.
                db.execute("UPDATE jobs SET payload='{}' WHERE id=?", (ident,))

    def jobs(self, scope):
        with self.db() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,session,kind,status,error,time FROM jobs WHERE scope=? ORDER BY time DESC LIMIT 100",
                    (scope,),
                )
            ]

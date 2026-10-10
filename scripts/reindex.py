#!/usr/bin/env python3
"""Offline migration into a NEW root. Never modify/delete the source index."""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
import lancedb
import portalocker

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--target", required=True)
a = p.parse_args()
source = Path(a.source).resolve()
target = Path(a.target).resolve()
if (
    not source.is_dir()
    or target.exists()
    or target.is_relative_to(source)
    or source.is_relative_to(target)
):
    p.error("source must exist; target must be new and outside source")
with portalocker.Lock(str(source / ".lock"), timeout=0):
    shutil.copytree(
        source,
        target,
        ignore=lambda directory, names: [
            n
            for n in names
            if n in (".lock", "server.log")
            or (
                Path(directory) == source / ".index"
                and n in ("lancedb", "embedding.json")
            )
        ],
    )
    env = {**os.environ, "EVEROS_ROOT": str(target)}
    subprocess.run(
        [
            os.environ.get("UNIFIED_EVEROS_BIN", "everos"),
            "cascade",
            "--root",
            str(target),
            "rebuild",
            "--yes",
        ],
        env=env,
        check=True,
    )
    state_db = target / ".index/sqlite/system.db"
    with sqlite3.connect(f"file:{state_db}?mode=ro", uri=True) as db:
        queue = dict(
            db.execute("SELECT status, count(*) FROM md_change_state GROUP BY status")
        )
    conn = lancedb.connect(str(target / ".index/lancedb"))
    vectors = {}
    for name in conn.list_tables().tables:
        table = conn.open_table(name)
        if "vector" in table.schema.names:
            vectors[name] = {
                "rows": table.count_rows(),
                "embedded": table.count_rows(filter="vector IS NOT NULL"),
            }
    fingerprint = target / ".index/embedding.json"
    valid = (
        fingerprint.is_file()
        and not any(queue.get(status, 0) for status in ("pending", "processing", "failed"))
        and all(v["rows"] == v["embedded"] for v in vectors.values())
    )
    report = {
        "source": str(source),
        "target": str(target),
        "markdown_files": sum(
            1 for x in target.rglob("*.md") if ".index" not in x.parts
        ),
        "queue": queue,
        "vectors": vectors,
        "fingerprint_present": fingerprint.is_file(),
        "status": "validated" if valid else "failed; inspect target before any switch",
    }
    (target / "reindex-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if not valid:
        raise SystemExit("Reindex validation failed; source was not modified")

"""Unified tool operations; all persistent semantic content is owned by EverOS."""

from __future__ import annotations
import asyncio
import fnmatch
import json
import re
import time
from pathlib import Path
from .catalog import catalog, resolve, scope_id, validate_scope
from .store import ControlStore, digest

TOOLS = (
    "memory_retrieve",
    "search",
    "vsearch",
    "get",
    "multi_get",
    "status",
    "session_log",
    "memory_forget",
    "memory_pin",
    "memory_snooze",
    "memory_compact",
    "memory_save",
    "trajectory_search",
    "run_consolidation",
    "graph_search",
    "trace_knowledge",
    "generate_templates",
)


def contaminated(text):
    return bool(
        re.search(
            r"(?i)(GAIA.{0,60}(ground.?truth|standard.answer|标准答案)|GAIA_web\.jsonl|\bFINAL ANSWER\b)",
            text,
        )
    )


def clean_text(text):
    return re.sub(
        r"<(?:/?)(?:everos_memory|unified_memory)>.*?</(?:everos_memory|unified_memory)>",
        "",
        text,
        flags=re.S | re.I,
    )


class UnifiedService:
    def __init__(self, root: Path, search_fn=None):
        self.root = root
        self.store = ControlStore(root)
        self.search_fn = search_fn

    def _skill_history(self, scope, item):
        from everos.core.persistence import sanitize_dirname

        name = str(item["metadata"].get("name", ""))
        if not name:
            return []
        expected = (
            self.root.resolve()
            / ".index"
            / "skill-history"
            / sanitize_dirname(scope["app_id"], "app")
            / sanitize_dirname(scope["project_id"], "project")
            / sanitize_dirname(scope["agent_id"], "agent")
            / sanitize_dirname(name, "skill")
        )
        if expected.resolve() != expected or not expected.is_dir():
            return []
        files = sorted(
            (p for p in expected.glob("*.md") if re.fullmatch(r"[0-9a-f]{64}", p.stem)),
            key=lambda p: p.stat().st_mtime,
        )
        return [
            {
                "sha256": p.stem,
                "content": p.read_text(encoding="utf-8")[:100000],
                "truncated": p.stat().st_size > 100000,
            }
            for p in files[-20:]
            if p.resolve().is_relative_to(expected)
        ]

    async def _search(self, scope, query, method, limit):
        from everos.memory.search import SearchMethod, SearchRequest
        from everos.service import search

        search_fn = self.search_fn or search
        results = []
        warnings = []
        for owner_key in ("user_id", "agent_id"):
            request = SearchRequest(
                **{owner_key: scope[owner_key]},
                app_id=scope["app_id"],
                project_id=scope["project_id"],
                query=query,
                method=method,
                top_k=min(100, max(limit * 3, 10)),
            )
            try:
                response = await search_fn(request)
            except Exception as exc:
                if method == "keyword":
                    raise
                warnings.append(
                    f"{method} unavailable; keyword fallback: {type(exc).__name__}"
                )
                request = request.model_copy(update={"method": SearchMethod.KEYWORD})
                response = await search_fn(request)
            data = response.model_dump(mode="json")["data"]
            for key, kind in [
                ("episodes", "episode"),
                ("agent_cases", "agent_case"),
                ("agent_skills", "agent_skill"),
            ]:
                results.extend({**item, "kind": kind} for item in data.get(key, []))
        items = await catalog(self.root, scope)
        by_id = {x["id"]: x for x in items}
        flags = self.store.flags(scope_id(scope))
        now = time.time()
        filtered = []
        for hit in results:
            item = by_id.get(hit["id"])
            if not item:
                continue  # index lag / dangling row: markdown is authoritative
            f = flags.get(item["id"], {})
            if f.get("inactive") or f.get("snooze_until", 0) > now:
                continue
            if scope.get("evaluation_mode") and contaminated(item["body"]):
                continue
            filtered.append(
                {
                    **item,
                    "score": float(hit.get("score") or 0),
                    "pinned": bool(f.get("pinned")),
                }
            )
        filtered.sort(key=lambda x: (x["pinned"], x["score"]), reverse=True)
        seen = set()
        unique = []
        for item in filtered:
            if item["id"] not in seen:
                unique.append(item)
                seen.add(item["id"])
        return {"results": unique[:limit], "warnings": warnings}

    async def call(self, name, args, scope):
        validate_scope(scope)
        key = scope_id(scope)
        session = scope["session_id"]
        limit = max(1, min(int(args.get("limit", 10)), 100))
        if name in ("memory_retrieve", "search", "vsearch"):
            query = str(args.get("query", "")).strip()
            if not query:
                raise ValueError("query is required")
            if args.get("vault") not in (None, "", "default", "_clawmem"):
                raise ValueError(
                    "Legacy vaults must be migrated explicitly; cross-project vault access is disabled"
                )
            method = {"search": "keyword", "vsearch": "vector"}.get(name, "hybrid")
            result = await self._search(scope, query, method, limit)
            self.store.audit(
                key, session, "retrieved", {"ids": [x["id"] for x in result["results"]]}
            )
            return result
        if name == "status":
            from everos.config import load_settings

            cfg = load_settings().embedding
            items = await catalog(self.root, scope)
            return {
                "backend": "everos",
                "scope": scope,
                "tools": list(TOOLS),
                "memory_count": len(items),
                "jobs": self.store.jobs(key),
                "embedding": {
                    "model": cfg.model,
                    "dimension": cfg.index_dimension,
                    "fingerprint": cfg.fingerprint,
                },
            }
        if name == "session_log":
            events = self.store.events(key, limit=1000)
            sessions = {}
            for event in events:
                sessions.setdefault(
                    event["session"],
                    {
                        "session_id": event["session"],
                        "latest_at": event["time"],
                        "events": [],
                    },
                )["events"].append(event)
            return {"sessions": list(sessions.values())[:limit]}
        if name == "trajectory_search":
            query = str(args.get("query", "")).lower()
            events = self.store.events(key, limit=1000, kind="tool")
            return {
                "results": [
                    e
                    for e in events
                    if query in json.dumps(e["payload"], ensure_ascii=False).lower()
                ][:limit],
                "window": "most recent 1000 tool events in this project",
            }
        if name == "memory_compact":
            steps = {}
            for field in ("completed_steps", "pending_steps"):
                value = args.get(field, [])
                if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                    raise ValueError(f"{field} must be a list of strings")
                steps[field] = value[:100]
            cp = {
                "reason": args.get("reason", "routine"),
                "goal": args.get("goal"),
                **steps,
                "verification": args.get("verification"),
                "saved_at": time.time(),
                "events": self.store.events(key, session, limit=30),
            }
            self.store.checkpoint(key, session, cp)
            return {
                "checkpoint": cp,
                "note": "Working state saved; this tool does not compact the host context window.",
            }
        if name == "memory_save":
            if args.get("path"):
                raise ValueError(
                    "Legacy memory_save path cannot be mapped to EverOS; omit path and use the returned job id"
                )
            title = str(args.get("title", "")).strip()
            body = clean_text(str(args.get("body", ""))).strip()
            if not title or not body:
                raise ValueError("title and body are required")
            if len(body) > 100000:
                raise ValueError("body exceeds 100000 characters")
            if args.get("collection", "_clawmem") not in ("_clawmem", "default"):
                raise ValueError(
                    "Collection migration is required; cannot silently remap a custom collection"
                )
            if scope.get("evaluation_mode") and contaminated(title + "\n" + body):
                self.store.audit(
                    key,
                    session,
                    "rejected",
                    {"reason": "benchmark_answer_contamination", "hash": digest(body)},
                )
                return {"accepted": False, "reason": "benchmark_answer_contamination"}
            content = f"Candidate evidence (agent-reported, not independently verified)\nTitle: {title}\nType: {args.get('contentType', 'discovery')}\n{body}"
            stamp = int(time.time() * 1000)
            save_key = digest([title, body, args.get("contentType")])
            payload = {
                "app_id": scope["app_id"],
                "project_id": scope["project_id"],
                "session_id": session,
                "messages": [
                    {
                        "sender_id": scope["agent_id"],
                        "role": "assistant",
                        "timestamp": stamp,
                        "content": content,
                    }
                ],
            }
            # deterministic id + timestamp survives a repeated memory_save tool call
            with self.store.db() as db:
                old = db.execute(
                    "SELECT id,status,error FROM jobs WHERE id=?",
                    (digest([key, session, "capture", save_key]),),
                ).fetchone()
            job = (
                dict(old)
                if old
                else self.store.enqueue(key, session, "capture", payload, save_key)
            )
            self.store.enqueue(
                key,
                session,
                "flush",
                {
                    "app_id": scope["app_id"],
                    "project_id": scope["project_id"],
                    "session_id": session,
                    "messages": [],
                },
                save_key,
            )
            return {
                "accepted": True,
                "job": job,
                "note": "Queued for EverOS extraction; not yet a verified or indexed Skill.",
            }
        if name == "run_consolidation":
            pending = [x for x in self.store.jobs(key) if x["status"] == "pending"]
            if args.get("dry_run"):
                return {
                    "pending_jobs": pending,
                    "backend": "EverOS extraction and skill clustering",
                }
            job = self.store.enqueue(
                key,
                session,
                "flush",
                {
                    "app_id": scope["app_id"],
                    "project_id": scope["project_id"],
                    "session_id": session,
                    "messages": [],
                },
                str(time.time_ns()),
            )
            return {
                "job": job,
                "note": "Flush current session; EverOS downstream strategies perform consolidation. Other projects are untouched.",
            }
        items = await catalog(self.root, scope)
        flags = self.store.flags(key)
        now = time.time()
        active = [
            x
            for x in items
            if not flags.get(x["id"], {}).get("inactive")
            and flags.get(x["id"], {}).get("snooze_until", 0) <= now
        ]
        if scope.get("evaluation_mode"):
            active = [x for x in active if not contaminated(x["body"])]
        if name == "get":
            return {"document": resolve(active, args["path"])}
        if name == "multi_get":
            patterns = [p.strip() for p in str(args["paths"]).split(",") if p.strip()]
            matches = [
                x
                for x in active
                if any(
                    fnmatch.fnmatchcase(x["path"], p) or x["id"] == p for p in patterns
                )
            ]
            return {"documents": matches[:100], "truncated": len(matches) > 100}
        if name in ("memory_forget", "memory_pin", "memory_snooze"):
            query = str(args.get("query", "")).strip()
            try:
                item = resolve(items, query)
            except ValueError:
                hits = await self._search(scope, query, "keyword", 3)
                if len(hits["results"]) != 1:
                    return {
                        "changed": False,
                        "reason": "Use exact id/path to select a single memory",
                        "candidates": hits["results"],
                    }
                item = hits["results"][0]
            if name == "memory_forget" and not args.get("confirm", False):
                return {"changed": False, "preview": item, "confirm_required": True}
            if name == "memory_forget":
                field, value = "inactive", 1
            elif name == "memory_pin":
                field, value = "pinned", int(not args.get("unpin", False))
            else:
                days = float(args.get("days", 30))
                if not 0 <= days <= 36500:
                    raise ValueError("days must be between 0 and 36500")
                field, value = "snooze_until", now + days * 86400
            self.store.flag(key, item["id"], field, value)
            self.store.audit(key, session, "control", {"id": item["id"], field: value})
            return {"changed": True, "id": item["id"], field: value}
        if name in ("graph_search", "trace_knowledge"):
            seeds = (
                args.get("seed_ids", [])
                if name == "graph_search"
                else [args["node_id"]]
            )
            if not seeds:
                raise ValueError("Supply EverOS ids from memory_retrieve")
            by_id = {x["id"]: x for x in active}
            adjacency = {i: set() for i in by_id}
            for x in active:
                for source in x["sources"]:
                    matches = [
                        v["id"]
                        for v in active
                        if source in (v["id"], v.get("entry_id"))
                    ]
                    for y in matches:
                        adjacency[x["id"]].add(y)
                        adjacency[y].add(x["id"])
            frontier = {resolve(active, str(seed))["id"] for seed in seeds}
            visited = set()
            edges = []
            hops = (
                max(
                    0,
                    min(int(args.get("max_hops", 3)), int(args.get("max_steps", 5)), 6),
                )
                if name == "graph_search"
                else 6
            )
            decay = float(args.get("decay_factor", 0.5))
            minimum = float(args.get("min_confidence", 0.3))
            if not 0 < decay <= 1 or not 0 <= minimum <= 1:
                raise ValueError("Invalid graph thresholds")
            results = []
            for depth in range(hops + 1):
                next_set = set()
                for ident in sorted(frontier - visited):
                    visited.add(ident)
                    strength = decay**depth
                    if name == "graph_search" and strength < minimum:
                        continue
                    results.append(
                        {**by_id[ident], "hop": depth, "path_weight": strength}
                    )
                    for other in adjacency[ident]:
                        edges.append(
                            {"from": ident, "to": other, "relation": "provenance"}
                        )
                        next_set.add(other)
                frontier = next_set
            response = {
                "nodes": results,
                "edges": edges,
                "semantics": "EverOS provenance graph; path weight is not verification confidence",
                "legacy_note": "ClawMem numeric node IDs and Hebbian graph are not reused; use EverOS string IDs.",
            }
            if name == "trace_knowledge":
                response["skill_history"] = {
                    x["id"]: self._skill_history(scope, x)
                    for x in results
                    if x["kind"] == "agent_skill"
                }
            return response
        if name == "generate_templates":
            if args.get("force"):
                raise ValueError(
                    "force regeneration is unavailable; existing EverOS Skills are the template source"
                )
            domain = str(args.get("domain") or "").strip()
            skills = [
                x
                for x in active
                if x["kind"] == "agent_skill"
                and (
                    not domain
                    or domain.casefold() in (x["title"] + " " + x["body"]).casefold()
                )
            ]
            return {
                "domain": domain or "current project",
                "templates": [
                    {
                        "source_id": x["id"],
                        "content": x["body"],
                        "source_case_ids": x["sources"],
                    }
                    for x in skills
                ],
                "note": "Rendered from existing EverOS Skills; no second extractor or fabricated cold-start template.",
            }
        raise ValueError(f"Unknown operation: {name}")

    async def event(self, name, args, scope):
        validate_scope(scope)
        key = scope_id(scope)
        session = scope["session_id"]
        if name == "capture":
            from everos.entrypoints.api.routes.memorize import MemorizeAddRequest

            raw = {
                **args["payload"],
                "app_id": scope["app_id"],
                "project_id": scope["project_id"],
                "session_id": session,
                "defer_extraction": True,
            }
            payload = MemorizeAddRequest.model_validate(raw).model_dump(
                exclude={"defer_extraction"}
            )
            for m in payload["messages"]:
                if m["sender_id"] not in (scope["user_id"], scope["agent_id"]):
                    raise ValueError("Sender outside current scope")
                if isinstance(m["content"], str):
                    m["content"] = clean_text(m["content"])
            return self.store.enqueue(key, session, "capture", payload, args["key"])
        if name == "seal":
            return self.store.enqueue(
                key,
                session,
                "flush",
                {
                    "app_id": scope["app_id"],
                    "project_id": scope["project_id"],
                    "session_id": session,
                    "messages": [],
                },
                args["key"],
            )
        if name == "checkpoint":
            return {
                "checkpoint": self.store.checkpoint(key, session, args.get("payload"))
            }
        if name == "audit":
            allowed = (
                "session_start",
                "prompt",
                "tool",
                "injected",
                "quality",
                "session_end",
                "adoption_candidate",
                "validation_candidate",
            )
            if args["kind"] not in allowed:
                raise ValueError("Invalid audit kind")
            self.store.audit(
                key, session, args["kind"], args.get("payload", {}), args.get("key")
            )
            return {"ok": True}
        raise ValueError("Unknown event")

    async def worker(self):
        from everos.service import memorize

        self.store.recover()
        while True:
            job = self.store.claim()
            if job is None:
                await asyncio.sleep(0.25)
                continue
            try:
                result = await memorize(
                    job["payload"],
                    is_final=job["kind"] == "flush",
                    defer_extraction=job["kind"] == "capture",
                )
                self.store.finish(job["id"], "done")
                self.store.audit(
                    job["scope"],
                    job["session"],
                    "job_completed",
                    {"id": job["id"], "result": result.model_dump()},
                )
            except asyncio.CancelledError:
                self.store.finish(
                    job["id"], "uncertain", "Worker stopped while executing"
                )
                raise
            except Exception as exc:
                self.store.finish(job["id"], "failed", f"{type(exc).__name__}: {exc}")

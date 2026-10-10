import sys
import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from unified_memory.service import UnifiedService, TOOLS
from unified_memory.catalog import catalog, scope_id
from everos.memory.search import SearchMethod

S = {
    "app_id": "app",
    "project_id": "project",
    "user_id": "user",
    "agent_id": "agent",
    "session_id": "session",
}


@pytest.fixture
def service(tmp_path):
    base = tmp_path / "app/project"
    files = {
        "users/user/.episodes/episode-2026-10-10.md": """---
type: episode_daily
---
<!-- entry:ep_20261010_00000001 -->
**parent_id**: cell1
### Subject
Repair a test
### Content
The test failed, a fix was applied, and the test passed.
<!-- /entry:ep_20261010_00000001 -->
""",
        "agents/agent/.cases/agent_case-2026-10-10.md": """---
type: agent_case_daily
---
<!-- entry:ac_20261010_00000001 -->
**parent_id**: cell1
**quality_score**: 0.9
### TaskIntent
Repair a test
### Approach
Run test, inspect error, apply fix, rerun test.
<!-- /entry:ac_20261010_00000001 -->
""",
        "agents/agent/skills/skill_debug/SKILL.md": """---
type: agent_skill
name: debug
source_case_ids: [ac_20261010_00000001]
confidence: 0.9
---
Run the failing test. Inspect evidence. Repair and verify.
""",
    }
    for name, body in files.items():
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    async def search(req):
        items = await catalog(tmp_path, S)
        data = {"episodes": [], "agent_cases": [], "agent_skills": []}
        for x in items:
            if (x["kind"] == "episode") != (req.user_id is not None):
                continue
            name = {
                "episode": "episodes",
                "agent_case": "agent_cases",
                "agent_skill": "agent_skills",
            }[x["kind"]]
            data[name].append({"id": x["id"], "score": 0.9})
        return SimpleNamespace(model_dump=lambda **_: {"data": data})

    return UnifiedService(tmp_path, search_fn=search)


@pytest.mark.asyncio
async def test_catalog_and_provenance(service):
    results = await service.call("memory_retrieve", {"query": "test"}, S)
    assert len(results["results"]) == 3
    doc = await service.call("get", {"path": "agent_ac_20261010_00000001"}, S)
    assert "apply fix" in doc["document"]["body"]
    trace = await service.call("trace_knowledge", {"node_id": "agent_debug"}, S)
    assert {x["kind"] for x in trace["nodes"]} == {
        "agent_skill",
        "agent_case",
        "episode",
    }
    assert len(trace["edges"]) >= 2


@pytest.mark.asyncio
async def test_trace_exposes_prior_skill_versions(service):
    history = (
        service.root
        / ".index/skill-history/app/project/agent/debug"
    )
    history.mkdir(parents=True)
    previous = history / ("a" * 64 + ".md")
    previous.write_text("Prior instruction version")
    trace = await service.call("trace_knowledge", {"node_id": "agent_debug"}, S)
    assert trace["skill_history"]["agent_debug"] == [
        {"sha256": "a" * 64, "content": "Prior instruction version", "truncated": False}
    ]


@pytest.mark.asyncio
async def test_control_flags_apply_to_tools_and_auto_recall(service):
    result = await service.call("memory_forget", {"query": "agent_debug"}, S)
    assert result["confirm_required"]
    await service.call("memory_pin", {"query": "agent_debug"}, S)
    assert (await service.call("search", {"query": "test"}, S))["results"][0][
        "id"
    ] == "agent_debug"
    await service.call("memory_forget", {"query": "agent_debug", "confirm": True}, S)
    assert "agent_debug" not in {
        x["id"]
        for x in (await service.call("memory_retrieve", {"query": "test"}, S))[
            "results"
        ]
    }
    with pytest.raises(ValueError):
        await service.call("get", {"path": "agent_debug"}, S)
    await service.call("memory_snooze", {"query": "agent_ac_20261010_00000001"}, S)
    assert len((await service.call("search", {"query": "test"}, S))["results"]) == 1
    await service.call(
        "memory_snooze", {"query": "agent_ac_20261010_00000001", "days": 0}, S
    )
    assert len((await service.call("search", {"query": "test"}, S))["results"]) == 2


@pytest.mark.asyncio
async def test_project_scope_cannot_read_other_project(service):
    scope = {**S, "project_id": "another"}
    assert await catalog(service.root, scope) == []
    with pytest.raises(ValueError):
        await service.call("get", {"path": "agent_debug"}, scope)
    result = await service.call("memory_retrieve", {"query": "test"}, scope)
    assert result["results"] == []


@pytest.mark.asyncio
async def test_project_symlink_cannot_redirect_catalog(service):
    outside = service.root / "other_data"
    target = outside / "agents" / "agent" / "skills" / "skill_secret" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("---\ntype: agent_skill\nname: secret\n---\nSecret")
    (service.root / "app" / "another").symlink_to(outside, target_is_directory=True)
    scope = {**S, "project_id": "another"}
    assert await catalog(service.root, scope) == []


@pytest.mark.asyncio
async def test_memory_save_is_durable_candidate_not_second_document_store(service):
    args = {"title": "Test fix", "body": "Run tests after repair."}
    first = await service.call("memory_save", args, S)
    second = await service.call("memory_save", args, S)
    assert first["job"]["id"] == second["job"]["id"]
    assert len(service.store.jobs(scope_id(S))) == 2
    service.store.finish(first["job"]["id"], "done")
    third = await service.call("memory_save", args, S)
    assert third["job"]["status"] == "done"
    assert len(await catalog(service.root, S)) == 3
    with service.store.db() as db:
        names = {
            x[0]
            for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert not names & {"documents", "semantic_memory", "semantic_edges"}


@pytest.mark.asyncio
async def test_all_17_tools_execute_real_paths(service):
    examples = {
        "memory_retrieve": {"query": "test"},
        "search": {"query": "test"},
        "vsearch": {"query": "test"},
        "get": {"path": "agent_debug"},
        "multi_get": {"paths": "agents/*/skills/*/SKILL.md"},
        "status": {},
        "session_log": {},
        "memory_forget": {"query": "agent_debug"},
        "memory_pin": {"query": "agent_debug"},
        "memory_snooze": {"query": "agent_debug", "days": 0},
        "memory_compact": {},
        "memory_save": {"title": "Fix", "body": "A new observed result"},
        "trajectory_search": {"query": "test"},
        "run_consolidation": {"dry_run": True},
        "graph_search": {"query": "test", "seed_ids": ["agent_debug"]},
        "trace_knowledge": {"node_id": "agent_debug"},
        "generate_templates": {},
    }
    assert set(examples) == set(TOOLS)
    for name, args in examples.items():
        result = await service.call(name, args, S)
        assert isinstance(result, dict), name


@pytest.mark.asyncio
async def test_evaluation_pollution_rejected(service):
    result = await service.call(
        "memory_save",
        {"title": "GAIA ground truth", "body": "FINAL ANSWER: 42"},
        {**S, "evaluation_mode": True},
    )
    assert result["accepted"] is False
    assert not service.store.jobs(scope_id(S))


@pytest.mark.asyncio
async def test_normal_uuid_is_not_treated_as_benchmark_answer(service):
    result = await service.call(
        "memory_save",
        {"title": "Issue identifier", "body": "Trace 123e4567-e89b-12d3-a456-426614174000 reproduced the timeout."},
        {**S, "evaluation_mode": True},
    )
    assert result["accepted"] is True


@pytest.mark.asyncio
async def test_unsupported_legacy_write_options_fail_explicitly(service):
    with pytest.raises(ValueError, match="Legacy memory_save path"):
        await service.call(
            "memory_save", {"title": "Fix", "body": "Body", "path": "notes/fix.md"}, S
        )
    with pytest.raises(ValueError, match="force regeneration"):
        await service.call("generate_templates", {"force": True}, S)


@pytest.mark.asyncio
async def test_explicit_working_state_is_saved_for_resume(service):
    result = await service.call(
        "memory_compact",
        {
            "goal": "Repair timeout",
            "completed_steps": ["Reproduced failure"],
            "pending_steps": ["Check retry path"],
            "verification": "Test still failing",
        },
        S,
    )
    saved = service.store.checkpoint(scope_id(S), S["session_id"])
    assert result["checkpoint"] == saved
    assert saved["pending_steps"] == ["Check retry path"]


@pytest.mark.asyncio
async def test_capture_and_flush_worker_are_durable_and_idempotent(service, monkeypatch):
    from everos import service as everos_service

    calls = []

    async def memorize(payload, *, is_final, defer_extraction):
        calls.append((payload, is_final, defer_extraction))
        return SimpleNamespace(model_dump=lambda: {"status": "extracted"})

    monkeypatch.setattr(everos_service, "memorize", memorize)
    payload = {
        "messages": [
            {
                "sender_id": S["user_id"],
                "role": "user",
                "timestamp": int(time.time() * 1000),
                "content": "Please repair the timeout.",
            }
        ]
    }
    first = await service.event("capture", {"payload": payload, "key": "capture-1"}, S)
    repeat = await service.event("capture", {"payload": payload, "key": "capture-1"}, S)
    assert first["id"] == repeat["id"]
    await service.event("seal", {"key": "seal-1"}, S)
    worker = asyncio.create_task(service.worker())
    try:
        async def completed():
            while any(x["status"] != "done" for x in service.store.jobs(scope_id(S))):
                await asyncio.sleep(0.05)

        await asyncio.wait_for(completed(), timeout=3)
    finally:
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
    assert [(final, deferred) for _, final, deferred in calls] == [
        (False, True),
        (True, False),
    ]
    with service.store.db() as db:
        assert all(
            row[0] == "{}" for row in db.execute("SELECT payload FROM jobs")
        )


@pytest.mark.asyncio
async def test_hybrid_fallback_uses_valid_keyword_enum(service):
    calls = []

    async def search(req):
        calls.append(req.method)
        assert isinstance(req.method, SearchMethod)
        if req.method == SearchMethod.HYBRID:
            raise RuntimeError("reranker unavailable")
        return SimpleNamespace(
            model_dump=lambda **_: {"data": {"episodes": [], "agent_cases": [], "agent_skills": []}}
        )

    service.search_fn = search
    result = await service.call("memory_retrieve", {"query": "test"}, S)
    assert calls == [
        SearchMethod.HYBRID,
        SearchMethod.KEYWORD,
        SearchMethod.HYBRID,
        SearchMethod.KEYWORD,
    ]
    assert len(result["warnings"]) == 2


@pytest.mark.asyncio
async def test_mcp_exports_only_original_17_names():
    from unified_memory.mcp_server import mcp

    tools = await mcp.list_tools()
    assert {t.name for t in tools} == set(TOOLS)

import importlib.util
import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "hooks" / "lib")]
from unified_memory.store import ControlStore
from unified_memory.catalog import validate_scope
from identity import project_id_of


def test_idempotent_queue_survives_reopen(tmp_path):
    store = ControlStore(tmp_path)
    data = {"messages": ["same"]}
    first = store.enqueue("project", "session", "capture", data, "key")
    second = ControlStore(tmp_path).enqueue(
        "project", "session", "capture", data, "key"
    )
    assert first["id"] == second["id"]
    assert len(store.jobs("project")) == 1
    job = store.claim()
    store.finish(job["id"], "done")
    assert (
        store.enqueue("project", "session", "capture", data, "key")["status"] == "done"
    )
    with pytest.raises(ValueError):
        store.enqueue("project", "session", "capture", {"different": 1}, "key")


def test_uncertain_extraction_blocks_following_session_jobs(tmp_path):
    s = ControlStore(tmp_path)
    s.enqueue("p", "s", "capture", {}, "a")
    s.claim()
    s.recover()
    s.enqueue("p", "s", "flush", {}, "b")
    assert s.claim() is None
    other = s.enqueue("p", "s2", "capture", {}, "c")
    assert s.claim()["id"] == other["id"]
    assert s.jobs("p")[-1]["status"] == "uncertain"


def test_scope_isolation_and_path_validation(tmp_path):
    s = ControlStore(tmp_path)
    s.flag("a", "same", "inactive", 1)
    assert not s.flags("b")
    s.checkpoint("a", "s", {"goal": "A"})
    assert s.checkpoint("b", "s") is None
    assert project_id_of("/repo/a", None) != project_id_of("/repo/b", None)
    assert project_id_of("/repo/a", "shared") == project_id_of("/repo/b", "shared")
    with pytest.raises(ValueError):
        validate_scope({"app_id": "../x"})


def test_audit_deduplication(tmp_path):
    s = ControlStore(tmp_path)
    s.audit("a", "s", "tool", {"x": 1}, "event")
    s.audit("a", "s", "tool", {"x": 1}, "event")
    assert len(s.events("a")) == 1
    assert not s.events("b")


def test_render_complete_objects_and_budget():
    spec = importlib.util.spec_from_file_location(
        "unified_hook", ROOT / "hooks/unified.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    items = [
        {"id": str(i), "kind": "agent_case", "path": f"p/{i}", "body": "中文" * 1000}
        for i in range(4)
    ]
    text, ids = module.render(items, 7000)
    assert len(ids) == 1 and len(text.encode()) <= 7000
    assert text.count("<unified_memory>") == 1
    assert items[0]["body"] in text


def test_original_tool_names_exactly_preserved():
    from unified_memory.service import TOOLS

    assert len(TOOLS) == len(set(TOOLS)) == 17
    assert not {"memory_feedback", "memory_checkpoint", "memory_reindex"} & set(TOOLS)
    hooks = json.loads((ROOT / "hooks/hooks.json").read_text())["hooks"]
    assert set(hooks) == {
        "SessionStart",
        "UserPromptSubmit",
        "PostToolUse",
        "PreCompact",
        "Stop",
        "SessionEnd",
    }
    for entries in hooks.values():
        assert len(entries) == 1 and len(entries[0]["hooks"]) == 1
        assert "unified.py" in entries[0]["hooks"][0]["command"]


def test_memory_lookup_is_not_counted_as_external_evidence():
    spec = importlib.util.spec_from_file_location(
        "unified_hook", ROOT / "hooks/unified.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.is_external_evidence("memory_retrieve")
    assert not module.is_external_evidence("mcp__unified-memory__search")
    assert module.is_external_evidence("Bash")
    assert module.is_external_evidence("mcp__browser__search")

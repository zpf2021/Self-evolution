#!/usr/bin/env python3
"""Isolated real Markdown -> LanceDB -> unified-tool smoke check.

Run only against a fresh /tmp/unified-e2e-* root initialized with `everos init`.
The embedding endpoint and fingerprint must be supplied through EVEROS_EMBEDDING__*.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from everos.core.persistence import MemoryRoot
from everos.infra.persistence.markdown.mds import AgentSkillFrontmatter
from everos.infra.persistence.markdown.writers.agent_case_writer import AgentCaseWriter
from everos.infra.persistence.markdown.writers.agent_skill_writer import AgentSkillWriter
from everos.infra.persistence.markdown.writers.episode_writer import EpisodeWriter
from unified_memory.service import UnifiedService


SCOPE = {
    "app_id": "smoke_app",
    "project_id": "smoke_project",
    "user_id": "smoke_user",
    "agent_id": "smoke_agent",
    "session_id": "smoke_session",
}


async def seed(root: Path) -> None:
    memory = MemoryRoot(root)
    timestamp = datetime.now(timezone.utc).isoformat()
    common = {"session_id": SCOPE["session_id"], "timestamp": timestamp, "parent_id": "smoke_memcell"}
    episode = EpisodeWriter(memory)
    await episode.append_entry(
        SCOPE["user_id"],
        inline=common,
        sections={
            "Subject": "Frobnicator timeout repair",
            "Content": "The frobnicator timed out. Inspecting the retry log revealed a stale cursor. Resetting the cursor fixed the timeout.",
        },
        app_id=SCOPE["app_id"],
        project_id=SCOPE["project_id"],
    )
    case = AgentCaseWriter(memory)
    case_id = await case.append_entry(
        SCOPE["agent_id"],
        inline={**common, "quality_score": 0.9},
        sections={
            "TaskIntent": "Repair frobnicator timeout",
            "Approach": "Inspect retry log, reset stale cursor, rerun timeout check.",
        },
        app_id=SCOPE["app_id"],
        project_id=SCOPE["project_id"],
    )
    skill = AgentSkillWriter(memory)
    await skill.write_main(
        SCOPE["agent_id"],
        "fix_frobnicator",
        frontmatter=AgentSkillFrontmatter(
            id="smoke_agent_fix_frobnicator",
            agent_id=SCOPE["agent_id"],
            name="fix_frobnicator",
            description="Repair frobnicator timeout after a stale cursor",
            confidence=0.9,
            maturity_score=0.9,
            source_case_ids=[f"{SCOPE['agent_id']}_{case_id.format()}"],
        ),
        body="Inspect the retry log, reset the stale cursor, and rerun the timeout check.",
        app_id=SCOPE["app_id"],
        project_id=SCOPE["project_id"],
    )


async def check(root: Path) -> None:
    service = UnifiedService(root)
    status = await service.call("status", {}, SCOPE)
    assert status["memory_count"] == 3, status
    assert status["embedding"]["dimension"] == 768, status
    for name in ("search", "vsearch", "memory_retrieve"):
        result = await service.call(name, {"query": "frobnicator timeout", "limit": 10}, SCOPE)
        assert result["results"], (name, result)
        assert any(x["kind"] == "agent_case" for x in result["results"]), (name, result)
        print(f"{name}: {[(x['id'], round(x['score'], 3)) for x in result['results']]}")
    trace = await service.call(
        "trace_knowledge", {"node_id": "smoke_agent_fix_frobnicator"}, SCOPE
    )
    assert {x["kind"] for x in trace["nodes"]} == {
        "episode", "agent_case", "agent_skill"
    }, trace
    isolated = await service.call(
        "memory_retrieve",
        {"query": "frobnicator timeout"},
        {**SCOPE, "project_id": "other_project"},
    )
    assert not isolated["results"], isolated
    print("provenance, project isolation and all three search modes: ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--everos-bin", required=True)
    parser.add_argument("--skip-seed", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not str(root).startswith("/tmp/unified-e2e-") or not (root / "everos.toml").is_file():
        parser.error("Use an initialized /tmp/unified-e2e-* test root")
    if not os.environ.get("EVEROS_EMBEDDING__FINGERPRINT"):
        parser.error("Set EVEROS_EMBEDDING__FINGERPRINT for this test")
    os.environ["EVEROS_ROOT"] = str(root)
    if not args.skip_seed:
        asyncio.run(seed(root))
    subprocess.run(
        [args.everos_bin, "cascade", "--root", str(root), "rebuild", "--yes"],
        check=True,
        env=os.environ.copy(),
    )
    asyncio.run(check(root))


if __name__ == "__main__":
    main()

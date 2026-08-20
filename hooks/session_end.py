#!/usr/bin/env python3
"""SessionEnd hook — seal step, port of DSH's `session/disposed` -> runtime.seal().

Reads: {"reason","cwd","transcript_path","session_id",...} on stdin.

Captures any trailing transcript lines not yet picked up by the last Stop call, then force-
flushes regardless of the locally tracked pending count (idempotent — EverOS returns
{"status": "no_extraction"} harmlessly if there is nothing to do, verified empirically), and
finally deletes the session's state file.

No separate step is needed to cancel background flush_watcher.py sleepers: once the state
file is deleted, a stale watcher's staleness check (pending_messages <= 0 / missing file)
makes it a no-op on its own — see flush_watcher.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from capture import capture_messages_since  # noqa: E402
from config import resolve_config  # noqa: E402
from everos_client import EverosClient, EverosError  # noqa: E402
from identity import agent_id_of, project_id_of, resolve_user_id, session_id_of  # noqa: E402
import state  # noqa: E402

ADD_MAX_MESSAGES = 500


class _Logger:
    def warn(self, message: str) -> None:
        print(message, file=sys.stderr)


def main() -> None:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        sys.exit(0)

    transcript_path = data.get("transcript_path", "") or ""
    cwd = data.get("cwd", "") or ""
    raw_session_id = data.get("session_id", "") or ""
    if not raw_session_id:
        sys.exit(0)

    config = resolve_config()
    logger = _Logger()
    project_id = project_id_of(cwd, config.project_id)
    session_id = session_id_of(raw_session_id)
    client = EverosClient(base_url=config.base_url, api_version=config.api_version)

    try:
        if transcript_path:
            before = state.load_state(session_id)
            items, new_cursor = capture_messages_since(
                transcript_path,
                before.get("cursor", 0),
                resolve_user_id(config.user_id),
                agent_id_of(config.agent_id),
                config.capture_max_chars,
            )
            if items:
                for i in range(0, len(items), ADD_MAX_MESSAGES):
                    batch = items[i : i + ADD_MAX_MESSAGES]
                    client.add(
                        {
                            "session_id": session_id,
                            "app_id": config.app_id,
                            "project_id": project_id,
                            "messages": batch,
                            "defer_extraction": True,
                        },
                        config.capture_timeout_ms / 1000,
                    )
            state.update_state(session_id, cursor=new_cursor)
    except Exception as exc:  # noqa: BLE001 — hooks must never crash the CC session
        logger.warn(f"everos-memory: final capture failed open: {exc}")

    try:
        client.flush(
            {"session_id": session_id, "app_id": config.app_id, "project_id": project_id},
            config.capture_timeout_ms / 1000,
        )
    except EverosError as exc:
        logger.warn(f"everos-memory: seal flush failed (ignored): {exc}")
    finally:
        state.delete_state(session_id)


if __name__ == "__main__":
    main()

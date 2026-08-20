#!/usr/bin/env python3
"""Stop hook — capture step, port of DSH's `agent/turn-stopping` + memory-runtime.ts's
capture()/armTimers() threshold logic.

Reads: {"transcript_path","cwd","session_id",...} on stdin.

1. Convert new transcript lines since the per-session cursor into EverOS MessageItem dicts
   (lib/capture.py) and POST them to /add with defer_extraction=true — cheap, no LLM call on
   EverOS's side (verified empirically: no chat/completions or embeddings HTTP calls appear
   in the server log for a defer_extraction=true request).
2. Update the per-session pending-buffer state (message/token counters, first-pending
   timestamp, epoch).
3. If the message/token threshold is already reached, flush immediately (no need to wait for
   a background watcher).
4. Otherwise arm two independent background watchers (see flush_watcher.py): an idle watcher
   that resets on every subsequent Stop call, and a max-delay watcher that is spawned once per
   buffer-open window and is not reset by ordinary activity.

Fails open: any capture/flush error is logged to stderr and swallowed — never blocks the
turn from completing, matching store_memory.py's proven behavior in ../../../everos_evolution.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from capture import capture_messages_since, estimate_message_tokens  # noqa: E402
from config import resolve_config  # noqa: E402
from everos_client import EverosClient, EverosError  # noqa: E402
from identity import agent_id_of, project_id_of, resolve_user_id, session_id_of  # noqa: E402
import state  # noqa: E402

ADD_MAX_MESSAGES = 500


class _Logger:
    def warn(self, message: str) -> None:
        print(message, file=sys.stderr)


def _spawn_watcher(session_id: str, project_id: str, delay_ms: int, check_field: str, check_value: str) -> None:
    script = str(Path(__file__).resolve().parent / "flush_watcher.py")
    subprocess.Popen(
        [
            sys.executable, script,
            "--session", session_id,
            "--project", project_id,
            "--delay-ms", str(delay_ms),
            "--check-field", check_field,
            "--check-value", check_value,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def main() -> None:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        sys.exit(0)

    transcript_path = data.get("transcript_path", "") or ""
    cwd = data.get("cwd", "") or ""
    raw_session_id = data.get("session_id", "") or ""
    if not transcript_path or not raw_session_id:
        sys.exit(0)

    config = resolve_config()
    logger = _Logger()
    project_id = project_id_of(cwd, config.project_id)
    session_id = session_id_of(raw_session_id)
    user_id = resolve_user_id(config.user_id)
    agent_id = agent_id_of(config.agent_id)
    client = EverosClient(base_url=config.base_url, api_version=config.api_version)

    try:
        before = state.load_state(session_id)
        cursor = before.get("cursor", 0)
        items, new_cursor = capture_messages_since(
            transcript_path, cursor, user_id, agent_id, config.capture_max_chars
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

        was_empty = before.get("pending_messages", 0) == 0
        added_tokens = sum(estimate_message_tokens(item) for item in items)
        now = time.time()
        after = state.update_state(
            session_id,
            cursor=new_cursor,
            pending_messages=before.get("pending_messages", 0) + len(items),
            pending_tokens=before.get("pending_tokens", 0) + added_tokens,
            first_pending_at=before.get("first_pending_at") if not (was_empty and items) else now,
            epoch=before.get("epoch", 0) + 1,
            project_id=project_id,
            app_id=config.app_id,
        )
    except Exception as exc:  # noqa: BLE001 — hooks must never crash the CC session
        logger.warn(f"everos-memory: capture failed open: {exc}")
        sys.exit(0)

    if after.get("pending_messages", 0) == 0:
        sys.exit(0)

    threshold_reached = (
        after["pending_messages"] >= config.flush_message_threshold
        or after["pending_tokens"] >= config.flush_token_threshold
    )
    if threshold_reached:
        try:
            client.flush(
                {"session_id": session_id, "app_id": config.app_id, "project_id": project_id},
                config.capture_timeout_ms / 1000,
            )
            state.update_state(session_id, pending_messages=0, pending_tokens=0, first_pending_at=None)
        except EverosError as exc:
            logger.warn(f"everos-memory: threshold flush failed (ignored): {exc}")
        return

    try:
        _spawn_watcher(session_id, project_id, config.flush_idle_ms, "epoch", str(after["epoch"]))
        if was_empty and items:
            _spawn_watcher(
                session_id, project_id, config.flush_max_delay_ms,
                "first_pending_at", str(after["first_pending_at"]),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warn(f"everos-memory: failed to arm flush timers (ignored): {exc}")


if __name__ == "__main__":
    main()

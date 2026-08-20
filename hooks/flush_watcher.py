#!/usr/bin/env python3
"""Background sleeper spawned by stop.py — the substitute for DSH's in-memory setTimeout
idle/max-delay flush timers (memory-runtime.ts's armTimers()).

Each Stop hook invocation is a fresh short-lived process, so nothing can hold a live
setTimeout across turns. Instead, stop.py spawns this script detached (new session, stdio
closed) with `--check-field/--check-value` set to whatever the state file held at spawn
time. After sleeping, it re-reads the state file: if the checked field has changed, some
newer activity (or an already-completed flush) superseded this watcher and it exits as a
no-op — equivalent to DSH's clearTimeout-on-new-activity. Otherwise it performs the flush
itself.

Two independent watchers are spawned per Stop call (see stop.py):
  --check-field epoch            -> idle watcher: cancelled by ANY new Stop call (epoch bumps
                                     every call), matching DSH's idleTimer reset-on-activity.
  --check-field first_pending_at -> max-delay watcher: only cancelled when the buffer is
                                     actually flushed (first_pending_at reset to null), NOT by
                                     ordinary new activity — matching DSH's maxTimer, which is
                                     armed once per buffer-open window and never reset by
                                     capture() the way idleTimer is.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from config import resolve_config  # noqa: E402
from everos_client import EverosClient, EverosError  # noqa: E402
import state  # noqa: E402


def _stale(check_field: str, check_value: str, current_state: dict) -> bool:
    current = current_state.get(check_field)
    if check_field == "epoch":
        return str(current) != check_value
    # first_pending_at: float comparison with tolerance; None means already flushed
    if current is None:
        return True
    try:
        return abs(float(current) - float(check_value)) > 1e-6
    except (TypeError, ValueError):
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--delay-ms", required=True, type=int)
    parser.add_argument("--check-field", required=True, choices=["epoch", "first_pending_at"])
    parser.add_argument("--check-value", required=True)
    args = parser.parse_args()

    time.sleep(max(0.0, args.delay_ms / 1000))

    current_state = state.load_state(args.session)
    if _stale(args.check_field, args.check_value, current_state):
        return  # superseded — nothing to do
    if current_state.get("pending_messages", 0) <= 0:
        return  # already flushed by something else

    config = resolve_config()
    client = EverosClient(base_url=config.base_url, api_version=config.api_version)
    try:
        client.flush(
            {"session_id": args.session, "app_id": config.app_id, "project_id": args.project},
            config.capture_timeout_ms / 1000,
        )
    except EverosError:
        return  # best-effort background maintenance; a later Stop/seal will retry
    state.update_state(args.session, pending_messages=0, pending_tokens=0, first_pending_at=None)


if __name__ == "__main__":
    main()

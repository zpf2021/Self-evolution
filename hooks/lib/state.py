#!/usr/bin/env python3
"""Per-session pending-buffer state, persisted to disk.

Claude Code hooks are short-lived one-shot processes (unlike DSH's long-running plugin host),
so there is no in-memory MemoryRuntime to hold counters or setTimeout timers across turns.
This module is the substitute: one small JSON file per session under
$CLAUDE_PLUGIN_DATA/state/, holding the transcript cursor, pending message/token counters,
and an "epoch" counter that background flush_watcher.py sleepers use to detect staleness
(equivalent to DSH's clearTimeout-on-new-activity).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    base = os.environ.get("CLAUDE_PLUGIN_DATA")
    root = Path(base) if base else Path.home() / ".everos_cc_plugin"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _state_dir() -> Path:
    d = data_dir() / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path(session_id: str) -> Path:
    return _state_dir() / f"{session_id}.json"


def _active_marker_path(project_id: str) -> Path:
    d = data_dir() / "active"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{project_id}.json"


@contextlib.contextmanager
def exclusive(path: Path):
    """Cross-process file lock on a sidecar .lock file next to `path`."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else dict(default)
    except (OSError, json.JSONDecodeError):
        return dict(default)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


DEFAULT_STATE: dict[str, Any] = {
    "cursor": 0,
    "pending_messages": 0,
    "pending_tokens": 0,
    "first_pending_at": None,
    "epoch": 0,
    "project_id": None,
    "app_id": None,
}


def load_state(session_id: str) -> dict[str, Any]:
    return _read_json(state_path(session_id), DEFAULT_STATE)


def save_state(session_id: str, state: dict[str, Any]) -> None:
    with exclusive(state_path(session_id)):
        _write_json(state_path(session_id), state)


def update_state(session_id: str, **updates: Any) -> dict[str, Any]:
    """Read-modify-write under the file lock; returns the merged state."""
    path = state_path(session_id)
    with exclusive(path):
        state = _read_json(path, DEFAULT_STATE)
        state.update(updates)
        _write_json(path, state)
        return state


def delete_state(session_id: str) -> None:
    path = state_path(session_id)
    with contextlib.suppress(OSError):
        path.unlink()
    with contextlib.suppress(OSError):
        path.with_suffix(path.suffix + ".lock").unlink()


def list_pending_sessions(project_id: str, exclude_session_id: str) -> list[tuple[str, dict[str, Any]]]:
    """Other sessions in the same project that still have an un-flushed buffer."""
    results: list[tuple[str, dict[str, Any]]] = []
    for path in _state_dir().glob("*.json"):
        session_id = path.stem
        if session_id == exclude_session_id:
            continue
        state = _read_json(path, DEFAULT_STATE)
        if state.get("project_id") == project_id and state.get("pending_messages", 0) > 0:
            results.append((session_id, state))
    return results


def read_active_session(project_id: str) -> str | None:
    data = _read_json(_active_marker_path(project_id), {})
    value = data.get("session_id")
    return value if isinstance(value, str) and value else None


def write_active_session(project_id: str, session_id: str) -> None:
    path = _active_marker_path(project_id)
    with exclusive(path):
        _write_json(path, {"session_id": session_id, "updated_at": time.time()})

"""Dependency-light transport shared by Hooks and MCP. No direct long-term writes."""

from __future__ import annotations
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_ROOT / "hooks" / "lib"))
from identity import project_id_of, resolve_user_id, agent_id_of, safe_id
from config import resolve_config


def scope_for(data=None):
    data = data or {}
    config = resolve_config()
    session = (
        os.environ.get("UNIFIED_SESSION_ID")
        or data.get("session_id")
        or os.environ.get("CLAUDE_SESSION_ID")
    )
    if not session:
        raise ValueError(
            "A session_id is required; configure UNIFIED_SESSION_ID for MCP"
        )
    return {
        "app_id": config.app_id,
        "project_id": project_id_of(data.get("cwd") or os.getcwd(), config.project_id),
        "session_id": safe_id(session, "session"),
        "user_id": resolve_user_id(config.user_id),
        "agent_id": agent_id_of(config.agent_id),
        "evaluation_mode": os.environ.get("UNIFIED_EVALUATION_MODE", "0") == "1",
    }


def call(path, scope, args=None, timeout=20):
    base = os.environ.get("UNIFIED_EVEROS_URL") or resolve_config().base_url
    request = urllib.request.Request(
        base.rstrip("/") + "/api/v2/unified/" + path,
        data=json.dumps(
            {"scope": scope, "arguments": args or {}}, ensure_ascii=False
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["data"]


def runtime_dir(scope):
    root = Path(
        os.environ.get("UNIFIED_RUNTIME_DIR")
        or os.environ.get("CLAUDE_PLUGIN_DATA")
        or Path.home() / ".unified-memory"
    )
    scoped = (
        root
        / "sessions"
        / hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    )
    scoped.mkdir(parents=True, exist_ok=True)
    return scoped

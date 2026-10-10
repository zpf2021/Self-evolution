#!/usr/bin/env python3
"""Stable, path-safe EverOS identity mapping — port of examples/dsh/src/identity.ts."""

from __future__ import annotations

import getpass
import hashlib
import os
import re
import unicodedata

_PATH_SAFE = re.compile(r"^[a-zA-Z0-9_.@+-]+$")
_RESERVED = {".", ".."}
_MAX_ID_CHARS = 128


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def safe_id(raw: str | None, fallback: str) -> str:
    """Keep readable identifiers while making collisions and path traversal unlikely."""
    source = (raw or "").strip() or fallback
    if len(source) <= _MAX_ID_CHARS and _PATH_SAFE.match(source) and source not in _RESERVED:
        return source

    stem = unicodedata.normalize("NFKD", source)
    stem = re.sub(r"[^a-zA-Z0-9_.@+-]+", "-", stem)
    stem = stem.strip(".-")
    suffix = _short_hash(source)
    readable = fallback if (not stem or stem in _RESERVED) else stem
    head = readable[: max(0, _MAX_ID_CHARS - len(suffix) - 1)]
    return f"{head}-{suffix}"


def session_id_of(session_id: str | None) -> str:
    return safe_id(session_id, "claude-code-session")


def project_id_of(cwd: str | None, configured: str | None) -> str:
    """Project-isolated by default; sharing requires an explicit project override."""
    if configured:
        return safe_id(configured, "default")
    canonical = os.path.realpath(cwd or os.getcwd())
    return safe_id(os.path.basename(canonical), "project")[:90] + "-" + _short_hash(canonical)



def agent_id_of(configured: str | None) -> str:
    return safe_id(configured, "claude-code")


def _os_username() -> str | None:
    try:
        return getpass.getuser().strip() or None
    except Exception:
        return None


def resolve_user_id(configured: str | None) -> str:
    raw = configured or os.environ.get("USER") or os.environ.get("USERNAME") or _os_username()
    return safe_id(raw, "local-user")

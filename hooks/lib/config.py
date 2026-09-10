#!/usr/bin/env python3
"""Plugin configuration — port of examples/dsh/src/config.ts's DEFAULTS + resolveConfig.

userConfig values declared in .claude-plugin/plugin.json are exposed to hook processes as
environment variables named CLAUDE_PLUGIN_OPTION_<UPPER_SNAKE_CASE_KEY> (e.g. baseUrl ->
CLAUDE_PLUGIN_OPTION_BASE_URL) — see Claude Code's plugins-reference.md. This module reads
those, falling back to the same defaults as the DSH reference plugin (minus autoStart/
startCommand/everosDir/readiness* — this plugin does not launch its own EverOS process; the
server is expected to already be running, matching the current pre-Docker phase of this
project).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

DEFAULTS = {
    "base_url": "http://127.0.0.1:8124",
    "api_version": "auto",
    "app_id": "claude-code",
    "agent_id": "claude-code",
    "recall_method": "hybrid",
    "enable_llm_rerank": True,
    "query_n": 3,
    "query_max_chars": 2_000,
    "recall_top_k": 5,
    "recall_episode_min_score": 0.85,
    "recall_case_min_score": 0.60,
    # Real recall against this deployment measured 15106 chars of combined content
    # (5 skills + 5 cases + 5 episodes + profile) against the old 12000-char DSH-inherited
    # default — the overflow silently ate whichever section rendered last. Section order is
    # now priority-sorted (skills/cases first, see render_memory) so overflow now trims the
    # least-critical content instead of the most-actionable. Round-4 GAIA memory measured
    # well above the original 12k default, especially once answer + feedback episodes are
    # both present. Keep a 100k hard ceiling while excluding user.md from this task-oriented
    # hook; this leaves room for recalled skills, cases, and evaluated episodes without the
    # unbounded 200k injection used during initial diagnosis.
    "recall_max_chars": 100_000,
    # hybrid+enable_llm_rerank makes a real LLM call for reranking (not just vector/BM25
    # fusion, which alone resolves in well under a second) — measured real-world latency
    # against this deployment's proxy: single calls have taken anywhere from ~8s to over
    # 20s. 5s (the DSH reference default, tuned for its plain "keyword" method with no LLM
    # call at all) silently truncated every hybrid+rerank recall in real testing. 20s per
    # attempt plus one retry, then fall back to "keyword" (see recall.py) — mirrors the
    # bounded-retry-then-fallback pattern already validated in ../../../everos_evolution's
    # inject_prompt.py for the same class of problem.
    "recall_timeout_ms": 20_000,
    "recall_retries": 2,
    "recall_fallback_method": "keyword",
    "recall_fallback_timeout_ms": 5_000,
    "capture_timeout_ms": 15_000,
    "capture_max_chars": 50_000,
    "flush_idle_ms": 30_000,
    "flush_token_threshold": 12_000,
    "flush_message_threshold": 50,
    "flush_max_delay_ms": 300_000,
    "flush_on_session_switch": True,
}

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _env_name(camel_key: str) -> str:
    snake = _CAMEL_RE.sub("_", camel_key).upper()
    return f"CLAUDE_PLUGIN_OPTION_{snake}"


def _read_str(camel_key: str, fallback: str) -> str:
    value = os.environ.get(_env_name(camel_key))
    return value.strip() if value and value.strip() else fallback


def _read_int(camel_key: str, fallback: int) -> int:
    value = os.environ.get(_env_name(camel_key))
    if not value or not value.strip():
        return fallback
    try:
        parsed = int(value)
    except ValueError:
        return fallback
    return parsed if parsed > 0 else fallback


def _read_float(camel_key: str, fallback: float | None) -> float | None:
    value = os.environ.get(_env_name(camel_key))
    if not value or not value.strip():
        return fallback
    try:
        parsed = float(value)
    except ValueError:
        return fallback
    return parsed if 0.0 <= parsed <= 1.0 else fallback


def _read_bool(camel_key: str, fallback: bool) -> bool:
    value = os.environ.get(_env_name(camel_key))
    if value is None or not value.strip():
        return fallback
    return value.strip().lower() in ("1", "true", "yes", "on")


def _normalize_base_url(raw: str) -> str:
    value = raw.strip() or DEFAULTS["base_url"]
    if not re.match(r"^https?://", value, re.IGNORECASE):
        value = f"http://{value}"
    return value.rstrip("/")


@dataclass(frozen=True)
class ResolvedConfig:
    base_url: str
    api_version: str
    app_id: str
    project_id: str | None
    user_id: str | None
    agent_id: str
    recall_method: str
    enable_llm_rerank: bool
    query_n: int
    query_max_chars: int
    recall_top_k: int
    recall_episode_min_score: float | None
    recall_case_min_score: float | None
    recall_max_chars: int
    recall_timeout_ms: int
    recall_retries: int
    recall_fallback_method: str
    recall_fallback_timeout_ms: int
    capture_timeout_ms: int
    capture_max_chars: int
    flush_idle_ms: int
    flush_token_threshold: int
    flush_message_threshold: int
    flush_max_delay_ms: int
    flush_on_session_switch: bool


def resolve_config() -> ResolvedConfig:
    project_id = _read_str("projectId", "") or None
    user_id = _read_str("userId", "") or None
    return ResolvedConfig(
        base_url=_normalize_base_url(_read_str("baseUrl", DEFAULTS["base_url"])),
        api_version=_read_str("apiVersion", DEFAULTS["api_version"]),
        app_id=_read_str("appId", DEFAULTS["app_id"]),
        project_id=project_id,
        user_id=user_id,
        agent_id=_read_str("agentId", DEFAULTS["agent_id"]),
        recall_method=_read_str("recallMethod", DEFAULTS["recall_method"]),
        enable_llm_rerank=_read_bool("enableLlmRerank", DEFAULTS["enable_llm_rerank"]),
        query_n=_read_int("queryN", DEFAULTS["query_n"]),
        query_max_chars=_read_int("queryMaxChars", DEFAULTS["query_max_chars"]),
        recall_top_k=_read_int("recallTopK", DEFAULTS["recall_top_k"]),
        recall_episode_min_score=_read_float(
            "recallEpisodeMinScore", DEFAULTS["recall_episode_min_score"]
        ),
        recall_case_min_score=_read_float(
            "recallCaseMinScore", DEFAULTS["recall_case_min_score"]
        ),
        recall_max_chars=_read_int("recallMaxChars", DEFAULTS["recall_max_chars"]),
        recall_timeout_ms=_read_int("recallTimeoutMs", DEFAULTS["recall_timeout_ms"]),
        recall_retries=_read_int("recallRetries", DEFAULTS["recall_retries"]),
        recall_fallback_method=_read_str("recallFallbackMethod", DEFAULTS["recall_fallback_method"]),
        recall_fallback_timeout_ms=_read_int(
            "recallFallbackTimeoutMs", DEFAULTS["recall_fallback_timeout_ms"]
        ),
        capture_timeout_ms=_read_int("captureTimeoutMs", DEFAULTS["capture_timeout_ms"]),
        capture_max_chars=_read_int("captureMaxChars", DEFAULTS["capture_max_chars"]),
        flush_idle_ms=_read_int("flushIdleMs", DEFAULTS["flush_idle_ms"]),
        flush_token_threshold=_read_int("flushTokenThreshold", DEFAULTS["flush_token_threshold"]),
        flush_message_threshold=_read_int(
            "flushMessageThreshold", DEFAULTS["flush_message_threshold"]
        ),
        flush_max_delay_ms=_read_int("flushMaxDelayMs", DEFAULTS["flush_max_delay_ms"]),
        flush_on_session_switch=_read_bool(
            "flushOnSessionSwitch", DEFAULTS["flush_on_session_switch"]
        ),
    )

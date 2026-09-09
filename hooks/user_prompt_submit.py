#!/usr/bin/env python3
"""UserPromptSubmit hook — recall step, port of examples/dsh's `agent/pre-step` handler.

Reads: {"prompt","cwd","session_id","transcript_path",...} on stdin.
Writes: {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
         "additionalContext": "<everos_memory>...</everos_memory>"}} on stdout.

Two responsibilities per DSH's index.ts apply():
  1. flushBeforeRecall — if the session id changed since the last prompt in this project,
     flush any other still-pending sessions first so their content is searchable.
  2. recallMessage — search user + agent tracks in parallel, inject a fenced memory block.

Fails open throughout: any EverOS/network error is caught and silently skipped, never
blocks the user's prompt (matches inject_prompt.py's proven behavior in ../../../everos_evolution).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from config import resolve_config  # noqa: E402
from everos_client import EverosClient, EverosError  # noqa: E402
from identity import agent_id_of, project_id_of, resolve_user_id, session_id_of  # noqa: E402
from recall import build_recall_query, recall_message  # noqa: E402
import state  # noqa: E402

# Character count, not whitespace-split word count: everos_evolution's original MIN_WORDS
# guard (`len(prompt.split()) < 3`) silently skips almost every CJK prompt, since a whole
# Chinese sentence with no spaces counts as exactly one "word" — this plugin is not tied to
# a single language, so the short-prompt skip has to be script-agnostic.
MIN_CHARS = 8

# Claude Code delivers background-agent completion notices through the same hook
# event as genuine user prompts. Recalling memory for those notices can trigger a
# redundant answer after the original task has already been completed.
_SYNTHETIC_PROMPT_PREFIXES = ("<task-notification>",)


def _is_synthetic_prompt(prompt: str) -> bool:
    normalized = prompt.lstrip().lower()
    return any(normalized.startswith(prefix) for prefix in _SYNTHETIC_PROMPT_PREFIXES)


class _Logger:
    def warn(self, message: str) -> None:
        print(message, file=sys.stderr)


def _flush_before_recall(client: EverosClient, config, project_id: str, current_session_id: str, logger) -> None:
    if not config.flush_on_session_switch:
        state.write_active_session(project_id, current_session_id)
        return
    active = state.read_active_session(project_id)
    state.write_active_session(project_id, current_session_id)
    if active is None or active == current_session_id:
        return
    for sid, st in state.list_pending_sessions(project_id, exclude_session_id=current_session_id):
        try:
            client.flush(
                {"session_id": sid, "app_id": config.app_id, "project_id": project_id},
                config.capture_timeout_ms / 1000,
            )
            state.update_state(sid, pending_messages=0, pending_tokens=0, first_pending_at=None)
        except EverosError as exc:
            logger.warn(f"everos-memory: session-switch flush failed (ignored) for {sid}: {exc}")


def main() -> None:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        sys.exit(0)

    prompt = data.get("prompt", "") or ""
    cwd = data.get("cwd", "") or ""
    raw_session_id = data.get("session_id", "") or ""
    transcript_path = data.get("transcript_path", "") or ""

    if _is_synthetic_prompt(prompt):
        sys.exit(0)

    config = resolve_config()
    logger = _Logger()
    project_id = project_id_of(cwd, config.project_id)
    session_id = session_id_of(raw_session_id)
    client = EverosClient(base_url=config.base_url, api_version=config.api_version)

    try:
        _flush_before_recall(client, config, project_id, session_id, logger)
    except Exception as exc:  # noqa: BLE001 — hooks must never crash the CC session
        logger.warn(f"everos-memory: session-switch flush failed open: {exc}")

    if len(prompt.strip()) < MIN_CHARS:
        sys.exit(0)  # silent skip for greetings/one-liners too short to carry a real query

    try:
        query = build_recall_query(transcript_path, prompt, config.query_n, config.query_max_chars)
        text = recall_message(
            client=client,
            query=query,
            app_id=config.app_id,
            project_id=project_id,
            user_id=resolve_user_id(config.user_id),
            agent_id=agent_id_of(config.agent_id),
            method=config.recall_method,
            top_k=config.recall_top_k,
            enable_llm_rerank=config.enable_llm_rerank,
            timeout_s=config.recall_timeout_ms / 1000,
            max_chars=config.recall_max_chars,
            logger=logger,
            retries=config.recall_retries,
            fallback_method=config.recall_fallback_method,
            fallback_timeout_s=config.recall_fallback_timeout_ms / 1000,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warn(f"everos-memory: recall failed open: {exc}")
        sys.exit(0)

    if not text:
        sys.exit(0)  # nothing relevant found, silent

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text,
        },
    }))


if __name__ == "__main__":
    main()

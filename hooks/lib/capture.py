#!/usr/bin/env python3
"""Convert new Claude Code transcript lines into EverOS MessageItem dicts.

Port of examples/dsh/src/capture.ts, but reading Claude Code's own transcript JSONL instead
of DSH's in-memory Session.events, and producing plain EverOS wire-format dicts instead of
DSH's typed MessageItem. The line-shape parsing (text / tool_use / tool_result blocks) mirrors
trajectory_to_memcell.py in ../../../everos_evolution, which is already verified against real
Claude Code transcripts in this environment — reused here rather than re-derived.

Raw line shapes handled:
  {"type": "user",      "message": {"role": "user", "content": str | list[block]}, "timestamp": iso, "uuid": str}
  {"type": "assistant", "message": {"role": "assistant", "content": list[block]}, "timestamp": iso, "uuid": str}
Content blocks: {"type": "text", "text": str}
                {"type": "tool_use", "id": str, "name": str, "input": dict}
                {"type": "tool_result", "tool_use_id": str, "content": str | list[block], "is_error": bool}
                {"type": "thinking", ...}  -> dropped, matches DSH excluding raw reasoning
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRUNCATE_MARKER = "\n[truncated by everos-memory]"


def _iso_to_epoch_ms(iso_ts: str) -> int:
    dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(_TRUNCATE_MARKER))
    return f"{text[:keep]}{_TRUNCATE_MARKER}"


def _bounded_arguments(arguments_text: str, max_chars: int) -> str:
    if len(arguments_text) <= max_chars:
        return arguments_text
    preview = arguments_text[: max(0, max_chars - 100)]
    return json.dumps({"everos_truncated": True, "preview": preview}, ensure_ascii=False)


def _block_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _convert_user_line(
    msg: dict[str, Any], timestamp_ms: int, user_id: str, agent_id: str, max_chars: int
) -> list[dict[str, Any]]:
    content = msg.get("content", "")
    if isinstance(content, str):
        text = _clip_text(content, max_chars)
        if not text:
            return []
        return [{"sender_id": user_id, "role": "user", "timestamp": timestamp_ms, "content": text}]

    items: list[dict[str, Any]] = []
    text_parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_result":
            result_text = _block_text(block.get("content", ""))
            status = "[tool error]\n" if block.get("is_error") else ""
            body = _clip_text(f"{status}{result_text}".strip(), max_chars) or "[empty tool result]"
            items.append(
                {
                    "sender_id": agent_id,
                    "sender_name": "tool",
                    "role": "tool",
                    "timestamp": timestamp_ms,
                    "content": body,
                    "tool_call_id": block.get("tool_use_id", ""),
                }
            )
    text = _clip_text("\n".join(text_parts), max_chars)
    if text:
        items.append({"sender_id": user_id, "role": "user", "timestamp": timestamp_ms, "content": text})
    return items


def _convert_assistant_line(
    msg: dict[str, Any], timestamp_ms: int, agent_id: str, max_chars: int
) -> list[dict[str, Any]]:
    content = msg.get("content", "")
    if isinstance(content, str):
        text = _clip_text(content, max_chars)
        if not text:
            return []
        return [
            {
                "sender_id": agent_id,
                "sender_name": "claude-code",
                "role": "assistant",
                "timestamp": timestamp_ms,
                "content": text,
            }
        ]

    text_parts = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            arguments = json.dumps(block.get("input", {}), ensure_ascii=False)
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _bounded_arguments(arguments, max_chars),
                    },
                }
            )
        # "thinking" blocks intentionally dropped — internal reasoning, not tool evidence.

    text = _clip_text("\n".join(text_parts), max_chars)
    if not text and not tool_calls:
        return []
    item: dict[str, Any] = {
        "sender_id": agent_id,
        "sender_name": "claude-code",
        "role": "assistant",
        "timestamp": timestamp_ms,
        "content": text,
    }
    if tool_calls:
        item["tool_calls"] = tool_calls
    return [item]


def estimate_message_tokens(item: dict[str, Any]) -> int:
    """Cheap deterministic estimate used only for deciding when to batch-flush."""
    content = item.get("content", "")
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    text += json.dumps(item.get("tool_calls", []), ensure_ascii=False) if item.get("tool_calls") else ""
    text += item.get("tool_call_id", "") or ""
    ascii_chars = sum(1 for ch in text if ord(ch) <= 0x7F)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, -(-ascii_chars // 4) + non_ascii_chars)


def capture_messages_since(
    transcript_path: str, cursor: int, user_id: str, agent_id: str, max_chars: int
) -> tuple[list[dict[str, Any]], int]:
    """Parse transcript lines after `cursor` into EverOS MessageItem dicts.

    Returns (messages, new_cursor). new_cursor is the total line count seen so far —
    unconditionally, whether or not a given line produced a message — so lines the parser
    doesn't understand (queue-operation/system/summary) still advance the cursor and are
    never re-scanned.
    """
    path = Path(transcript_path)
    if not path.exists():
        return [], cursor

    all_lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = all_lines[cursor:]
    if not new_lines:
        return [], cursor

    items: list[dict[str, Any]] = []
    last_ts = 0
    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        entry_type = entry.get("type")
        if entry_type not in ("user", "assistant"):
            continue
        msg = entry.get("message", {})
        ts_raw = entry.get("timestamp")
        timestamp_ms = _iso_to_epoch_ms(ts_raw) if isinstance(ts_raw, str) else last_ts
        last_ts = timestamp_ms
        if entry_type == "user":
            items.extend(_convert_user_line(msg, timestamp_ms, user_id, agent_id, max_chars))
        else:
            items.extend(_convert_assistant_line(msg, timestamp_ms, agent_id, max_chars))

    return items, len(all_lines)

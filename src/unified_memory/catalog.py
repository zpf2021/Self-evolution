"""Read full objects directly from EverOS Markdown, never clone a content database."""

from __future__ import annotations
import asyncio
import json
import re
from pathlib import Path


def scope_id(scope):
    return json.dumps(
        [scope["app_id"], scope["project_id"], scope["user_id"], scope["agent_id"]],
        separators=(",", ":"),
    )


def validate_scope(scope):
    for key in ("app_id", "project_id", "user_id", "agent_id", "session_id"):
        value = scope.get(key, "")
        if (
            not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z0-9_.@+-]{1,128}", value)
            or value in (".", "..")
        ):
            raise ValueError(f"Invalid {key}")
    return scope


def _load(root, scope):
    from everos.core.persistence.markdown import MarkdownReader

    app = "default_app" if scope["app_id"] == "default" else scope["app_id"]
    project = (
        "default_project" if scope["project_id"] == "default" else scope["project_id"]
    )
    base = root / app / project
    expected_base = root.resolve() / app / project
    if base.resolve() != expected_base:
        return []
    result = []
    groups = [("agents", scope["agent_id"]), ("users", scope["user_id"])]
    for group, owner in groups:
        directory = base / group / owner
        expected_directory = expected_base / group / owner
        if directory.resolve() != expected_directory:
            continue
        for path in sorted(directory.rglob("*.md")):
            if (
                not path.resolve().is_relative_to(expected_directory)
                or ".history" in path.parts
            ):
                continue
            parsed = MarkdownReader.parse(path.read_text(encoding="utf-8"))
            meta = parsed.frontmatter
            kind = meta.get("type", "")
            if kind == "agent_skill":
                ident = f"{owner}_{meta['name']}"
                refs = []
                for ref in sorted((path.parent / "references").glob("*.md")):
                    if ref.resolve().is_relative_to(expected_directory):
                        refs.append(ref.read_text(encoding="utf-8"))
                result.append(
                    dict(
                        id=ident,
                        kind="agent_skill",
                        owner=owner,
                        title=meta["name"],
                        body=parsed.body + "\n" + "\n".join(refs),
                        metadata=meta,
                        sources=meta.get("source_case_ids", []),
                        path=str(path.relative_to(base)),
                    )
                )
            elif kind in ("agent_case_daily", "episode_daily") or meta.get(
                "file_type"
            ) in ("agent_case_daily", "episode_daily"):
                record_kind = (
                    "agent_case"
                    if "case" in kind or "case" in str(meta.get("file_type"))
                    else "episode"
                )
                for entry in parsed.entries:
                    fields = entry.as_structured()
                    result.append(
                        dict(
                            id=f"{owner}_{entry.id}",
                            entry_id=entry.id,
                            kind=record_kind,
                            owner=owner,
                            title=fields.sections.get(
                                "TaskIntent", fields.sections.get("Subject", entry.id)
                            ),
                            body=entry.body,
                            metadata={**meta, **fields.inline},
                            sources=[],
                            path=str(path.relative_to(base)) + "#" + entry.id,
                        )
                    )
    # Case and Episode share the EverOS memcell parent_id; this is a real provenance edge.
    for item in result:
        parent = item["metadata"].get("parent_id")
        if item["kind"] == "agent_case" and parent:
            item["sources"] = [
                x["id"]
                for x in result
                if x["kind"] == "episode" and x["metadata"].get("parent_id") == parent
            ]
    return result


async def catalog(root: Path, scope):
    return await asyncio.to_thread(_load, root, validate_scope(scope))


def resolve(items, ident):
    matches = [
        x for x in items if str(ident) in (x["id"], x.get("entry_id"), x["path"])
    ]
    if len(matches) != 1:
        raise ValueError(
            "Memory identifier missing or ambiguous; use an exact id/path from search"
        )
    return matches[0]

"""Private loopback API for the unified plugin's original 17 tools and Hook events."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from everos.core.persistence import MemoryRoot
from unified_memory.service import UnifiedService, TOOLS

router = APIRouter(prefix="/unified", tags=["unified-memory"])
_service = None


def service():
    global _service
    if _service is None:
        _service = UnifiedService(MemoryRoot.resolve().root)
    return _service


class RequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: dict[str, Any]
    arguments: dict[str, Any] = Field(default_factory=dict)


@router.post("/tools/{name}")
async def tool(name: str, request: RequestBody):
    if name not in TOOLS:
        raise HTTPException(404, "Unknown tool")
    try:
        return {"data": await service().call(name, request.arguments, request.scope)}
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/events/{name}")
async def event(name: str, request: RequestBody):
    try:
        return {"data": await service().event(name, request.arguments, request.scope)}
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc

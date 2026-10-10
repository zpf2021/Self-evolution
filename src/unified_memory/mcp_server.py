"""Original 17 names retained. Numeric ClawMem graph ids become EverOS string ids."""

from __future__ import annotations
import asyncio
from mcp.server.fastmcp import FastMCP
from .client import call, scope_for

mcp = FastMCP("unified-memory")


async def invoke(name, args):
    return await asyncio.to_thread(call, "tools/" + name, scope_for(), args, 60)


@mcp.tool()
async def memory_retrieve(query: str, limit: int = 10, vault: str | None = None):
    """Search EverOS memories within the current project, with keyword fallback."""
    return await invoke("memory_retrieve", locals())


@mcp.tool()
async def search(query: str, limit: int = 10):
    """Keyword search within the current project's EverOS memory."""
    return await invoke("search", locals())


@mcp.tool()
async def vsearch(query: str, limit: int = 10):
    """Vector search; reports keyword degradation explicitly if unavailable."""
    return await invoke("vsearch", locals())


@mcp.tool()
async def get(path: str):
    """Read full memory by an exact EverOS id or path returned by search."""
    return await invoke("get", locals())


@mcp.tool()
async def multi_get(paths: str):
    """Read memories using comma-separated paths or glob patterns (up to 100)."""
    return await invoke("multi_get", locals())


@mcp.tool()
async def status():
    """Report EverOS scope, embedding identity and ingestion job outcomes."""
    return await invoke("status", {})


@mcp.tool()
async def session_log(limit: int = 10):
    """Read recent session audit history in this project."""
    return await invoke("session_log", locals())


@mcp.tool()
async def memory_forget(query: str, confirm: bool = False):
    """Preview or soft-disable a memory. Prefer an exact id; confirmation is required."""
    return await invoke("memory_forget", locals())


@mcp.tool()
async def memory_pin(query: str, unpin: bool = False):
    """Prioritize/deprioritize a relevant memory; never bypass scope or filtering."""
    return await invoke("memory_pin", locals())


@mcp.tool()
async def memory_snooze(query: str, days: float = 30):
    """Temporarily hide a memory. days=0 restores visibility."""
    return await invoke("memory_snooze", locals())


@mcp.tool()
async def memory_compact(
    reason: str = "routine",
    goal: str | None = None,
    completed_steps: list[str] | None = None,
    pending_steps: list[str] | None = None,
    verification: str | None = None,
):
    """Save current session checkpoint. Does not itself compact the host context."""
    args = locals()
    args["completed_steps"] = completed_steps or []
    args["pending_steps"] = pending_steps or []
    return await invoke("memory_compact", args)


@mcp.tool()
async def memory_save(
    title: str,
    body: str,
    path: str | None = None,
    contentType: str = "discovery",
    collection: str = "_clawmem",
):
    """Submit candidate evidence to EverOS. Completion/indexing is asynchronous; inspect status."""
    return await invoke("memory_save", locals())


@mcp.tool()
async def trajectory_search(query: str, limit: int = 10):
    """Search recent actual tool evidence in this project's audit log."""
    return await invoke("trajectory_search", locals())


@mcp.tool()
async def run_consolidation(dry_run: bool = False):
    """Flush current session into EverOS extraction/skill evolution, or inspect pending jobs."""
    return await invoke("run_consolidation", locals())


@mcp.tool()
async def graph_search(
    query: str,
    seed_ids: list[str],
    max_hops: int = 3,
    max_steps: int = 5,
    decay_factor: float = 0.5,
    min_confidence: float = 0.3,
):
    """Traverse actual EverOS provenance edges. Uses string ids; not ClawMem's Hebbian graph."""
    return await invoke("graph_search", locals())


@mcp.tool()
async def trace_knowledge(node_id: str):
    """Trace an EverOS memory's Case/Episode/Skill provenance using a string id."""
    return await invoke("trace_knowledge", locals())


@mcp.tool()
async def generate_templates(domain: str | None = None, force: bool = False):
    """Render current EverOS Skills as methodology templates; no separate LLM extraction."""
    return await invoke("generate_templates", locals())


if __name__ == "__main__":
    mcp.run()

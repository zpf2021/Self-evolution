from __future__ import annotations
import asyncio
from contextlib import suppress
from everos.core.lifespan import LifespanProvider


class UnifiedLifespanProvider(LifespanProvider):
    def __init__(self):
        super().__init__("unified-memory", order=1000)
        self.task = None

    async def startup(self, app):
        from everos.entrypoints.api.routes.unified import service

        self.task = asyncio.create_task(
            service().worker(), name="unified-memory-ingestion"
        )
        return {"worker": self.task}

    async def shutdown(self, app):
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task

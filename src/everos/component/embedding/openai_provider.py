"""OpenAI-compatible embedding provider.

Wraps :class:`openai.AsyncOpenAI` so any OpenAI-protocol endpoint
(DeepInfra, OpenAI, Together, Fireworks, …) works without per-provider
forks. When ``dimensions`` is set, the parameter is forwarded to the
API so MRL-capable models (OpenAI text-embedding-3-*, Qwen3-Embedding,
…) can do server-side truncation with proper re-normalization.
Responses must match the configured index width; invalid vectors are rejected.

Concurrency model:

- ``embed_batch`` splits the inputs into chunks of ``batch_size``.
- An :class:`asyncio.Semaphore` capped at ``max_concurrent`` bounds
  in-flight requests; remaining chunks queue and start as slots free.
- Retries / timeouts come from the openai SDK (``max_retries``,
  ``timeout`` constructor args).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

import openai

from everos.core.observability.tracing import memory_span, set_generation_usage

from .protocol import EmbeddingServiceError


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embedding provider with batching + concurrency.

    Args:
        model: Embedding model id (e.g. ``"Qwen/Qwen3-Embedding-4B"``).
        api_key: Bearer credential as a plain ``str``.
        base_url: OpenAI-protocol endpoint
            (e.g. ``"https://api.deepinfra.com/v1/openai"``).
        dim: Exact target vector dimension; mismatches are rejected.
        dimensions: Optional API-level ``dimensions`` parameter for
            MRL-capable models.  When set, forwarded to the embedding
            API for server-side truncation with re-normalization.
            When ``None``, the parameter is omitted from the request.
        timeout: Per-request timeout, seconds.
        max_retries: Retry budget exposed via the openai SDK.
        batch_size: How many inputs per ``/embeddings`` call.
        max_concurrent: Cap on in-flight chunked requests.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        dim: int = 1024,
        text_format: str = "plain",
        dimensions: int | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ) -> None:
        self.dim = dim
        self._text_format = text_format
        self._dimensions = dimensions
        self._model = model
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        vectors = await self._embed_chunk([self._format(text, "document")])
        return vectors[0]

    def _format(self, text: str, role: str) -> str:
        if self._text_format != "embeddinggemma":
            return text
        return f"task: search result | query: {text}" if role == "query" else f"title: none | text: {text}"

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed_chunk([self._format(text, "query")]))[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many strings, preserving input order."""
        if not texts:
            return []
        chunks = [
            [self._format(t, "document") for t in texts[i : i + self._batch_size]]
            for i in range(0, len(texts), self._batch_size)
        ]
        results = await asyncio.gather(*(self._embed_chunk(chunk) for chunk in chunks))
        # gather preserves order across awaitables, and each chunk preserves
        # its internal order — so flattening yields the input order back.
        return [vec for chunk in results for vec in chunk]

    async def _embed_chunk(self, chunk: list[str]) -> list[list[float]]:
        """One ``/embeddings`` call, semaphore-guarded."""
        # Wrap in an EMBEDDING-typed span so token usage lands on an embedding
        # observation (which Langfuse can price) rather than the enclosing
        # retriever span. nested_only: skip when there is no active trace (e.g.
        # cascade-time indexing) so we don't spawn one root trace per chunk.
        with memory_span(
            "everos.embedding", observation_type="embedding", nested_only=True
        ):
            async with self._semaphore:
                try:
                    response = await self._client.embeddings.create(
                        model=self._model,
                        input=chunk,
                        encoding_format="float",
                        dimensions=self._dimensions
                        if self._dimensions is not None
                        else openai.NOT_GIVEN,
                    )
                except openai.OpenAIError as exc:
                    raise EmbeddingServiceError(str(exc)) from exc
            if not response.data:
                raise EmbeddingServiceError(
                    f"Embedding API returned empty data for {len(chunk)} inputs"
                )
            # Embeddings report only input (prompt) tokens.
            usage = getattr(response, "usage", None)
            set_generation_usage(
                model=self._model,
                input_tokens=usage.prompt_tokens if usage else None,
            )
            rows = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in rows] != list(range(len(chunk))):
                raise EmbeddingServiceError("Embedding response count/index mismatch")
            vectors = [list(item.embedding) for item in rows]
            for vector in vectors:
                if len(vector) != self.dim or not all(math.isfinite(x) for x in vector):
                    raise EmbeddingServiceError(f"Expected {self.dim} finite dimensions, got {len(vector)}")
                if not any(vector):
                    raise EmbeddingServiceError("Embedding response is a zero vector")
            return vectors

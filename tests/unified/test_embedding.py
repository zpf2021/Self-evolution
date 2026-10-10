import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from everos.component.embedding.openai_provider import OpenAIEmbeddingProvider
from everos.component.embedding.protocol import EmbeddingServiceError
from unified_memory.fingerprint import ensure_index_identity


def provider():
    p = OpenAIEmbeddingProvider(
        model="local",
        api_key="local-only",
        base_url="http://localhost:8125/v1",
        dim=3,
        text_format="embeddinggemma",
    )
    p._client = SimpleNamespace(embeddings=SimpleNamespace(create=AsyncMock()))
    return p


def response(rows):
    return SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=v) for i, v in rows], usage=None
    )


@pytest.mark.asyncio
async def test_query_document_formats_and_response_order():
    p = provider()
    p._client.embeddings.create.return_value = response([(0, [1.0, 0.0, 0.0])])
    await p.embed_query("查询")
    assert p._client.embeddings.create.call_args.kwargs["input"] == [
        "task: search result | query: 查询"
    ]
    assert p._client.embeddings.create.call_args.kwargs["encoding_format"] == "float"
    await p.embed("文档")
    assert p._client.embeddings.create.call_args.kwargs["input"] == [
        "title: none | text: 文档"
    ]
    p._client.embeddings.create.return_value = response(
        [(1, [0.0, 1.0, 0.0]), (0, [1.0, 0.0, 0.0])]
    )
    assert await p.embed_batch(["a", "b"]) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [
        [(0, [1.0, 2.0])],
        [(0, [1.0, 2.0, 3.0, 4.0])],
        [(0, [float("nan"), 1.0, 0.0])],
        [(0, [0.0, 0.0, 0.0])],
        [(1, [1.0, 0.0, 0.0])],
    ],
)
async def test_never_silently_truncate_pad_or_accept_invalid_vector(rows):
    p = provider()
    p._client.embeddings.create.return_value = response(rows)
    with pytest.raises(EmbeddingServiceError):
        await p.embed("text")


def test_fingerprint_cannot_mix_same_width_models(tmp_path):
    a = SimpleNamespace(
        model="a",
        index_dimension=768,
        dimensions=None,
        text_format="embeddinggemma",
        fingerprint="sha-a",
    )
    ensure_index_identity(tmp_path, a)
    ensure_index_identity(tmp_path, a)
    a.fingerprint = "sha-b"
    with pytest.raises(ValueError):
        ensure_index_identity(tmp_path, a)


def test_existing_unidentified_index_is_not_adopted(tmp_path):
    (tmp_path / ".index/lancedb/episode.lance").mkdir(parents=True)
    a = SimpleNamespace(
        model="a",
        index_dimension=768,
        dimensions=None,
        text_format="embeddinggemma",
        fingerprint="sha-a",
    )
    with pytest.raises(ValueError):
        ensure_index_identity(tmp_path, a)


@pytest.mark.asyncio
async def test_mismatched_index_is_rejected_before_lancedb_opens(tmp_path, monkeypatch):
    from everos.core.persistence import MemoryRoot
    from everos.infra.persistence.lancedb import lancedb_manager

    embedding = SimpleNamespace(
        model="local",
        index_dimension=768,
        dimensions=None,
        text_format="embeddinggemma",
        fingerprint="first-model",
    )
    ensure_index_identity(tmp_path, embedding)
    embedding.fingerprint = "second-model"
    opened = AsyncMock()
    monkeypatch.setattr(lancedb_manager, "_conn", None)
    monkeypatch.setattr(
        lancedb_manager, "load_settings", lambda: SimpleNamespace(embedding=embedding)
    )
    monkeypatch.setattr(
        lancedb_manager.MemoryRoot, "resolve", lambda: MemoryRoot(tmp_path)
    )
    monkeypatch.setattr(lancedb_manager, "open_lancedb_connection", opened)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        await lancedb_manager.get_connection()
    opened.assert_not_called()

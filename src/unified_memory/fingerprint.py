"""An index must never mix embedding spaces, even when widths are equal."""

import json
from pathlib import Path


def expected(settings):
    return {
        "model": settings.model,
        "index_dimension": settings.index_dimension,
        "dimensions": settings.dimensions,
        "text_format": settings.text_format,
        "model_fingerprint": settings.fingerprint,
        "format_version": 2,
    }


def ensure_index_identity(root: Path, settings):
    if not settings.fingerprint:
        return  # legacy configurations require an explicit migration to fingerprint enforcement
    path = root / ".index" / "embedding.json"
    value = expected(settings)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(
                "Embedding fingerprint mismatch: rebuild into a NEW memory root; keyword-only until corrected"
            )
        return
    index = root / ".index" / "lancedb"
    if index.exists() and any(index.glob("*.lance")):
        raise ValueError(
            "Existing index has no fingerprint; migrate instead of adopting it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as f:
            json.dump(value, f, sort_keys=True)
    except FileExistsError:
        if json.loads(path.read_text()) != value:
            raise ValueError("Concurrent index fingerprint mismatch")

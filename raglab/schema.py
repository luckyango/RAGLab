"""Shared data shapes for RAGLab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved evidence chunk with source metadata."""

    content: str
    source: str
    relevance: float
    metadata: dict[str, Any]
    chunk_id: str = ""
    retrieval_method: str = "vector"

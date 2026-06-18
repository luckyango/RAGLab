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


@dataclass(frozen=True)
class Citation:
    """A compact citation derived from a retrieved chunk."""

    source: str
    chunk_id: str
    quote: str
    relevance: float
    retrieval_method: str


@dataclass(frozen=True)
class RetrievalTrace:
    """Trace data for inspecting a RAG answer."""

    query: str
    retrieval_mode: str
    reranker: str
    context: str
    retrieved_chunks: list[RetrievedChunk]


@dataclass(frozen=True)
class RAGAnswer:
    """Structured answer with citations and retrieval trace."""

    text: str
    citations: list[Citation]
    trace: RetrievalTrace

"""Citation and trace helpers for grounded RAG answers."""

from __future__ import annotations

from raglab.schema import Citation, RetrievedChunk


def excerpt(text: str, max_chars: int = 240) -> str:
    """Return a compact single-line excerpt for citation display."""
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    trimmed = normalized[: max_chars - 3].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed + "..."


def build_citations(
    chunks: list[RetrievedChunk],
    limit: int = 3,
    quote_chars: int = 240,
) -> list[Citation]:
    """Build source citations from retrieved chunks."""
    citations: list[Citation] = []
    for chunk in chunks[:limit]:
        citations.append(
            Citation(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                quote=excerpt(chunk.content, max_chars=quote_chars),
                relevance=chunk.relevance,
                retrieval_method=chunk.retrieval_method,
                parent_id=str(chunk.metadata.get("parent_id", "")),
            )
        )
    return citations

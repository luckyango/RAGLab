"""Context expansion helpers for parent-child retrieval."""

from __future__ import annotations

from raglab.schema import RetrievedChunk


def expand_parent_context(
    chunks: list[RetrievedChunk],
    max_parent_chars: int = 1600,
) -> list[RetrievedChunk]:
    """Return prompt chunks that use parent text when available.

    Retrieval and reranking still operate on small child chunks. This function
    expands only the context passed to generation, preserving the child hit in
    metadata for traceability.
    """
    expanded: list[RetrievedChunk] = []
    for chunk in chunks:
        parent_text = str(chunk.metadata.get("parent_text", "")).strip()
        if not parent_text:
            expanded.append(chunk)
            continue

        parent_context = _truncate(parent_text, max_parent_chars)
        expanded.append(
            RetrievedChunk(
                content=parent_context,
                source=chunk.source,
                relevance=chunk.relevance,
                metadata={
                    **chunk.metadata,
                    "matched_child_text": chunk.content,
                    "context_expanded_from_parent": True,
                },
                chunk_id=chunk.chunk_id,
                retrieval_method=chunk.retrieval_method,
            )
        )
    return expanded


def _truncate(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."

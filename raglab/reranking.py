"""Reranking interfaces and dependency-free fallback rerankers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from raglab.retrieval import tokenize
from raglab.schema import RetrievedChunk


class Reranker(ABC):
    """Interface for reranking retrieved evidence chunks."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return reranked chunks."""


class NoOpReranker(Reranker):
    """Keep retrieval order unchanged."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        return chunks[:top_k]


class LexicalOverlapReranker(Reranker):
    """Deterministic fallback reranker based on query-token overlap.

    This is intentionally simple and dependency-free. It gives the project a
    reranking stage that can be tested locally, while leaving a clean interface
    for cross-encoder or API-based rerankers later.
    """

    def __init__(self, retrieval_weight: float = 0.35) -> None:
        self.retrieval_weight = retrieval_weight

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []

        query_terms = set(tokenize(query))
        if not query_terms:
            return chunks[:top_k]

        scored = []
        for rank, chunk in enumerate(chunks, 1):
            chunk_terms = set(tokenize(chunk.content))
            overlap = len(query_terms & chunk_terms) / len(query_terms)
            retrieval_prior = 1 / rank
            score = overlap + self.retrieval_weight * retrieval_prior
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                content=chunk.content,
                source=chunk.source,
                relevance=round(score, 4),
                metadata={
                    **chunk.metadata,
                    "rerank_score": score,
                    "pre_rerank_relevance": chunk.relevance,
                },
                chunk_id=chunk.chunk_id,
                retrieval_method=f"{chunk.retrieval_method}+rerank",
            )
            for score, chunk in scored[:top_k]
        ]


def build_reranker(name: str | None) -> Reranker:
    """Construct a reranker by name."""
    if name in (None, "none"):
        return NoOpReranker()
    if name == "lexical":
        return LexicalOverlapReranker()
    raise ValueError("reranker must be one of: none, lexical")

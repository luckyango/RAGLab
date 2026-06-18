"""Reranking interfaces and dependency-free fallback rerankers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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


class CohereReranker(Reranker):
    """API-based reranker using Cohere's rerank endpoint.

    Pass an initialized Cohere client for tests or custom configuration. If no
    client is passed, the optional `cohere` package is imported lazily.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "rerank-v3.5",
    ) -> None:
        self.client = client or self._build_default_client()
        self.model = model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0 or not chunks:
            return []

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=[chunk.content for chunk in chunks],
            top_n=min(top_k, len(chunks)),
        )
        return [
            _copy_with_rerank_score(
                chunks[result.index],
                score=float(result.relevance_score),
                method_suffix="cohere_rerank",
            )
            for result in response.results
        ]

    @staticmethod
    def _build_default_client() -> Any:
        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "CohereReranker requires the optional 'cohere' package or a "
                "preconfigured client."
            ) from exc
        return cohere.Client()


class CrossEncoderReranker(Reranker):
    """Local cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model: Any | None = None,
        model_name: str = "BAAI/bge-reranker-base",
    ) -> None:
        self.model = model or self._load_model(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0 or not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs)
        scored = [
            (float(score), chunk)
            for score, chunk in zip(scores, chunks)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            _copy_with_rerank_score(
                chunk,
                score=score,
                method_suffix="cross_encoder_rerank",
                extra_metadata={"cross_encoder_model": self.model_name},
            )
            for score, chunk in scored[:top_k]
        ]

    @staticmethod
    def _load_model(model_name: str) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires the optional "
                "'sentence-transformers' package or a preloaded model."
            ) from exc
        return CrossEncoder(model_name)


def build_reranker(name: str | None) -> Reranker:
    """Construct a reranker by name."""
    if name in (None, "none"):
        return NoOpReranker()
    if name == "lexical":
        return LexicalOverlapReranker()
    if name == "cohere":
        return CohereReranker()
    if name in {"cross_encoder", "cross-encoder"}:
        return CrossEncoderReranker()
    raise ValueError(
        "reranker must be one of: none, lexical, cohere, cross_encoder"
    )


def _copy_with_rerank_score(
    chunk: RetrievedChunk,
    score: float,
    method_suffix: str,
    extra_metadata: dict[str, Any] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        content=chunk.content,
        source=chunk.source,
        relevance=round(score, 4),
        metadata={
            **chunk.metadata,
            "rerank_score": score,
            "pre_rerank_relevance": chunk.relevance,
            "reranker": method_suffix,
            **(extra_metadata or {}),
        },
        chunk_id=chunk.chunk_id,
        retrieval_method=f"{chunk.retrieval_method}+{method_suffix}",
    )

"""Query transformation strategies for retrieval."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryTransformResult:
    """Generated retrieval queries plus strategy metadata."""

    original_query: str
    queries: list[str]
    mode: str
    notes: str = ""


class QueryTransformer(ABC):
    """Interface for query rewriting and decomposition."""

    mode: str

    @abstractmethod
    def transform(self, query: str) -> QueryTransformResult:
        """Generate retrieval queries from the user's original query."""


class OriginalQueryTransformer(QueryTransformer):
    mode = "original"

    def transform(self, query: str) -> QueryTransformResult:
        return QueryTransformResult(
            original_query=query,
            queries=[query],
            mode=self.mode,
            notes="Use the original user query.",
        )


class MultiQueryTransformer(QueryTransformer):
    mode = "multi_query"

    def transform(self, query: str) -> QueryTransformResult:
        keywords = _keywords(query)
        queries = [
            query,
            " ".join(keywords[:6]) if keywords else query,
            f"definition context {' '.join(keywords[:4])}".strip(),
            f"implementation details {' '.join(keywords[:4])}".strip(),
        ]
        return QueryTransformResult(
            original_query=query,
            queries=_dedupe(queries),
            mode=self.mode,
            notes="Deterministic keyword-focused query expansion.",
        )


class HyDEQueryTransformer(QueryTransformer):
    mode = "hyde"

    def transform(self, query: str) -> QueryTransformResult:
        hypothetical_answer = (
            "Hypothetical answer document: "
            f"This passage directly answers the question '{query}' with relevant "
            "definitions, constraints, examples, and implementation details."
        )
        return QueryTransformResult(
            original_query=query,
            queries=[query, hypothetical_answer],
            mode=self.mode,
            notes="Deterministic HyDE-style hypothetical answer expansion.",
        )


class DecompositionQueryTransformer(QueryTransformer):
    mode = "decompose"

    def transform(self, query: str) -> QueryTransformResult:
        parts = _split_complex_query(query)
        queries = [query, *parts]
        return QueryTransformResult(
            original_query=query,
            queries=_dedupe(queries),
            mode=self.mode,
            notes="Deterministic split of compound questions into subqueries.",
        )


def build_query_transformer(mode: str | QueryTransformer | None) -> QueryTransformer:
    """Construct a query transformer by mode."""
    if isinstance(mode, QueryTransformer):
        return mode
    if mode in (None, "original"):
        return OriginalQueryTransformer()
    if mode == "multi_query":
        return MultiQueryTransformer()
    if mode == "hyde":
        return HyDEQueryTransformer()
    if mode == "decompose":
        return DecompositionQueryTransformer()
    raise ValueError(
        "query_mode must be one of: original, multi_query, hyde, decompose"
    )


def _keywords(query: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
    }
    words = re.findall(r"[\w]+", query.lower())
    return [word for word in words if word not in stopwords]


def _split_complex_query(query: str) -> list[str]:
    normalized = query.strip().rstrip("?")
    parts = [
        part.strip(" ,;:")
        for part in re.split(r"\b(?:and|also|then|vs|versus)\b|[;]", normalized)
        if part.strip(" ,;:")
    ]
    if len(parts) <= 1:
        return []
    return [part if part.endswith("?") else f"{part}?" for part in parts]


def _dedupe(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        normalized = " ".join(query.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped

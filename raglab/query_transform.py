"""Query transformation strategies for retrieval."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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


class OpenAIQueryTransformer(QueryTransformer):
    """LLM-backed query transformer with deterministic fallback."""

    def __init__(
        self,
        mode: str = "multi_query",
        client: Any | None = None,
        model: str = "gpt-4.1-mini",
        max_queries: int = 4,
        fallback: QueryTransformer | None = None,
    ) -> None:
        self.mode = f"openai_{mode}"
        self.strategy_mode = mode
        self.client = client
        self.model = model
        self.max_queries = max_queries
        self.fallback = fallback or build_query_transformer(mode)

    def transform(self, query: str) -> QueryTransformResult:
        if self.client is None:
            fallback_result = self.fallback.transform(query)
            return QueryTransformResult(
                original_query=query,
                queries=fallback_result.queries,
                mode=self.mode,
                notes=f"Fallback used: no OpenAI client. {fallback_result.notes}",
            )

        try:
            content = self._call_model(query)
            queries = _dedupe([query, *_extract_queries(content)])
        except Exception as exc:
            fallback_result = self.fallback.transform(query)
            return QueryTransformResult(
                original_query=query,
                queries=fallback_result.queries,
                mode=self.mode,
                notes=f"Fallback used after LLM rewrite failed: {exc}",
            )

        if len(queries) <= 1:
            fallback_result = self.fallback.transform(query)
            return QueryTransformResult(
                original_query=query,
                queries=fallback_result.queries,
                mode=self.mode,
                notes="Fallback used because LLM returned no additional queries.",
            )

        return QueryTransformResult(
            original_query=query,
            queries=queries[: self.max_queries],
            mode=self.mode,
            notes=f"OpenAI-backed {self.strategy_mode} query transformation.",
        )

    def _call_model(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": _TRANSFORM_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Strategy: {self.strategy_mode}\n"
                        f"Max queries: {self.max_queries}\n"
                        f"User query: {query}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content or ""


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


_TRANSFORM_SYSTEM_PROMPT = """You generate retrieval queries for a RAG system.
Return JSON only:
{"queries": ["query 1", "query 2"]}

Rules:
- Keep queries concise and retrieval-focused.
- Preserve named entities, acronyms, numbers, and constraints.
- For multi_query, create diverse paraphrases and keyword-focused queries.
- For hyde, include a short hypothetical answer passage as one query.
- For decompose, split compound questions into subquestions."""


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


def _extract_queries(content: str) -> list[str]:
    parsed = json.loads(content)
    queries = parsed.get("queries", [])
    if not isinstance(queries, list):
        return []
    return [str(query) for query in queries if str(query).strip()]

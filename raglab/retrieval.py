"""Retrieval strategies and fusion helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from raglab.schema import RetrievedChunk


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class CorpusDocument:
    """A searchable document chunk loaded from storage."""

    chunk_id: str
    content: str
    source: str
    metadata: dict[str, Any]


def tokenize(text: str) -> list[str]:
    """Tokenize text for simple lexical retrieval."""
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Retriever:
    """Small dependency-free BM25 retriever for lexical matching."""

    def __init__(
        self,
        documents: list[CorpusDocument],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document.content) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.document_frequency = self._build_document_frequency()

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the highest scoring BM25 matches."""
        if not self.documents or top_k <= 0:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[float, CorpusDocument]] = []
        for index, document in enumerate(self.documents):
            score = self._score(query_terms, index)
            if score > 0:
                scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                content=document.content,
                source=document.source,
                relevance=round(score, 3),
                metadata={**document.metadata, "bm25_score": score},
                chunk_id=document.chunk_id,
                retrieval_method="bm25",
            )
            for score, document in scored[:top_k]
        ]

    def _build_document_frequency(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _score(self, query_terms: list[str], document_index: int) -> float:
        tokens = self.doc_tokens[document_index]
        if not tokens:
            return 0.0

        term_frequency: dict[str, int] = {}
        for token in tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1

        score = 0.0
        total_documents = len(self.documents)
        doc_length = self.doc_lengths[document_index]
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue

            doc_frequency = self.document_frequency.get(term, 0)
            idf = math.log(
                1 + (total_documents - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * doc_length / (self.avg_doc_length or 1)
            )
            score += idf * (frequency * (self.k1 + 1)) / denominator

        return score


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse ranked retrieval lists with Reciprocal Rank Fusion."""
    if top_k <= 0:
        return []

    by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    methods: dict[str, set[str]] = {}

    for ranked_chunks in ranked_lists:
        for rank, chunk in enumerate(ranked_chunks, 1):
            chunk_key = chunk.chunk_id or f"{chunk.source}:{chunk.content}"
            by_id.setdefault(chunk_key, chunk)
            scores[chunk_key] = scores.get(chunk_key, 0.0) + 1 / (rrf_k + rank)
            methods.setdefault(chunk_key, set()).add(chunk.retrieval_method)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[RetrievedChunk] = []
    for chunk_key, score in ordered[:top_k]:
        chunk = by_id[chunk_key]
        method = "+".join(sorted(methods[chunk_key]))
        fused.append(
            RetrievedChunk(
                content=chunk.content,
                source=chunk.source,
                relevance=round(score, 4),
                metadata={**chunk.metadata, "rrf_score": score},
                chunk_id=chunk.chunk_id,
                retrieval_method=method,
            )
        )

    return fused

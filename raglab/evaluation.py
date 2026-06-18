"""Deterministic retrieval evaluation for RAGLab."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from raglab.schema import RetrievedChunk


@dataclass(frozen=True)
class GoldenQuestion:
    """A labelled retrieval example."""

    question: str
    expected_sources: list[str]
    expected_terms: list[str]
    notes: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    """Per-question retrieval evaluation result."""

    question: str
    retrieved_sources: list[str]
    hit: bool
    reciprocal_rank: float
    expected_term_coverage: float
    retrieved_chunks: list[RetrievedChunk]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate retrieval metrics."""

    total: int
    recall_at_k: float
    mrr_at_k: float
    expected_term_coverage: float
    results: list[EvaluationResult]


class RetrieverLike(Protocol):
    def retrieve(self, query: str, n: int = 5) -> list[RetrievedChunk]:
        """Retrieve chunks for a query."""


def load_golden_questions(path: str | Path) -> list[GoldenQuestion]:
    """Load JSONL golden retrieval questions."""
    questions: list[GoldenQuestion] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        questions.append(
            GoldenQuestion(
                question=payload["question"],
                expected_sources=list(payload.get("expected_sources", [])),
                expected_terms=list(payload.get("expected_terms", [])),
                notes=payload.get("notes", ""),
            )
        )
        if not questions[-1].expected_sources and not questions[-1].expected_terms:
            raise ValueError(
                f"Line {line_number} must include expected_sources or expected_terms"
            )
    return questions


def evaluate_retrieval(
    retriever: RetrieverLike,
    questions: list[GoldenQuestion],
    k: int = 5,
) -> EvaluationSummary:
    """Evaluate retrieval recall, MRR, and expected term coverage."""
    if k <= 0:
        raise ValueError("k must be positive")

    results = [_evaluate_question(retriever, question, k=k) for question in questions]
    if not results:
        return EvaluationSummary(
            total=0,
            recall_at_k=0.0,
            mrr_at_k=0.0,
            expected_term_coverage=0.0,
            results=[],
        )

    return EvaluationSummary(
        total=len(results),
        recall_at_k=sum(result.hit for result in results) / len(results),
        mrr_at_k=sum(result.reciprocal_rank for result in results) / len(results),
        expected_term_coverage=(
            sum(result.expected_term_coverage for result in results) / len(results)
        ),
        results=results,
    )


def format_summary(summary: EvaluationSummary) -> str:
    """Format evaluation metrics for terminal output."""
    lines = [
        "Retrieval Evaluation",
        f"Total: {summary.total}",
        f"Recall@k: {summary.recall_at_k:.3f}",
        f"MRR@k: {summary.mrr_at_k:.3f}",
        f"Expected term coverage: {summary.expected_term_coverage:.3f}",
        "",
        "Questions:",
    ]
    for result in summary.results:
        status = "PASS" if result.hit else "MISS"
        lines.append(
            f"- {status} | rr={result.reciprocal_rank:.3f} | "
            f"terms={result.expected_term_coverage:.3f} | {result.question}"
        )
    return "\n".join(lines)


def _evaluate_question(
    retriever: RetrieverLike,
    question: GoldenQuestion,
    k: int,
) -> EvaluationResult:
    chunks = retriever.retrieve(question.question, n=k)
    retrieved_sources = [chunk.source for chunk in chunks]
    reciprocal_rank = _reciprocal_rank(chunks, question)
    return EvaluationResult(
        question=question.question,
        retrieved_sources=retrieved_sources,
        hit=reciprocal_rank > 0,
        reciprocal_rank=reciprocal_rank,
        expected_term_coverage=_expected_term_coverage(chunks, question.expected_terms),
        retrieved_chunks=chunks,
    )


def _reciprocal_rank(chunks: list[RetrievedChunk], question: GoldenQuestion) -> float:
    expected_sources = {source.lower() for source in question.expected_sources}
    expected_terms = [term.lower() for term in question.expected_terms]

    for index, chunk in enumerate(chunks, 1):
        source_hit = (
            chunk.source.lower() in expected_sources if expected_sources else False
        )
        content = chunk.content.lower()
        term_hit = any(term in content for term in expected_terms)
        if source_hit or term_hit:
            return 1 / index
    return 0.0


def _expected_term_coverage(
    chunks: list[RetrievedChunk],
    expected_terms: list[str],
) -> float:
    if not expected_terms:
        return 0.0
    combined = "\n".join(chunk.content for chunk in chunks).lower()
    hits = sum(1 for term in expected_terms if term.lower() in combined)
    return hits / len(expected_terms)


def main() -> None:
    from raglab.agent import DocumentQAAgent

    parser = argparse.ArgumentParser(description="Evaluate RAGLab retrieval.")
    parser.add_argument("--dataset", required=True, help="Path to golden QA JSONL.")
    parser.add_argument("--persist-dir", default="./qa_db", help="Chroma persist dir.")
    parser.add_argument("--retrieval-mode", default="hybrid")
    parser.add_argument("--query-mode", default="original")
    parser.add_argument("--reranker", default="lexical")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    agent = DocumentQAAgent(
        persist_dir=args.persist_dir,
        retrieval_mode=args.retrieval_mode,
        query_mode=args.query_mode,
        reranker=args.reranker,
    )
    questions = load_golden_questions(args.dataset)
    summary = evaluate_retrieval(agent, questions, k=args.k)
    print(format_summary(summary))


if __name__ == "__main__":
    main()

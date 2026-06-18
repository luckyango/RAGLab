import tempfile
import unittest
from pathlib import Path

from raglab.evaluation import (
    GoldenQuestion,
    evaluate_retrieval,
    format_summary,
    load_golden_questions,
)
from raglab.schema import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, query, n=5):
        self.calls.append((query, n))
        return self.chunks[:n]


class EvaluationTests(unittest.TestCase):
    def test_loads_golden_questions_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "golden.jsonl"
            path.write_text(
                '{"question": "Q?", "expected_sources": ["Doc"], '
                '"expected_terms": ["answer"]}\n',
                encoding="utf-8",
            )

            questions = load_golden_questions(path)

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].question, "Q?")
        self.assertEqual(questions[0].expected_sources, ["Doc"])

    def test_evaluate_retrieval_computes_recall_and_mrr(self):
        retriever = FakeRetriever(
            [
                RetrievedChunk("unrelated", "Other", 0.9, {}, "1"),
                RetrievedChunk("contains answer", "Doc", 0.8, {}, "2"),
            ]
        )
        questions = [
            GoldenQuestion(
                question="Where is the answer?",
                expected_sources=["Doc"],
                expected_terms=["answer"],
            )
        ]

        summary = evaluate_retrieval(retriever, questions, k=2)

        self.assertEqual(summary.total, 1)
        self.assertEqual(summary.recall_at_k, 1.0)
        self.assertEqual(summary.mrr_at_k, 0.5)
        self.assertEqual(summary.expected_term_coverage, 1.0)
        self.assertEqual(retriever.calls, [("Where is the answer?", 2)])

    def test_format_summary_includes_question_status(self):
        retriever = FakeRetriever([RetrievedChunk("answer", "Doc", 0.9, {}, "1")])
        summary = evaluate_retrieval(
            retriever,
            [GoldenQuestion("Q?", ["Doc"], ["answer"])],
            k=1,
        )

        text = format_summary(summary)

        self.assertIn("Retrieval Evaluation", text)
        self.assertIn("Recall@k: 1.000", text)
        self.assertIn("PASS", text)

    def test_empty_dataset_returns_zero_metrics(self):
        summary = evaluate_retrieval(FakeRetriever([]), [], k=1)

        self.assertEqual(summary.total, 0)
        self.assertEqual(summary.recall_at_k, 0.0)


if __name__ == "__main__":
    unittest.main()

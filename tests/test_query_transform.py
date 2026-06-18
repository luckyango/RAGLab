import unittest

from raglab.query_transform import (
    DecompositionQueryTransformer,
    HyDEQueryTransformer,
    MultiQueryTransformer,
    OriginalQueryTransformer,
    build_query_transformer,
)


class QueryTransformTests(unittest.TestCase):
    def test_original_returns_user_query_only(self):
        result = OriginalQueryTransformer().transform("What is RAG?")

        self.assertEqual(result.mode, "original")
        self.assertEqual(result.queries, ["What is RAG?"])

    def test_multi_query_generates_distinct_expansions(self):
        result = MultiQueryTransformer().transform("How does hybrid RAG retrieval work?")

        self.assertEqual(result.mode, "multi_query")
        self.assertGreater(len(result.queries), 1)
        self.assertEqual(len(result.queries), len(set(result.queries)))
        self.assertIn("hybrid", " ".join(result.queries))

    def test_hyde_adds_hypothetical_answer_query(self):
        result = HyDEQueryTransformer().transform("What is chunking?")

        self.assertEqual(result.mode, "hyde")
        self.assertEqual(result.queries[0], "What is chunking?")
        self.assertIn("Hypothetical answer document", result.queries[1])

    def test_decompose_splits_compound_query(self):
        result = DecompositionQueryTransformer().transform(
            "What is BM25 and how does reranking work?"
        )

        self.assertEqual(result.mode, "decompose")
        self.assertIn("What is BM25?", result.queries)
        self.assertIn("how does reranking work?", result.queries)

    def test_build_query_transformer_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            build_query_transformer("unknown")


if __name__ == "__main__":
    unittest.main()

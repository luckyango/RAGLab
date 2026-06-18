import unittest

from raglab.tracing import build_citations, excerpt
from raglab.schema import RetrievedChunk


class TracingTests(unittest.TestCase):
    def test_excerpt_normalizes_whitespace_and_truncates(self):
        text = "Alpha\n\nBeta   Gamma Delta"

        result = excerpt(text, max_chars=16)

        self.assertEqual(result, "Alpha Beta...")

    def test_build_citations_uses_retrieved_chunk_metadata(self):
        chunks = [
            RetrievedChunk(
                content="Python was created by Guido van Rossum.",
                source="Python Basics",
                relevance=0.88,
                metadata={"parent_id": "parent-1"},
                chunk_id="chunk-1",
                retrieval_method="hybrid+rerank",
            )
        ]

        citations = build_citations(chunks)

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].source, "Python Basics")
        self.assertEqual(citations[0].chunk_id, "chunk-1")
        self.assertEqual(citations[0].parent_id, "parent-1")
        self.assertEqual(citations[0].retrieval_method, "hybrid+rerank")
        self.assertIn("Guido", citations[0].quote)


if __name__ == "__main__":
    unittest.main()

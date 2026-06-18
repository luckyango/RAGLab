import unittest

from raglab.reranking import LexicalOverlapReranker, NoOpReranker, build_reranker
from raglab.schema import RetrievedChunk


class RerankerTests(unittest.TestCase):
    def test_noop_keeps_existing_order(self):
        chunks = [
            RetrievedChunk("first", "doc", 0.2, {}, "1", "vector"),
            RetrievedChunk("second", "doc", 0.9, {}, "2", "vector"),
        ]

        reranked = NoOpReranker().rerank("anything", chunks, top_k=2)

        self.assertEqual([chunk.chunk_id for chunk in reranked], ["1", "2"])

    def test_lexical_overlap_promotes_query_match(self):
        chunks = [
            RetrievedChunk(
                "general python text",
                "doc",
                0.9,
                {},
                "general",
                "vector",
            ),
            RetrievedChunk(
                "uvicorn is an ASGI server for FastAPI",
                "doc",
                0.4,
                {},
                "target",
                "bm25",
            ),
        ]

        reranked = LexicalOverlapReranker().rerank(
            "FastAPI uvicorn server",
            chunks,
            top_k=2,
        )

        self.assertEqual(reranked[0].chunk_id, "target")
        self.assertEqual(reranked[0].retrieval_method, "bm25+rerank")
        self.assertIn("rerank_score", reranked[0].metadata)
        self.assertIn("pre_rerank_relevance", reranked[0].metadata)

    def test_build_reranker_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            build_reranker("unknown")


if __name__ == "__main__":
    unittest.main()

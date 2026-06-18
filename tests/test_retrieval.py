import unittest

from raglab.retrieval import BM25Retriever, CorpusDocument, reciprocal_rank_fusion
from raglab.schema import RetrievedChunk


class BM25RetrieverTests(unittest.TestCase):
    def test_keyword_match_ranks_exact_document_first(self):
        documents = [
            CorpusDocument(
                chunk_id="a",
                content="FastAPI uses uvicorn as an ASGI server.",
                source="FastAPI",
                metadata={},
            ),
            CorpusDocument(
                chunk_id="b",
                content="Python supports object oriented programming.",
                source="Python",
                metadata={},
            ),
        ]

        results = BM25Retriever(documents).search("uvicorn server", top_k=2)

        self.assertEqual(results[0].chunk_id, "a")
        self.assertEqual(results[0].retrieval_method, "bm25")


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_merges_duplicate_chunks_by_id(self):
        vector_result = RetrievedChunk(
            chunk_id="same",
            content="shared content",
            source="doc",
            relevance=0.9,
            metadata={},
            retrieval_method="vector",
        )
        bm25_result = RetrievedChunk(
            chunk_id="same",
            content="shared content",
            source="doc",
            relevance=2.5,
            metadata={},
            retrieval_method="bm25",
        )

        fused = reciprocal_rank_fusion([[vector_result], [bm25_result]], top_k=5)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].chunk_id, "same")
        self.assertEqual(fused[0].retrieval_method, "bm25+vector")
        self.assertIn("rrf_score", fused[0].metadata)

    def test_prefers_documents_ranked_high_in_multiple_lists(self):
        shared_late = RetrievedChunk("shared", "doc", 0.5, {}, "shared", "vector")
        vector_first = RetrievedChunk("vector", "doc", 0.9, {}, "vector", "vector")
        bm25_first = RetrievedChunk("bm25", "doc", 3.0, {}, "bm25", "bm25")

        fused = reciprocal_rank_fusion(
            [[vector_first, shared_late], [bm25_first, shared_late]],
            top_k=3,
        )

        self.assertEqual(fused[0].chunk_id, "shared")


if __name__ == "__main__":
    unittest.main()

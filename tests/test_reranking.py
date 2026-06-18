import importlib.util
import unittest

from raglab.reranking import (
    CohereReranker,
    CrossEncoderReranker,
    LexicalOverlapReranker,
    NoOpReranker,
    build_reranker,
)
from raglab.schema import RetrievedChunk


class _FakeCohereResult:
    def __init__(self, index, relevance_score):
        self.index = index
        self.relevance_score = relevance_score


class _FakeCohereResponse:
    def __init__(self, results):
        self.results = results


class _FakeCohereClient:
    def __init__(self):
        self.calls = []

    def rerank(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCohereResponse(
            [
                _FakeCohereResult(index=1, relevance_score=0.95),
                _FakeCohereResult(index=0, relevance_score=0.25),
            ]
        )


class _FakeCrossEncoderModel:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return self.scores


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

    def test_cohere_reranker_uses_api_scores(self):
        chunks = [
            RetrievedChunk("first", "doc", 0.5, {}, "1", "hybrid"),
            RetrievedChunk("second", "doc", 0.4, {}, "2", "hybrid"),
        ]
        client = _FakeCohereClient()

        reranked = CohereReranker(client=client).rerank("query", chunks, top_k=2)

        self.assertEqual([chunk.chunk_id for chunk in reranked], ["2", "1"])
        self.assertEqual(reranked[0].relevance, 0.95)
        self.assertEqual(reranked[0].metadata["reranker"], "cohere_rerank")
        self.assertEqual(reranked[0].retrieval_method, "hybrid+cohere_rerank")
        self.assertEqual(client.calls[0]["documents"], ["first", "second"])
        self.assertEqual(client.calls[0]["top_n"], 2)

    def test_cross_encoder_reranker_uses_model_scores(self):
        chunks = [
            RetrievedChunk("first", "doc", 0.5, {}, "1", "hybrid"),
            RetrievedChunk("second", "doc", 0.4, {}, "2", "hybrid"),
        ]
        model = _FakeCrossEncoderModel(scores=[0.1, 0.9])

        reranked = CrossEncoderReranker(model=model, model_name="fake").rerank(
            "query",
            chunks,
            top_k=1,
        )

        self.assertEqual([chunk.chunk_id for chunk in reranked], ["2"])
        self.assertEqual(reranked[0].metadata["reranker"], "cross_encoder_rerank")
        self.assertEqual(reranked[0].metadata["cross_encoder_model"], "fake")
        self.assertEqual(model.calls[0], [("query", "first"), ("query", "second")])

    def test_optional_cohere_dependency_error_is_clear(self):
        if importlib.util.find_spec("cohere"):
            self.skipTest("cohere is installed")

        with self.assertRaisesRegex(ImportError, "cohere"):
            CohereReranker()

    def test_optional_cross_encoder_dependency_error_is_clear(self):
        if importlib.util.find_spec("sentence_transformers"):
            self.skipTest("sentence-transformers is installed")

        with self.assertRaisesRegex(ImportError, "sentence-transformers"):
            CrossEncoderReranker()


if __name__ == "__main__":
    unittest.main()

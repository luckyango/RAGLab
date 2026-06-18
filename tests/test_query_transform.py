import unittest

from raglab.query_transform import (
    DecompositionQueryTransformer,
    HyDEQueryTransformer,
    MultiQueryTransformer,
    OpenAIQueryTransformer,
    OriginalQueryTransformer,
    build_query_transformer,
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return _FakeResponse(self.content)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, content=None, error=None):
        self.chat = _FakeChat(_FakeCompletions(content=content, error=error))


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

    def test_openai_transformer_uses_json_queries(self):
        client = _FakeClient(
            content='{"queries": ["hybrid retrieval BM25", "RAG reranking"]}'
        )

        result = OpenAIQueryTransformer(
            mode="multi_query",
            client=client,
            max_queries=3,
        ).transform("How does hybrid retrieval work?")

        self.assertEqual(result.mode, "openai_multi_query")
        self.assertEqual(result.queries[0], "How does hybrid retrieval work?")
        self.assertIn("hybrid retrieval BM25", result.queries)
        self.assertIn("OpenAI-backed", result.notes)
        self.assertEqual(
            client.chat.completions.calls[0]["response_format"],
            {"type": "json_object"},
        )

    def test_openai_transformer_falls_back_without_client(self):
        result = OpenAIQueryTransformer(mode="hyde", client=None).transform(
            "What is chunking?"
        )

        self.assertEqual(result.mode, "openai_hyde")
        self.assertGreater(len(result.queries), 1)
        self.assertIn("Fallback used", result.notes)

    def test_openai_transformer_falls_back_on_error(self):
        client = _FakeClient(error=RuntimeError("boom"))

        result = OpenAIQueryTransformer(mode="decompose", client=client).transform(
            "What is BM25 and how does reranking work?"
        )

        self.assertEqual(result.mode, "openai_decompose")
        self.assertIn("What is BM25?", result.queries)
        self.assertIn("Fallback used", result.notes)


if __name__ == "__main__":
    unittest.main()

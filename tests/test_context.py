import unittest

from raglab.context import expand_parent_context
from raglab.schema import RetrievedChunk


class ContextExpansionTests(unittest.TestCase):
    def test_expands_chunk_content_with_parent_text(self):
        child = RetrievedChunk(
            content="matched child",
            source="guide.md",
            relevance=0.7,
            metadata={"parent_text": "full parent context", "parent_id": "parent-1"},
            chunk_id="child-1",
            retrieval_method="hybrid+rerank",
        )

        expanded = expand_parent_context([child])

        self.assertEqual(expanded[0].content, "full parent context")
        self.assertEqual(expanded[0].metadata["matched_child_text"], "matched child")
        self.assertTrue(expanded[0].metadata["context_expanded_from_parent"])
        self.assertEqual(expanded[0].chunk_id, "child-1")

    def test_keeps_chunk_when_parent_text_missing(self):
        child = RetrievedChunk("child only", "guide.md", 0.5, {}, "child-1")

        expanded = expand_parent_context([child])

        self.assertEqual(expanded[0], child)


if __name__ == "__main__":
    unittest.main()

import unittest

from raglab.prompting import build_system_prompt, format_context
from raglab.schema import RetrievedChunk


class PromptingTests(unittest.TestCase):
    def test_format_context_includes_source_and_relevance(self):
        chunks = [
            RetrievedChunk(
                content="Python was created by Guido van Rossum.",
                source="Python Basics",
                relevance=0.91,
                metadata={"chunk_index": 0},
            )
        ]

        context = format_context(chunks)

        self.assertIn("[Document Chunk 1]", context)
        self.assertIn("Source: Python Basics", context)
        self.assertIn("Retrieval: vector", context)
        self.assertIn("Relevance: 0.91", context)

    def test_system_prompt_contains_grounding_requirements(self):
        prompt = build_system_prompt("DocAssistant", "reference context")

        self.assertIn("Answer based ONLY", prompt)
        self.assertIn("[Reference Documents]", prompt)
        self.assertIn("reference context", prompt)

    def test_format_context_includes_matched_child_when_parent_expanded(self):
        chunks = [
            RetrievedChunk(
                content="Full parent paragraph.",
                source="Guide",
                relevance=0.8,
                metadata={"matched_child_text": "Small matched chunk."},
                chunk_id="child-1",
            )
        ]

        context = format_context(chunks)

        self.assertIn("Full parent paragraph.", context)
        self.assertIn("Matched child chunk:", context)
        self.assertIn("Small matched chunk.", context)


if __name__ == "__main__":
    unittest.main()

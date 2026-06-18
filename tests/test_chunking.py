import unittest

from raglab.chunking import (
    ChunkingConfig,
    chunk_document_text,
    chunk_source_document,
    chunk_text,
    count_tokens,
)
from raglab.ingestion import SourceDocument


class ChunkTextTests(unittest.TestCase):
    def test_keeps_short_paragraphs_together(self):
        text = "Alpha paragraph.\n\nBeta paragraph."

        chunks = chunk_text(text, chunk_size=100)

        self.assertEqual(chunks, ["Alpha paragraph.\nBeta paragraph."])

    def test_splits_long_paragraph_on_sentence_boundaries(self):
        text = "First sentence. Second sentence. Third sentence."

        chunks = chunk_text(text, chunk_size=25)

        self.assertEqual(
            chunks,
            ["First sentence.", "Second sentence.", "Third sentence."],
        )

    def test_rejects_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=0)

    def test_counts_tokens_with_punctuation(self):
        self.assertEqual(count_tokens("FastAPI uses uvicorn."), 4)

    def test_header_sections_add_section_metadata(self):
        text = "# Intro\nAlpha beta.\n\n# Usage\nGamma delta."

        chunks = chunk_document_text(text, config=ChunkingConfig(chunk_size=10))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["section"], "Intro")
        self.assertEqual(chunks[1].metadata["section"], "Usage")

    def test_chunks_include_parent_child_metadata(self):
        document = SourceDocument(
            text="One two three four five six. Seven eight nine ten.",
            source="manual",
            metadata={"source": "manual", "section": "Numbers"},
        )

        chunks = chunk_source_document(
            document,
            config=ChunkingConfig(chunk_size=5, chunk_overlap=1),
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["section"], "Numbers")
        self.assertIn("parent_id", chunks[0].metadata)
        self.assertIn("parent_text", chunks[0].metadata)
        self.assertIn("token_count", chunks[0].metadata)
        self.assertEqual(chunks[0].metadata["chunking_strategy"], "recursive_token")

    def test_overlap_prefixes_later_chunks(self):
        text = "one two three four five six seven eight"

        chunks = chunk_document_text(
            text,
            config=ChunkingConfig(chunk_size=4, chunk_overlap=2),
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[1].text.startswith("three four"))


if __name__ == "__main__":
    unittest.main()

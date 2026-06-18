import unittest

from raglab.chunking import chunk_text


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


if __name__ == "__main__":
    unittest.main()

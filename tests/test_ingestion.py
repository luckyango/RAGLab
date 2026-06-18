import tempfile
import unittest
from pathlib import Path

from raglab.ingestion import IngestionError, load_documents


class IngestionTests(unittest.TestCase):
    def test_loads_txt_with_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.txt"
            path.write_text("Plain text document.", encoding="utf-8")

            documents = load_documents(path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source, "notes.txt")
        self.assertEqual(documents[0].metadata["source"], "notes.txt")
        self.assertEqual(documents[0].metadata["document_type"], "txt")

    def test_loads_markdown_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "guide.md"
            path.write_text(
                "# Intro\nFirst section.\n\n## Usage\nSecond section.",
                encoding="utf-8",
            )

            documents = load_documents(path)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["section"], "Intro")
        self.assertEqual(documents[1].metadata["section"], "Usage")

    def test_loads_html_text_and_heading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "page.html"
            path.write_text(
                "<html><body><h1>Overview</h1><p>HTML content.</p></body></html>",
                encoding="utf-8",
            )

            documents = load_documents(path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["section"], "Overview")
        self.assertIn("HTML content", documents[0].text)

    def test_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.csv"
            path.write_text("a,b", encoding="utf-8")

            with self.assertRaises(IngestionError):
                load_documents(path)

    def test_optional_pdf_dependency_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4")

            try:
                load_documents(path)
            except IngestionError as exc:
                self.assertIn("pypdf", str(exc))

    def test_optional_docx_dependency_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.docx"
            path.write_bytes(b"not a real docx")

            try:
                load_documents(path)
            except IngestionError as exc:
                self.assertIn("python-docx", str(exc))


if __name__ == "__main__":
    unittest.main()

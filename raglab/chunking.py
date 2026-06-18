"""Chunking utilities for document ingestion."""

from __future__ import annotations

import re


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？.!?])")


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Split text into paragraph-aware chunks.

    This keeps the original baseline behavior but moves it behind a testable
    function so later chunking strategies can be added without touching the
    agent orchestration code.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk = (
                current_chunk + "\n" + paragraph if current_chunk else paragraph
            )
            continue

        if current_chunk:
            chunks.append(current_chunk)

        if len(paragraph) > chunk_size:
            current_chunk = ""
            for sentence in SENTENCE_BOUNDARY_RE.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        else:
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

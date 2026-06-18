"""Chunking utilities for document ingestion."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from raglab.ingestion import SourceDocument


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\u3002\uff01\uff1f.!?])")


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for recursive token-aware chunking."""

    chunk_size: int = 400
    chunk_overlap: int = 40
    preserve_parent_text: bool = True
    max_parent_chars: int = 2000
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            object.__setattr__(self, "chunk_overlap", max(0, self.chunk_size // 5))


@dataclass(frozen=True)
class TextChunk:
    """A chunk with metadata for retrieval and citation."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def count_tokens(text: str) -> int:
    """Approximate token count without requiring tokenizer dependencies."""
    return len(TOKEN_RE.findall(text))


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Split text into paragraph-aware chunks and return plain strings."""
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


def chunk_source_document(
    document: SourceDocument,
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    """Chunk a loaded source document while preserving source metadata."""
    config = config or ChunkingConfig()
    chunks = chunk_document_text(document.text, config=config)
    enriched: list[TextChunk] = []
    for chunk in chunks:
        enriched.append(
            TextChunk(
                text=chunk.text,
                metadata={**document.metadata, **chunk.metadata},
            )
        )
    return enriched


def chunk_document_text(
    text: str,
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    """Split text by headers, then recursively into token-aware chunks."""
    config = config or ChunkingConfig()
    chunks: list[TextChunk] = []

    for section_index, section in enumerate(_split_header_sections(text)):
        parent_id = str(uuid.uuid4())
        parent_text = _truncate_parent_text(section["text"], config.max_parent_chars)
        parts = _recursive_split(section["text"], config)
        parts = _with_token_overlap(parts, config)

        for chunk_index, part in enumerate(parts):
            metadata: dict[str, Any] = {
                "section_index": section_index,
                "parent_id": parent_id,
                "child_index": chunk_index,
                "token_count": count_tokens(part),
                "chunking_strategy": "recursive_token",
            }
            if section["section"]:
                metadata["section"] = section["section"]
            if config.preserve_parent_text:
                metadata["parent_text"] = parent_text
            chunks.append(TextChunk(text=part, metadata=metadata))

    return chunks


def _split_header_sections(text: str) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        match = HEADER_RE.match(line.strip())
        if match:
            if current_lines:
                sections.append(
                    {
                        "section": current_heading,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []
            current_heading = match.group(2).strip()
            continue
        current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section": current_heading,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["text"]]


def _recursive_split(text: str, config: ChunkingConfig) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= config.chunk_size:
        return [text]

    for separator in config.separators:
        parts = [part.strip() for part in text.split(separator) if part.strip()]
        if len(parts) <= 1:
            continue
        return _pack_parts(parts, separator, config)

    return _split_by_tokens(text, config.chunk_size)


def _pack_parts(parts: list[str], separator: str, config: ChunkingConfig) -> list[str]:
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = part if not current else f"{current}{separator}{part}"
        if count_tokens(candidate) <= config.chunk_size:
            current = candidate
            continue

        if current:
            chunks.extend(_recursive_split(current, config))
        if count_tokens(part) > config.chunk_size:
            chunks.extend(_recursive_split(part, config))
            current = ""
        else:
            current = part

    if current:
        chunks.extend(_recursive_split(current, config))
    return chunks


def _split_by_tokens(text: str, chunk_size: int) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    chunks: list[str] = []
    for start in range(0, len(tokens), chunk_size):
        chunks.append(" ".join(tokens[start : start + chunk_size]))
    return chunks


def _with_token_overlap(parts: list[str], config: ChunkingConfig) -> list[str]:
    if config.chunk_overlap == 0 or len(parts) <= 1:
        return parts

    overlapped = [parts[0]]
    for previous, current in zip(parts, parts[1:]):
        previous_tokens = TOKEN_RE.findall(previous)
        overlap = previous_tokens[-config.chunk_overlap :]
        prefix = " ".join(overlap)
        candidate = f"{prefix} {current}".strip()
        if count_tokens(candidate) <= config.chunk_size + config.chunk_overlap:
            overlapped.append(candidate)
        else:
            overlapped.append(current)
    return overlapped


def _truncate_parent_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."

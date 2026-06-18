"""Prompt construction helpers."""

from __future__ import annotations

from raglab.schema import RetrievedChunk


SYSTEM_REQUIREMENTS = """Requirements:
1. Answer based ONLY on the provided reference documents.
2. If the reference documents do not contain relevant information, state it clearly.
3. Cite specific sources using the provided chunk labels, e.g. [Document Chunk 1].
4. Keep the answer concise and accurate, avoid fabricating information."""


def format_context(chunks: list[RetrievedChunk], limit: int = 3) -> str:
    """Format retrieved chunks for the answer-generation prompt."""
    context_parts = []
    for index, chunk in enumerate(chunks[:limit], 1):
        context_parts.append(
            f"[Document Chunk {index}] "
            f"(Source: {chunk.source}, Chunk ID: {chunk.chunk_id or 'unknown'}, "
            f"Retrieval: {chunk.retrieval_method}, Relevance: {chunk.relevance})\n"
            f"{chunk.content}"
        )
        child_text = chunk.metadata.get("matched_child_text")
        if child_text:
            context_parts[-1] += f"\n\nMatched child chunk:\n{child_text}"
    return "\n\n".join(context_parts)


def build_system_prompt(agent_name: str, context: str) -> str:
    """Build the grounded QA system prompt."""
    return f"""You are {agent_name}, a QA assistant based on the user's document knowledge base.

{SYSTEM_REQUIREMENTS}

[Reference Documents]
{context}"""

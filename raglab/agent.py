"""Baseline document QA agent backed by OpenAI embeddings and ChromaDB."""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

from raglab.chunking import chunk_text
from raglab.prompting import build_system_prompt, format_context
from raglab.schema import RetrievedChunk

load_dotenv()


class DocumentQAAgent:
    """Document QA agent with persistent vector retrieval."""

    def __init__(
        self,
        name: str = "DocAssistant",
        persist_dir: str = "./qa_db",
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4.1",
        client: OpenAI | None = None,
        console: Console | None = None,
    ) -> None:
        self.name = name
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.client = client or OpenAI()
        self.console = console or Console()

        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.chat_history: list[dict[str, str]] = []

        count = self.collection.count()
        self.console.print(
            f"[dim]{name} started, knowledge base contains {count} chunks[/dim]"
        )

    def add_text(
        self,
        text: str,
        source: str = "manual",
        chunk_size: int = 400,
    ) -> int:
        """Add text to the vector knowledge base and return chunk count."""
        chunks = chunk_text(text, chunk_size=chunk_size)
        if not chunks:
            return 0

        embeddings_response = self.client.embeddings.create(
            input=chunks,
            model=self.embedding_model,
        )
        embeddings = [item.embedding for item in embeddings_response.data]

        ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": source,
                    "chunk_index": index,
                    "added_at": dt.datetime.now().isoformat(),
                }
                for index in range(len(chunks))
            ],
        )

        self.console.print(
            f"[green]Added {len(chunks)} chunks (Source: {source})[/green]"
        )
        return len(chunks)

    def add_file(self, file_path: str) -> int:
        """Load a UTF-8 text file and add it to the knowledge base."""
        path = Path(file_path)
        if not path.exists():
            self.console.print(f"[red]File does not exist: {file_path}[/red]")
            return 0

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            self.console.print(f"[red]Load failed: {exc}[/red]")
            return 0

        return self.add_text(content, source=path.name)

    def list_sources(self) -> list[str]:
        """List unique source names in the knowledge base."""
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas"])
        sources = {
            metadata.get("source", "unknown")
            for metadata in results["metadatas"]
        }
        return sorted(sources)

    def retrieve(self, query: str, n: int = 5) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        if self.collection.count() == 0:
            return []

        response = self.client.embeddings.create(
            input=query.replace("\n", " "),
            model=self.embedding_model,
        )
        query_embedding = response.data[0].embedding

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        if results["documents"] and results["documents"][0]:
            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                relevance = 1 - distance
                if relevance > 0.3:
                    chunks.append(
                        RetrievedChunk(
                            content=document,
                            source=metadata.get("source", "Unknown"),
                            relevance=round(relevance, 3),
                            metadata=metadata,
                        )
                    )

        return chunks

    def ask(self, question: str) -> str:
        """Answer a question using retrieved document context."""
        chunks = self.retrieve(question)
        if not chunks:
            return (
                "Sorry, no relevant information was found in my knowledge base. "
                "Please add relevant documents first."
            )

        context = format_context(chunks)
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(self.name, context),
            },
            *self.chat_history[-6:],
            {"role": "user", "content": question},
        ]

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            max_tokens=800,
        )
        answer = response.choices[0].message.content

        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer or ""

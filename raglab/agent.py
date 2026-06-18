"""Baseline document QA agent backed by OpenAI embeddings and ChromaDB."""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

from raglab.chunking import chunk_text
from raglab.ingestion import IngestionError, SourceDocument, load_documents
from raglab.prompting import build_system_prompt, format_context
from raglab.reranking import Reranker, build_reranker
from raglab.retrieval import BM25Retriever, CorpusDocument, reciprocal_rank_fusion
from raglab.schema import RAGAnswer, RetrievalTrace, RetrievedChunk
from raglab.tracing import build_citations

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
        retrieval_mode: str = "hybrid",
        reranker: str | Reranker | None = "lexical",
        client: OpenAI | None = None,
        console: Console | None = None,
    ) -> None:
        self.name = name
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.retrieval_mode = retrieval_mode
        self.reranker = (
            reranker if isinstance(reranker, Reranker) else build_reranker(reranker)
        )
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
            f"[dim]{name} started, knowledge base contains {count} chunks "
            f"(retrieval={retrieval_mode}, reranker={reranker or 'none'})[/dim]"
        )

    def add_text(
        self,
        text: str,
        source: str = "manual",
        chunk_size: int = 400,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add text to the vector knowledge base and return chunk count."""
        document = SourceDocument(
            text=text,
            source=source,
            metadata={"source": source, **(metadata or {})},
        )
        return self.add_documents([document], chunk_size=chunk_size)

    def add_documents(
        self,
        documents: list[SourceDocument],
        chunk_size: int = 400,
    ) -> int:
        """Add loaded source documents to the vector knowledge base."""
        chunk_records: list[tuple[str, dict[str, str | int | float | bool]]] = []
        created_at = dt.datetime.now().isoformat()

        for document in documents:
            chunks = chunk_text(document.text, chunk_size=chunk_size)
            for index, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                metadata = {
                    **document.metadata,
                    "source": document.source,
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "created_at": created_at,
                    "added_at": created_at,
                }
                chunk_records.append((chunk, self._sanitize_metadata(metadata)))

        if not chunk_records:
            return 0

        chunks = [chunk for chunk, _ in chunk_records]
        metadatas = [metadata for _, metadata in chunk_records]

        embeddings_response = self.client.embeddings.create(
            input=chunks,
            model=self.embedding_model,
        )
        embeddings = [item.embedding for item in embeddings_response.data]
        ids = [str(metadata["chunk_id"]) for metadata in metadatas]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        sources = sorted({str(metadata["source"]) for metadata in metadatas})
        self.console.print(
            f"[green]Added {len(chunks)} chunks (Sources: {', '.join(sources)})[/green]"
        )
        return len(chunks)

    @staticmethod
    def _sanitize_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, str | int | float | bool]:
        """Keep metadata compatible with Chroma's scalar metadata types."""
        sanitized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    def add_file(self, file_path: str) -> int:
        """Load a supported file and add it to the knowledge base."""
        path = Path(file_path)
        if not path.exists():
            self.console.print(f"[red]File does not exist: {file_path}[/red]")
            return 0

        try:
            documents = load_documents(path)
        except IngestionError as exc:
            self.console.print(f"[red]Load failed: {exc}[/red]")
            return 0

        return self.add_documents(documents)

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

        if self.retrieval_mode == "vector":
            chunks = self._retrieve_vector(query, n=self._candidate_count(n))
        elif self.retrieval_mode == "bm25":
            chunks = self._retrieve_bm25(query, n=self._candidate_count(n))
        elif self.retrieval_mode == "hybrid":
            chunks = self._retrieve_hybrid(query, n=self._candidate_count(n))
        else:
            raise ValueError(
                "retrieval_mode must be one of: vector, bm25, hybrid"
            )

        return self.reranker.rerank(query, chunks, top_k=n)

    def _retrieve_vector(self, query: str, n: int = 5) -> list[RetrievedChunk]:
        """Retrieve chunks with dense vector search."""
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
            for chunk_id, document, metadata, distance in zip(
                results["ids"][0],
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
                            metadata={**metadata, "vector_relevance": relevance},
                            chunk_id=chunk_id,
                            retrieval_method="vector",
                        )
                    )

        return chunks

    def _retrieve_bm25(self, query: str, n: int = 5) -> list[RetrievedChunk]:
        """Retrieve chunks with lexical BM25 search."""
        documents = self._load_corpus_documents()
        return BM25Retriever(documents).search(query, top_k=n)

    def _retrieve_hybrid(self, query: str, n: int = 5) -> list[RetrievedChunk]:
        """Retrieve chunks with vector search, BM25, and RRF fusion."""
        vector_chunks = self._retrieve_vector(query, n=n)
        bm25_chunks = self._retrieve_bm25(query, n=n)
        return reciprocal_rank_fusion([vector_chunks, bm25_chunks], top_k=n)

    def _candidate_count(self, n: int) -> int:
        """Use a wider candidate pool before reranking."""
        return min(max(n * 4, 10), self.collection.count())

    def _load_corpus_documents(self) -> list[CorpusDocument]:
        """Load stored Chroma chunks into a lexical-search corpus."""
        results = self.collection.get(include=["documents", "metadatas"])
        documents: list[CorpusDocument] = []
        for chunk_id, content, metadata in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            documents.append(
                CorpusDocument(
                    chunk_id=chunk_id,
                    content=content,
                    source=metadata.get("source", "Unknown"),
                    metadata={**metadata, "chunk_id": chunk_id},
                )
            )
        return documents

    def ask(self, question: str) -> str:
        """Answer a question using retrieved document context."""
        return self.ask_with_trace(question).text

    def ask_with_trace(self, question: str) -> RAGAnswer:
        """Answer a question and return citations plus retrieval trace."""
        chunks = self.retrieve(question)
        if not chunks:
            answer = (
                "Sorry, no relevant information was found in my knowledge base. "
                "Please add relevant documents first."
            )
            return RAGAnswer(
                text=answer,
                citations=[],
                trace=RetrievalTrace(
                    query=question,
                    retrieval_mode=self.retrieval_mode,
                    reranker=self.reranker.__class__.__name__,
                    context="",
                    retrieved_chunks=[],
                ),
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

        return RAGAnswer(
            text=answer or "",
            citations=build_citations(chunks),
            trace=RetrievalTrace(
                query=question,
                retrieval_mode=self.retrieval_mode,
                reranker=self.reranker.__class__.__name__,
                context=context,
                retrieved_chunks=chunks,
            ),
        )

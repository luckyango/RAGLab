"""Command-line interface for the baseline RAGLab agent."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from raglab.agent import DocumentQAAgent


SAMPLE_DOCS = [
    (
        "Python Basics",
        """
Python is a high-level programming language known for its simplicity and readability. Python was created by Guido van Rossum in 1991.
Python supports object-oriented programming, functional programming, and procedural programming.
The package management tool for Python is pip, and recommended virtual environment tools are venv or conda.
""",
    ),
    (
        "FastAPI Introduction",
        """
FastAPI is a modern, high-performance Web framework for Python.
FastAPI is based on Python 3.7+ type hints and automatically generates API documentation.
FastAPI's performance is close to NodeJS and Go, making it one of the fastest Python Web frameworks available.
It uses uvicorn as the ASGI server, command: uvicorn main:app --reload
""",
    ),
    (
        "LangChain Introduction",
        """
LangChain is an open-source framework for building LLM applications.
LangChain provides components such as Chain (processing pipelines), Agent, Memory, and RAG.
LangChain supports multiple LLM providers like OpenAI, Anthropic, and local models.
The latest version LangChain 0.3 adopts LCEL (LangChain Expression Language) as the standard way to build chains.
""",
    ),
]


def main() -> None:
    """Run the interactive CLI."""
    console = Console()
    agent = DocumentQAAgent("SmartDocAssistant", console=console)

    for title, content in SAMPLE_DOCS:
        agent.add_text(content, source=title)

    console.print(
        Panel(
            f"[bold]{agent.name} is ready[/bold]\n"
            f"Knowledge base contains {len(SAMPLE_DOCS)} sample topics\n\n"
            "Commands:\n"
            "  sources -> View knowledge base sources\n"
            "  add <file_path> -> Add a UTF-8 text file\n"
            "  quit -> Exit",
            title="System Startup",
            border_style="blue",
        )
    )

    while True:
        user_input = input("\nYour Question: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "sources":
            sources = agent.list_sources()
            console.print(f"[cyan]Knowledge base sources ({len(sources)}):[/cyan]")
            for source in sources:
                console.print(f"  - {source}")
            continue

        if user_input.lower().startswith("add "):
            file_path = user_input[4:].strip()
            agent.add_file(file_path)
            continue

        answer = agent.ask(user_input)
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(Markdown(answer))


if __name__ == "__main__":
    main()

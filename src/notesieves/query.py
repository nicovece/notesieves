from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import Config
from .llm import LLMService
from .prompts import (
    BROAD_SYSTEM_PROMPT,
    QUIZ_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_broad_user_prompt,
    build_quiz_start_prompt,
    build_user_prompt,
)
from .vectorstore import VectorStore

console = Console()


def run_query(config: Config, question: str, broad: bool = False):
    """Run the full query pipeline."""
    with console.status("Loading embedding model..."):
        from .embeddings import EmbeddingService
        embedder = EmbeddingService()

    store = VectorStore(config.paths.database_directory)
    llm = LLMService(model=config.llm.model, max_tokens=config.llm.max_tokens)

    if store.count() == 0:
        console.print("[red]No notes indexed yet. Run 'nsie ingest' first.[/red]")
        return

    if broad:
        with console.status("Scanning notes..."):
            query_embedding = embedder.embed_text(question)
            file_map = store.search_broad(query_embedding, top_k=30)

        if not file_map:
            console.print("[yellow]No relevant content found in your notes.[/yellow]")
            return

        with console.status("Generating answer..."):
            user_prompt = build_broad_user_prompt(question, file_map)
            answer = llm.generate(BROAD_SYSTEM_PROMPT, user_prompt)
    else:
        with console.status("Searching notes..."):
            query_embedding = embedder.embed_text(question)
            chunks = store.search(query_embedding, top_k=config.retrieval.top_k)

        if not chunks:
            console.print("[yellow]No relevant content found in your notes.[/yellow]")
            return

        with console.status("Generating answer..."):
            user_prompt = build_user_prompt(question, chunks)
            answer = llm.generate(SYSTEM_PROMPT, user_prompt)

    console.print()
    console.print(Panel(
        Markdown(answer),
        title="[bold]Answer[/bold]",
        border_style="green",
    ))


def run_quiz(config: Config, topic: str):
    """Run an interactive quiz session."""
    with console.status("Loading embedding model..."):
        from .embeddings import EmbeddingService
        embedder = EmbeddingService()

    store = VectorStore(config.paths.database_directory)
    llm = LLMService(model=config.llm.model, max_tokens=config.llm.max_tokens)

    if store.count() == 0:
        console.print("[red]No notes indexed yet. Run 'nsie ingest' first.[/red]")
        return

    with console.status("Searching notes..."):
        query_embedding = embedder.embed_text(topic)
        chunks = store.search(query_embedding, top_k=15)

    if not chunks:
        console.print("[yellow]No relevant content found in your notes.[/yellow]")
        return

    # Build the first message with context
    first_message = build_quiz_start_prompt(topic, chunks)
    messages = [{"role": "user", "content": first_message}]

    console.print()
    console.print(f"[bold]Quiz: {topic}[/bold]")
    console.print("[dim]Type your answer, or 'quit' to stop.[/dim]\n")

    try:
        while True:
            with console.status("Thinking..."):
                response = llm.generate_multiturn(QUIZ_SYSTEM_PROMPT, messages)

            messages.append({"role": "assistant", "content": response})

            console.print(Panel(
                Markdown(response),
                border_style="blue",
            ))

            answer = console.input("[bold]Your answer:[/bold] ")

            if answer.strip().lower() in ("quit", "q", "exit"):
                break

            messages.append({"role": "user", "content": answer})

    except KeyboardInterrupt:
        pass

    console.print("\n[dim]Quiz ended.[/dim]")

from pathlib import Path

import typer
from rich.console import Console

from .config import load_config
from .ingest import run_ingest
from .query import run_query

app = typer.Typer(
    name="notesieves",
    help="Query your markdown notes with AI",
)
console = Console()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to markdown directory"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear existing index first"),
):
    """Index markdown files from a directory."""
    config = load_config()
    notes_path = Path(path).expanduser().resolve()
    run_ingest(config, notes_path, clear=clear)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the notes"),
):
    """Ask a question about your indexed notes."""
    config = load_config()
    run_query(config, question)


@app.command()
def status():
    """Show index statistics."""
    console.print("[dim]Would show index status[/dim]")


if __name__ == "__main__":
    app()

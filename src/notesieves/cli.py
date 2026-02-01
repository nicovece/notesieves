from pathlib import Path

import anthropic
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


def _load_config_or_exit():
    """Load config, exit with a helpful message if it fails."""
    try:
        return load_config()
    except FileNotFoundError:
        console.print("[red]config.yaml not found.[/red] Create one in the project root.")
        console.print("See config.yaml.example or the spec for the expected format.")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Invalid config:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to markdown directory"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear existing index first"),
):
    """Index markdown files from a directory."""
    config = _load_config_or_exit()
    notes_path = Path(path).expanduser().resolve()

    if not notes_path.is_dir():
        console.print(f"[red]Directory not found:[/red] {notes_path}")
        raise typer.Exit(1)

    run_ingest(config, notes_path, clear=clear)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the notes"),
):
    """Ask a question about your indexed notes."""
    config = _load_config_or_exit()

    try:
        run_query(config, question)
    except anthropic.AuthenticationError:
        console.print("[red]Invalid API key.[/red] Check ANTHROPIC_API_KEY in your .env file.")
        raise typer.Exit(1)
    except anthropic.APIConnectionError:
        console.print("[red]Could not connect to the Anthropic API.[/red] Check your internet connection.")
        raise typer.Exit(1)


@app.command()
def status():
    """Show index statistics."""
    config = _load_config_or_exit()

    from .vectorstore import VectorStore

    store = VectorStore(config.paths.database_directory)
    count = store.count()

    console.print()
    console.print("[bold]NoteSieves Index Status[/bold]")
    console.print(f"  Notes directory:    {config.paths.notes_directory}")
    console.print(f"  Database directory: {config.paths.database_directory}")
    console.print(f"  Chunks indexed:     {count}")
    if count == 0:
        console.print("\n  [yellow]No notes indexed yet. Run 'nsie ingest <path>' first.[/yellow]")
    console.print()


if __name__ == "__main__":
    app()

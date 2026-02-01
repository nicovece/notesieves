import typer
from rich.console import Console

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
    console.print(f"[dim]Would ingest from:[/dim] {path}")
    if clear:
        console.print("[dim]Would clear existing index first[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the notes"),
):
    """Ask a question about your indexed notes."""
    console.print(f"[dim]Would ask:[/dim] {question}")


@app.command()
def status():
    """Show index statistics."""
    console.print("[dim]Would show index status[/dim]")


if __name__ == "__main__":
    app()

from pathlib import Path

from rich.console import Console

from .chunker import MarkdownChunker
from .config import Config
from .embeddings import EmbeddingService
from .vectorstore import VectorStore

console = Console()


def run_ingest(config: Config, notes_path: Path, clear: bool = False):
    """Run the full ingestion pipeline."""
    chunker = MarkdownChunker(
        max_tokens=config.chunking.max_chunk_tokens,
        overlap_tokens=config.chunking.overlap_tokens,
    )
    embedder = EmbeddingService()
    store = VectorStore(config.paths.database_directory)

    if clear:
        console.print("[yellow]Clearing existing index...[/yellow]")
        store.clear()

    md_files = list(notes_path.rglob("*.md"))
    if not md_files:
        console.print("[red]No markdown files found.[/red]")
        return

    console.print(f"Found [green]{len(md_files)}[/green] markdown files")

    all_chunks = []
    for file_path in md_files:
        chunks = chunker.chunk_file(file_path)
        all_chunks.extend(chunks)

    console.print(f"Created [green]{len(all_chunks)}[/green] chunks")

    console.print("Generating embeddings...")
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed_batch(texts)

    console.print("Storing in vector database...")
    store.add_chunks(all_chunks, embeddings)

    console.print(f"[green]Done.[/green] Indexed {len(md_files)} files ({len(all_chunks)} chunks)")

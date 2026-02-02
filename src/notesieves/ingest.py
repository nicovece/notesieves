import hashlib
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .chunker import MarkdownChunker
from .config import Config
from .vectorstore import VectorStore

console = Console()


def _hash_file(file_path: Path) -> str:
    """Compute a short SHA-256 hash of a file's content."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]


def run_ingest(config: Config, notes_path: Path, clear: bool = False):
    """Run the full ingestion pipeline."""
    chunker = MarkdownChunker(
        max_tokens=config.chunking.max_chunk_tokens,
        overlap_tokens=config.chunking.overlap_tokens,
    )

    with console.status("Loading embedding model..."):
        from .embeddings import EmbeddingService
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

    # Compute hashes for all files on disk
    disk_files = {}
    for file_path in md_files:
        disk_files[str(file_path)] = _hash_file(file_path)

    # Get stored hashes from the database
    stored_hashes = store.get_file_hashes()

    # Determine what changed
    disk_paths = set(disk_files.keys())
    stored_paths = set(stored_hashes.keys())

    new_files = disk_paths - stored_paths
    deleted_files = stored_paths - disk_paths
    common_files = disk_paths & stored_paths
    changed_files = {fp for fp in common_files if disk_files[fp] != stored_hashes[fp]}
    unchanged_files = common_files - changed_files

    files_to_process = new_files | changed_files

    # Report what was found
    if unchanged_files:
        console.print(f"  [dim]Unchanged: {len(unchanged_files)} files (skipped)[/dim]")
    if new_files:
        console.print(f"  [green]New: {len(new_files)} files[/green]")
    if changed_files:
        console.print(f"  [yellow]Changed: {len(changed_files)} files[/yellow]")
    if deleted_files:
        console.print(f"  [red]Deleted: {len(deleted_files)} files[/red]")

    # Delete chunks for changed and deleted files
    files_to_delete = changed_files | deleted_files
    if files_to_delete:
        with console.status("Removing outdated chunks..."):
            for fp in files_to_delete:
                store.delete_by_file(fp)

    if not files_to_process:
        console.print("[green]Everything up to date.[/green]")
        return

    # Chunk new and changed files
    all_chunks = []
    files_list = [Path(fp) for fp in sorted(files_to_process)]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking files...", total=len(files_list))
        for file_path in files_list:
            file_hash = disk_files[str(file_path)]
            chunks = chunker.chunk_file(file_path, file_hash=file_hash)
            all_chunks.extend(chunks)
            progress.advance(task)

    console.print(f"Created [green]{len(all_chunks)}[/green] chunks")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating embeddings...", total=len(all_chunks))
        texts = [chunk.text for chunk in all_chunks]
        embeddings = embedder.embed_batch(texts)
        progress.update(task, completed=len(all_chunks))

    with console.status("Storing in vector database..."):
        store.add_chunks(all_chunks, embeddings)

    console.print(
        f"[green]Done.[/green] Processed {len(files_to_process)} files "
        f"({len(all_chunks)} chunks)"
    )
    if deleted_files:
        console.print(f"  Removed {len(deleted_files)} deleted files from index")

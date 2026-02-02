from pathlib import Path

import chromadb


class VectorStore:
    def __init__(self, persist_directory: Path, collection_name: str = "notes"):
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list, embeddings: list[list[float]]):
        """Add chunks with their embeddings to the store."""
        self.collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return chunks

    def clear(self):
        """Delete all documents in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search_broad(self, query_embedding: list[float], top_k: int = 30) -> dict[str, list[str]]:
        """Search and return unique file→headings map (no document text)."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["metadatas"],
        )

        file_map: dict[str, set[str]] = {}
        for meta in results["metadatas"][0]:
            name = meta.get("file_name", "Unknown")
            heading = meta.get("heading_hierarchy", "")
            if name not in file_map:
                file_map[name] = set()
            if heading:
                file_map[name].add(heading)

        return {name: sorted(headings) for name, headings in sorted(file_map.items())}

    def list_sources(self) -> list[str]:
        """Return sorted unique file names from all indexed chunks."""
        results = self.collection.get(include=["metadatas"])
        names = {m["file_name"] for m in results["metadatas"] if "file_name" in m}
        return sorted(names)

    def count(self) -> int:
        """Return number of chunks in the store."""
        return self.collection.count()

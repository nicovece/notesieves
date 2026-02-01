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

    def clear(self):
        """Delete all documents in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return number of chunks in the store."""
        return self.collection.count()

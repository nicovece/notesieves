import logging

from sentence_transformers import SentenceTransformer

# Suppress the harmless "position_ids UNEXPECTED" warning from transformers
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        return self.model.encode(texts, show_progress_bar=True).tolist()

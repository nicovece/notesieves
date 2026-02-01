from pathlib import Path

import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    notes_directory: Path
    database_directory: Path = Path("./data/chroma")


class ChunkingConfig(BaseModel):
    max_chunk_tokens: int = 500
    overlap_tokens: int = 50
    split_on_headings: bool = True


class RetrievalConfig(BaseModel):
    top_k: int = 5
    include_metadata_in_context: bool = True


class LLMConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024


class Config(BaseModel):
    paths: PathsConfig
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()


def load_config(config_path: Path = Path("config.yaml")) -> Config:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return Config(**data)

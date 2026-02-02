import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    metadata: dict


class MarkdownChunker:
    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_file(self, file_path: Path, file_hash: str = "") -> list[Chunk]:
        """Main entry point: file → list of chunks."""
        content = file_path.read_text(encoding="utf-8")
        file_name = file_path.stem

        sections = self._parse_sections(content)

        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, file_path, file_name)
            chunks.extend(section_chunks)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks_in_file"] = len(chunks)
            chunk.metadata["file_hash"] = file_hash

        return chunks

    def _parse_sections(self, content: str) -> list[dict]:
        """Split content by markdown headings, preserving hierarchy."""
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        sections = []
        current_hierarchy = []
        last_end = 0

        for match in heading_pattern.finditer(content):
            text_before = content[last_end:match.start()]
            if text_before.strip():
                sections.append({
                    "heading_hierarchy": current_hierarchy.copy(),
                    "content": text_before.strip(),
                })

            level = len(match.group(1))
            heading_text = match.group(2)
            current_hierarchy = current_hierarchy[:level - 1]
            current_hierarchy.append(heading_text)

            last_end = match.end()

        remaining = content[last_end:]
        if remaining.strip():
            sections.append({
                "heading_hierarchy": current_hierarchy.copy(),
                "content": remaining.strip(),
            })

        return sections

    def _chunk_section(self, section: dict, file_path: Path, file_name: str) -> list[Chunk]:
        """Split a section into chunks if too large."""
        content = section["content"]
        hierarchy = section["heading_hierarchy"]

        if self._estimate_tokens(content) <= self.max_tokens:
            return [Chunk(
                text=content,
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_name,
                    "heading_hierarchy": " > ".join(hierarchy),
                },
            )]

        paragraphs = content.split("\n\n")
        chunks = []
        current_parts = []
        current_tokens = 0

        for paragraph in paragraphs:
            para_tokens = self._estimate_tokens(paragraph)

            if current_tokens + para_tokens > self.max_tokens and current_parts:
                chunks.append(Chunk(
                    text="\n\n".join(current_parts),
                    metadata={
                        "file_path": str(file_path),
                        "file_name": file_name,
                        "heading_hierarchy": " > ".join(hierarchy),
                    },
                ))
                overlap_parts = self._get_overlap_parts(current_parts)
                current_parts = overlap_parts
                current_tokens = self._estimate_tokens("\n\n".join(overlap_parts))

            current_parts.append(paragraph)
            current_tokens += para_tokens

        if current_parts:
            chunks.append(Chunk(
                text="\n\n".join(current_parts),
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_name,
                    "heading_hierarchy": " > ".join(hierarchy),
                },
            ))

        return chunks

    def _get_overlap_parts(self, parts: list[str]) -> list[str]:
        """Get trailing paragraphs that fit within overlap_tokens."""
        overlap = []
        tokens = 0
        for part in reversed(parts):
            part_tokens = self._estimate_tokens(part)
            if tokens + part_tokens > self.overlap_tokens and overlap:
                break
            overlap.insert(0, part)
            tokens += part_tokens
        return overlap

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return len(text) // 4

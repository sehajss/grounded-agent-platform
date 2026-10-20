from __future__ import annotations

import re
from dataclasses import dataclass


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    token_count: int
    attributes: dict[str, str]


def _markdown_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_heading = ""

    def save_section() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match:
            save_section()
            level = len(match.group(1))
            heading_stack[level - 1 :] = [match.group(2).strip()]
            current_heading = " > ".join(heading_stack)
            current_lines = []
        else:
            current_lines.append(line)

    save_section()
    return sections


def chunk_markdown(
    markdown: str,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []

    for heading, section_text in _markdown_sections(markdown):
        words = section_text.split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            body = " ".join(words[start:end])
            content = f"{heading}\n\n{body}" if heading else body

            chunks.append(
                TextChunk(
                    index=len(chunks),
                    content=content,
                    token_count=end - start,
                    attributes={"heading": heading},
                )
            )

            if end == len(words):
                break
            start = end - chunk_overlap

    return chunks
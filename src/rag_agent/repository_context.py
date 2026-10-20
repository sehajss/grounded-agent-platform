"""Build bounded, task-relevant context from a local repository."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_agent.agent_tools import RepositoryTools


@dataclass(frozen=True)
class ContextBundle:
    text: str
    files: tuple[str, ...]
    estimated_tokens: int


class RepositoryContextBuilder:
    """Select relevant repository files within fixed context limits."""

    _STOP_WORDS = {
        "about", "after", "again", "also", "and", "are", "build", "change",
        "code", "could", "does", "file", "fix", "from", "have", "into",
        "make", "need", "please", "project", "should", "that", "task",
        "than", "then", "this", "what", "when", "where", "which", "with",
        "would", "your",
    }

    _PREFERRED_FILES = {
        "readme.md": 5,
        "pyproject.toml": 4,
        "requirements.txt": 3,
        "setup.py": 3,
        "setup.cfg": 3,
    }

    def __init__(
        self,
        tools: RepositoryTools,
        *,
        max_files: int = 12,
        max_file_chars: int = 8_000,
        max_context_chars: int = 36_000,
    ) -> None:
        self.tools = tools
        self.max_files = max_files
        self.max_file_chars = max_file_chars
        self.max_context_chars = max_context_chars

    @staticmethod
    def _output_text(result: object) -> str:
        """Extract readable output from a tool result."""
        if isinstance(result, str):
            return result

        for attribute in ("output", "text", "content"):
            value = getattr(result, attribute, None)
            if isinstance(value, str):
                return value

        if isinstance(result, dict):
            for key in ("output", "text", "content"):
                value = result.get(key)
                if isinstance(value, str):
                    return value

        return str(result)

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        words = set(re.findall(r"[a-zA-Z0-9_]{2,}", text.lower()))
        return words - cls._STOP_WORDS

    @classmethod
    def _path_score(cls, path: str, task_terms: set[str]) -> int:
        normalized = path.lower()
        path_terms = cls._terms(normalized)
        score = len(path_terms & task_terms) * 4
        score += cls._PREFERRED_FILES.get(normalized.rsplit("/", 1)[-1], 0)

        # Give source and test files a small preference over unrelated assets.
        if normalized.endswith((".py", ".toml", ".yaml", ".yml", ".md")):
            score += 1
        if any(part in normalized for part in ("/tests/", "/test/")):
            score += 1
        return score

    @classmethod
    def _content_score(cls, content: str, task_terms: set[str]) -> int:
        content_terms = cls._terms(content)
        return len(content_terms & task_terms)

    def build(self, task: str) -> ContextBundle:
        task_terms = self._terms(task)

        listing = self._output_text(self.tools.list_files("."))
        paths = sorted(
            line.strip()
            for line in listing.splitlines()
            if line.strip() and not line.startswith("[")
        )

        # Rank paths before reading file contents, limiting repository reads.
        candidates = sorted(
            paths,
            key=lambda path: (-self._path_score(path, task_terms), path),
        )[:60]

        ranked_files: list[tuple[int, str, str]] = []
        for path in candidates:
            result = self.tools.read_file(path)
            content = self._output_text(result)
            if not content.strip():
                continue

            content = content[: self.max_file_chars]
            score = (
                self._path_score(path, task_terms) * 2
                + self._content_score(content, task_terms)
            )
            ranked_files.append((score, path, content))

        ranked_files.sort(key=lambda item: (-item[0], item[1]))

        sections: list[str] = []
        selected_paths: list[str] = []
        used_chars = 0

        for _, path, content in ranked_files:
            if len(selected_paths) >= self.max_files:
                break

            section = f"### {path}\n```text\n{content.rstrip()}\n```"
            remaining = self.max_context_chars - used_chars
            if remaining <= 0:
                break

            if len(section) > remaining:
                if remaining < 300:
                    break
                section = section[:remaining].rstrip() + "\n[context truncated]"
            sections.append(section)
            selected_paths.append(path)
            used_chars += len(section)

        context = "\n\n".join(sections)
        return ContextBundle(
            text=context,
            files=tuple(selected_paths),
            estimated_tokens=max(1, len(context) // 4),
        )
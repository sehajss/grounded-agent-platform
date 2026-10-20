from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_agent.agent_policy import AgentPolicy, PolicyViolation


MAX_READ_BYTES = 1_000_000
MAX_SEARCH_FILE_BYTES = 256_000
MAX_TOOL_OUTPUT_CHARS = 20_000
MAX_LIST_FILES = 200
MAX_SEARCH_FILES = 500

_IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
_TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
_SAFE_ENV_VARS = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "VIRTUAL_ENV",
    "WINDIR",
}

TOOL_DESCRIPTIONS = [
    {
        "name": "list_files",
        "description": "List repository files under a relative directory.",
        "parameters": {"path": "string, optional"},
    },
    {
        "name": "read_file",
        "description": "Read one UTF-8 text file inside the repository.",
        "parameters": {"path": "string"},
    },
    {
        "name": "search_code",
        "description": "Search repository text files for a literal string.",
        "parameters": {
            "query": "string",
            "path": "string, optional",
        },
    },
    {
        "name": "run_check",
        "description": "Run one configured verification command by its short name.",
        "parameters": {"name": "pytest | ruff | mypy"},
    },
]


@dataclass(frozen=True)
class ToolResult:
    tool: str
    success: bool
    output: str
    changed_files: tuple[str, ...] = ()
    policy_blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RepositoryTools:
    def __init__(self, policy: AgentPolicy) -> None:
        self.policy = policy

    def descriptions(self) -> list[dict[str, Any]]:
        descriptions = list(TOOL_DESCRIPTIONS)
        if self.policy.allow_writes:
            descriptions.append(
                {
                    "name": "write_file",
                    "description": "Write UTF-8 text to a repository-relative file.",
                    "parameters": {"path": "string", "content": "string"},
                }
            )
        return descriptions

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            handlers = {
                "list_files": self.list_files,
                "read_file": self.read_file,
                "search_code": self.search_code,
                "run_check": self.run_check,
            }
            if self.policy.allow_writes:
                handlers["write_file"] = self.write_file

            handler = handlers.get(name)
            if handler is None:
                raise PolicyViolation(f"Unknown or unavailable tool: {name}")

            result = handler(**arguments)
            return result.to_dict()
        except PolicyViolation as exc:
            return ToolResult(
                tool=name,
                success=False,
                output=str(exc),
                policy_blocked=True,
            ).to_dict()
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            return ToolResult(
                tool=name,
                success=False,
                output=f"Tool failed: {type(exc).__name__}: {exc}",
            ).to_dict()

    def _iter_files(
        self,
        relative_directory: str,
        *,
        limit: int,
    ) -> tuple[list[Path], bool]:
        root = self.policy.resolve_path(relative_directory)
        if not root.is_dir():
            raise ValueError("The requested path is not a directory")

        found: list[Path] = []
        truncated = False

        for current, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in _IGNORED_DIRS
            )

            for filename in sorted(filenames):
                path = Path(current) / filename
                relative_path = path.relative_to(self.policy.repository_root).as_posix()
                try:
                    safe_path = self.policy.resolve_path(relative_path)
                except PolicyViolation:
                    continue

                if not safe_path.is_file():
                    continue
                found.append(safe_path)
                if len(found) >= limit:
                    truncated = True
                    return found, truncated

        return found, truncated

    def list_files(self, path: str = ".") -> ToolResult:
        files, truncated = self._iter_files(path, limit=MAX_LIST_FILES)
        relative_files = [
            item.relative_to(self.policy.repository_root).as_posix()
            for item in files
        ]
        suffix = "\n[output truncated at file limit]" if truncated else ""
        return ToolResult(
            tool="list_files",
            success=True,
            output="\n".join(relative_files) + suffix,
        )

    def read_file(self, path: str) -> ToolResult:
        target = self.policy.resolve_path(path)
        if not target.is_file():
            raise ValueError("The requested path is not a file")
        if target.stat().st_size > MAX_READ_BYTES:
            raise ValueError("File exceeds the 1 MB read limit")

        content = target.read_text(encoding="utf-8")
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            content = content[:MAX_TOOL_OUTPUT_CHARS] + "\n[output truncated]"
        return ToolResult(tool="read_file", success=True, output=content)

    def search_code(self, query: str, path: str = ".") -> ToolResult:
        if not query.strip() or len(query) > 200:
            raise ValueError("Search query must contain 1–200 characters")

        files, truncated = self._iter_files(path, limit=MAX_SEARCH_FILES)
        matches: list[str] = []

        for file_path in files:
            if file_path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            if file_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue

            relative = file_path.relative_to(self.policy.repository_root).as_posix()
            for line_number, line in enumerate(lines, start=1):
                if query.casefold() in line.casefold():
                    matches.append(f"{relative}:{line_number}: {line.strip()[:300]}")
                    if len(matches) >= 100:
                        break
            if len(matches) >= 100:
                break

        if truncated:
            matches.append("[search limited to the first 500 files]")
        return ToolResult(
            tool="search_code",
            success=True,
            output="\n".join(matches) or "No matches found.",
        )

    def write_file(self, path: str, content: str) -> ToolResult:
        target = self.policy.authorize_write(path, content)
        target.parent.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=".agent-write-",
                delete=False,
            ) as temporary_file:
                temporary_file.write(content)
                temporary_path = Path(temporary_file.name)

            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

        relative = target.relative_to(self.policy.repository_root).as_posix()
        return ToolResult(
            tool="write_file",
            success=True,
            output=f"Wrote {len(content.encode('utf-8'))} bytes to {relative}",
            changed_files=(relative,),
        )

    def run_check(self, name: str) -> ToolResult:
        command = self.policy.verification_command(name)
        safe_environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in _SAFE_ENV_VARS
        }

        try:
            completed = subprocess.run(
                command,
                cwd=self.policy.repository_root,
                env=safe_environment,
                shell=False,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            output = (completed.stdout + completed.stderr).strip()
            if len(output) > MAX_TOOL_OUTPUT_CHARS:
                output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n[output truncated]"

            return ToolResult(
                tool="run_check",
                success=completed.returncode == 0,
                output=output or f"Command exited with code {completed.returncode}.",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool="run_check",
                success=False,
                output="Verification command timed out after 120 seconds.",
            )
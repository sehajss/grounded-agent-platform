from __future__ import annotations

from pathlib import Path

from rag_agent.config import Settings, get_settings


class PolicyViolation(ValueError):
    """Raised when a developer-agent action violates the local policy."""


class AgentPolicy:
    _BLOCKED_PARTS = {
        ".aws",
        ".git",
        ".ssh",
        "credentials",
        "secrets",
    }
    _CHECK_COMMANDS = {
        "pytest": ("pytest", "-q"),
        "ruff": ("ruff", "check", "."),
        "mypy": ("mypy", "src"),
    }

    def __init__(
        self,
        repository_root: Path,
        *,
        allow_writes: bool = False,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository_root = repository_root.expanduser().resolve(strict=True)
        if not self.repository_root.is_dir():
            raise ValueError("repository_root must be a directory")

        # Both the environment setting and this run's explicit opt-in are required.
        self.allow_writes = allow_writes and self.settings.agent_allow_writes
        self.max_write_bytes = self.settings.agent_max_write_bytes
        self.allowed_checks = (
            self.settings.allowed_checks & self._CHECK_COMMANDS.keys()
        )

    def resolve_path(self, relative_path: str) -> Path:
        if not relative_path.strip():
            raise PolicyViolation("A repository-relative path is required")

        requested = Path(relative_path)
        if requested.is_absolute() or ".." in requested.parts:
            raise PolicyViolation("Absolute paths and path traversal are not allowed")

        candidate = (self.repository_root / requested).resolve()
        if not candidate.is_relative_to(self.repository_root):
            raise PolicyViolation("The path resolves outside the repository")

        parts = {part.lower() for part in requested.parts}
        resolved_parts = {
            part.lower()
            for part in candidate.relative_to(self.repository_root).parts
        }
        all_parts = parts | resolved_parts

        if all_parts & self._BLOCKED_PARTS:
            raise PolicyViolation("Access to sensitive paths is blocked")

        for part in all_parts:
            if part == ".env" or (
                part.startswith(".env.")
                and part not in {".env.example", ".env.sample"}
            ):
                raise PolicyViolation("Access to environment secrets is blocked")
            if part.endswith((".pem", ".key")) or part in {
                "id_rsa",
                "id_ed25519",
            }:
                raise PolicyViolation("Access to private-key files is blocked")

        return candidate

    def authorize_write(self, relative_path: str, content: str) -> Path:
        if not self.allow_writes:
            raise PolicyViolation(
                "Writes require AGENT_ALLOW_WRITES=true and an explicit "
                "write opt-in for this run"
            )

        target = self.resolve_path(relative_path)
        content_size = len(content.encode("utf-8"))
        if content_size > self.max_write_bytes:
            raise PolicyViolation(
                f"Write exceeds the {self.max_write_bytes}-byte limit"
            )
        return target

    def verification_command(self, name: str) -> tuple[str, ...]:
        if name not in self.allowed_checks:
            raise PolicyViolation(f"Verification command is not allowed: {name}")
        return self._CHECK_COMMANDS[name]
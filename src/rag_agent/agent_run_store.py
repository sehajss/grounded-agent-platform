"""Append developer-agent run records to a JSONL file."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rag_agent.developer_agent import AgentRun


class AgentRunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, run: AgentRun) -> None:
        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **asdict(run),
        }

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str))
            file.write("\n")
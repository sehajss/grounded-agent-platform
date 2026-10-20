"""Create and validate bounded plans for repository tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class PlannedAction:
    tool: str
    arguments: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class AgentPlan:
    summary: str
    actions: tuple[PlannedAction, ...]


class AgentPlanner:
    """Turn a user task and bounded repo context into validated tool calls."""

    def __init__(
        self,
        complete: Callable[[str], Awaitable[str]],
        *,
        allowed_tools: set[str],
        max_actions: int = 8,
    ) -> None:
        self.complete = complete
        self.allowed_tools = set(allowed_tools)
        self.max_actions = max_actions

    async def plan(self, task: str, repository_context: str) -> AgentPlan:
        tool_names = ", ".join(sorted(self.allowed_tools))
        prompt = f"""
You are planning a small software change in a local repository.

Return only valid JSON with this shape:
{{
  "summary": "one sentence describing the approach",
  "actions": [
    {{
      "tool": "one allowed tool name",
      "arguments": {{"key": "value"}},
      "reason": "brief reason for this action"
    }}
  ]
}}

Rules:
- Use only these tools: {tool_names}
- Make the fewest actions needed, at most {self.max_actions}.
- Read relevant files before proposing edits.
- Do not invent file contents, test results, or tool outputs.
- If the task is unclear or cannot be completed safely, return no actions
  and explain that in the summary.
- Tool arguments must be JSON objects.

User task:
{task}

Relevant repository context:
{repository_context}
""".strip()

        raw = await self.complete(prompt)
        payload = self._parse_json(raw)

        summary = payload.get("summary", "")
        if not isinstance(summary, str):
            raise ValueError("Planner response field 'summary' must be text.")

        raw_actions = payload.get("actions", [])
        if not isinstance(raw_actions, list):
            raise ValueError("Planner response field 'actions' must be a list.")
        if len(raw_actions) > self.max_actions:
            raise ValueError(
                f"Planner returned {len(raw_actions)} actions; "
                f"the limit is {self.max_actions}."
            )

        actions: list[PlannedAction] = []
        for index, item in enumerate(raw_actions):
            if not isinstance(item, dict):
                raise ValueError(f"Planner action {index} must be an object.")

            tool = item.get("tool")
            arguments = item.get("arguments", {})
            reason = item.get("reason", "")

            if not isinstance(tool, str) or tool not in self.allowed_tools:
                raise ValueError(f"Planner selected disallowed tool: {tool!r}")
            if not isinstance(arguments, dict):
                raise ValueError(f"Arguments for action {index} must be an object.")
            if not isinstance(reason, str):
                raise ValueError(f"Reason for action {index} must be text.")

            actions.append(
                PlannedAction(tool=tool, arguments=arguments, reason=reason)
            )

        return AgentPlan(summary=summary, actions=tuple(actions))

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = raw.strip()

        # Accept a fenced JSON response, while still requiring valid JSON inside.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Planner did not return valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Planner JSON response must be an object.")
        return payload
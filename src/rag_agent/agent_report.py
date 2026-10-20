"""Format developer-agent run records as Markdown."""

from __future__ import annotations

from rag_agent.developer_agent import AgentRun


def format_agent_report(run: AgentRun) -> str:
    lines = [
        "# Repository Agent Run",
        "",
        f"**Task:** {run.task}",
        "",
        f"**Plan:** {run.plan.summary}",
        "",
        "## Actions",
        "",
    ]

    if not run.trace:
        lines.append("No tools were run.")
    else:
        for index, item in enumerate(run.trace, start=1):
            lines.extend(
                [
                    f"### {index}. `{item.tool}`",
                    "",
                    f"**Reason:** {item.reason}",
                    "",
                    "**Arguments:**",
                    "",
                    "```json",
                    _pretty_json(item.arguments),
                    "```",
                    "",
                    "**Result:**",
                    "",
                    "```text",
                    item.result.rstrip(),
                    "```",
                    "",
                ]
            )

    lines.extend(["## Summary", "", run.final_response, ""])
    return "\n".join(lines)


def _pretty_json(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
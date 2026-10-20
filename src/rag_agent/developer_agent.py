"""Execute one bounded repository task and record its tool trace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from rag_agent.agent_planner import AgentPlan, AgentPlanner
from rag_agent.agent_tools import RepositoryTools
from rag_agent.repository_context import RepositoryContextBuilder


@dataclass(frozen=True)
class ToolTrace:
    tool: str
    arguments: dict[str, Any]
    reason: str
    result: str


@dataclass(frozen=True)
class AgentRun:
    task: str
    plan: AgentPlan
    trace: tuple[ToolTrace, ...]
    final_response: str


class DeveloperAgent:
    """Plan and execute a single bounded task in a local repository."""

    def __init__(
        self,
        *,
        tools: RepositoryTools,
        context_builder: RepositoryContextBuilder,
        planner: AgentPlanner,
        complete: Callable[[str], Awaitable[str]],
        max_tool_calls: int = 8,
    ) -> None:
        self.tools = tools
        self.context_builder = context_builder
        self.planner = planner
        self.complete = complete
        self.max_tool_calls = max_tool_calls

    async def run(self, task: str) -> AgentRun:
        context = self.context_builder.build(task)
        plan = await self.planner.plan(task, context.text)

        trace: list[ToolTrace] = []
        for action in plan.actions[: self.max_tool_calls]:
            try:
                raw_result = self.tools.execute(
                    action.tool,
                    action.arguments,
                )
                result_text = self._result_text(raw_result)
            except Exception as exc:
                result_text = f"Tool failed: {type(exc).__name__}: {exc}"

            trace.append(
                ToolTrace(
                    tool=action.tool,
                    arguments=action.arguments,
                    reason=action.reason,
                    result=result_text[:12_000],
                )
            )

        final_prompt = self._final_prompt(task, plan, trace)
        final_response = await self.complete(final_prompt)

        return AgentRun(
            task=task,
            plan=plan,
            trace=tuple(trace),
            final_response=final_response.strip(),
        )

    @staticmethod
    def _result_text(result: object) -> str:
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

    @staticmethod
    def _final_prompt(
        task: str,
        plan: AgentPlan,
        trace: list[ToolTrace],
    ) -> str:
        steps = []
        for index, item in enumerate(trace, start=1):
            steps.append(
                f"{index}. Tool: {item.tool}\n"
                f"   Reason: {item.reason}\n"
                f"   Result:\n{item.result}"
            )

        trace_text = "\n".join(steps) if steps else "(No tools were run.)"
        return f"""
You are reporting the outcome of a repository task.

User task:
{task}

Plan summary:
{plan.summary}

Actual tool calls and outputs:
{trace_text}

Report only what the recorded tool outputs support. Clearly state if the task
was not completed or if a tool failed. Do not claim that tests passed unless
the trace includes successful test output.
""".strip()
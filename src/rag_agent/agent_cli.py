"""Command-line entry point for running a repository agent task."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from rag_agent.agent_completion import GroqAgentCompletion
from rag_agent.agent_planner import AgentPlanner
from rag_agent.agent_policy import AgentPolicy
from rag_agent.agent_report import format_agent_report
from rag_agent.agent_run_store import AgentRunStore
from rag_agent.agent_tools import RepositoryTools
from rag_agent.developer_agent import DeveloperAgent
from rag_agent.repository_context import RepositoryContextBuilder


TOOL_NAMES = {
    "list_files",
    "read_file",
    "search_text",
    "write_file",
    "run_check",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded AI task against a local repository."
    )
    parser.add_argument("--task", required=True, help="Task for the agent.")
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository folder to inspect (defaults to the current folder).",
    )
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="Allow file edits if AGENT_ENABLE_WRITES=true is also set.",
    )
    parser.add_argument(
        "--report",
        default=".agent-runs.jsonl",
        help="JSONL run log path (defaults to .agent-runs.jsonl in the repo).",
    )
    return parser.parse_args()


async def run_task(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo).expanduser().resolve()
    if not repo_root.is_dir():
        raise SystemExit(f"Repository folder does not exist: {repo_root}")

    api_key = os.getenv("GROQ_API_KEY", "")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    writes_enabled = (
        args.allow_writes
        and os.getenv("AGENT_ENABLE_WRITES", "").strip().lower()
        in {"1", "true", "yes"}
    )

    policy = AgentPolicy(root=repo_root, allow_writes=writes_enabled)
    tools = RepositoryTools(policy=policy)

    allowed_tools = set(TOOL_NAMES)
    if not writes_enabled:
        allowed_tools.discard("write_file")

    completion = GroqAgentCompletion(api_key=api_key, model=model_name)
    try:
        planner = AgentPlanner(
            completion,
            allowed_tools=allowed_tools,
        )
        context_builder = RepositoryContextBuilder(tools)
        agent = DeveloperAgent(
            tools=tools,
            context_builder=context_builder,
            planner=planner,
            complete=completion,
        )

        run = await agent.run(args.task)
        report_path = Path(args.report).expanduser()
        if not report_path.is_absolute():
            report_path = repo_root / report_path

        AgentRunStore(report_path).append(run)
        print(format_agent_report(run))
        print(f"\nRun record saved to: {report_path.resolve()}")
    finally:
        await completion.close()


def main() -> None:
    asyncio.run(run_task(parse_args()))


if __name__ == "__main__":
    main()
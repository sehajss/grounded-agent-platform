# Grounded Agent Platform

An API-first platform for workspace-scoped document retrieval and bounded
repository-agent tasks.

The project combines:

- Document ingestion, chunking, embeddings, and hybrid search with PostgreSQL
  and pgvector.
- A LangGraph-based RAG workflow with source citations and basic input/output
  guardrails.
- A local repository agent that selects relevant files, plans tool calls, and
  records its actions.
- Run metrics and JSONL records for inspecting agent activity.

## Requirements

- Python 3.11 or newer
- Docker Desktop
- A Groq API key for chat completions
- Git, if you want to fetch a documentation corpus

## Setup

Create a virtual environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy the example environment file and fill in the values:

```bash
cp .env.example .env
```

Start the local PostgreSQL database:

```bash
docker compose up -d db
```

Apply the database migration:

```bash
alembic upgrade head
```

Start the API:

```bash
uvicorn rag_agent.api:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive API
documentation is at `http://127.0.0.1:8000/docs`.

## Run a repository-agent task

Set the Groq credentials in your shell:

```bash
export GROQ_API_KEY="your-groq-api-key"
export GROQ_MODEL="llama-3.3-70b-versatile"
```

Run a task in the current repository:

```bash
python -m rag_agent.agent_cli \
  --repo . \
  --task "Explain how authentication works"
```

The agent reads repository files and runs only its allowlisted checks. Each run
is appended to `.agent-runs.jsonl` by default.

### Allow file edits

The agent starts in read-only mode. To permit edits, enable writes in the
environment and pass the command-line option:

```bash
export AGENT_ENABLE_WRITES=true

python -m rag_agent.agent_cli \
  --repo . \
  --allow-writes \
  --task "Add input validation to the upload endpoint"
```

Both settings are required for the CLI to enable write access. Review the
generated run report and resulting changes before using them.

## Project layout

```text
src/rag_agent/
  agent_cli.py             Local repository-agent command
  agent_completion.py      Groq completion adapter
  agent_planner.py         Plan generation and validation
  agent_policy.py          Repository access and write policy
  agent_tools.py           File, search, and check tools
  repository_context.py    Bounded repository context
  developer_agent.py       Tool execution and run trace
  agent_run_store.py       JSONL run log
  agent_report.py          Markdown report formatting
  api.py                   FastAPI application
  ingestion.py             Document ingestion
  retrieval.py             Hybrid document retrieval
  query_service.py         RAG query workflow
```

## Security notes

- Keep `.env` and API keys out of version control.
- Run the repository agent only against folders you trust.
- Keep write access disabled unless the task requires edits.
- The agent’s tool policy and allowed checks are the boundary for repository
  operations. Review those rules before exposing the agent to other users.

## Current scope

This is a development project. Authentication, authorization, retrieval,
guardrails, and agent tooling should be reviewed and hardened before deploying
to a public or production environment.

## Retrieval evaluation

The local FastAPI documentation evaluation uses 10 labeled questions and
reports document-level Recall@5, mean reciprocal rank, and NDCG@5. The latest
run returned Recall@5 **0.85**, mean reciprocal rank **0.67**, and NDCG@5
**0.69**. These are results from a small project evaluation set, not a
comparison against other systems.

After starting the API and uploading the documentation corpus, run:

```bash
python scripts/run_retrieval_eval.py --workspace-id "$WORKSPACE_ID"
```

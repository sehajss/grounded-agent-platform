from prometheus_client import Counter, Histogram


LLM_CALLS = Counter(
    "grounded_agent_llm_calls_total",
    "Language-model calls.",
    ("provider", "model", "outcome"),
)

LLM_LATENCY = Histogram(
    "grounded_agent_llm_duration_seconds",
    "Language-model call duration in seconds.",
    ("provider", "model"),
)

RETRIEVAL_CALLS = Counter(
    "grounded_agent_retrieval_calls_total",
    "Workspace retrieval operations.",
    ("outcome",),
)

RETRIEVAL_LATENCY = Histogram(
    "grounded_agent_retrieval_duration_seconds",
    "Hybrid retrieval duration in seconds.",
)

RETRIEVED_CHUNKS = Histogram(
    "grounded_agent_retrieved_chunks",
    "Number of chunks returned by hybrid retrieval.",
    buckets=(0, 1, 2, 4, 8, 12, 20, 30),
)

INGESTION_DOCUMENTS = Counter(
    "grounded_agent_documents_ingested_total",
    "Document ingestion attempts.",
    ("outcome",),
)

INGESTION_CHUNKS = Counter(
    "grounded_agent_chunks_created_total",
    "Document chunks created.",
)

INGESTION_LATENCY = Histogram(
    "grounded_agent_ingestion_duration_seconds",
    "Document ingestion duration in seconds.",
)

GUARDRAIL_BLOCKS = Counter(
    "grounded_agent_guardrail_blocks_total",
    "Requests blocked by an input guardrail.",
)

INVALID_CITATIONS = Counter(
    "grounded_agent_invalid_citation_markers_total",
    "Generated citation markers that did not map to retrieved excerpts.",
)

AGENT_RUNS = Counter(
    "grounded_agent_developer_runs_total",
    "Developer-agent runs.",
    ("outcome",),
)

AGENT_TOOL_CALLS = Counter(
    "grounded_agent_tool_calls_total",
    "Developer-agent tool invocations.",
    ("tool", "outcome"),
)

AGENT_POLICY_BLOCKS = Counter(
    "grounded_agent_policy_blocks_total",
    "Developer-agent actions blocked by policy.",
    ("tool", "reason"),
)
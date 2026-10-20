from __future__ import annotations

import re
from typing import TypedDict
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agent.config import get_settings
from rag_agent.embeddings import Embedder
from rag_agent.guardrails import (
    redact_sensitive_output,
    screen_question,
    validate_citation_markers,
)
from rag_agent.llm import GroqChat
from rag_agent.retrieval import HybridRetriever, RetrievedChunk
from rag_agent.schemas import Citation, QueryResponse


_STOP_WORDS = {
    "about", "after", "also", "and", "are", "can", "could", "does", "for",
    "from", "how", "into", "is", "its", "the", "this", "that", "then", "there",
    "these", "they", "what", "when", "where", "which", "with", "would",
}


class QueryState(TypedDict, total=False):
    question: str
    query: str
    workspace_id: UUID
    db: AsyncSession
    top_k: int
    retries: int
    candidates: list[RetrievedChunk]
    documents: list[RetrievedChunk]
    low_confidence: bool
    blocked: bool
    answer: str
    grounded: bool


class QueryService:
    def __init__(self, embedder: Embedder) -> None:
        self.settings = get_settings()
        self.retriever = HybridRetriever(embedder)
        self._llm: GroqChat | None = None
        self._graph = None

    @property
    def llm(self) -> GroqChat:
        if self._llm is None:
            self._llm = GroqChat()
        return self._llm

    def _build_graph(self):
        async def guard_node(state: QueryState) -> dict:
            reason = screen_question(state["question"])
            if reason:
                return {
                    "blocked": True,
                    "answer": reason,
                    "documents": [],
                    "grounded": False,
                }
            return {"blocked": False}

        async def retrieve_node(state: QueryState) -> dict:
            documents = await self.retriever.search(
                state["db"],
                workspace_id=state["workspace_id"],
                query=state["query"],
                top_k=state["top_k"],
            )
            return {"candidates": documents}

        async def grade_node(state: QueryState) -> dict:
            candidates = state.get("candidates", [])
            terms = {
                term
                for term in re.findall(r"[a-z0-9_]+", state["query"].lower())
                if len(term) > 2 and term not in _STOP_WORDS
            }
            retrieved_terms = {
                term
                for document in candidates[:3]
                for term in re.findall(r"[a-z0-9_]+", document.content.lower())
            }
            coverage = len(terms & retrieved_terms) / max(len(terms), 1)
            return {
                "documents": candidates,
                "low_confidence": not candidates or coverage < 0.20,
            }

        async def rewrite_node(state: QueryState) -> dict:
            rewritten = await self.llm.rewrite(
                state["question"],
                state["query"],
            )
            return {
                "query": rewritten or state["question"],
                "retries": state.get("retries", 0) + 1,
            }

        async def answer_node(state: QueryState) -> dict:
            documents = state.get("documents", [])
            if not documents or state.get("low_confidence", False):
                return {
                    "answer": (
                        "I could not find sufficiently relevant evidence in this "
                        "workspace’s documents to answer confidently."
                    ),
                    "grounded": False,
                }

            context = "\n\n".join(
                (
                    f"[{index}] File: {document.filename}\n"
                    f"Section: {document.heading or '(no heading)'}\n"
                    f"{document.content}"
                )
                for index, document in enumerate(documents, start=1)
            )
            generated = await self.llm.answer(state["question"], context)
            cleaned, citations_valid, _unknown = validate_citation_markers(
                redact_sensitive_output(generated),
                len(documents),
            )
            valid_markers = re.findall(r"\[(\d+)\]", cleaned)
            grounded = citations_valid and bool(valid_markers)

            if not valid_markers:
                cleaned = (
                    "I could not produce an answer with verifiable citations "
                    "from the retrieved excerpts."
                )

            return {"answer": cleaned, "grounded": grounded}

        def route_after_guard(state: QueryState) -> str:
            return "blocked" if state.get("blocked") else "retrieve"

        def route_after_grade(state: QueryState) -> str:
            if state.get("low_confidence") and state.get("retries", 0) < 1:
                return "rewrite"
            return "answer"

        graph = StateGraph(QueryState)
        graph.add_node("guard", guard_node)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("grade", grade_node)
        graph.add_node("rewrite", rewrite_node)
        graph.add_node("answer", answer_node)

        graph.add_edge(START, "guard")
        graph.add_conditional_edges(
            "guard",
            route_after_guard,
            {"blocked": END, "retrieve": "retrieve"},
        )
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade",
            route_after_grade,
            {"rewrite": "rewrite", "answer": "answer"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("answer", END)

        return graph.compile()

    async def answer(
        self,
        db: AsyncSession,
        *,
        workspace_id: UUID,
        question: str,
        request_id: str | None = None,
        top_k: int | None = None,
    ) -> QueryResponse:
        if self._graph is None:
            self._graph = self._build_graph()

        state: QueryState = await self._graph.ainvoke(
            {
                "question": question,
                "query": question,
                "workspace_id": workspace_id,
                "db": db,
                "top_k": top_k or self.settings.retrieval_top_k,
                "retries": 0,
            }
        )
        documents = state.get("documents", [])

        citations = [
            Citation(
                document_id=document.document_id,
                filename=document.filename,
                chunk_index=document.chunk_index,
                excerpt=document.content[:500],
            )
            for document in documents
        ]

        return QueryResponse(
            question=question,
            answer=state.get("answer", "No answer was generated."),
            citations=citations,
            grounded=state.get("grounded", False),
            request_id=request_id or uuid4().hex,
        )
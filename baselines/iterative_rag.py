"""Iterative RAG baseline: repeated retrieval with checklist evaluation."""

from __future__ import annotations

import json
import time

from core.data_models import GenerationResult, RetrievalResult
from generation.llm_client import LLMClient
from generation.prompt_templates import (
    ITERATIVE_RAG_EVALUATOR,
    ITERATIVE_RAG_FOLLOWUP,
    VANILLA_RAG_SYSTEM,
    VANILLA_RAG_USER,
)
from retrieval.vector_store import VectorStore


class IterativeRAG:
    """Multi-turn RAG: retrieve, generate, evaluate coverage, repeat if needed.

    This baseline measures how many queries and tokens traditional RAG needs
    to achieve the same information coverage as Sentinel-RAG in a single pass.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLMClient,
        top_k: int = 10,
        max_iterations: int = 5,
        coverage_threshold: float = 0.9,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.coverage_threshold = coverage_threshold

    def query(self, question: str, information_checklist: list[str] | None = None) -> GenerationResult:
        start = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        all_chunks = []
        all_scores = []

        retrieval = self.vector_store.search(question, top_k=self.top_k)
        all_chunks.extend(retrieval.chunks)
        all_scores.extend(retrieval.scores)

        context = self._format_context(retrieval)
        prompt = VANILLA_RAG_USER.format(context=context, query=question)
        llm_resp = self.llm.generate(prompt, system_prompt=VANILLA_RAG_SYSTEM)

        current_answer = llm_resp.text
        total_prompt_tokens += llm_resp.prompt_tokens
        total_completion_tokens += llm_resp.completion_tokens
        iterations = 1

        if information_checklist:
            for _ in range(self.max_iterations - 1):
                eval_result = self._evaluate_coverage(question, current_answer, information_checklist)
                coverage = eval_result.get("coverage_ratio", 1.0)
                follow_up = eval_result.get("follow_up_query")

                if coverage >= self.coverage_threshold or not follow_up:
                    break

                follow_retrieval = self.vector_store.search(follow_up, top_k=self.top_k)
                all_chunks.extend(follow_retrieval.chunks)
                all_scores.extend(follow_retrieval.scores)

                follow_context = self._format_context(follow_retrieval)
                follow_prompt = ITERATIVE_RAG_FOLLOWUP.format(
                    context=follow_context,
                    previous_answer=current_answer,
                    follow_up_query=follow_up,
                )
                follow_resp = self.llm.generate(follow_prompt, system_prompt=VANILLA_RAG_SYSTEM)

                current_answer = follow_resp.text
                total_prompt_tokens += follow_resp.prompt_tokens
                total_completion_tokens += follow_resp.completion_tokens
                iterations += 1

        total_latency = time.time() - start

        combined_retrieval = RetrievalResult(
            chunks=all_chunks,
            scores=all_scores,
            retrieval_method="iterative_faiss",
            latency_seconds=total_latency,
        )

        return GenerationResult(
            answer=current_answer,
            retrieved_context=self._format_context(combined_retrieval),
            retrieval_result=combined_retrieval,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            latency_seconds=total_latency,
            num_iterations=iterations,
            model_name=self.llm.model,
        )

    def _evaluate_coverage(self, query: str, answer: str, checklist: list[str]) -> dict:
        checklist_str = "\n".join(f"- {item}" for item in checklist)
        prompt = ITERATIVE_RAG_EVALUATOR.format(
            query=query, answer=answer, checklist=checklist_str,
        )
        resp = self.llm.generate(prompt, max_tokens=512, temperature=0.0)
        try:
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {"coverage_ratio": 0.0, "follow_up_query": None, "covered": [], "missing": checklist}

    @staticmethod
    def _format_context(retrieval: RetrievalResult) -> str:
        parts = []
        for i, chunk in enumerate(retrieval.chunks):
            parts.append(
                f"[Source {i+1}: {chunk.source_document}, {chunk.section_id}]\n{chunk.text}"
            )
        return "\n\n".join(parts)

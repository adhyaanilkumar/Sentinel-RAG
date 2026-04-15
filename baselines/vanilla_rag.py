"""Vanilla RAG baseline: top-k retrieval + single LLM call."""

from __future__ import annotations

import time

from core.data_models import GenerationResult, RetrievalResult
from generation.llm_client import LLMClient
from generation.prompt_templates import VANILLA_RAG_SYSTEM, VANILLA_RAG_USER
from retrieval.vector_store import VectorStore


class VanillaRAG:
    """Single-shot RAG: retrieve top-k chunks, generate answer."""

    def __init__(self, vector_store: VectorStore, llm: LLMClient, top_k: int = 10):
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k

    def query(self, question: str) -> GenerationResult:
        start = time.time()

        retrieval_result = self.vector_store.search(question, top_k=self.top_k)
        context = self._format_context(retrieval_result)
        prompt = VANILLA_RAG_USER.format(context=context, query=question)
        llm_resp = self.llm.generate(prompt, system_prompt=VANILLA_RAG_SYSTEM)

        total_latency = time.time() - start

        return GenerationResult(
            answer=llm_resp.text,
            retrieved_context=context,
            retrieval_result=retrieval_result,
            prompt_tokens=llm_resp.prompt_tokens,
            completion_tokens=llm_resp.completion_tokens,
            total_tokens=llm_resp.total_tokens,
            latency_seconds=total_latency,
            num_iterations=1,
            model_name=llm_resp.model,
        )

    @staticmethod
    def _format_context(retrieval: RetrievalResult) -> str:
        parts = []
        for i, chunk in enumerate(retrieval.chunks):
            parts.append(
                f"[Source {i+1}: {chunk.source_document}, {chunk.section_id}]\n{chunk.text}"
            )
        return "\n\n".join(parts)

"""Prompt templates for RAG generation."""

VANILLA_RAG_SYSTEM = (
    "You are a military doctrine expert assistant. Answer questions using ONLY the "
    "provided context from Army Field Manuals. If the context does not contain enough "
    "information to fully answer the question, say so explicitly. Always cite the "
    "specific FM and section when possible."
)

VANILLA_RAG_USER = """Context:
{context}

Question: {query}

Provide a thorough answer based solely on the context above. Cite specific field manual references."""


ITERATIVE_RAG_EVALUATOR = """You are evaluating whether an answer fully addresses a military doctrine question.

Question: {query}

Current Answer: {answer}

Information Checklist (items the answer should cover):
{checklist}

For each checklist item, determine if it is adequately covered in the answer.
Return a JSON object with:
- "covered": list of checklist items that ARE adequately covered
- "missing": list of checklist items that are NOT covered or only partially covered
- "coverage_ratio": float between 0 and 1 (covered / total)
- "follow_up_query": a specific follow-up query to retrieve the missing information, or null if coverage is complete

Return ONLY valid JSON."""


ITERATIVE_RAG_FOLLOWUP = """Context (additional):
{context}

Previous partial answer: {previous_answer}

Follow-up question: {follow_up_query}

Provide an updated, comprehensive answer that incorporates both the previous answer and any new information from the additional context. Cite specific field manual references."""


SENTINEL_RAG_SYSTEM = (
    "You are Sentinel-RAG, a military doctrine expert with access to a knowledge graph "
    "built from Army Field Manuals. The context provided has been curated through graph-based "
    "retrieval that follows cross-references between documents and sections. Answer thoroughly "
    "using the provided context. When the context includes cross-referenced material from "
    "different FMs or sections, explicitly explain how they relate to the question. "
    "Always cite specific FM and section references."
)

SENTINEL_RAG_USER = """Retrieved Context (graph-optimized, cross-referenced):
{context}

Cross-Reference Notes:
{cross_ref_notes}

Question: {query}

Provide a thorough answer that synthesizes information across all retrieved sections and documents. Pay special attention to any overriding directives, definitions from other sources, and scattered components that together form the complete answer."""

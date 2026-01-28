"""API routes for Sentinel RAG."""

from app.api.routes import analyze, chat, health, ingest, knowledge

__all__ = ["analyze", "chat", "health", "ingest", "knowledge"]

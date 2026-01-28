"""
Sentinel RAG - FastAPI Application Entry Point

An AI-powered military intelligence analysis system combining
real-time sensor data with historical knowledge using RAG.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import analyze, ingest, knowledge, chat, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Debug mode: {settings.DEBUG}")
    print(f"Mock mode: {settings.MOCK_MODE}")
    
    yield
    
    # Shutdown
    print("Shutting down Sentinel RAG...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "AI-powered military intelligence analysis system using "
            "Retrieval-Augmented Generation (RAG) with multimodal capabilities."
        ),
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(
        analyze.router, 
        prefix=settings.API_PREFIX, 
        tags=["Analysis"]
    )
    app.include_router(
        ingest.router, 
        prefix=settings.API_PREFIX, 
        tags=["Ingestion"]
    )
    app.include_router(
        knowledge.router, 
        prefix=settings.API_PREFIX, 
        tags=["Knowledge Base"]
    )
    app.include_router(
        chat.router, 
        prefix=settings.API_PREFIX, 
        tags=["Chat"]
    )
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )

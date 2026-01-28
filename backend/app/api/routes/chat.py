"""
Chat endpoint - Interactive follow-up questions.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from app.api.dependencies import LLMDep, RAGDep
from app.models.chat import ChatRequest, ChatResponse, ChatMessage

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag: RAGDep,
    llm: LLMDep,
):
    """
    Interactive chat for follow-up questions.
    
    Maintains conversation context and retrieves relevant documents
    for each question.
    """
    try:
        # Build conversation context
        conversation_context = "\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in request.messages[-5:]  # Last 5 messages for context
        ])
        
        # Get the latest user message
        latest_message = request.messages[-1].content
        
        # Retrieve relevant documents for the question
        retrieved_docs = await rag.retrieve(
            query=latest_message,
            top_k=3,
        )
        
        # Build context from retrieved documents
        doc_context = "\n\n".join([
            f"[{doc.metadata.get('title', 'Unknown')}]\n{doc.content}"
            for doc in retrieved_docs
        ])
        
        # Generate response
        response = await llm.generate_chat_response(
            conversation=conversation_context,
            question=latest_message,
            retrieved_context=doc_context,
            previous_analysis=request.previous_analysis,
        )
        
        return ChatResponse(
            success=True,
            message=ChatMessage(role="assistant", content=response),
            retrieved_documents=[
                {
                    "id": doc.id,
                    "title": doc.metadata.get("title", "Unknown"),
                    "relevance_score": doc.score,
                }
                for doc in retrieved_docs
            ],
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

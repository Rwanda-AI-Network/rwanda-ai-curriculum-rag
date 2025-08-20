
# app/schemas/chat.py
from app.schemas.base import BaseRequest, BaseResponse
from pydantic import Field
from typing import List, Optional, Dict, Any

class ChatRequest(BaseRequest):
    """
    Request model for chat endpoints.
    Contains the user's question and optional session information.
    """
    
    # The user's question or message
    message: str = Field(..., min_length=1, max_length=1000, 
                        description="User's question or message")
    
    # Optional session ID for conversation tracking
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    
    # Optional context or additional information
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")
    
    # Whether to include sources in the response
    include_sources: bool = Field(True, description="Whether to include document sources")


class ChatResponse(BaseResponse):
    """
    Response model for chat endpoints.
    Contains the AI-generated answer and source information.
    """
    
    # The AI-generated answer
    answer: str = Field(..., description="AI-generated answer to the user's question")
    
    # Sources used to generate the answer
    sources: List[Dict[str, Any]] = Field(default_factory=list, 
                                         description="Source documents used for the answer")
    
    # Session ID for conversation tracking
    session_id: Optional[str] = Field(None, description="Session ID for this conversation")
    
    # Confidence score for the answer
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, 
                                       description="Confidence score for the answer")
    
    # Processing time in milliseconds
    processing_time_ms: Optional[int] = Field(None, description="Time taken to process the request")


class ChatHistoryResponse(BaseResponse):
    """
    Response model for chat history endpoints.
    """
    
    # List of previous messages in the conversation
    history: List[Dict[str, Any]] = Field(default_factory=list, 
                                         description="Conversation history")
    
    # Total number of messages in the session
    total_messages: int = Field(0, description="Total number of messages in the session")
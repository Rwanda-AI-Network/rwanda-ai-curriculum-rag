

# app/schemas/search.py
from app.schemas.base import BaseRequest, BaseResponse
from pydantic import Field
from typing import List, Optional, Dict, Any

class SearchRequest(BaseRequest):
    """
    Request model for document search endpoints.
    """
    
    # Search query
    query: str = Field(..., min_length=1, max_length=500, 
                      description="Search query text")
    
    # Maximum number of results to return
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    
    # Search filters (document type, date range, etc.)
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    
    # Minimum similarity threshold
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, 
                                 description="Minimum similarity threshold")


class SearchResponse(BaseResponse):
    """
    Response model for search endpoints.
    """
    
    # Search results
    results: List[Dict[str, Any]] = Field(default_factory=list, 
                                         description="Search results")
    
    # Total number of results found
    total_results: int = Field(0, description="Total number of results found")
    
    # Query that was searched
    query: str = Field(..., description="The search query")
    
    # Search time in milliseconds
    search_time_ms: Optional[int] = Field(None, description="Time taken for the search")


class SearchSuggestionResponse(BaseResponse):
    """
    Response model for search suggestion endpoints.
    """
    
    # List of suggested search terms
    suggestions: List[str] = Field(default_factory=list, 
                                  description="Suggested search terms")
    
    # Original query that suggestions are based on
    original_query: str = Field(..., description="Original partial query")

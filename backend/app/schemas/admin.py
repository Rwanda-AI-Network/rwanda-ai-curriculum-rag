
# app/schemas/admin.py
from app.schemas.base import BaseRequest, BaseResponse
from pydantic import Field
from typing import List, Optional, Dict, Any

class UploadRequest(BaseRequest):
    """
    Request model for document upload.
    Note: File data is handled separately by FastAPI's UploadFile
    """
    
    # Optional metadata for the uploaded documents
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata for uploaded documents")
    
    # Whether to process documents immediately
    process_immediately: bool = Field(True, description="Process documents immediately after upload")


class UploadResponse(BaseResponse):
    """
    Response model for document upload endpoints.
    """
    
    # List of uploaded file information
    uploaded_files: List[Dict[str, Any]] = Field(default_factory=list, 
                                                 description="Information about uploaded files")
    
    # Processing results for each file
    processing_results: List[Dict[str, Any]] = Field(default_factory=list, 
                                                    description="Processing results for each file")
    
    # Total number of files uploaded
    total_files: int = Field(0, description="Total number of files uploaded")
    
    # Number of files successfully processed
    successful_files: int = Field(0, description="Number of successfully processed files")


class IndexStatsResponse(BaseResponse):
    """
    Response model for index statistics.
    """
    
    # Total number of documents in the index
    total_documents: int = Field(0, description="Total number of documents")
    
    # Total number of text chunks in the index
    total_chunks: int = Field(0, description="Total number of text chunks")
    
    # Index size information
    index_size: Dict[str, Any] = Field(default_factory=dict, 
                                      description="Index size information")
    
    # Last update timestamp
    last_updated: Optional[str] = Field(None, description="Last index update timestamp")
    
    # Health status of the index
    health_status: str = Field("unknown", description="Health status of the index")


class RebuildIndexRequest(BaseRequest):
    """
    Request model for index rebuild.
    """
    
    # Optional directory to rebuild from
    source_directory: Optional[str] = Field(None, description="Directory to rebuild index from")
    
    # Whether to backup existing index before rebuild
    backup_existing: bool = Field(True, description="Backup existing index before rebuild")


class RebuildIndexResponse(BaseResponse):
    """
    Response model for index rebuild.
    """
    
    # Rebuild statistics
    rebuild_stats: Dict[str, Any] = Field(default_factory=dict, 
                                         description="Rebuild statistics")
    
    # Time taken for rebuild in seconds
    rebuild_time_seconds: Optional[float] = Field(None, description="Time taken for rebuild")
    
    # Number of documents processed
    documents_processed: int = Field(0, description="Number of documents processed")
    
    # Any errors encountered during rebuild
    errors: List[str] = Field(default_factory=list, description="Errors encountered during rebuild")

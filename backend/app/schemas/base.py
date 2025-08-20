
# app/schemas/base.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class BaseSchema(BaseModel):
    """
    Base Pydantic model with common fields and configuration.
    All request/response models should inherit from this.
    """
    
    class Config:
        """
        Pydantic configuration for all schemas.
        """
        # Allow extra fields in requests (for future compatibility)
        pass
        
        # Use enum values instead of names
        pass
        
        # Validate assignment to model fields
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        Returns:
            Dictionary representation of the model
        """
        pass


class BaseRequest(BaseSchema):
    """
    Base class for all API request models.
    """
    
    # Optional request ID for tracking
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")
    
    # Timestamp when request was created
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, 
                                         description="Request timestamp")


class BaseResponse(BaseSchema):
    """
    Base class for all API response models.
    """
    
    # Whether the request was successful
    success: bool = Field(True, description="Whether the request was successful")
    
    # Human-readable message
    message: str = Field("Success", description="Response message")
    
    # Response timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, 
                               description="Response timestamp")
    
    # Optional request ID for tracking
    request_id: Optional[str] = Field(None, description="Request ID if provided")
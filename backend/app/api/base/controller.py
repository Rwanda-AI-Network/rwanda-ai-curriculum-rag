# app/api/base/controller.py
from fastapi import HTTPException
import logging
from typing import Any, Dict, Optional

class BaseController:
    """
    Base controller class for all API endpoints.
    Provides common functionality like error handling, logging, and validation.
    
    Usage in your API endpoints:
    controller = BaseController("chat")
    controller.log_request("/chat", {"query": "Hello"})
    """
    
    def __init__(self, service_name: str):
        """
        Initialize the controller for a specific service.
        Args:
            service_name: Name of the service (e.g., "chat", "search", "admin")
        """
        self.service_name = service_name
        self.logger = logging.getLogger(f"api.{service_name}")
    
    def handle_error(self, error: Exception, message: str = "Operation failed") -> HTTPException:
        """
        Handle and log errors, then raise appropriate HTTP exception.
        Args:
            error: The exception that occurred
            message: User-friendly error message
        Raises:
            HTTPException with appropriate status code
        """
        pass
    
    def log_request(self, endpoint: str, data: Dict[str, Any] = None):
        """
        Log incoming API requests for monitoring and debugging.
        Args:
            endpoint: The API endpoint being called (e.g., "/chat", "/search")
            data: Request data to log (be careful with sensitive info)
        """
        pass
    
    def validate_input(self, data: Dict[str, Any], required_fields: list) -> bool:
        """
        Validate that required fields are present in the request.
        Args:
            data: Request data to validate
            required_fields: List of required field names
        Returns:
            True if valid, raises HTTPException if invalid
        """
        pass
    
    def success_response(self, data: Any, message: str = "Success") -> Dict[str, Any]:
        """
        Create a standardized success response.
        Args:
            data: The response data
            message: Success message
        Returns:
            Formatted response dictionary
        """
        pass

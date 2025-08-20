
# app/core/base/service.py
from abc import ABC
import logging
from typing import Any, Dict, Optional

class BaseService(ABC):
    """
    Base class for all services in the application.
    Provides common functionality like logging and error handling.
    
    When creating a new service, inherit from this class:
    class MyService(BaseService):
        def __init__(self):
            super().__init__("my_service")
    """
    
    def __init__(self, name: str):
        """
        Initialize the service with a name.
        Args:
            name: Service name for logging (e.g., "rag", "chat", "search")
        """
        self.name = name
        self.logger = logging.getLogger(f"service.{name}")
        self._setup()
    
    def _setup(self):
        """
        Override this method in child classes to add initialization logic.
        Called automatically after the service is created.
        Example: Initialize database connections, load models, etc.
        """
        pass
    
    def log_info(self, message: str, extra: Dict[str, Any] = None):
        """
        Log an informational message.
        Args:
            message: The message to log
            extra: Additional data to include in the log
        """
        pass
    
    def log_error(self, message: str, error: Exception = None):
        """
        Log an error message.
        Args:
            message: The error message
            error: The exception object (optional)
        """
        pass
    
    def log_debug(self, message: str, extra: Dict[str, Any] = None):
        """
        Log a debug message (only shown in development).
        Args:
            message: The debug message
            extra: Additional debug data
        """
        pass







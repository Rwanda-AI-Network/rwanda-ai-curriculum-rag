
# app/db/base/connection.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseConnection(ABC):
    """
    Base class for database connections.
    Provides a standard interface for different types of databases
    (vector databases like ChromaDB, traditional databases like PostgreSQL).
    """
    
    def __init__(self, connection_string: str, config: Dict[str, Any] = None):
        """
        Initialize database connection.
        Args:
            connection_string: Database connection URL or path
            config: Additional configuration options
        """
        self.connection_string = connection_string
        self.config = config or {}
        self._connection = None
    
    @abstractmethod
    async def connect(self):
        """
        Establish connection to the database.
        Should handle authentication, connection pooling, etc.
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """
        Close the database connection and clean up resources.
        """
        pass
    
    @abstractmethod
    async def is_healthy(self) -> bool:
        """
        Check if the database connection is working properly.
        Returns:
            True if database is accessible and responsive
        """
        pass
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get information about the current connection.
        Returns:
            Dictionary with connection status, database type, etc.
        """
        pass

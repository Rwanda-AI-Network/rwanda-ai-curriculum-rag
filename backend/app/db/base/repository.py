
# app/db/base/repository.py
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
from app.db.base.connection import BaseConnection
from uuid import UUID

class BaseRepository(ABC):
    """
    Base repository class implementing the Repository pattern.
    Handles common database operations (CRUD) for any type of data.
    
    To create a repository for a specific data type:
    1. Inherit from this class
    2. Implement the abstract methods
    3. Add any specific methods your data type needs
    """
    
    def __init__(self, connection: BaseConnection):
        """
        Initialize repository with a database connection.
        Args:
            connection: Database connection instance
        """
        self.connection = connection
    
    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> str:
        """
        Create a new record in the database.
        Args:
            data: Dictionary containing the data to store
        Returns:
            Unique identifier of the created record
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a record by its unique identifier.
        Args:
            record_id: Unique identifier of the record
        Returns:
            Dictionary with record data, or None if not found
        """
        pass
    
    @abstractmethod
    async def update(self, record_id: str, data: Dict[str, Any]) -> bool:
        """
        Update an existing record.
        Args:
            record_id: Unique identifier of the record
            data: Dictionary with updated data
        Returns:
            True if update was successful
        """
        pass
    
    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """
        Delete a record from the database.
        Args:
            record_id: Unique identifier of the record
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    async def list(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List records with pagination.
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
        Returns:
            List of record dictionaries
        """
        pass
    
    async def count(self) -> int:
        """
        Get the total number of records.
        Returns:
            Total count of records in the collection/table
        """
        pass
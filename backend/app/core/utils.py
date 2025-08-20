
# app/core/utils.py
import asyncio
import hashlib
import uuid
from typing import Any, List, Dict, Optional
from datetime import datetime

class Utils:
    """
    Utility functions used throughout the application.
    """
    
    @staticmethod
    def generate_id() -> str:
        """
        Generate a unique ID for records, sessions, etc.
        Returns:
            Unique string identifier
        """
        pass
    
    @staticmethod
    def hash_text(text: str) -> str:
        """
        Create a hash of text for deduplication or caching.
        Args:
            text: Text to hash
        Returns:
            Hash string
        """
        pass
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for safe storage.
        Args:
            filename: Original filename
        Returns:
            Sanitized filename
        """
        pass
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        Args:
            size_bytes: Size in bytes
        Returns:
            Formatted size string (e.g., "1.5 MB")
        """
        pass
    
    @staticmethod
    def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
        """
        Validate that file type is allowed.
        Args:
            filename: Name of the file
            allowed_types: List of allowed file extensions
        Returns:
            True if file type is allowed
        """
        pass
    
    @staticmethod
    async def retry_async(func, max_retries: int = 3, delay: float = 1.0):
        """
        Retry an async function with exponential backoff.
        Args:
            func: Async function to retry
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries
        Returns:
            Function result or raises last exception
        """
        pass
    
    @staticmethod
    def measure_time(func):
        """
        Decorator to measure function execution time.
        Args:
            func: Function to measure
        Returns:
            Decorated function that logs execution time
        """
        pass
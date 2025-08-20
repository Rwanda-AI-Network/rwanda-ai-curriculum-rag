
# app/services/base/processor.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDocumentProcessor(ABC):
    """
    Base class for processing different types of documents.
    Each document type (PDF, Word, PowerPoint) has its own processor.
    
    To add support for a new document type:
    1. Inherit from this class
    2. Implement extract_text method
    3. Add the file extensions to supported_types
    """
    
    def __init__(self, supported_types: List[str]):
        """
        Initialize the processor with supported file types.
        Args:
            supported_types: List of file extensions (e.g., ['.pdf', '.docx'])
        """
        self.supported_types = supported_types
    
    def can_process(self, file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        Args:
            file_path: Path to the file
        Returns:
            True if this processor supports the file type
        """
        pass
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from the document.
        Args:
            file_path: Path to the document file
        Returns:
            Extracted text content
        """
        pass
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and extract metadata + text.
        Args:
            file_path: Path to the document
        Returns:
            Dictionary with text, metadata, and processing info
        """
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """
        Check if the file exists and is readable.
        Args:
            file_path: Path to the file
        Returns:
            True if file is valid and accessible
        """
        pass


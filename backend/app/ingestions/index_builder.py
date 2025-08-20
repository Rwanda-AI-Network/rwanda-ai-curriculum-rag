
# app/ingestion/index_builder.py
from app.core.base.service import BaseService
from app.ingestions.base.processor import BaseDocumentProcessor 
from app.ingestions.text_chunker import TextChunker 
from app.db.base.repository import BaseRepository
from app.sevices.base.embedding_service import BaseEmbeddingService


from typing import List, Dict, Any

class IndexBuilder(BaseService):
    """
    Service for building and rebuilding the search index.
    Processes documents, creates embeddings, and stores them in the vector database.
    """
    
    def __init__(self, 
                 document_processor: BaseDocumentProcessor,
                 embedding_service: BaseEmbeddingService,
                 document_repository: BaseRepository,
                 chunker: TextChunker):
        """
        Initialize index builder with required services.
        Args:
            document_processor: Service for extracting text from documents
            embedding_service: Service for creating embeddings
            document_repository: Repository for storing document data
            chunker: Service for splitting text into chunks
        """
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.document_repository = document_repository
        self.chunker = chunker
        super().__init__("index_builder")
    
    async def process_document(self, file_path: str) -> List[str]:
        """
        Process a single document and add it to the index.
        
        Steps:
        1. Extract text from document
        2. Split text into chunks
        3. Create embeddings for each chunk
        4. Store chunks and embeddings in database
        
        Args:
            file_path: Path to the document file
        Returns:
            List of chunk IDs that were created
        """
        pass
    
    async def rebuild_index(self, document_directory: str):
        """
        Rebuild the entire search index from a directory of documents.
        
        Steps:
        1. Clear existing index
        2. Process all documents in the directory
        3. Update index statistics
        
        Args:
            document_directory: Path to directory containing documents
        """
        pass
    
    async def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Add multiple documents to the index.
        Args:
            file_paths: List of paths to document files
        Returns:
            Summary of processing results (success count, errors, etc.)
        """
        pass
    
    async def remove_document(self, document_id: str) -> bool:
        """
        Remove a document and all its chunks from the index.
        Args:
            document_id: ID of the document to remove
        Returns:
            True if removal was successful
        """
        pass
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        Returns:
            Dictionary with document count, chunk count, index size, etc.
        """
        pass

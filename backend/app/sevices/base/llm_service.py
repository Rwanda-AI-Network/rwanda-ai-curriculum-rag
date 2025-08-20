
# ===================================================================
# SERVICE LAYER - Business logic and AI services
# ===================================================================

# app/services/base/llm_service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLLMService(ABC):
    """
    Base class for all Language Model services (OpenAI, HuggingFace, etc.).
    Define the interface that all LLM providers must implement.
    
    To add a new LLM provider:
    1. Create a new class that inherits from this
    2. Implement all the abstract methods
    3. Register it in the service registry
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        """
        Initialize the LLM service with model and configuration.
        Args:
            model_name: Name of the model (e.g., "gpt-3.5-turbo", "llama-2-7b")
            config: Configuration dictionary (API keys, timeouts, etc.)
        """
        self.model_name = model_name
        self.config = config or {}
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """
        Initialize the specific LLM provider.
        Examples: 
        - Set up API client for OpenAI
        - Load model for HuggingFace
        - Configure authentication
        """
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response from the LLM.
        Args:
            prompt: Input prompt to send to the LLM
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def is_healthy(self) -> bool:
        """
        Check if the LLM service is working properly.
        Returns:
            True if service is healthy, False otherwise
        """
        pass
    
    def prepare_rag_prompt(self, question: str, context: List[str]) -> str:
        """
        Create a RAG prompt combining question and retrieved context.
        This method is the same for all LLM providers.
        Args:
            question: User's question
            context: List of relevant document chunks
        Returns:
            Formatted prompt ready for the LLM
        """
        pass
    
    def prepare_chat_prompt(self, message: str, chat_history: List[Dict] = None) -> str:
        """
        Create a chat prompt with conversation history.
        Args:
            message: Current user message
            chat_history: Previous messages in the conversation
        Returns:
            Formatted prompt for chat-style interaction
        """
        pass


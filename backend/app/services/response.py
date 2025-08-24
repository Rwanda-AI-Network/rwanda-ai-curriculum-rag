"""
Response Processing Service

This service handles response processing, formatting, and post-processing
for the AI curriculum assistant system.

Key Features:
- Response formatting and validation
- Language-specific processing
- Response enrichment and enhancement
- Quality assessment
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
import re


@dataclass
class ResponseMetadata:
    """Metadata for AI responses."""
    
    response_id: str
    timestamp: datetime
    language: str
    confidence_score: float
    source_type: str  # "rag", "llm", "hybrid"
    processing_time: float
    tokens_used: int = 0
    retrieval_context: Optional[str] = None


class ResponseProcessor:
    """
    Main response processing service.
    
    This service handles all response post-processing operations
    including formatting, validation, and enhancement.
    """
    
    def __init__(self, 
                 default_language: str = "en",
                 enable_enrichment: bool = True):
        """
        Initialize response processor.
        
        Args:
            default_language: Default response language
            enable_enrichment: Enable response enrichment
        """
        self.default_language = default_language
        self.enable_enrichment = enable_enrichment
        self.response_cache = {}
    
    def process_response(self,
                        raw_response: str,
                        question: str,
                        metadata: Optional[ResponseMetadata] = None) -> Dict[str, Any]:
        """
        Process and format AI response.
        
        Implementation Guide:
        1. Clean and validate response
        2. Format for display
        3. Add metadata
        4. Enhance if enabled
        5. Return processed response
        
        Args:
            raw_response: Raw AI response text
            question: Original user question
            metadata: Response metadata
            
        Returns:
            Processed response dictionary
        """
        # TODO: Implement comprehensive response processing
        
        # Basic cleaning
        cleaned_response = self._clean_response(raw_response)
        
        # Format response
        formatted_response = self._format_response(cleaned_response, metadata)
        
        # Add enhancements if enabled
        if self.enable_enrichment:
            formatted_response = self._enrich_response(formatted_response, question)
        
        return {
            "response": formatted_response,
            "metadata": metadata.__dict__ if metadata else {},
            "processed_at": datetime.now().isoformat(),
            "language": self._detect_language(cleaned_response),
            "quality_score": self._assess_quality(cleaned_response)
        }
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and sanitize response text.
        
        Args:
            response: Raw response text
            
        Returns:
            Cleaned response text
        """
        # TODO: Implement comprehensive cleaning
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', response.strip())
        
        # Remove any potential harmful content markers
        cleaned = re.sub(r'<script.*?</script>', '', cleaned, flags=re.DOTALL)
        
        # Ensure proper sentence endings
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned
    
    def _format_response(self, 
                        response: str, 
                        metadata: Optional[ResponseMetadata] = None) -> str:
        """
        Format response for display.
        
        Args:
            response: Cleaned response text
            metadata: Response metadata
            
        Returns:
            Formatted response text
        """
        # TODO: Implement rich formatting
        
        # Basic paragraph formatting
        paragraphs = response.split('\n\n')
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Clean up paragraph spacing
                clean_paragraph = re.sub(r'\s+', ' ', paragraph.strip())
                formatted_paragraphs.append(clean_paragraph)
        
        return '\n\n'.join(formatted_paragraphs)
    
    def _enrich_response(self, response: str, question: str) -> str:
        """
        Enrich response with additional context.
        
        Args:
            response: Formatted response
            question: Original question
            
        Returns:
            Enriched response
        """
        # TODO: Implement response enrichment
        
        enriched = response
        
        # Add helpful context for AI/ML topics
        if any(term in question.lower() for term in ['machine learning', 'ai', 'algorithm']):
            if 'practical applications' not in response.lower():
                enriched += "\n\nFor practical applications and examples, feel free to ask for more specific use cases."
        
        return enriched
    
    def _detect_language(self, text: str) -> str:
        """
        Detect primary language of response.
        
        Args:
            text: Response text to analyze
            
        Returns:
            Language code
        """
        # TODO: Implement proper language detection
        
        # Simple heuristic detection
        kinyarwanda_words = ['mu', 'ku', 'wa', 'ba', 'ubushobozi', 'amahoro']
        
        word_count = 0
        kinyarwanda_count = 0
        
        words = text.lower().split()
        for word in words:
            word_count += 1
            if word in kinyarwanda_words or any(kw in word for kw in kinyarwanda_words):
                kinyarwanda_count += 1
        
        if word_count > 0 and kinyarwanda_count / word_count > 0.2:
            return "rw"
        
        return "en"
    
    def _assess_quality(self, response: str) -> float:
        """
        Assess response quality.
        
        Args:
            response: Response text to assess
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        # TODO: Implement sophisticated quality assessment
        
        score = 0.5  # Base score
        
        # Length check
        if len(response) > 50:
            score += 0.1
        if len(response) > 200:
            score += 0.1
        
        # Structure check
        if '\n' in response:  # Has paragraphs
            score += 0.1
        
        # Completeness check
        if response.endswith('.'):
            score += 0.1
        
        # Information density
        if len(response.split()) > 20:
            score += 0.2
        
        return min(1.0, score)
    
    def validate_response(self, response: str) -> Tuple[bool, List[str]]:
        """
        Validate response content.
        
        Args:
            response: Response to validate
            
        Returns:
            Tuple of (is_valid, issues)
        """
        # TODO: Implement comprehensive validation
        issues = []
        
        if not response.strip():
            issues.append("Response is empty")
        
        if len(response) < 10:
            issues.append("Response is too short")
        
        # Check for placeholder text
        placeholders = ["TODO", "[PLACEHOLDER]", "Not implemented"]
        for placeholder in placeholders:
            if placeholder in response:
                issues.append(f"Contains placeholder: {placeholder}")
        
        return len(issues) == 0, issues
    
    def generate_followup_suggestions(self, 
                                    response: str, 
                                    question: str) -> List[str]:
        """
        Generate follow-up question suggestions.
        
        Args:
            response: AI response
            question: Original question
            
        Returns:
            List of suggested follow-up questions
        """
        # TODO: Implement intelligent suggestion generation
        
        suggestions = []
        
        # Topic-based suggestions
        if 'machine learning' in question.lower() or 'ml' in question.lower():
            suggestions.extend([
                "Can you explain this with a practical example?",
                "What are the prerequisites for learning this?",
                "How is this used in real-world applications?"
            ])
        
        if 'algorithm' in question.lower():
            suggestions.extend([
                "What's the time complexity of this algorithm?",
                "Are there alternative approaches?",
                "Can you show the step-by-step process?"
            ])
        
        # Generic suggestions
        suggestions.extend([
            "Can you provide more details about this topic?",
            "What are common challenges with this approach?",
            "How does this relate to other AI concepts?"
        ])
        
        return suggestions[:5]  # Return top 5


class ResponseCache:
    """Simple response caching mechanism."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        return self.cache.get(key)
    
    def set(self, key: str, response: Dict[str, Any]) -> None:
        """Cache response."""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = response
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


def create_error_response(error_message: str, 
                         error_code: str = "PROCESSING_ERROR") -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        error_message: Error description
        error_code: Error code
        
    Returns:
        Formatted error response
    """
    return {
        "response": f"I apologize, but I encountered an error while processing your request: {error_message}",
        "metadata": {
            "error": True,
            "error_code": error_code,
            "timestamp": datetime.now().isoformat()
        },
        "processed_at": datetime.now().isoformat(),
        "language": "en",
        "quality_score": 0.0
    }


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format source information for display.
    
    Args:
        sources: List of source documents
        
    Returns:
        Formatted source string
    """
    if not sources:
        return ""
    
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        title = source.get('title', 'Unknown Source')
        url = source.get('url', '')
        
        if url:
            formatted_sources.append(f"{i}. {title} ({url})")
        else:
            formatted_sources.append(f"{i}. {title}")
    
    return "\n\nSources:\n" + "\n".join(formatted_sources)

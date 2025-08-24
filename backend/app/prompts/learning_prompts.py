"""
Rwanda AI Curriculum RAG - Learning Prompts

This module contains prompt templates for various learning scenarios
in the curriculum RAG system, supporting both English and Kinyarwanda.
"""

from typing import Dict, List, Optional, Any
from enum import Enum

class PromptType(Enum):
    """Types of learning prompts."""
    EXPLANATION = "explanation"
    QUESTION_ANSWER = "question_answer"
    EXAMPLE = "example"
    SUMMARY = "summary"
    EXERCISE = "exercise"

class LearningPromptTemplate:
    """
    Template system for generating learning prompts.
    
    Implementation Guide:
    1. Define templates for different learning scenarios
    2. Support parameter substitution
    3. Handle both English and Kinyarwanda
    4. Include context awareness (grade level, subject)
    5. Provide prompt validation and formatting
    
    Example:
        template = LearningPromptTemplate()
        prompt = template.generate_explanation_prompt(
            topic="photosynthesis",
            grade_level=5,
            language="en"
        )
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize prompt template system.
        
        Args:
            language: Default language for prompts (en/rw)
        """
        self.language = language
        self._templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Initialize prompt templates for different languages.
        
        Implementation Guide:
        1. Define templates for each prompt type
        2. Include both English and Kinyarwanda versions
        3. Use placeholder syntax for parameter substitution
        4. Ensure templates are pedagogically sound
        5. Include system prompts and user prompts
        
        Returns:
            Dictionary of templates organized by type and language
        """
        # TODO: Define comprehensive prompt templates
        # 1. Create templates for all prompt types
        # 2. Include both languages
        # 3. Add parameter placeholders
        # 4. Validate template structure
        return {}  # Placeholder return
        
    def generate_explanation_prompt(self, 
                                   topic: str,
                                   context: Optional[str] = None,
                                   grade_level: Optional[int] = None,
                                   language: Optional[str] = None) -> str:
        """
        Generate prompt for explaining a concept.
        
        Implementation Guide:
        1. Select appropriate explanation template
        2. Customize for grade level and subject
        3. Include context if provided
        4. Format with clear structure
        5. Add pedagogical guidance
        
        Args:
            topic: Topic to explain
            context: Optional context information
            grade_level: Target grade level
            language: Language for explanation
            
        Returns:
            Formatted explanation prompt
        """
        # TODO: Generate explanation prompts
        # 1. Select template based on parameters
        # 2. Substitute topic and context
        # 3. Adjust for grade level
        # 4. Format final prompt
        return ""  # Placeholder return
        
    def generate_qa_prompt(self,
                          question: str,
                          context: Optional[str] = None,
                          grade_level: Optional[int] = None,
                          language: Optional[str] = None) -> str:
        """
        Generate prompt for answering questions.
        
        Implementation Guide:
        1. Use Q&A template structure
        2. Include context for RAG scenarios
        3. Add instructions for age-appropriate answers
        4. Include source citation requirements
        5. Handle follow-up question scenarios
        
        Args:
            question: Question to answer
            context: Retrieved context for RAG
            grade_level: Student grade level
            language: Response language
            
        Returns:
            Formatted Q&A prompt
        """
        # TODO: Generate Q&A prompts
        # 1. Format question with context
        # 2. Add answering guidelines
        # 3. Include citation instructions
        # 4. Adjust for grade level
        return ""  # Placeholder return
        
    def generate_summary_prompt(self,
                               content: str,
                               max_length: int = 200,
                               focus_areas: Optional[List[str]] = None,
                               language: Optional[str] = None) -> str:
        """
        Generate prompt for summarizing content.
        
        Implementation Guide:
        1. Include summarization instructions
        2. Specify length requirements
        3. Highlight focus areas if provided
        4. Maintain key concepts and facts
        5. Use appropriate reading level
        
        Args:
            content: Content to summarize
            max_length: Maximum summary length
            focus_areas: Key areas to emphasize
            language: Summary language
            
        Returns:
            Formatted summary prompt
        """
        # TODO: Generate summary prompts
        # 1. Format content for summarization
        # 2. Add length and focus constraints
        # 3. Include quality requirements
        return ""  # Placeholder return
        
    def generate_exercise_prompt(self,
                                topic: str,
                                difficulty: str = "medium",
                                exercise_type: str = "practice",
                                grade_level: Optional[int] = None,
                                language: Optional[str] = None) -> str:
        """
        Generate prompt for creating practice exercises.
        
        Implementation Guide:
        1. Specify exercise requirements
        2. Include difficulty level guidance
        3. Define exercise format expectations
        4. Add assessment criteria
        5. Include answer key instructions
        
        Args:
            topic: Topic for exercises
            difficulty: Difficulty level (easy/medium/hard)
            exercise_type: Type of exercise (practice/assessment/review)
            grade_level: Target grade level
            language: Exercise language
            
        Returns:
            Formatted exercise generation prompt
        """
        # TODO: Generate exercise prompts
        # 1. Format topic and requirements
        # 2. Add difficulty constraints
        # 3. Include exercise type specifications
        return ""  # Placeholder return
        
    def customize_for_grade(self, 
                           base_prompt: str,
                           grade_level: int) -> str:
        """
        Customize prompt for specific grade level.
        
        Implementation Guide:
        1. Adjust vocabulary complexity
        2. Modify explanation depth
        3. Include grade-appropriate examples
        4. Set appropriate expectations
        5. Consider cognitive development stage
        
        Args:
            base_prompt: Base prompt template
            grade_level: Target grade level (1-12)
            
        Returns:
            Grade-appropriate prompt
        """
        # TODO: Implement grade-level customization
        # 1. Analyze vocabulary level
        # 2. Adjust complexity
        # 3. Add appropriate context
        return ""  # Placeholder return
        
    def add_context_instructions(self,
                                prompt: str,
                                context: str,
                                citation_required: bool = True) -> str:
        """
        Add context and citation instructions to prompt.
        
        Implementation Guide:
        1. Format context clearly
        2. Add citation requirements
        3. Specify how to use context
        4. Include accuracy instructions
        5. Handle multiple sources
        
        Args:
            prompt: Base prompt
            context: Context information
            citation_required: Whether to require citations
            
        Returns:
            Enhanced prompt with context instructions
        """
        # TODO: Add context handling instructions
        # 1. Format context clearly
        # 2. Add usage guidelines
        # 3. Include citation requirements
        return ""  # Placeholder return

# Pre-defined prompt templates
SYSTEM_PROMPTS = {
    "en": {
        "general": """You are an AI tutor helping students learn curriculum content. 
        Provide clear, accurate, and age-appropriate explanations. Always cite your sources when using provided context.""",
        
        "explanation": """Explain the given topic clearly and simply. Use examples that students can relate to. 
        Break complex concepts into smaller, manageable parts.""",
        
        "qa": """Answer the student's question accurately and completely. Use the provided context to support your answer. 
        If you're not certain about something, say so clearly."""
    },
    
    "rw": {
        "general": """Uri mwarimu w'ubwenge bw'ubuhanga ufasha abanyeshuri kwiga ibinyambo. 
        Tanga ibisobanuro byoroshye, byo'ukuri, kandi bikwiriye imyaka yabo.""",
        
        "explanation": """Sobanura ingingo yatanzwe mu buryo bwuroshye kandi bwumvikana. 
        Koresha ingero abanyeshuri bashobora guhuza nabyo.""",
        
        "qa": """Subiza ikibazo cy'umunyeshuri mu buryo bwukuri kandi buzuye. 
        Koresha imiterere yatanzwe kugira ngo ushyigikire igisubizo cyawe."""
    }
}

def get_system_prompt(prompt_type: str = "general", language: str = "en") -> str:
    """
    Get system prompt for given type and language.
    
    Args:
        prompt_type: Type of system prompt
        language: Language code (en/rw)
        
    Returns:
        System prompt string
    """
    return SYSTEM_PROMPTS.get(language, {}).get(prompt_type, SYSTEM_PROMPTS["en"]["general"])

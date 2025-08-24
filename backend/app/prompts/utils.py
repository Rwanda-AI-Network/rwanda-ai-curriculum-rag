"""
Prompt Utilities Module

This module provides utility functions for working with prompts,
including prompt templates, formatting, and language-specific handling.

Key Features:
- Prompt template management
- Dynamic prompt generation
- Language-specific prompt handling
- Prompt validation and formatting
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import re


class PromptUtils:
    """
    Utility class for prompt operations.
    
    This class provides helper methods for prompt formatting,
    template management, and language handling.
    """
    
    @staticmethod
    def format_template(template: str, variables: Dict[str, Any]) -> str:
        """
        Format prompt template with variables.
        
        Implementation Guide:
        1. Parse template placeholders
        2. Replace with variable values
        3. Handle missing variables
        4. Return formatted prompt
        
        Args:
            template: Template string with placeholders
            variables: Dictionary of variable values
            
        Returns:
            Formatted prompt string
        """
        # TODO: Implement robust template formatting
        try:
            return template.format(**variables)
        except KeyError as e:
            # Handle missing variables gracefully
            missing_var = str(e).strip("'")
            return template.replace(f"{{{missing_var}}}", f"[{missing_var}]")
    
    @staticmethod
    def extract_placeholders(template: str) -> List[str]:
        """
        Extract placeholder variables from template.
        
        Args:
            template: Template string
            
        Returns:
            List of placeholder variable names
        """
        # TODO: Implement placeholder extraction
        pattern = r'\{(\w+)\}'
        return re.findall(pattern, template)
    
    @staticmethod
    def validate_prompt(prompt: str, 
                       max_length: Optional[int] = None,
                       required_elements: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate prompt content.
        
        Args:
            prompt: Prompt to validate
            max_length: Maximum allowed length
            required_elements: Required elements in prompt
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        # TODO: Implement comprehensive validation
        errors = []
        
        if not prompt.strip():
            errors.append("Prompt is empty")
        
        if max_length and len(prompt) > max_length:
            errors.append(f"Prompt exceeds maximum length of {max_length}")
        
        if required_elements:
            for element in required_elements:
                if element not in prompt:
                    errors.append(f"Required element '{element}' missing from prompt")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def adapt_for_language(prompt: str, 
                          target_language: str = "rw") -> str:
        """
        Adapt prompt for specific language.
        
        Implementation Guide:
        1. Identify language-specific elements
        2. Apply language rules
        3. Adjust formatting
        4. Return adapted prompt
        
        Args:
            prompt: Original prompt
            target_language: Target language code
            
        Returns:
            Language-adapted prompt
        """
        # TODO: Implement language adaptation
        if target_language == "rw":
            # Add Kinyarwanda context markers
            return f"Kinyarwanda Context:\n{prompt}\n\nPlease respond in Kinyarwanda when appropriate."
        return prompt
    
    @staticmethod
    def truncate_safely(prompt: str, 
                       max_tokens: int,
                       preserve_structure: bool = True) -> str:
        """
        Safely truncate prompt while preserving structure.
        
        Args:
            prompt: Prompt to truncate
            max_tokens: Maximum number of tokens
            preserve_structure: Whether to preserve prompt structure
            
        Returns:
            Truncated prompt
        """
        # TODO: Implement smart truncation
        # Simple character-based truncation for now
        if len(prompt) <= max_tokens * 4:  # Rough token estimation
            return prompt
        
        if preserve_structure:
            # Try to preserve the end of the prompt (usually instructions)
            lines = prompt.split('\n')
            if len(lines) > 2:
                # Keep first and last few lines
                truncated = '\n'.join(lines[:2]) + '\n...\n' + '\n'.join(lines[-2:])
                return truncated
        
        return prompt[:max_tokens * 4] + "..."


def create_system_prompt(role: str = "assistant",
                        context: str = "",
                        language: str = "en") -> str:
    """
    Create a system prompt for the AI assistant.
    
    Implementation Guide:
    1. Define role and capabilities
    2. Add context information
    3. Set language preferences
    4. Include safety guidelines
    
    Args:
        role: Assistant role
        context: Additional context
        language: Primary language
        
    Returns:
        Formatted system prompt
    """
    # TODO: Implement comprehensive system prompt
    base_prompt = f"""You are a helpful AI {role} for the Rwanda AI Curriculum project.
    
Your primary role is to assist with educational content related to artificial intelligence,
focusing on curriculum development and learning support.

Language: {'Kinyarwanda and English' if language == 'rw' else 'English'}
Context: {context if context else 'General AI curriculum assistance'}

Guidelines:
- Provide accurate, educational information
- Adapt explanations to the user's level
- Support both English and Kinyarwanda when appropriate
- Focus on practical, actionable guidance
"""
    
    return base_prompt.strip()


def create_rag_prompt(question: str,
                     context: str,
                     language: str = "en") -> str:
    """
    Create RAG prompt combining question and context.
    
    Args:
        question: User question
        context: Retrieved context
        language: Response language
        
    Returns:
        Formatted RAG prompt
    """
    # TODO: Implement sophisticated RAG prompt
    language_instruction = ""
    if language == "rw":
        language_instruction = "Please respond in Kinyarwanda when appropriate.\n"
    
    prompt = f"""Based on the following context, please answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

{language_instruction}
Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to answer the question completely, please indicate what information is missing."""
    
    return prompt


def extract_key_concepts(text: str) -> List[str]:
    """
    Extract key concepts from text for prompt enhancement.
    
    Implementation Guide:
    1. Identify important terms
    2. Extract technical concepts
    3. Find domain-specific vocabulary
    4. Return ranked concepts
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of key concepts
    """
    # TODO: Implement sophisticated concept extraction
    # Simple keyword extraction for now
    words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', text)
    
    # Filter for potential concepts
    concepts = []
    concept_indicators = ['algorithm', 'model', 'data', 'learning', 'neural', 'intelligence']
    
    for word in words:
        if any(indicator in word.lower() for indicator in concept_indicators):
            concepts.append(word)
    
    return list(set(concepts))


def generate_follow_up_questions(response: str, 
                               original_question: str) -> List[str]:
    """
    Generate follow-up questions based on response.
    
    Args:
        response: AI response
        original_question: Original user question
        
    Returns:
        List of suggested follow-up questions
    """
    # TODO: Implement intelligent follow-up generation
    followups = [
        "Can you provide more examples?",
        "What are the practical applications?",
        "How does this relate to other AI concepts?",
        "What are the current limitations?",
        "Are there any prerequisites I should know?"
    ]
    
    return followups[:3]  # Return top 3 for now

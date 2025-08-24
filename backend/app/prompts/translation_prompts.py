"""
Rwanda AI Curriculum RAG - Translation Prompts

This module contains prompt templates and guidance for translating
content between English and Kinyarwanda, ensuring quality and
cultural appropriateness.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum

class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    KINYARWANDA = "rw"

class TranslationStyle(Enum):
    """Translation styles"""
    LITERAL = "literal"      # Direct word-for-word
    NATURAL = "natural"      # Flowing, natural language
    ACADEMIC = "academic"    # Formal academic style
    SIMPLIFIED = "simple"    # Simplified for students

class TranslationPrompts:
    """
    Translation prompt templates and guidelines.
    
    Implementation Guide:
    1. Support multiple styles:
       - Literal translation
       - Natural language
       - Academic format
    2. Handle subject-specific:
       - Technical terms
       - Mathematical notation
       - Scientific concepts
    3. Preserve meaning:
       - Cultural context
       - Educational value
    4. Quality control:
       - Accuracy checks
       - Cultural review
    
    Example:
        prompts = TranslationPrompts()
        
        template = prompts.get_template(
            source_lang=Language.ENGLISH,
            target_lang=Language.KINYARWANDA,
            style=TranslationStyle.ACADEMIC
        )
    """
    
    _templates: Dict[Tuple[Language, Language, TranslationStyle, Optional[str]], str]
    
    def __init__(self):
        """
        Initialize prompt templates.
        
        Implementation Guide:
        1. Load templates:
           - Read files
           - Parse formats
        2. Setup validation:
           - Quality checks
           - Format rules
        3. Initialize helpers:
           - Term dictionary
           - Style guides
        4. Configure logging:
           - Track usage
           - Monitor quality
        """
        self._load_templates()
        
    def get_template(self,
                    source_lang: Language,
                    target_lang: Language,
                    style: TranslationStyle = TranslationStyle.NATURAL,
                    subject: Optional[str] = None) -> str:
        """
        Get translation prompt template.
        
        Implementation Guide:
        1. Select template:
           - Match languages
           - Apply style
        2. Add context:
           - Subject area
           - Grade level
        3. Include guidance:
           - Quality rules
           - Common issues
        4. Format output:
           - Add placeholders
           - Set markers
           
        Args:
            source_lang: Source language
            target_lang: Target language
            style: Translation style
            subject: Optional subject
            
        Returns:
            Prompt template string
        """
        # Create a key that matches the template dict structure
        key = (source_lang, target_lang, style, subject)
        
        # Try with subject first, then without subject
        template = self._templates.get(key)
        if template is None and subject is not None:
            # Try without subject
            key_no_subject = (source_lang, target_lang, style, None)
            template = self._templates.get(key_no_subject)
        
        return template or self._get_default_template(source_lang, target_lang, style)
    
    def _get_default_template(self, source_lang: Language, target_lang: Language, style: TranslationStyle) -> str:
        """Get default template for language/style combination."""
        return f"Translate the following text from {source_lang.value} to {target_lang.value} in {style.value} style:\\n\\n{{text}}"
        
    def _load_templates(self) -> None:
        """
        Load translation templates.
        
        Implementation Guide:
        1. Read files:
           - Template JSON
           - Config YAML
        2. Parse content:
           - Validate format
           - Check required
        3. Process rules:
           - Style guides
           - Term lists
        4. Initialize cache:
           - Store templates
           - Set indices
        """
        self._templates = {
            # English to Kinyarwanda - Academic
            (Language.ENGLISH, Language.KINYARWANDA, TranslationStyle.ACADEMIC, None):
            '''
            You are an expert academic translator specializing in {subject} content.
            Translate the following English text to Kinyarwanda, maintaining academic
            rigor while ensuring clarity for students.
            
            Important guidelines:
            1. Preserve technical terms
            2. Maintain formal tone
            3. Keep mathematical notation
            4. Add explanatory notes if needed
            
            Text to translate:
            {text}
            
            Translation:
            ''',
            
            # Kinyarwanda to English - Natural
            (Language.KINYARWANDA, Language.ENGLISH, TranslationStyle.NATURAL, None):
            '''
            You are a skilled translator focusing on natural, flowing English.
            Translate the following Kinyarwanda text to English, maintaining
            the original meaning while using natural expression.
            
            Important guidelines:
            1. Use natural phrasing
            2. Preserve key concepts
            3. Maintain tone
            4. Consider context
            
            Text to translate:
            {text}
            
            Translation:
            '''
        }
        
    def get_quality_checklist(self,
                            source_lang: Language,
                            target_lang: Language) -> List[str]:
        """
        Get translation quality checklist.
        
        Implementation Guide:
        1. Select checks:
           - Language pair
           - Content type
        2. Add rules:
           - Grammar points
           - Common errors
        3. Include examples:
           - Good/bad cases
           - Explanations
        4. Format output:
           - Organize points
           - Add details
           
        Args:
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            List of quality checks
        """
        return [
            "Verify technical term accuracy",
            "Check grammar and structure",
            "Confirm cultural appropriateness",
            "Validate educational clarity",
            "Review tone and register"
        ]
        
    def get_subject_terms(self,
                         subject: str,
                         source_lang: Language,
                         target_lang: Language) -> Dict[str, str]:
        """
        Get subject-specific term translations.
        
        Implementation Guide:
        1. Load dictionary:
           - Get subject
           - Match languages
        2. Filter terms:
           - Check context
           - Verify usage
        3. Add metadata:
           - Usage notes
           - Examples
        4. Format output:
           - Sort terms
           - Add context
           
        Args:
            subject: Academic subject
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Term translation dictionary
        """
        # TODO: Implement this function

        return {}
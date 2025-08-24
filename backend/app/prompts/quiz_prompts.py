"""
Rwanda AI Curriculum RAG - Quiz Generation Prompts

This module contains comprehensive prompt templates for generating various types
of quiz questions aligned with the Rwandan curriculum. It supports multiple
question types, difficulty levels, languages, and subject areas.

Key Features:
- Multiple choice, true/false, short answer, essay questions
- Grade-level appropriate content (P1-P6, S1-S6)
- Bilingual support (English/Kinyarwanda)
- Subject-specific prompts
- Competency-based assessment alignment
- Bloom's taxonomy integration
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Supported quiz question types."""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false" 
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    FILL_IN_BLANK = "fill_in_blank"
    MATCHING = "matching"
    ORDERING = "ordering"


class DifficultyLevel(Enum):
    """Question difficulty levels based on Bloom's taxonomy."""
    REMEMBER = "remember"  # Level 1 - Recall facts
    UNDERSTAND = "understand"  # Level 2 - Explain concepts
    APPLY = "apply"  # Level 3 - Use knowledge
    ANALYZE = "analyze"  # Level 4 - Break down information
    EVALUATE = "evaluate"  # Level 5 - Make judgments
    CREATE = "create"  # Level 6 - Produce new work


class GradeLevel(Enum):
    """Rwanda education system grade levels."""
    P1 = "primary_1"
    P2 = "primary_2"
    P3 = "primary_3"
    P4 = "primary_4"
    P5 = "primary_5"
    P6 = "primary_6"
    S1 = "secondary_1"
    S2 = "secondary_2"
    S3 = "secondary_3"
    S4 = "secondary_4"
    S5 = "secondary_5"
    S6 = "secondary_6"


class Subject(Enum):
    """Major subjects in Rwandan curriculum."""
    MATHEMATICS = "mathematics"
    ENGLISH = "english"
    KINYARWANDA = "kinyarwanda"
    SOCIAL_STUDIES = "social_studies"
    SCIENCE = "science"
    ICT = "ict"
    ENTREPRENEURSHIP = "entrepreneurship"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    LITERATURE = "literature"


class QuizPromptGenerator:
    """
    Main class for generating quiz prompts for different question types.
    
    Implementation Guide:
    1. Question Generation:
       - Use curriculum-aligned content
       - Ensure appropriate difficulty level
       - Include clear instructions
       - Provide answer explanations
    
    2. Language Support:
       - Generate in English or Kinyarwanda
       - Maintain cultural relevance
       - Use appropriate vocabulary
    
    3. Assessment Quality:
       - Align with learning objectives
       - Follow assessment best practices
       - Ensure fair and unbiased questions
    
    Example:
        generator = QuizPromptGenerator()
        
        prompt = generator.generate_multiple_choice_prompt(
            topic="Fractions",
            grade_level=GradeLevel.P4,
            subject=Subject.MATHEMATICS,
            language="en"
        )
    """
    
    def __init__(self):
        """Initialize the quiz prompt generator."""
        # TODO: Load configuration and templates
        self.base_templates = self._load_base_templates()
        self.subject_specific_templates = self._load_subject_templates()
        self.language_templates = self._load_language_templates()
        
        logger.info("QuizPromptGenerator initialized")
    
    def generate_multiple_choice_prompt(self,
                                      topic: str,
                                      content: str,
                                      grade_level: GradeLevel,
                                      subject: Subject,
                                      difficulty: DifficultyLevel = DifficultyLevel.UNDERSTAND,
                                      language: str = "en",
                                      num_questions: int = 5,
                                      **kwargs) -> str:
        """
        Generate prompt for multiple choice questions.
        
        Implementation Guide:
        1. Content Analysis:
           - Extract key concepts from content
           - Identify testable knowledge
           - Ensure curriculum alignment
        
        2. Question Design:
           - Create clear, unambiguous questions
           - Design plausible distractors
           - Avoid trick questions
           - Ensure one clearly correct answer
        
        3. Format Requirements:
           - Consistent structure
           - Clear numbering
           - Proper answer key
        
        Args:
            topic: The topic/lesson focus
            content: Source content for questions
            grade_level: Target grade level
            subject: Subject area
            difficulty: Question difficulty level
            language: Language for questions (en/rw)
            num_questions: Number of questions to generate
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt for multiple choice generation
        """
        # TODO: Implement multiple choice prompt generation
        
        # Base prompt structure
        base_prompt = f"""
        You are an expert educational assessment designer for the Rwanda curriculum.
        Your task is to create {num_questions} high-quality multiple-choice questions.
        
        CONTENT CONTEXT:
        Topic: {topic}
        Subject: {subject.value}
        Grade Level: {grade_level.value}
        Difficulty: {difficulty.value}
        Language: {language}
        
        SOURCE CONTENT:
        {content}
        
        INSTRUCTIONS:
        1. Create {num_questions} multiple-choice questions based on the content
        2. Each question should have 4 options (A, B, C, D)
        3. Ensure only one correct answer per question
        4. Make distractors plausible but clearly incorrect
        5. Questions should be appropriate for {grade_level.value} students
        6. Align with Rwanda curriculum competencies
        7. Use clear, simple language appropriate for the grade level
        """
        
        # Add subject-specific guidelines
        subject_guidelines = self._get_subject_specific_guidelines(subject, grade_level)
        
        # Add language-specific formatting
        language_guidelines = self._get_language_guidelines(language)
        
        # Add difficulty-specific requirements
        difficulty_guidelines = self._get_difficulty_guidelines(difficulty, grade_level)
        
        full_prompt = f"{base_prompt}\n\n{subject_guidelines}\n\n{language_guidelines}\n\n{difficulty_guidelines}"
        
        return full_prompt
    
    def generate_true_false_prompt(self,
                                 topic: str,
                                 content: str,
                                 grade_level: GradeLevel,
                                 subject: Subject,
                                 difficulty: DifficultyLevel = DifficultyLevel.REMEMBER,
                                 language: str = "en",
                                 num_questions: int = 10,
                                 **kwargs) -> str:
        """
        Generate prompt for true/false questions.
        
        Implementation Guide:
        1. Statement Design:
           - Create clear, unambiguous statements
           - Avoid partially true statements
           - Balance true and false questions
           - Avoid negative language confusion
        
        2. Content Coverage:
           - Cover key facts and concepts
           - Test important information
           - Avoid trivial details
        
        Args:
            topic: The topic/lesson focus
            content: Source content for questions
            grade_level: Target grade level
            subject: Subject area
            difficulty: Question difficulty level
            language: Language for questions (en/rw)
            num_questions: Number of questions to generate
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt for true/false generation
        """
        # TODO: Implement true/false prompt generation
        
        base_prompt = f"""
        Create {num_questions} true/false questions for {grade_level.value} students
        studying {subject.value}.
        
        Topic: {topic}
        Content: {content}
        
        Requirements:
        - Balance of true and false statements
        - Clear, unambiguous statements
        - No trick questions
        - Grade-appropriate language
        - Include explanations for answers
        """
        
        return self._enhance_prompt_with_context(base_prompt, subject, grade_level, difficulty, language)
    
    def generate_short_answer_prompt(self,
                                   topic: str,
                                   content: str,
                                   grade_level: GradeLevel,
                                   subject: Subject,
                                   difficulty: DifficultyLevel = DifficultyLevel.APPLY,
                                   language: str = "en",
                                   num_questions: int = 5,
                                   **kwargs) -> str:
        """
        Generate prompt for short answer questions.
        
        Implementation Guide:
        1. Question Design:
           - Ask for specific information
           - Require brief, focused responses
           - Test understanding, not memorization
           - Provide clear success criteria
        
        2. Answer Guidelines:
           - Specify expected answer length
           - Provide scoring criteria
           - Include sample answers
        
        Args:
            topic: The topic/lesson focus
            content: Source content for questions
            grade_level: Target grade level
            subject: Subject area
            difficulty: Question difficulty level
            language: Language for questions (en/rw)
            num_questions: Number of questions to generate
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt for short answer generation
        """
        # TODO: Implement short answer prompt generation
        
        base_prompt = f"""
        Design {num_questions} short answer questions requiring 1-3 sentence responses.
        
        Context:
        - Topic: {topic}
        - Subject: {subject.value}
        - Grade: {grade_level.value}
        - Content: {content}
        
        Each question should:
        - Test understanding, not just recall
        - Have clear, specific answers
        - Be appropriate for {grade_level.value} level
        - Include marking criteria
        """
        
        return self._enhance_prompt_with_context(base_prompt, subject, grade_level, difficulty, language)
    
    def generate_essay_prompt(self,
                            topic: str,
                            content: str,
                            grade_level: GradeLevel,
                            subject: Subject,
                            difficulty: DifficultyLevel = DifficultyLevel.EVALUATE,
                            language: str = "en",
                            num_questions: int = 2,
                            **kwargs) -> str:
        """
        Generate prompt for essay questions.
        
        Implementation Guide:
        1. Question Design:
           - Require higher-order thinking
           - Allow for multiple valid approaches
           - Test synthesis and analysis
           - Provide clear expectations
        
        2. Assessment Criteria:
           - Define evaluation rubric
           - Specify required elements
           - Set word count guidelines
           - Include sample responses
        
        Args:
            topic: The topic/lesson focus
            content: Source content for questions
            grade_level: Target grade level
            subject: Subject area
            difficulty: Question difficulty level
            language: Language for questions (en/rw)
            num_questions: Number of questions to generate
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt for essay generation
        """
        # TODO: Implement essay prompt generation
        
        word_counts = self._get_appropriate_word_count(grade_level)
        
        base_prompt = f"""
        Create {num_questions} essay questions for comprehensive assessment.
        
        Parameters:
        - Topic: {topic}
        - Subject: {subject.value}
        - Grade Level: {grade_level.value}
        - Expected Length: {word_counts['min']}-{word_counts['max']} words
        
        Source Content:
        {content}
        
        Requirements:
        - Questions should require analysis, synthesis, or evaluation
        - Include clear rubric criteria
        - Provide example thesis statements
        - Specify required components
        """
        
        return self._enhance_prompt_with_context(base_prompt, subject, grade_level, difficulty, language)
    
    def generate_fill_in_blank_prompt(self,
                                    topic: str,
                                    content: str,
                                    grade_level: GradeLevel,
                                    subject: Subject,
                                    difficulty: DifficultyLevel = DifficultyLevel.REMEMBER,
                                    language: str = "en",
                                    num_questions: int = 10,
                                    **kwargs) -> str:
        """
        Generate prompt for fill-in-the-blank questions.
        
        Implementation Guide:
        1. Sentence Selection:
           - Choose sentences with key terms
           - Ensure context provides clues
           - Avoid ambiguous blanks
           - Test important vocabulary
        
        2. Blank Design:
           - Remove one key term per sentence
           - Provide word bank when appropriate
           - Ensure single correct answer
        
        Args:
            topic: The topic/lesson focus
            content: Source content for questions
            grade_level: Target grade level
            subject: Subject area
            difficulty: Question difficulty level
            language: Language for questions (en/rw)
            num_questions: Number of questions to generate
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt for fill-in-blank generation
        """
        # TODO: Implement fill-in-blank prompt generation
        
        base_prompt = f"""
        Create {num_questions} fill-in-the-blank questions for {grade_level.value} students.
        
        Context:
        - Topic: {topic}
        - Subject: {subject.value}
        - Content: {content}
        
        Requirements:
        - One blank per sentence
        - Context should provide clues
        - Include word bank when helpful
        - Test key vocabulary and concepts
        """
        
        return self._enhance_prompt_with_context(base_prompt, subject, grade_level, difficulty, language)
    
    def generate_matching_prompt(self,
                               topic: str,
                               content: str,
                               grade_level: GradeLevel,
                               subject: Subject,
                               difficulty: DifficultyLevel = DifficultyLevel.UNDERSTAND,
                               language: str = "en",
                               num_pairs: int = 8,
                               **kwargs) -> str:
        """
        Generate prompt for matching questions.
        
        Implementation Guide:
        1. Item Selection:
           - Choose related concepts/terms
           - Ensure clear relationships
           - Avoid ambiguous matches
           - Balance difficulty
        
        2. Format Design:
           - Create two columns
           - Add extra distractors
           - Provide clear instructions
        
        Args:
            topic: The topic/lesson focus
            content: Source content for questions
            grade_level: Target grade level
            subject: Subject area
            difficulty: Question difficulty level
            language: Language for questions (en/rw)
            num_pairs: Number of matching pairs
            **kwargs: Additional parameters
            
        Returns:
            Formatted prompt for matching generation
        """
        # TODO: Implement matching prompt generation
        
        base_prompt = f"""
        Create {num_pairs} matching question pairs for {grade_level.value} students.
        
        Context:
        - Topic: {topic}
        - Subject: {subject.value}
        - Content: {content}
        
        Format:
        - Two columns: terms and definitions/descriptions
        - Include {num_pairs + 2} items in second column (extra distractors)
        - Clear relationship between matched items
        - Appropriate difficulty for grade level
        """
        
        return self._enhance_prompt_with_context(base_prompt, subject, grade_level, difficulty, language)
    
    def _load_base_templates(self) -> Dict[str, str]:
        """
        Load base prompt templates.
        
        Implementation Guide:
        1. Load from configuration files
        2. Organize by question type
        3. Include common elements
        4. Support template variables
        
        Returns:
            Dictionary of base templates
        """
        # TODO: Implement template loading
        return {
            'multiple_choice': 'Base MC template...',
            'true_false': 'Base T/F template...',
            'short_answer': 'Base SA template...',
            'essay': 'Base essay template...'
        }
    
    def _load_subject_templates(self) -> Dict[Subject, Dict[str, str]]:
        """
        Load subject-specific prompt templates.
        
        Implementation Guide:
        1. Load subject-specific guidelines
        2. Include curriculum standards
        3. Add subject terminology
        4. Include assessment criteria
        
        Returns:
            Dictionary of subject templates
        """
        # TODO: Implement subject-specific templates
        return {
            Subject.MATHEMATICS: {
                'guidelines': 'Math-specific assessment guidelines...',
                'vocabulary': 'Mathematical terminology...',
                'standards': 'Rwanda math curriculum standards...'
            },
            Subject.SCIENCE: {
                'guidelines': 'Science assessment guidelines...',
                'vocabulary': 'Scientific terminology...',
                'standards': 'Rwanda science curriculum standards...'
            }
            # Add other subjects...
        }
    
    def _load_language_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Load language-specific templates.
        
        Implementation Guide:
        1. Support English and Kinyarwanda
        2. Include cultural considerations
        3. Adapt vocabulary levels
        4. Format instructions appropriately
        
        Returns:
            Dictionary of language templates
        """
        # TODO: Implement language-specific templates
        return {
            'en': {
                'instructions': 'Answer all questions clearly...',
                'format': 'English formatting guidelines...'
            },
            'rw': {
                'instructions': 'Subiza ibibazo byose neza...',
                'format': 'Kinyarwanda formatting guidelines...'
            }
        }
    
    def _get_subject_specific_guidelines(self, subject: Subject, grade_level: GradeLevel) -> str:
        """
        Get subject-specific assessment guidelines.
        
        Implementation Guide:
        1. Map subject to guidelines
        2. Adjust for grade level
        3. Include curriculum alignment
        4. Add subject vocabulary
        
        Args:
            subject: Target subject
            grade_level: Target grade level
            
        Returns:
            Subject-specific guidelines string
        """
        # TODO: Implement subject guidelines
        guidelines_map = {
            Subject.MATHEMATICS: f"Focus on mathematical reasoning and problem-solving appropriate for {grade_level.value}.",
            Subject.SCIENCE: f"Emphasize scientific method and inquiry skills for {grade_level.value}.",
            Subject.SOCIAL_STUDIES: f"Include Rwanda's history, culture, and civic education for {grade_level.value}."
        }
        
        return guidelines_map.get(subject, f"Follow general curriculum guidelines for {subject.value} at {grade_level.value} level.")
    
    def _get_language_guidelines(self, language: str) -> str:
        """
        Get language-specific formatting guidelines.
        
        Implementation Guide:
        1. Adapt vocabulary complexity
        2. Include cultural context
        3. Format appropriately
        4. Ensure accessibility
        
        Args:
            language: Target language code
            
        Returns:
            Language-specific guidelines
        """
        # TODO: Implement language guidelines
        if language == "rw":
            return """
            LANGUAGE GUIDELINES (KINYARWANDA):
            - Use appropriate Kinyarwanda vocabulary
            - Include cultural context relevant to Rwanda
            - Ensure proper grammar and sentence structure
            - Use formal register appropriate for educational content
            """
        else:
            return """
            LANGUAGE GUIDELINES (ENGLISH):
            - Use clear, simple English appropriate for the grade level
            - Avoid complex sentence structures
            - Include Rwandan context in examples
            - Use vocabulary suitable for ESL learners
            """
    
    def _get_difficulty_guidelines(self, difficulty: DifficultyLevel, grade_level: GradeLevel) -> str:
        """
        Get difficulty-specific requirements based on Bloom's taxonomy.
        
        Implementation Guide:
        1. Map difficulty to cognitive level
        2. Adjust for grade level
        3. Include action verbs
        4. Provide examples
        
        Args:
            difficulty: Target difficulty level
            grade_level: Target grade level
            
        Returns:
            Difficulty-specific guidelines
        """
        # TODO: Implement difficulty guidelines
        difficulty_map = {
            DifficultyLevel.REMEMBER: "Focus on recall of facts, terms, and basic concepts.",
            DifficultyLevel.UNDERSTAND: "Test comprehension and explanation of ideas.",
            DifficultyLevel.APPLY: "Require use of knowledge in new situations.",
            DifficultyLevel.ANALYZE: "Ask students to break down complex information.",
            DifficultyLevel.EVALUATE: "Require judgment and decision-making skills.",
            DifficultyLevel.CREATE: "Challenge students to produce new work."
        }
        
        base_guideline = difficulty_map.get(difficulty, "Follow general assessment guidelines.")
        
        return f"""
        DIFFICULTY LEVEL ({difficulty.value.upper()}):
        {base_guideline}
        Adjust complexity for {grade_level.value} cognitive development.
        """
    
    def _get_appropriate_word_count(self, grade_level: GradeLevel) -> Dict[str, int]:
        """
        Get appropriate word count ranges for essays by grade level.
        
        Implementation Guide:
        1. Map grade levels to word counts
        2. Consider cognitive development
        3. Allow for individual differences
        4. Align with curriculum expectations
        
        Args:
            grade_level: Target grade level
            
        Returns:
            Dictionary with min and max word counts
        """
        # TODO: Implement grade-appropriate word counts
        word_count_map = {
            GradeLevel.P1: {'min': 25, 'max': 50},
            GradeLevel.P2: {'min': 30, 'max': 60},
            GradeLevel.P3: {'min': 50, 'max': 100},
            GradeLevel.P4: {'min': 75, 'max': 150},
            GradeLevel.P5: {'min': 100, 'max': 200},
            GradeLevel.P6: {'min': 150, 'max': 250},
            GradeLevel.S1: {'min': 200, 'max': 300},
            GradeLevel.S2: {'min': 250, 'max': 350},
            GradeLevel.S3: {'min': 300, 'max': 400},
            GradeLevel.S4: {'min': 350, 'max': 500},
            GradeLevel.S5: {'min': 400, 'max': 600},
            GradeLevel.S6: {'min': 500, 'max': 800}
        }
        
        return word_count_map.get(grade_level, {'min': 100, 'max': 200})
    
    def _enhance_prompt_with_context(self, 
                                   base_prompt: str, 
                                   subject: Subject, 
                                   grade_level: GradeLevel, 
                                   difficulty: DifficultyLevel, 
                                   language: str) -> str:
        """
        Enhance a base prompt with contextual guidelines.
        
        Implementation Guide:
        1. Add subject-specific context
        2. Include grade-level adjustments
        3. Apply difficulty requirements
        4. Add language formatting
        
        Args:
            base_prompt: Base prompt text
            subject: Target subject
            grade_level: Target grade level
            difficulty: Target difficulty
            language: Target language
            
        Returns:
            Enhanced prompt with full context
        """
        # TODO: Implement prompt enhancement
        subject_context = self._get_subject_specific_guidelines(subject, grade_level)
        language_context = self._get_language_guidelines(language)
        difficulty_context = self._get_difficulty_guidelines(difficulty, grade_level)
        
        enhanced_prompt = f"""
        {base_prompt}
        
        {subject_context}
        
        {language_context}
        
        {difficulty_context}
        
        OUTPUT FORMAT:
        Please format your response as structured JSON with the following fields:
        - questions: Array of question objects
        - answer_key: Correct answers
        - explanations: Answer explanations
        - metadata: Question metadata (difficulty, topic, etc.)
        """
        
        return enhanced_prompt


class QuizPromptValidator:
    """
    Validator for quiz prompts and generated questions.
    
    Implementation Guide:
    1. Validate prompt structure and completeness
    2. Check curriculum alignment
    3. Verify language appropriateness
    4. Ensure assessment quality
    """
    
    def __init__(self):
        """Initialize the validator."""
        # TODO: Load validation rules and criteria
        pass
    
    def validate_prompt(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a quiz prompt for completeness and quality.
        
        Implementation Guide:
        1. Check required elements present
        2. Validate curriculum alignment
        3. Verify language appropriateness
        4. Ensure clear instructions
        
        Args:
            prompt: The prompt text to validate
            context: Context information (subject, grade, etc.)
            
        Returns:
            Validation result with issues and recommendations
        """
        # TODO: Implement prompt validation
        return {
            'valid': True,
            'issues': [],
            'recommendations': [],
            'score': 0.95
        }
    
    def validate_generated_questions(self, questions: List[Dict], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate generated quiz questions for quality and appropriateness.
        
        Implementation Guide:
        1. Check question clarity and structure
        2. Validate answer options
        3. Ensure appropriate difficulty
        4. Verify curriculum alignment
        
        Args:
            questions: List of generated questions
            context: Context information
            
        Returns:
            Validation result for questions
        """
        # TODO: Implement question validation
        return {
            'valid': True,
            'question_scores': [],
            'overall_score': 0.90,
            'recommendations': []
        }

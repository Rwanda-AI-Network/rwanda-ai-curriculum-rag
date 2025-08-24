"""
Rwanda AI Curriculum RAG - Constants

This module defines system-wide constants and enums used
throughout the application, with proper typing and documentation.
"""

from enum import Enum
from typing import Dict, List, Set, Final, Any
from pathlib import Path

# Path Constants
ROOT_DIR: Final[Path] = Path(__file__).parent.parent.parent
DATA_DIR: Final[Path] = ROOT_DIR / "datasets"
MODEL_DIR: Final[Path] = ROOT_DIR / "models"
CACHE_DIR: Final[Path] = ROOT_DIR / "cache"

class Environment(Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class ModelType(Enum):
    """Supported model types"""
    RAG = "rag"
    LLM = "llm"
    HYBRID = "hybrid"
    FINE_TUNED = "fine_tuned"

class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    KINYARWANDA = "rw"

class Subject(Enum):
    """Academic subjects"""
    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    SOCIAL_STUDIES = "social_studies"
    LANGUAGE = "language"
    COMPUTER_SCIENCE = "computer_science"

class Grade(Enum):
    """Grade levels"""
    GRADE_1 = 1
    GRADE_2 = 2
    GRADE_3 = 3
    GRADE_4 = 4
    GRADE_5 = 5
    GRADE_6 = 6
    GRADE_7 = 7
    GRADE_8 = 8
    GRADE_9 = 9
    GRADE_10 = 10
    GRADE_11 = 11
    GRADE_12 = 12

# API Version
API_VERSION: Final[str] = "v1"
API_PREFIX: Final[str] = f"/api/{API_VERSION}"

# Security Constants
JWT_ALGORITHM: Final[str] = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: Final[int] = 30
REFRESH_TOKEN_EXPIRE_DAYS: Final[int] = 7

# Rate Limiting
RATE_LIMIT_PER_MINUTE: Final[int] = 20
RATE_LIMIT_BURST: Final[int] = 5

# Cache Settings
CACHE_TTL: Final[int] = 3600  # 1 hour
MAX_CACHE_SIZE: Final[int] = 1000  # items

# Model Settings
MAX_SEQUENCE_LENGTH: Final[int] = 512
DEFAULT_TEMPERATURE: Final[float] = 0.7
TOP_P: Final[float] = 0.9
TOP_K: Final[int] = 40

# Vector Store Settings
VECTOR_DIMENSION: Final[int] = 384
SIMILARITY_TOP_K: Final[int] = 5
SIMILARITY_THRESHOLD: Final[float] = 0.7

# Supported File Types
SUPPORTED_DOCUMENT_TYPES: Final[Set[str]] = {
    ".pdf",
    ".txt",
    ".doc",
    ".docx",
    ".csv",
    ".xlsx"
}

# Error Messages
ERROR_MESSAGES: Final[Dict[str, str]] = {
    "auth_failed": "Authentication failed",
    "token_expired": "Token has expired",
    "rate_limited": "Too many requests",
    "invalid_input": "Invalid input provided",
    "not_found": "Resource not found",
    "server_error": "Internal server error"
}

# Success Messages
SUCCESS_MESSAGES: Final[Dict[str, str]] = {
    "login_success": "Successfully logged in",
    "logout_success": "Successfully logged out",
    "update_success": "Successfully updated",
    "delete_success": "Successfully deleted"
}

# Curriculum Competencies
COMPETENCIES: Final[Dict[str, List[str]]] = {
    "mathematics": [
        "Problem Solving",
        "Critical Thinking",
        "Mathematical Communication",
        "Logical Reasoning"
    ],
    "science": [
        "Scientific Inquiry",
        "Data Analysis",
        "Experimental Design",
        "Scientific Communication"
    ],
    "language": [
        "Reading Comprehension",
        "Writing Skills",
        "Oral Communication",
        "Literature Analysis"
    ]
}

# Quiz Settings
QUIZ_TYPES: Final[List[str]] = [
    "multiple_choice",
    "true_false",
    "short_answer",
    "essay",
    "matching"
]

DEFAULT_QUIZ_LENGTH: Final[int] = 10
MIN_QUIZ_LENGTH: Final[int] = 5
MAX_QUIZ_LENGTH: Final[int] = 20

# Content Moderation
FORBIDDEN_TOPICS: Final[Set[str]] = {
    "violence",
    "hate_speech",
    "discrimination",
    "adult_content"
}

# Logging Levels
LOG_LEVELS: Final[Dict[str, int]] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

# Default Settings
DEFAULTS: Final[Dict[str, Any]] = {
    "language": Language.ENGLISH,
    "model_type": ModelType.RAG,
    "cache_enabled": True,
    "offline_mode": False,
    "debug_mode": False
}

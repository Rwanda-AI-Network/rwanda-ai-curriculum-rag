# Contributing to Rwanda AI Curriculum RAG System 🇷🇼

Welcome to the Rwanda AI Curriculum RAG System! We're excited to have you contribute to advancing education through artificial intelligence in Rwanda and beyond. This guide will help you get started with contributing to our project.

[![Contributors](https://img.shields.io/github/contributors/Rwanda-AI-Network/rwanda-ai-curriculum-rag.svg)](https://github.com/Rwanda-AI-Network/rwanda-ai-curriculum-rag/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/Rwanda-AI-Network/rwanda-ai-curriculum-rag.svg)](https://github.com/Rwanda-AI-Network/rwanda-ai-curriculum-rag/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/Rwanda-AI-Network/rwanda-ai-curriculum-rag.svg)](https://github.com/Rwanda-AI-Network/rwanda-ai-curriculum-rag/pulls)

---

## 📋 Table of Contents

1. [🎯 Project Mission](#project-mission)
2. [🚀 Quick Start Guide](#quick-start-guide)
3. [📁 Project Architecture](#project-architecture)
4. [🛠️ Development Environment](#development-environment)
5. [🔧 Contribution Types](#contribution-types)
6. [📝 Development Guidelines](#development-guidelines)
7. [🧪 Testing Standards](#testing-standards)
8. [📚 Documentation Standards](#documentation-standards)
9. [🔄 Git Workflow](#git-workflow)
10. [👥 Community Guidelines](#community-guidelines)
11. [🏆 Recognition](#recognition)
12. [❓ Getting Help](#getting-help)

---

## 🎯 Project Mission

The Rwanda AI Curriculum RAG System aims to **democratize access to quality education through AI-powered learning tools**. Our mission is to:

- 🎓 **Enhance Learning Outcomes** - Provide personalized, context-aware educational assistance
- 🌍 **Bridge Educational Gaps** - Make quality educational resources accessible to all
- 🤖 **Advance AI in Education** - Pioneer innovative applications of AI technology in learning
- 🇷🇼 **Support Local Context** - Ensure cultural relevance and linguistic diversity
- 🤝 **Foster Collaboration** - Build a community-driven educational platform

### 🌟 Impact Areas

- **Students** - Personalized learning assistance and interactive content
- **Educators** - Enhanced teaching tools and curriculum management
- **Institutions** - Scalable educational technology solutions
- **Community** - Open-source educational innovation

---

## 🚀 Quick Start Guide

### ✅ Prerequisites

Before contributing, ensure you have:

- **Python 3.13+** (Required for latest type system features)
- **Git** for version control
- **[uv](https://github.com/astral-sh/uv)** - Modern Python package manager
- **Node.js 18+** (if contributing to documentation)
- **Docker** (optional, for containerized development)

### ⚡ 5-Minute Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/rwanda-ai-curriculum-rag.git
cd rwanda-ai-curriculum-rag/backend

# 2. Set up development environment
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
uv sync

# 4. Configure environment
cp .env.example app/config/env_files/.env

# 5. Run tests to verify setup
uv run pytest tests/test_response.py -v

# 6. Start development server
uv run uvicorn app.main:app --reload
```

### 🎯 First Contribution Ideas

Perfect for newcomers:

- 📝 **Documentation** - Improve code comments or README sections
- 🐛 **Bug Fixes** - Fix issues labeled `good-first-issue`
- 🧪 **Tests** - Add test cases for existing functionality
- 🌍 **Localization** - Add support for local languages
- 📊 **Examples** - Create usage examples and tutorials

---

## 📁 Project Architecture

Understanding our architecture helps you contribute effectively:

### 🏗️ High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │────│   REST API      │────│  AI Services    │
│   (Future)      │    │   (FastAPI)     │    │  (LLM/RAG)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                       ┌───────┼───────┐
                       │               │
                ┌─────────────┐ ┌─────────────┐
                │   Vector    │ │  Knowledge  │
                │   Store     │ │   Base      │
                └─────────────┘ └─────────────┘
```

### 📦 Module Structure

```
app/
├── 📡 api/                    # REST API Layer
│   ├── middleware.py          # CORS, auth, logging
│   └── v1/                    # API version 1
│       ├── auth.py            # 🔐 Authentication
│       ├── curriculum.py      # 📚 Content management
│       ├── quiz.py            # 📊 Assessment tools
│       ├── search.py          # 🔍 Search endpoints
│       ├── chat.py            # 💬 AI chat interface
│       └── admin.py           # 👨‍💼 Admin functions
│
├── 🧠 Core Services           # Business Logic
│   ├── data_loader/           # 📥 Data ingestion
│   ├── embeddings/            # 🔢 Vector operations
│   ├── models/                # 🤖 AI model management
│   ├── prompts/               # 💭 Prompt engineering
│   └── services/              # ⚙️ Core orchestration
│
└── 🔧 Infrastructure          # Support Systems
    ├── config/                # ⚙️ Configuration
    ├── logger.py              # 📋 Logging
    └── main.py                # 🚀 Application entry
```

### 🔄 Data Flow

1. **Request** → API endpoint receives user request
2. **Authentication** → Verify user permissions
3. **Processing** → Route to appropriate service
4. **AI Pipeline** → RAG system processes query
5. **Response** → Format and return results

---

## 🛠️ Development Environment

### 📦 Package Management with uv

We use **uv** for fast, reliable dependency management:

```bash
# Install production dependencies
uv sync

# Install with development dependencies
uv sync --dev

# Add new dependency
uv add fastapi
uv add --dev pytest

# Remove dependency
uv remove package_name

# Update dependencies
uv sync --update
```

### 🐳 Docker Development (Optional)

```bash
# Build development image
docker build -t rwanda-ai-rag-dev .

# Run with mounted source
docker run -v $(pwd):/app -p 8000:8000 rwanda-ai-rag-dev

# Using docker-compose
docker-compose up --build
```

### 🔧 IDE Configuration

#### VS Code Setup

Recommended extensions:
- Python
- Pylance
- Black Formatter
- GitLens
- Docker

#### Settings (.vscode/settings.json)

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

---

## 🔧 Contribution Types

### 💻 Code Contributions

#### 🎯 Areas of Focus

1. **🔍 Search & RAG Enhancement**
   - Improve semantic search accuracy
   - Optimize vector similarity algorithms
   - Enhance context retrieval mechanisms

2. **🤖 AI Model Integration**
   - Add support for new LLM providers
   - Implement model fine-tuning pipelines
   - Optimize inference performance

3. **📚 Content Management**
   - Enhance document processing capabilities
   - Improve curriculum organization features
   - Add multimedia content support

4. **🔐 Security & Performance**
   - Implement security best practices
   - Optimize database queries
   - Add caching mechanisms

5. **🧪 Testing & Quality**
   - Increase test coverage
   - Add integration tests
   - Implement performance benchmarks

#### 🛠️ Technical Requirements

- **Type Hints** - All functions must include proper type annotations
- **Documentation** - Docstrings required for all public functions
- **Error Handling** - Graceful error handling with informative messages
- **Async Support** - Use async/await for I/O operations
- **Mock Strategy** - Development mocks for external dependencies

### 📖 Documentation Contributions

#### 📝 Documentation Types

1. **Code Documentation**
   ```python
   async def generate_quiz(
       content: str, 
       difficulty: DifficultyLevel,
       num_questions: int = 5
   ) -> QuizResponse:
       """
       Generate an AI-powered quiz from educational content.
       
       This function processes educational material and creates
       contextually relevant questions with multiple choice answers.
       
       Args:
           content: Source educational material to generate quiz from
           difficulty: Quiz difficulty level (EASY, MEDIUM, HARD)
           num_questions: Number of questions to generate (1-20)
           
       Returns:
           QuizResponse containing generated questions, answers, and metadata
           
       Raises:
           ValidationError: If content is too short or invalid
           AIServiceError: If quiz generation fails
           
       Example:
           >>> content = "Machine learning is a subset of AI..."
           >>> quiz = await generate_quiz(content, DifficultyLevel.MEDIUM, 3)
           >>> print(f"Generated {len(quiz.questions)} questions")
       """
   ```

2. **API Documentation**
   - OpenAPI/Swagger documentation
   - Endpoint descriptions and examples
   - Request/response schemas
   - Error code explanations

3. **User Guides**
   - Installation and setup guides
   - Feature usage tutorials
   - Best practices documentation
   - Troubleshooting guides

### 🧪 Testing Contributions

#### 🎯 Testing Strategy

Our testing approach ensures reliability and maintainability:

```python
"""
Example test structure following our patterns
"""
import pytest
from unittest.mock import AsyncMock, Mock

class TestCurriculumService:
    """Test curriculum management functionality."""
    
    @pytest.fixture
    async def curriculum_service(self):
        """Create curriculum service with mocked dependencies."""
        return CurriculumService()
    
    @pytest.mark.asyncio
    async def test_create_curriculum_success(self, curriculum_service):
        """Test successful curriculum creation."""
        # Arrange
        curriculum_data = {
            "title": "Introduction to AI",
            "description": "Basic AI concepts",
            "difficulty": "beginner"
        }
        
        # Act
        result = await curriculum_service.create_curriculum(curriculum_data)
        
        # Assert
        assert result.id is not None
        assert result.title == curriculum_data["title"]
        assert result.status == "created"
    
    @pytest.mark.asyncio
    async def test_create_curriculum_validation_error(self, curriculum_service):
        """Test curriculum creation with invalid data."""
        # Arrange
        invalid_data = {"title": ""}  # Missing required fields
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            await curriculum_service.create_curriculum(invalid_data)
        
        assert "title" in str(exc_info.value)
```

#### 📊 Current Test Status

- **✅ 75 tests passing** (Core functionality stable)
- **🔧 18 tests needing improvement** (Mock enhancements needed)
- **⚠️ 7 tests with fixture issues** (Dependency resolution needed)

#### 🧪 Running Tests

```bash
# Run all working tests
uv run pytest tests/test_response.py tests/test_models.py::TestLLMInferenceEngine tests/test_embeddings.py::TestEmbeddingGenerator tests/test_rag.py::TestRAGService -v

# Run specific test file
uv run pytest tests/test_curriculum.py -v

# Run with coverage
uv run pytest --cov=app tests/

# Run performance tests
uv run pytest -m performance
```

### 🌍 Localization & Cultural Contributions

#### 🗣️ Language Support

Help make the system accessible to all Rwandans:

1. **Kinyarwanda Integration**
   - Translate UI elements
   - Add Kinyarwanda curriculum content
   - Implement language detection

2. **French Support**
   - Bilingual documentation
   - French curriculum materials
   - Localized error messages

3. **Cultural Context**
   - Rwanda-specific examples
   - Local educational standards alignment
   - Cultural sensitivity in AI responses

---

## 📝 Development Guidelines

### 🎯 Code Quality Standards

#### 🐍 Python Style Guide

We follow **PEP 8** with modern enhancements:

```python
# ✅ Good: Clear, typed function with comprehensive docstring
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel

class DifficultyLevel(Enum):
    """Quiz difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class QuizQuestion(BaseModel):
    """Individual quiz question model."""
    question: str
    options: List[str]
    correct_answer: int
    explanation: str
    difficulty: DifficultyLevel

async def process_educational_content(
    content: str,
    content_type: str = "text",
    metadata: Optional[Dict[str, Any]] = None
) -> List[QuizQuestion]:
    """
    Process educational content and extract key concepts.
    
    This function analyzes educational material and identifies
    important concepts suitable for quiz generation.
    
    Args:
        content: Raw educational content to process
        content_type: Type of content (text, pdf, html)
        metadata: Optional content metadata and context
        
    Returns:
        List of processed quiz questions with explanations
        
    Raises:
        ContentProcessingError: If content cannot be processed
        ValidationError: If content format is invalid
    """
    if not content or not content.strip():
        raise ValidationError("Content cannot be empty")
    
    try:
        # Process content with error handling
        processed_questions = await _extract_key_concepts(content)
        return processed_questions
    
    except Exception as e:
        logger.error(f"Content processing failed: {e}")
        raise ContentProcessingError(f"Processing failed: {e}")

# ✅ Good: Async context manager for resource management
class DatabaseSession:
    """Async database session manager."""
    
    async def __aenter__(self):
        self.connection = await get_database_connection()
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.connection.rollback()
        else:
            await self.connection.commit()
        await self.connection.close()
```

#### 🚫 Anti-Patterns to Avoid

```python
# ❌ Bad: No type hints, unclear function purpose
def process_data(data):
    result = []
    for item in data:
        if item:
            result.append(item.upper())
    return result

# ❌ Bad: No error handling, blocking operations
def get_user_data(user_id):
    response = requests.get(f"http://api.example.com/users/{user_id}")
    return response.json()

# ❌ Bad: No docstring, unclear variable names
async def func1(x, y, z):
    a = x + y
    b = a * z
    return b if b > 0 else None
```

### 🔧 Development Patterns

#### 🎯 Dependency Injection Pattern

```python
from abc import ABC, abstractmethod
from typing import Protocol

class LLMProvider(Protocol):
    """Protocol for LLM service providers."""
    
    async def generate_text(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...

class OpenAIProvider:
    """OpenAI implementation of LLM provider."""
    
    async def generate_text(self, prompt: str) -> str:
        # Implementation
        pass

class QuizService:
    """Service for quiz generation using dependency injection."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
    
    async def generate_quiz(self, content: str) -> Quiz:
        prompt = f"Generate quiz from: {content}"
        response = await self.llm_provider.generate_text(prompt)
        return self._parse_quiz_response(response)
```

#### 🔄 Error Handling Strategy

```python
from enum import Enum
from typing import Optional
import logging

class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"

class ServiceError(Exception):
    """Base service error with context."""
    
    def __init__(
        self, 
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.severity = severity
        self.context = context or {}
        super().__init__(self.message)

class AIServiceError(ServiceError):
    """AI-specific service errors."""
    pass

async def safe_ai_operation(content: str) -> Optional[str]:
    """Safely perform AI operation with comprehensive error handling."""
    try:
        result = await ai_service.process(content)
        return result
        
    except AIServiceError as e:
        logger.error(f"AI service error: {e.message}", extra=e.context)
        if e.severity == ErrorSeverity.CRITICAL:
            raise
        return None
        
    except Exception as e:
        logger.exception("Unexpected error in AI operation")
        raise AIServiceError(
            f"Unexpected error: {str(e)}",
            severity=ErrorSeverity.ERROR,
            context={"content_length": len(content)}
        )
```

### 📊 Performance Guidelines

#### ⚡ Async Best Practices

```python
import asyncio
from typing import List, Coroutine, Any

# ✅ Good: Concurrent execution for independent operations
async def process_multiple_documents(documents: List[str]) -> List[ProcessedDocument]:
    """Process multiple documents concurrently."""
    tasks = [process_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results and exceptions
    processed_docs = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Document {i} processing failed: {result}")
        else:
            processed_docs.append(result)
    
    return processed_docs

# ✅ Good: Async context managers for resource management
async def batch_process_with_rate_limit(items: List[Any], rate_limit: int = 10):
    """Process items with rate limiting."""
    semaphore = asyncio.Semaphore(rate_limit)
    
    async def process_with_limit(item):
        async with semaphore:
            return await process_item(item)
    
    tasks = [process_with_limit(item) for item in items]
    return await asyncio.gather(*tasks)
```

#### 💾 Caching Strategy

```python
from functools import lru_cache
from typing import Dict, Any
import asyncio

class AsyncLRUCache:
    """Async LRU cache implementation."""
    
    def __init__(self, maxsize: int = 128):
        self.cache: Dict[str, Any] = {}
        self.maxsize = maxsize
    
    async def get_or_set(self, key: str, factory_func):
        """Get cached value or compute and cache new value."""
        if key in self.cache:
            return self.cache[key]
        
        value = await factory_func()
        if len(self.cache) >= self.maxsize:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
        return value

# Usage example
embedding_cache = AsyncLRUCache(maxsize=1000)

async def get_cached_embedding(text: str) -> List[float]:
    """Get text embedding with caching."""
    cache_key = f"embedding:{hash(text)}"
    return await embedding_cache.get_or_set(
        cache_key,
        lambda: ai_service.generate_embedding(text)
    )
```

---

## 🧪 Testing Standards

### 📋 Testing Principles

1. **🎯 Comprehensive Coverage** - Test all critical paths
2. **🔄 Fast Feedback** - Quick test execution for development
3. **🧩 Isolation** - Tests don't depend on each other
4. **📝 Clear Intent** - Test names describe expected behavior
5. **🔧 Maintainable** - Easy to update when code changes

### 🏗️ Test Structure

#### 📁 Test Organization

```
tests/
├── unit/                      # Fast, isolated unit tests
│   ├── test_services/
│   ├── test_models/
│   └── test_utils/
├── integration/               # Service integration tests  
│   ├── test_api_endpoints/
│   ├── test_database/
│   └── test_ai_services/
├── performance/               # Performance benchmarks
│   └── test_load_performance.py
├── fixtures/                  # Shared test data
│   ├── sample_curricula.json
│   └── mock_responses.py
└── conftest.py               # Pytest configuration
```

#### 🧪 Test Categories

```python
import pytest

# Unit tests - fast, isolated
class TestQuizGenerator:
    """Unit tests for quiz generation logic."""
    
    def test_question_generation_valid_content(self):
        """Test question generation with valid content."""
        pass
    
    def test_question_generation_empty_content(self):
        """Test error handling for empty content."""
        pass

# Integration tests - test service interactions
@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests for RAG system components."""
    
    async def test_end_to_end_query_processing(self):
        """Test complete query processing pipeline."""
        pass

# Performance tests - benchmark critical operations
@pytest.mark.performance  
class TestPerformance:
    """Performance tests for system bottlenecks."""
    
    def test_embedding_generation_performance(self):
        """Benchmark embedding generation speed."""
        pass
```

### 🎯 Writing Effective Tests

#### 📝 Test Naming Convention

```python
# Pattern: test_[method_name]_[condition]_[expected_result]

def test_create_curriculum_valid_data_returns_curriculum():
    """Test curriculum creation with valid data returns curriculum object."""
    pass

def test_create_curriculum_missing_title_raises_validation_error():
    """Test curriculum creation without title raises ValidationError."""
    pass

def test_search_content_empty_query_returns_empty_results():
    """Test content search with empty query returns empty results."""
    pass
```

#### 🎭 Mocking Strategy

```python
from unittest.mock import AsyncMock, Mock, patch
import pytest

class TestAIService:
    """Test AI service with proper mocking."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = AsyncMock()
        provider.generate_text.return_value = "Generated response"
        return provider
    
    @pytest.fixture
    def ai_service(self, mock_llm_provider):
        """Create AI service with mocked dependencies."""
        return AIService(llm_provider=mock_llm_provider)
    
    async def test_generate_response_success(self, ai_service, mock_llm_provider):
        """Test successful response generation."""
        # Arrange
        query = "What is machine learning?"
        expected_response = "Machine learning is..."
        mock_llm_provider.generate_text.return_value = expected_response
        
        # Act
        result = await ai_service.generate_response(query)
        
        # Assert
        assert result == expected_response
        mock_llm_provider.generate_text.assert_called_once_with(query)
    
    @patch('app.services.external_api.requests.post')
    async def test_external_api_failure_handling(self, mock_post, ai_service):
        """Test handling of external API failures."""
        # Arrange
        mock_post.side_effect = requests.RequestException("API down")
        
        # Act & Assert
        with pytest.raises(AIServiceError) as exc_info:
            await ai_service.call_external_api("test data")
        
        assert "API down" in str(exc_info.value)
```

### 📊 Test Data Management

#### 🎯 Fixture Design

```python
# conftest.py - Shared fixtures
import pytest
from pathlib import Path
import json

@pytest.fixture(scope="session")
def sample_curriculum_data():
    """Load sample curriculum data for tests."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_curricula.json"
    with open(fixture_path) as f:
        return json.load(f)

@pytest.fixture
def mock_database_session():
    """Create mock database session."""
    session = AsyncMock()
    session.add = Mock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session

@pytest.fixture(autouse=True)
def reset_caches():
    """Reset all caches before each test."""
    # Clear application caches
    yield
    # Cleanup after test
```

---

## 📚 Documentation Standards

### 📖 Documentation Types

#### 🔤 Code Documentation

```python
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class ContentType(Enum):
    """Supported content types for processing."""
    TEXT = "text"
    PDF = "pdf"  
    HTML = "html"
    MARKDOWN = "markdown"

class ContentProcessor:
    """
    Educational content processing service.
    
    This class provides methods for processing various types of educational
    content including text extraction, concept identification, and quiz
    generation preparation.
    
    Attributes:
        supported_types: Set of supported content types
        max_content_length: Maximum content length for processing
        
    Example:
        >>> processor = ContentProcessor()
        >>> content = "Machine learning is a subset of AI..."
        >>> concepts = await processor.extract_concepts(content)
        >>> print(f"Found {len(concepts)} key concepts")
    """
    
    def __init__(
        self, 
        max_content_length: int = 50000,
        supported_types: Optional[List[ContentType]] = None
    ):
        """
        Initialize content processor.
        
        Args:
            max_content_length: Maximum length of content to process
            supported_types: List of supported content types, defaults to all
        """
        self.max_content_length = max_content_length
        self.supported_types = supported_types or list(ContentType)
    
    async def process_content(
        self,
        content: str,
        content_type: ContentType = ContentType.TEXT,
        extract_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Process educational content and extract key information.
        
        This method analyzes educational content to extract key concepts,
        learning objectives, and other metadata useful for educational
        applications.
        
        Args:
            content: Raw content to process
            content_type: Type of content being processed
            extract_metadata: Whether to extract additional metadata
            
        Returns:
            Dictionary containing:
                - concepts: List of key concepts found
                - summary: Brief content summary
                - difficulty: Estimated difficulty level
                - metadata: Additional content metadata (if requested)
                
        Raises:
            ContentTooLongError: If content exceeds max_content_length
            UnsupportedContentTypeError: If content_type not supported
            ContentProcessingError: If processing fails
            
        Example:
            >>> processor = ContentProcessor()
            >>> result = await processor.process_content(
            ...     "Artificial intelligence (AI) is...",
            ...     ContentType.TEXT
            ... )
            >>> print(result['concepts'])
            ['artificial intelligence', 'machine learning', 'algorithms']
        """
        # Validation
        if len(content) > self.max_content_length:
            raise ContentTooLongError(
                f"Content length {len(content)} exceeds maximum {self.max_content_length}"
            )
        
        if content_type not in self.supported_types:
            raise UnsupportedContentTypeError(f"Content type {content_type} not supported")
        
        try:
            # Process content based on type
            processed_data = await self._process_by_type(content, content_type)
            
            if extract_metadata:
                processed_data['metadata'] = await self._extract_metadata(content)
            
            return processed_data
            
        except Exception as e:
            raise ContentProcessingError(f"Processing failed: {str(e)}") from e
```

#### 🌐 API Documentation

```python
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter(prefix="/api/v1/curriculum", tags=["curriculum"])

class CurriculumCreateRequest(BaseModel):
    """Request model for creating new curriculum."""
    title: str = Field(..., min_length=3, max_length=200, description="Curriculum title")
    description: str = Field(..., max_length=1000, description="Detailed description")
    difficulty: str = Field(..., regex="^(beginner|intermediate|advanced)$", description="Difficulty level")
    tags: Optional[List[str]] = Field(default=[], description="Content tags for categorization")

class CurriculumResponse(BaseModel):
    """Response model for curriculum data."""
    id: str = Field(..., description="Unique curriculum identifier")
    title: str = Field(..., description="Curriculum title")
    description: str = Field(..., description="Curriculum description") 
    difficulty: str = Field(..., description="Difficulty level")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

@router.post("/", response_model=CurriculumResponse)
async def create_curriculum(curriculum_data: CurriculumCreateRequest):
    """
    Create a new curriculum.
    
    Creates a new curriculum with the provided information. The curriculum
    will be validated and stored in the system for use in educational content
    generation and management.
    
    ## Request Body
    - **title**: Curriculum name (3-200 characters)
    - **description**: Detailed description (max 1000 characters)  
    - **difficulty**: Must be 'beginner', 'intermediate', or 'advanced'
    - **tags**: Optional list of content tags for categorization
    
    ## Response
    Returns the created curriculum with generated ID and timestamps.
    
    ## Errors
    - **400 Bad Request**: Invalid input data or validation errors
    - **409 Conflict**: Curriculum with same title already exists
    - **500 Internal Server Error**: Server processing error
    
    ## Example
    ```json
    {
        "title": "Introduction to Machine Learning",
        "description": "Basic concepts and applications of ML",
        "difficulty": "beginner",
        "tags": ["ai", "machine-learning", "basics"]
    }
    ```
    """
    try:
        # Create curriculum logic
        result = await curriculum_service.create_curriculum(curriculum_data.dict())
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DuplicateTitleError:
        raise HTTPException(status_code=409, detail="Curriculum title already exists")
```

### 📝 User Documentation

#### 🚀 README Structure

Each module should include:

1. **Purpose** - What the module does
2. **Usage** - How to use the module  
3. **Examples** - Practical usage examples
4. **API Reference** - Function/class documentation
5. **Configuration** - Setup and configuration options

#### 📖 Tutorial Documentation

```markdown
# Creating Your First Educational Quiz

This tutorial walks you through creating an AI-generated quiz using the Rwanda AI Curriculum RAG System.

## Prerequisites

- System installed and running
- Basic understanding of REST APIs
- Sample educational content

## Step 1: Prepare Your Content

First, prepare the educational content you want to create a quiz from:

```python
content = """
Machine learning is a subset of artificial intelligence (AI) that enables 
computers to learn and improve from experience without being explicitly 
programmed. It focuses on developing algorithms that can access data and 
use it to learn for themselves.
"""
```

## Step 2: Call the Quiz Generation API

Make a POST request to the quiz generation endpoint:

```bash
curl -X POST "http://localhost:8000/api/v1/quiz/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning is a subset of...",
    "difficulty": "beginner", 
    "num_questions": 3
  }'
```

## Step 3: Process the Response

The API returns a structured quiz response:

```json
{
  "quiz_id": "quiz_123",
  "questions": [
    {
      "question": "What is machine learning?",
      "options": ["A subset of AI", "A programming language", "A database"],
      "correct_answer": 0,
      "explanation": "Machine learning is indeed a subset of AI..."
    }
  ],
  "metadata": {
    "difficulty": "beginner",
    "estimated_time": "5 minutes"
  }
}
```

## Next Steps

- Try different difficulty levels
- Experiment with longer content
- Explore the chat interface for follow-up questions
```

---

## 🔄 Git Workflow

### 🌊 Branching Strategy

We use **Git Flow** with modifications for educational context:

```bash
main                    # Production-ready code
├── develop            # Integration branch for features
├── feature/           # New features and enhancements
│   ├── quiz-generator # Feature branches
│   └── search-api     
├── bugfix/            # Bug fixes
│   └── auth-token-fix
├── hotfix/            # Critical production fixes
│   └── security-patch
└── release/           # Release preparation
    └── v1.2.0
```

### 📝 Commit Message Convention

We follow **Conventional Commits** with educational context:

```bash
# Format: <type>[scope]: <description>
# 
# Types:
# feat     - New features
# fix      - Bug fixes  
# docs     - Documentation changes
# test     - Testing improvements
# refactor - Code refactoring
# perf     - Performance improvements
# chore    - Maintenance tasks

# Examples:
feat(quiz): add multi-language quiz generation
fix(auth): resolve JWT token expiration issue  
docs(api): update curriculum endpoint documentation
test(rag): add comprehensive RAG system tests
refactor(embeddings): optimize vector similarity calculations
perf(search): implement caching for search queries
chore(deps): update Python dependencies

# Educational context examples:
feat(curriculum): add Rwanda history curriculum template
fix(localization): correct Kinyarwanda text rendering
docs(tutorial): add beginner's guide to AI tutoring
test(content): add tests for multilingual content processing
```

### 🔄 Pull Request Process

#### 📋 PR Template

```markdown
## 🎯 Description
Brief description of changes and their educational impact.

## 🧪 Testing
- [ ] Unit tests pass
- [ ] Integration tests pass  
- [ ] Manual testing completed
- [ ] Performance impact assessed

## 📚 Documentation
- [ ] Code documentation updated
- [ ] API documentation updated (if applicable)
- [ ] User guides updated (if applicable)

## 🔍 Review Checklist
- [ ] Code follows style guidelines
- [ ] Error handling is comprehensive
- [ ] Security considerations addressed
- [ ] Performance impact is acceptable
- [ ] Accessibility requirements met

## 🎓 Educational Impact
Describe how this change improves the educational experience.

## 📸 Screenshots (if applicable)
Include screenshots for UI changes.

## 🔗 Related Issues
Closes #123
Related to #456
```

#### 🚀 PR Workflow

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Test**
   ```bash
   # Make changes
   git add .
   git commit -m "feat(feature): add new functionality"
   
   # Run tests
   uv run pytest tests/ -v
   
   # Push changes
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Use GitHub interface or CLI
   - Fill out PR template completely
   - Request review from maintainers

4. **Address Feedback**
   ```bash
   # Make requested changes
   git add .
   git commit -m "fix: address review feedback"
   git push origin feature/your-feature-name
   ```

5. **Merge Process**
   - Squash commits for clean history
   - Delete feature branch after merge
   - Update local develop branch

---

## 👥 Community Guidelines

### 🤝 Code of Conduct

We are committed to creating a welcoming, inclusive environment for all contributors. Our community values:

#### ✨ Our Values

- **🎓 Educational Excellence** - Prioritize learning outcomes and educational impact
- **🌍 Inclusivity** - Welcome contributors from all backgrounds and skill levels  
- **🤝 Collaboration** - Work together to solve educational challenges
- **🔓 Transparency** - Open communication and decision-making processes
- **🇷🇼 Cultural Respect** - Honor Rwandan culture and local educational context

#### 📋 Expected Behavior

- **Be Respectful** - Treat all community members with respect and kindness
- **Be Constructive** - Provide helpful feedback and suggestions
- **Be Patient** - Help newcomers learn and grow
- **Be Inclusive** - Use inclusive language and consider diverse perspectives
- **Be Educational** - Share knowledge and learn from others

#### 🚫 Unacceptable Behavior

- Harassment, discrimination, or hate speech
- Personal attacks or trolling
- Sharing others' private information
- Spam or self-promotion without value
- Disruptive behavior in discussions

### 💬 Communication Channels

#### 🗨️ Discussion Forums

- **💡 Ideas & Suggestions** - [GitHub Discussions](https://github.com/Rwanda-AI-Network/rwanda-ai-curriculum-rag/discussions)
- **❓ Questions & Help** - [GitHub Issues](https://github.com/Rwanda-AI-Network/rwanda-ai-curriculum-rag/issues) with `question` label
- **📢 Announcements** - Watch repository for release notifications

#### 🚀 Real-Time Chat

- **Discord Server** - [Join Rwanda AI Community](https://discord.gg/rwanda-ai)
  - `#general` - General discussion
  - `#contributors` - Contributor coordination
  - `#help` - Get help with setup and development
  - `#showcase` - Share your contributions

#### 📧 Direct Contact

- **Maintainers** - contribute@rwanda-ai.net
- **Security Issues** - security@rwanda-ai.net (for private security reports)
- **Partnership Inquiries** - partnerships@rwanda-ai.net

### 🎓 Mentorship Program

#### 👨‍🏫 For New Contributors

We provide mentorship for newcomers:

- **🌟 First-Time Contributors** - Guided introduction to project
- **🎯 Skill Development** - Help learning new technologies
- **📚 Educational Context** - Understanding of educational AI applications
- **🏆 Recognition** - Celebrate contributions and achievements

#### 🤝 Becoming a Mentor

Experienced contributors can become mentors by:

1. **Demonstrating Expertise** - Consistent quality contributions
2. **Helping Others** - Active in helping newcomers
3. **Communication Skills** - Clear, patient explanation abilities
4. **Time Commitment** - Regular availability for mentoring

---

## 🏆 Recognition

### 🌟 Contribution Recognition

We believe in recognizing all types of contributions:

#### 🎖️ Recognition Levels

1. **🥉 Community Contributor**
   - First merged pull request
   - Listed in CONTRIBUTORS.md
   - Special GitHub badge

2. **🥈 Active Contributor** 
   - 5+ merged pull requests
   - Helped other contributors
   - Featured in release notes

3. **🥇 Core Contributor**
   - 20+ merged pull requests
   - Significant feature contributions
   - Community leadership activities

4. **💎 Maintainer**
   - Ongoing project maintenance
   - Review responsibilities
   - Project direction input

#### 🎁 Contribution Rewards

- **📜 Certificate of Contribution** - Digital certificates for significant contributions
- **🎽 Rwanda AI Swag** - T-shirts and stickers for active contributors  
- **📱 Social Media Recognition** - Shoutouts on project social media
- **🎤 Conference Opportunities** - Speaking opportunities at events
- **💼 Professional References** - LinkedIn recommendations for outstanding contributors

### 📊 Contribution Tracking

We track various types of contributions:

```python
# Example contribution metrics we value
contributions = {
    "code": {
        "pull_requests": 15,
        "lines_added": 2500,
        "lines_removed": 800,
        "files_changed": 45
    },
    "documentation": {
        "docs_updated": 8,
        "tutorials_created": 3,
        "examples_added": 12
    },
    "community": {
        "issues_helped": 25,
        "reviews_provided": 18,
        "discussions_started": 7
    },
    "testing": {
        "tests_added": 50,
        "bugs_found": 8,
        "coverage_improved": "15%"
    }
}
```

---

## ❓ Getting Help

### 🆘 Where to Get Help

#### 🐛 Technical Issues

1. **Check Existing Issues** - Search [GitHub Issues](https://github.com/Rwanda-AI-Network/rwanda-ai-curriculum-rag/issues)
2. **Create New Issue** - Use appropriate issue template
3. **Provide Details** - Include error messages, environment info, and steps to reproduce

#### 💬 General Questions

1. **GitHub Discussions** - For open-ended questions and discussions
2. **Discord Chat** - For real-time help and community interaction  
3. **Documentation** - Check README and documentation first

#### 🔒 Security Issues

For security vulnerabilities:
1. **DO NOT** create public issues
2. **Email** security@rwanda-ai.net with details
3. **Include** steps to reproduce and potential impact

### 📝 Issue Templates

#### 🐛 Bug Report Template

```markdown
## 🐛 Bug Description
Clear description of the bug and expected vs actual behavior.

## 🔄 Steps to Reproduce  
1. Go to '...'
2. Click on '...'
3. See error

## 💻 Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.13.1]
- uv version: [e.g., 0.5.0]
- Browser: [if applicable]

## 📸 Screenshots
Add screenshots if helpful.

## 📋 Additional Context
Any other context about the problem.
```

#### ✨ Feature Request Template

```markdown
## 🎯 Feature Description
Clear description of the proposed feature and its educational value.

## 🎓 Educational Impact
How will this improve the learning experience?

## 💡 Proposed Solution
Describe your preferred solution.

## 🔄 Alternative Solutions
Other approaches you've considered.

## 📚 Additional Context
Examples, mockups, or related features.
```

### 📚 Learning Resources

#### 🐍 Python & FastAPI
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Async Programming in Python](https://docs.python.org/3/library/asyncio.html)

#### 🧠 AI & Machine Learning
- [Hugging Face Course](https://huggingface.co/course)
- [LangChain Documentation](https://docs.langchain.com/)
- [RAG Systems Guide](https://docs.llamaindex.ai/en/stable/)

#### 🧪 Testing
- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://realpython.com/python-testing/)

#### 🔧 Development Tools
- [uv Documentation](https://github.com/astral-sh/uv)
- [Git Best Practices](https://git-scm.com/book)

---

## 🎉 Final Words

Thank you for your interest in contributing to the Rwanda AI Curriculum RAG System! Your contributions help advance education through technology and support learning opportunities in Rwanda and beyond.

### 🌟 Remember

- **Every contribution matters** - From fixing typos to major features
- **Learning is a journey** - We're all here to grow and improve
- **Community first** - Be kind, helpful, and inclusive
- **Educational impact** - Focus on improving learning outcomes
- **Have fun** - Enjoy the process of building something meaningful!

### 🚀 Next Steps

1. **⭐ Star the repository** - Show your support
2. **🍴 Fork the project** - Create your own copy
3. **👀 Explore the issues** - Find something that interests you
4. **💬 Join the community** - Connect with other contributors
5. **🎯 Make your first contribution** - Start with something small

---

<div align="center">

## 🇷🇼 **Murabeho! Welcome to our community!** 🤝

**Built with ❤️ for education in Rwanda and beyond**

[![GitHub stars](https://img.shields.io/github/stars/Rwanda-AI-Network/rwanda-ai-curriculum-rag?style=social)](https://github.com/Rwanda-AI-Network/rwanda-ai-curriculum-rag)
[![Discord](https://img.shields.io/discord/YOUR_DISCORD_ID?logo=discord)](https://discord.gg/rwanda-ai)
[![Follow on Twitter](https://img.shields.io/twitter/follow/RwandaAI?style=social)](https://twitter.com/RwandaAI)

**Happy Contributing! 🚀**

</div>
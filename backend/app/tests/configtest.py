

# app/tests/conftest.py
"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import asyncio
import tempfile
import shutil
from typing import Generator, AsyncGenerator
from app.core.config import AppConfig
from app.db.chroma import ChromaConnection

@pytest.fixture(scope="session")
def event_loop():
    """
    Create event loop for async tests.
    """
    # Create and set event loop for the test session
    pass

@pytest.fixture(scope="session")
def test_config() -> AppConfig:
    """
    Provide test configuration that overrides production settings.
    Returns:
        Test configuration instance
    """
    pass

@pytest.fixture
def temp_directory():
    """
    Create temporary directory for test files.
    Yields:
        Path to temporary directory
    """
    # Create temporary directory
    pass
    
    # Yield directory path for tests
    pass
    
    # Clean up after tests
    pass

@pytest.fixture
async def test_database() -> AsyncGenerator:
    """
    Set up test database for integration tests.
    Yields:
        Test database connection
    """
    # Set up test database
    pass
    
    # Yield database connection
    pass
    
    # Clean up test database
    pass

@pytest.fixture
def sample_documents():
    """
    Provide sample documents for testing.
    Returns:
        Dictionary with sample document content
    """
    pass

@pytest.fixture
def mock_openai_api():
    """
    Mock OpenAI API responses for testing.
    """
    # Mock OpenAI API calls
    pass

@pytest.fixture
def mock_huggingface_api():
    """
    Mock Hugging Face API responses for testing.
    """
    # Mock Hugging Face API calls
    pass
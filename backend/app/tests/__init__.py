
# app/tests/__init__.py
"""
Test suite for the Education RAG API.

This package contains unit tests, integration tests, and test utilities
for all components of the application.
"""

import pytest
import asyncio
from typing import Generator


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """
    Create an event loop for async tests.
    """
    # Create new event loop for testing
    pass


@pytest.fixture
def test_config():
    """
    Provide test configuration.
    Returns:
        Test configuration object
    """
    pass


@pytest.fixture
async def test_database():
    """
    Set up test database for integration tests.
    Returns:
        Test database connection
    """
    pass

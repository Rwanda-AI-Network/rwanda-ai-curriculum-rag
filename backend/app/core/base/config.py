# app/core/base/config.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseConfig(ABC):
    """
    Base configuration class that handles environment variables and settings.
    Each service can have its own config class that inherits from this.
    """
    
    def __init__(self):
        """
        Initialize configuration by loading environment variables.
        """
        self._config_data: Dict[str, Any] = {}
        self._load_config()
    
    @abstractmethod
    def _load_config(self):
        """
        Override this to load specific configuration values.
        Example: Load database URLs, API keys, model names, etc.
        """
        pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        Args:
            key: Configuration key (e.g., "database_url", "openai_api_key")
            default: Default value if key doesn't exist
        Returns:
            The configuration value
        """
        pass
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        Args:
            key: Configuration key
            value: Configuration value
        """
        pass
    
    def is_development(self) -> bool:
        """
        Check if we're running in development mode.
        Returns:
            True if in development, False if in production
        """
        pass


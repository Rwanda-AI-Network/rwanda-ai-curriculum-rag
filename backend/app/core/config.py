# app/core/config.py
import os
from app.core.base.config import BaseConfig

class AppConfig(BaseConfig):
    """
    Main application configuration.
    Loads all environment variables and settings needed by the application.
    """
    
    def _load_config(self):
        """
        Load configuration from environment variables.
        Add new config values here as your application grows.
        """
        # Database configuration
        # Load database connection URL from environment
        pass
        
        # AI service configuration
        # Load API keys for OpenAI, HuggingFace, etc.
        pass
        
        # Application settings
        # Load debug mode, logging level, etc.
        pass
        
        # Security settings
        # Load JWT secrets, CORS settings, etc.
        pass


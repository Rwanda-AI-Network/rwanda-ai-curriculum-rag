"""
Rwanda AI Curriculum RAG - Configuration Settings

This module handles all configuration settings for the application, including:
- Environment-specific settings
- Model configurations
- API settings
- Database connections
- Security parameters

The configuration is loaded from environment variables with proper validation
and type conversion. It supports multiple deployment environments:
- Development
- Staging
- Production
- Testing

Usage:
    from app.config.settings import settings
    
    # Access configuration
    db_url = settings.database_url
    model_path = settings.model_path
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Implementation Guide:
    1. Load environment variables from the appropriate .env file
    2. Validate all settings on application startup
    3. Provide proper type hints and validation rules
    4. Include default values where appropriate
    5. Add descriptive error messages for validation failures
    
    Example:
        # .env.development
        DATABASE_URL=postgresql://user:pass@localhost:5432/db
        MODEL_PATH=/path/to/model
        ENABLE_OFFLINE_MODE=true
        
        # Usage in code
        settings = Settings()
        if settings.enable_offline_mode:
            model = load_local_model(settings.model_path)
    """
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
    
    # Environment
    ENV: str = "development"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/rwanda_ai_db"
    VECTOR_STORE_URL: Optional[str] = None
    
    # Model Settings
    MODEL_PATH: Path = Path("./models/default")
    ENABLE_OFFLINE_MODE: bool = False
    DEVICE: str = "cpu"  # or "cuda" for GPU
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 20
    
    # Language Settings
    ENABLE_KINYARWANDA: bool = True
    TRANSLATION_MODEL_PATH: Optional[Path] = None
    
    # Cache Settings
    ENABLE_CACHE: bool = True
    CACHE_URL: Optional[str] = None
    
    # Monitoring
    ENABLE_MONITORING: bool = True
    METRICS_PREFIX: str = "rwanda_ai_rag"
    
    # Model specific settings
    MODEL_CONFIG: Dict = {
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    # Content Moderation
    ENABLE_MODERATION: bool = True
    FORBIDDEN_TOPICS: List[str] = []
    
    class Config:
        """Pydantic configuration"""
        env_file = f".env.{os.getenv('ENV', 'development')}"
        case_sensitive = True
        
    def get_vector_store_settings(self) -> Dict:
        """
        Get vector store specific settings based on configuration.
        
        Implementation Guide:
        1. Check if offline mode is enabled
        2. Return appropriate vector store settings
        3. Include connection parameters
        4. Add index names and other metadata
        """
        # TODO: Implement vector store settings
        return {}  # TODO: Return actual vector store configuration
    
    def get_model_settings(self) -> Dict:
        """
        Get model specific settings based on hardware and environment.
        
        Implementation Guide:
        1. Check available hardware (CPU/GPU)
        2. Determine appropriate model settings
        3. Set quantization parameters if needed
        4. Configure batch sizes and threading
        """
        # TODO: Implement model settings
        return {}  # TODO: Return actual model configuration
    
    def validate_paths(self) -> None:
        """
        Validate all configured paths exist and are accessible.
        
        Implementation Guide:
        1. Check all configured paths
        2. Validate file permissions
        3. Create directories if needed
        4. Raise descriptive errors if validation fails
        """
        # TODO: Implement this function

        return None

# Create global settings instance
settings = Settings()

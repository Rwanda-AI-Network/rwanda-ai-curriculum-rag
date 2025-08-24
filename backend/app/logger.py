"""
Rwanda AI Curriculum RAG - Logging Configuration

This module sets up centralized logging with proper formatting,
rotation, and different handlers for various log levels.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """
    Custom log formatter with JSON output.
    
    Implementation Guide:
    1. Format fields:
       - Timestamp
       - Level
       - Module
       - Message
    2. Add context:
       - Request ID
       - User info
    3. Handle extras:
       - Custom fields
       - Metrics
    4. Format output:
       - JSON structure
       - Pretty print
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage()
        }
        
        # Add extra fields if present
        if hasattr(record, "extra") and record.extra:  # type: ignore[attr-defined]
            log_data.update(record.extra)  # type: ignore[attr-defined]
            
        return json.dumps(log_data)

class RwandaAILogger:
    """
    Centralized logging system.
    
    Implementation Guide:
    1. Configure handlers:
       - Console output
       - File rotation
       - Error alerts
    2. Set log levels:
       - Debug for dev
       - Info for prod
    3. Handle formatting:
       - JSON structure
       - Stack traces
    4. Manage files:
       - Rotation policy
       - Cleanup rules
    
    Example:
        logger = RwandaAILogger(
            log_dir="logs",
            app_name="rag-api",
            level="INFO"
        )
        
        logger.info("Processing request", extra={"user_id": "123"})
    """
    
    def __init__(self,
                 log_dir: Path,
                 app_name: str,
                 level: str = "INFO",
                 rotate_size: int = 10485760,  # 10MB
                 backup_count: int = 5):
        """
        Initialize logger.
        
        Implementation Guide:
        1. Create directory:
           - Make paths
           - Set permissions
        2. Setup handlers:
           - Configure each
           - Set formats
        3. Set rotation:
           - Size limits
           - Backup count
        4. Configure alerts:
           - Error levels
           - Notifications
           
        Args:
            log_dir: Log directory
            app_name: Application name
            level: Log level
            rotate_size: Rotation size
            backup_count: Number of backups
        """
        self.log_dir = log_dir
        self.app_name = app_name
        
        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(level)
        
        # Add handlers
        self._add_console_handler()
        self._add_file_handler(rotate_size, backup_count)
        self._add_error_handler()
        
    def _add_console_handler(self) -> None:
        """
        Add console output handler.
        
        Implementation Guide:
        1. Create handler:
           - Set stream
           - Configure level
        2. Set formatter:
           - Add colors
           - Format fields
        3. Add filters:
           - Level rules
           - Module rules
        4. Attach handler:
           - Add to logger
           - Enable output
        """
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())
        self.logger.addHandler(handler)
        
    def _add_file_handler(self,
                         rotate_size: int,
                         backup_count: int) -> None:
        """
        Add rotating file handler.
        
        Implementation Guide:
        1. Create handler:
           - Set file path
           - Configure rotation
        2. Set formatter:
           - JSON format
           - Add fields
        3. Configure rotation:
           - Size triggers
           - Backup naming
        4. Handle cleanup:
           - Old file removal
           - Space management
           
        Args:
            rotate_size: Size trigger
            backup_count: Number of backups
        """
        log_file = self.log_dir / f"{self.app_name}.log"
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=rotate_size,
            backupCount=backup_count
        )
        handler.setFormatter(CustomFormatter())
        self.logger.addHandler(handler)
        
    def _add_error_handler(self) -> None:
        """
        Add error notification handler.
        
        Implementation Guide:
        1. Create handler:
           - Set alert method
           - Configure trigger
        2. Set rules:
           - Error levels
           - Rate limits
        3. Add formatting:
           - Alert template
           - Add context
        4. Configure delivery:
           - Set channels
           - Add retry
        """
        # TODO: Implement this function

        return None
        
    def debug(self,
             msg: str,
             extra: Optional[Dict] = None) -> None:
        """Log debug message"""
        self.logger.debug(msg, extra=extra)
        
    def info(self,
            msg: str,
            extra: Optional[Dict] = None) -> None:
        """Log info message"""
        self.logger.info(msg, extra=extra)
        
    def warning(self,
               msg: str,
               extra: Optional[Dict] = None) -> None:
        """Log warning message"""
        self.logger.warning(msg, extra=extra)
        
    def error(self,
             msg: str,
             extra: Optional[Dict] = None) -> None:
        """Log error message"""
        self.logger.error(msg, extra=extra)
        
    def critical(self,
                msg: str,
                extra: Optional[Dict] = None) -> None:
        """Log critical message"""
        self.logger.critical(msg, extra=extra)

# Create global logger instance
logger = RwandaAILogger(
    log_dir=Path("logs"),
    app_name="rwanda-ai-rag",
    level="INFO"
)

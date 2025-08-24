"""
Rwanda AI Curriculum RAG - Helper Functions

Collection of utility functions used throughout the application for common
operations, data processing, and system utilities.

Key Features:
- File and path utilities
- Data validation and transformation
- String processing helpers
- Configuration management utilities
- Logging and monitoring helpers
- Performance measurement tools
"""

import os
import json
import yaml
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import time
import asyncio
import logging
from functools import wraps, lru_cache
import re

logger = logging.getLogger(__name__)


# File and Path Utilities
def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Implementation Guide:
    1. Convert string to Path object
    2. Create parent directories if needed
    3. Handle permission errors
    4. Log directory creation
    5. Return Path object
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the directory
    """
    # TODO: Implement directory creation
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")
    return directory


def get_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Generate hash for file content.
    
    Implementation Guide:
    1. Open file in binary mode
    2. Read in chunks for memory efficiency
    3. Update hash object
    4. Return hex digest
    5. Handle file access errors
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha256, sha512)
        
    Returns:
        Hex string of file hash
    """
    # TODO: Implement file hashing
    hash_func = hashlib.new(algorithm)
    file_path = Path(file_path)
    
    try:
        with open(file_path, 'rb') as f:
            # Read in 64KB chunks
            for chunk in iter(lambda: f.read(65536), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
        
    except Exception as e:
        logger.error(f"Failed to hash file {file_path}: {e}")
        raise


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Implementation Guide:
    1. Get file stats (size, dates)
    2. Extract file extension
    3. Check permissions
    4. Calculate hash
    5. Detect file type
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    # TODO: Implement file information extraction
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"error": "File does not exist"}
    
    stat = file_path.stat()
    
    return {
        "name": file_path.name,
        "path": str(file_path),
        "size": stat.st_size,
        "extension": file_path.suffix.lower(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
        "is_file": file_path.is_file(),
        "is_directory": file_path.is_dir(),
        "permissions": oct(stat.st_mode)[-3:]
    }


def safe_filename(filename: str) -> str:
    """
    Create safe filename by removing/replacing problematic characters.
    
    Implementation Guide:
    1. Remove or replace unsafe characters
    2. Handle unicode characters
    3. Limit filename length
    4. Preserve file extension
    5. Avoid reserved names
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    # TODO: Implement safe filename creation
    # Remove problematic characters
    safe_chars = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    safe_chars = safe_chars.strip('. ')
    
    # Limit length (preserve extension)
    name_part, ext = os.path.splitext(safe_chars)
    if len(name_part) > 200:
        name_part = name_part[:200]
    
    # Avoid reserved names on Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
        'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
        'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    base_name = name_part.upper()
    if base_name in reserved_names:
        name_part = f"_{name_part}"
    
    return name_part + ext


def copy_file_with_progress(src: Path, dst: Path, chunk_size: int = 64 * 1024) -> Dict[str, Any]:
    """
    Copy file with progress tracking.
    
    Implementation Guide:
    1. Open source and destination files
    2. Copy in chunks
    3. Track progress
    4. Calculate copy speed
    5. Handle errors gracefully
    
    Args:
        src: Source file path
        dst: Destination file path
        chunk_size: Size of chunks to copy
        
    Returns:
        Copy operation results
    """
    # TODO: Implement file copying with progress
    start_time = time.time()
    total_size = src.stat().st_size
    copied_size = 0
    
    try:
        with open(src, 'rb') as src_file, open(dst, 'wb') as dst_file:
            while True:
                chunk = src_file.read(chunk_size)
                if not chunk:
                    break
                dst_file.write(chunk)
                copied_size += len(chunk)
        
        elapsed_time = time.time() - start_time
        speed = copied_size / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "success": True,
            "bytes_copied": copied_size,
            "elapsed_time": elapsed_time,
            "speed_mb_s": speed / (1024 * 1024)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "bytes_copied": copied_size
        }


# Data Validation and Transformation
def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Implementation Guide:
    1. Use regex pattern for basic validation
    2. Check for common issues
    3. Handle internationalized domains
    4. Validate length limits
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    # TODO: Implement email validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string by removing/replacing problematic characters.
    
    Implementation Guide:
    1. Remove control characters
    2. Handle unicode normalization
    3. Trim whitespace
    4. Limit length if specified
    5. Preserve readability
    
    Args:
        text: Text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    # TODO: Implement string sanitization
    if not text:
        return ""
    
    # Remove control characters (except common whitespace)
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    # Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Limit length if specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip() + "..."
    
    return sanitized


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Implementation Guide:
    1. Handle nested dictionaries recursively
    2. Handle different data types
    3. Preserve original dictionaries
    4. Handle conflicts appropriately
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    # TODO: Implement deep dictionary merging
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Implementation Guide:
    1. Recursively process nested dictionaries
    2. Create flat keys with separator
    3. Handle lists and other types
    4. Preserve data types
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Key separator
        
    Returns:
        Flattened dictionary
    """
    # TODO: Implement dictionary flattening
    items = []
    
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}[{i}]", sep).items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, value))
    
    return dict(items)


# Configuration Management
def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Implementation Guide:
    1. Detect file format by extension
    2. Load and parse file content
    3. Handle parsing errors
    4. Validate configuration structure
    5. Apply default values
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # TODO: Implement configuration loading
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif file_path.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    except Exception as e:
        logger.error(f"Failed to load configuration from {file_path}: {e}")
        raise


def save_config_file(config: Dict[str, Any], file_path: Union[str, Path], format: str = 'auto') -> None:
    """
    Save configuration to file.
    
    Implementation Guide:
    1. Determine output format
    2. Create directory if needed
    3. Format and write data
    4. Handle write errors
    5. Validate saved data
    
    Args:
        config: Configuration data
        file_path: Output file path
        format: Output format (json, yaml, auto)
    """
    # TODO: Implement configuration saving
    file_path = Path(file_path)
    ensure_directory_exists(file_path.parent)
    
    # Determine format
    if format == 'auto':
        format = 'yaml' if file_path.suffix.lower() in ['.yaml', '.yml'] else 'json'
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if format == 'yaml':
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
            else:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {file_path}: {e}")
        raise


@lru_cache(maxsize=128)
def get_environment_variable(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with caching.
    
    Implementation Guide:
    1. Check environment variables
    2. Apply default if not found
    3. Cache results for performance
    4. Handle type conversion
    5. Log access for debugging
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    # TODO: Implement environment variable retrieval
    value = os.getenv(key, default)
    logger.debug(f"Environment variable {key}: {'[SET]' if value else '[NOT SET]'}")
    return value


# String Processing Helpers
def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Implementation Guide:
    1. Check if truncation needed
    2. Find appropriate cut point
    3. Add suffix if truncated
    4. Preserve word boundaries
    5. Handle edge cases
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    # TODO: Implement text truncation
    if len(text) <= max_length:
        return text
    
    if max_length <= len(suffix):
        return suffix[:max_length]
    
    # Try to break at word boundary
    max_text_length = max_length - len(suffix)
    truncated = text[:max_text_length]
    
    # Find last space to preserve word boundaries
    last_space = truncated.rfind(' ')
    if last_space > 0 and last_space > max_text_length * 0.8:
        truncated = truncated[:last_space]
    
    return truncated + suffix


def extract_keywords(text: str, max_keywords: int = 10, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text.
    
    Implementation Guide:
    1. Tokenize text
    2. Remove stop words
    3. Filter by length and frequency
    4. Apply stemming/lemmatization
    5. Return top keywords
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords
        min_length: Minimum keyword length
        
    Returns:
        List of extracted keywords
    """
    # TODO: Implement keyword extraction
    # Simple implementation - would be enhanced with NLP libraries
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter by length
    words = [word for word in words if len(word) >= min_length]
    
    # Remove common stop words (simplified list)
    stop_words = {
        'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'as', 'are',
        'was', 'will', 'been', 'be', 'have', 'had', 'has', 'do', 'did',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'in',
        'of', 'for', 'with', 'by', 'from', 'up', 'about', 'into', 'through'
    }
    
    words = [word for word in words if word not in stop_words]
    
    # Count frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison and search.
    
    Implementation Guide:
    1. Convert to lowercase
    2. Remove diacritics/accents
    3. Normalize whitespace
    4. Remove punctuation
    5. Handle unicode normalization
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # TODO: Implement text normalization
    import unicodedata
    
    # Convert to lowercase
    normalized = text.lower()
    
    # Remove diacritics
    normalized = ''.join(
        char for char in unicodedata.normalize('NFD', normalized)
        if unicodedata.category(char) != 'Mn'
    )
    
    # Remove punctuation and normalize whitespace
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


# Performance and Monitoring
def timer(func):
    """
    Decorator to measure function execution time.
    
    Implementation Guide:
    1. Record start time
    2. Execute function
    3. Calculate elapsed time
    4. Log timing information
    5. Return original result
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"Function {func.__name__} took {elapsed_time:.3f} seconds")
        return result
    return wrapper


async def async_timer(func):
    """
    Decorator to measure async function execution time.
    
    Implementation Guide:
    1. Record start time
    2. Await function execution
    3. Calculate elapsed time
    4. Log timing information
    5. Return original result
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"Async function {func.__name__} took {elapsed_time:.3f} seconds")
        return result
    return wrapper


def memory_usage_mb() -> float:
    """
    Get current memory usage in MB.
    
    Implementation Guide:
    1. Import psutil or use resource module
    2. Get current process memory usage
    3. Convert to MB
    4. Handle import errors gracefully
    
    Returns:
        Memory usage in MB
    """
    # TODO: Implement memory usage measurement
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate unique identifier.
    
    Implementation Guide:
    1. Generate UUID
    2. Add timestamp component
    3. Include prefix if provided
    4. Ensure uniqueness
    5. Return formatted ID
    
    Args:
        prefix: Optional prefix for ID
        
    Returns:
        Unique identifier string
    """
    # TODO: Implement unique ID generation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_part = str(uuid.uuid4()).split('-')[0]
    
    if prefix:
        return f"{prefix}_{timestamp}_{unique_part}"
    else:
        return f"{timestamp}_{unique_part}"


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator for retrying functions with exponential backoff.
    
    Implementation Guide:
    1. Wrap function execution
    2. Catch exceptions
    3. Apply exponential backoff
    4. Retry up to max_retries
    5. Log retry attempts
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff multiplier
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}), "
                        f"retrying in {wait_time:.1f} seconds: {e}"
                    )
                    time.sleep(wait_time)
            
            logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"Function {func.__name__} failed after {max_retries + 1} attempts")
        
        return wrapper
    return decorator


# Data Structure Utilities
class LRUCache:
    """
    Simple LRU Cache implementation.
    
    Implementation Guide:
    1. Use OrderedDict for O(1) operations
    2. Implement get/set operations
    3. Handle capacity limits
    4. Track cache statistics
    5. Provide cache management
    """
    
    def __init__(self, capacity: int = 128):
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum cache capacity
        """
        # TODO: Implement LRU cache initialization
        from collections import OrderedDict
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Any:
        """
        Get value from cache.
        
        Implementation Guide:
        1. Check if key exists
        2. Update access order
        3. Track cache hits/misses
        4. Return value or None
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # TODO: Implement cache get
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
        else:
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Implementation Guide:
        1. Check if key exists
        2. Update or add entry
        3. Maintain capacity limits
        4. Remove least recently used if needed
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # TODO: Implement cache set
        if key in self.cache:
            # Update existing entry
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used (first item)
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "capacity": self.capacity,
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


# Global cache instance
default_cache = LRUCache(capacity=256)


# Date and Time Utilities
def parse_flexible_date(date_string: str) -> Optional[datetime]:
    """
    Parse date from various string formats.
    
    Implementation Guide:
    1. Try common date formats
    2. Handle different separators
    3. Support relative dates
    4. Return None for invalid dates
    5. Log parsing attempts
    
    Args:
        date_string: Date string to parse
        
    Returns:
        Parsed datetime or None
    """
    # TODO: Implement flexible date parsing
    common_formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y"
    ]
    
    for fmt in common_formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date string: {date_string}")
    return None


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Implementation Guide:
    1. Convert seconds to appropriate units
    2. Handle different time scales
    3. Format with appropriate precision
    4. Return readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    # TODO: Implement duration formatting
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days"


def get_relative_time(dt: datetime) -> str:
    """
    Get relative time description (e.g., "2 hours ago").
    
    Implementation Guide:
    1. Calculate time difference
    2. Choose appropriate unit
    3. Format relative description
    4. Handle future dates
    
    Args:
        dt: Datetime to compare
        
    Returns:
        Relative time string
    """
    # TODO: Implement relative time formatting
    now = datetime.now()
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hours ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minutes ago"
    else:
        return "just now"


if __name__ == "__main__":
    # TODO: Add command-line utility functions
    pass

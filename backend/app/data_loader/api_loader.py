"""
Rwanda AI Curriculum RAG - API Data Loader

This module handles loading curriculum data from external APIs,
with proper rate limiting, error handling, and caching.
"""

from typing import Dict, List, Optional, Union, Any, cast
from pathlib import Path
import aiohttp
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from aiohttp import ClientTimeout
from .base import BaseDataLoader

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

class APIConfig:
    """API configuration and credentials"""
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    timeout: int = 30     # seconds
    retry_attempts: int = 3
    cache_ttl: int = 3600  # seconds

class APILoader(BaseDataLoader):
    """
    Load curriculum data from external APIs.
    
    Implementation Guide:
    1. Handle multiple endpoints:
       - REB API
       - Content APIs
       - Assessment APIs
    2. Manage authentication
    3. Handle rate limiting
    4. Implement caching
    5. Support async operations
    
    Example:
        loader = APILoader(
            config=APIConfig(
                base_url="https://api.reb.rw",
                api_key="your-key"
            )
        )
        
        data = await loader.load_curriculum(
            grade=5,
            subject="science"
        )
    """
    
    def __init__(self,
                 config: APIConfig,
                 cache_dir: Optional[Path] = None):
        """
        Initialize API loader.
        
        Implementation Guide:
        1. Validate config:
           - Check credentials
           - Verify URLs
        2. Setup client:
           - Configure timeout
           - Set headers
        3. Initialize cache:
           - Create directory
           - Set policies
        4. Setup rate limiting:
           - Track requests
           - Set windows
           
        Args:
            config: API configuration
            cache_dir: Cache directory
        """
        super().__init__()
        self.config = config
        self.cache_dir = cache_dir
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_timestamps: List[datetime] = []
        
    async def load(self, source: Union[str, Path], **params) -> Dict[str, Any]:
        """
        Load data from API endpoint.
        
        Implementation Guide:
        1. Prepare request:
           - Build URL
           - Set parameters
           - Add headers
        2. Check cache:
           - Look for data
           - Verify freshness
        3. Make request:
           - Handle rate limits
           - Manage retries
        4. Process response:
           - Parse data
           - Handle errors
           - Update cache
           
        Args:
            endpoint: API endpoint
            **params: Query parameters
            
        Returns:
            API response data
            
        Raises:
            APIError: For request failures
        """
        # Build full URL
        endpoint = str(source).lstrip('/') if isinstance(source, Path) else source.lstrip('/')
        url = f"{self.config.base_url}/{endpoint}"
        
        # Check cache first
        cache_key = f"{url}:{str(params)}"
        if cached_data := await self._get_from_cache(cache_key):
            return cached_data
            
        # Make API request
        try:
            response_data = await self._make_request(url, params=params)
            
            # Cache successful response
            await self._save_to_cache(cache_key, response_data)
            
            return response_data
            
        except aiohttp.ClientError as e:
            raise APIError(f"API request failed: {str(e)}")
        
    async def _make_request(self,
                          url: str,
                          method: str = "GET",
                          **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with retries.
        
        Implementation Guide:
        1. Check rate limit:
           - Count requests
           - Wait if needed
        2. Prepare request:
           - Add auth
           - Set timeout
        3. Handle retries:
           - Track attempts
           - Implement backoff
        4. Process response:
           - Check status
           - Parse JSON
           
        Args:
            url: Request URL
            method: HTTP method
            **kwargs: Request parameters
            
        Returns:
            Response data
        """
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
            
        for attempt in range(self.config.retry_attempts):
            try:
                # Check rate limiting
                await self._check_rate_limit()
                
                # Make request
                async with self.session.request(
                    method,
                    url,
                    timeout=ClientTimeout(total=self.config.timeout),
                    **kwargs
                ) as response:
                    response.raise_for_status()
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                if attempt == self.config.retry_attempts - 1:
                    raise APIError(f"Request failed after {attempt + 1} attempts: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            finally:
                self.request_timestamps.append(datetime.now())
        
        # Should not reach here, but return empty dict as fallback
        return {}
        
    async def _check_rate_limit(self) -> None:
        """
        Check and enforce rate limits.
        
        Implementation Guide:
        1. Clean old requests:
           - Remove expired
           - Update window
        2. Count requests:
           - Check window
           - Calculate wait
        3. Apply limit:
           - Sleep if needed
           - Update tracking
        4. Handle errors:
           - Timeout cases
           - Reset needed
           
        Raises:
            RateLimitError: If limit exceeded
        """
        now = datetime.now()
        window_start = now - timedelta(minutes=1)
        
        # Clean old timestamps
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if ts > window_start
        ]
        
        # Check if we're over the limit
        if len(self.request_timestamps) >= self.config.rate_limit:
            # Calculate sleep time
            oldest_timestamp = min(self.request_timestamps)
            sleep_time = (oldest_timestamp + timedelta(minutes=1) - now).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
    async def _get_from_cache(self,
                            key: str) -> Optional[Dict]:
        """
        Get data from cache.
        
        Implementation Guide:
        1. Generate key:
           - Hash parameters
           - Add version
        2. Check cache:
           - Find file
           - Read data
        3. Validate entry:
           - Check TTL
           - Verify format
        4. Return data:
           - Parse JSON
           - Handle missing
           
        Args:
            key: Cache key
            
        Returns:
            Cached data if found
        """
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / f"{hash(key)}.json"
        if not cache_file.exists():
            return None
            
        try:
            data = json.loads(cache_file.read_text())
            cached_time = datetime.fromisoformat(data['_cached_at'])
            
            # Check TTL
            if (datetime.now() - cached_time).total_seconds() > self.config.cache_ttl:
                cache_file.unlink()
                return None
                
            return data['content']
            
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
        
    async def _save_to_cache(self,
                           key: str,
                           data: Dict) -> None:
        """
        Save data to cache.
        
        Implementation Guide:
        1. Prepare data:
           - Add timestamp
           - Set TTL
        2. Generate path:
           - Create dirs
           - Set filename
        3. Write file:
           - Atomic write
           - Set permissions
        4. Cleanup old:
           - Check size
           - Remove expired
           
        Args:
            key: Cache key
            data: Data to cache
        """
        if not self.cache_dir:
            return
            
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = self.cache_dir / f"{hash(key)}.json"
        cache_data = {
            '_cached_at': datetime.now().isoformat(),
            'content': data
        }
        
        # Atomic write using temporary file
        temp_file = cache_file.with_suffix('.tmp')
        temp_file.write_text(json.dumps(cache_data))
        temp_file.replace(cache_file)
        
    async def close(self) -> None:
        """
        Cleanup resources.
        
        Implementation Guide:
        1. Close session:
           - Wait pending
           - Clear queue
        2. Clear cache:
           - Remove temp
           - Update index
        3. Reset state:
           - Clear counters
           - Reset tracking
        4. Log closure:
           - Save stats
           - Report errors
        """
        if self.session and not self.session.closed:
            await self.session.close()
            
        # Clear request tracking
        self.request_timestamps.clear()
        
        # Remove temporary cache files
        if self.cache_dir:
            for temp_file in self.cache_dir.glob('*.tmp'):
                try:
                    temp_file.unlink()
                except OSError:
                    # TODO: Implement this function

                    return None

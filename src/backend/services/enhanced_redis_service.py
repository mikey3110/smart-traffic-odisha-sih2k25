"""
Enhanced Redis service with connection pooling, retry logic, and reliability features
for Smart Traffic Management System
"""

import redis
import redis.connection
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import (
    ConnectionError,
    TimeoutError,
    RedisError,
    BusyLoadingError,
    ResponseError
)
import json
import asyncio
from typing import Optional, Dict, Any, List, Union, Callable
from contextlib import asynccontextmanager
import time
from functools import wraps

from config.logging_config import get_logger
from config.settings import settings
from exceptions import RedisConnectionError, RedisOperationError

logger = get_logger(__name__)


def retry_on_redis_error(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying Redis operations with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionError, TimeoutError, BusyLoadingError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Redis operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Redis operation failed after {max_retries + 1} attempts: {e}")
                        raise RedisOperationError(f"Operation failed after {max_retries + 1} attempts: {str(e)}")
                except RedisError as e:
                    # Don't retry on other Redis errors
                    logger.error(f"Redis operation failed with non-retryable error: {e}")
                    raise RedisOperationError(f"Operation failed: {str(e)}")
            
            # This should never be reached, but just in case
            if last_exception:
                raise RedisOperationError(f"Operation failed: {str(last_exception)}")
        
        return wrapper
    return decorator


class EnhancedRedisService:
    """
    Enhanced Redis service with connection pooling, retry logic, and reliability features
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        max_connections: int = None,
        socket_timeout: int = None,
        socket_connect_timeout: int = None,
        retry_on_timeout: bool = None
    ):
        """
        Initialize Redis service with enhanced configuration
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            max_connections: Maximum number of connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
            retry_on_timeout: Whether to retry on timeout
        """
        
        # Use settings if not provided
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self.db = db or settings.redis_db
        self.password = password or settings.redis_password
        self.max_connections = max_connections or settings.redis_max_connections
        self.socket_timeout = socket_timeout or settings.redis_socket_timeout
        self.socket_connect_timeout = socket_connect_timeout or settings.redis_socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout if retry_on_timeout is not None else settings.redis_retry_on_timeout
        
        # Connection pool and client
        self.connection_pool = None
        self.redis_client = None
        self.connected = False
        
        # Statistics
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "connection_errors": 0,
            "timeout_errors": 0,
            "last_connection_time": None
        }
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Redis connection with connection pooling"""
        try:
            # Create connection pool
            self.connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
                decode_responses=True
            )
            
            # Create Redis client with retry logic
            retry = Retry(ExponentialBackoff(), 3)
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                retry=retry,
                retry_on_timeout=self.retry_on_timeout
            )
            
            # Test connection
            self.redis_client.ping()
            self.connected = True
            self.stats["last_connection_time"] = time.time()
            
            logger.info(f"✅ Enhanced Redis service connected successfully to {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.connected = False
            self.stats["connection_errors"] += 1
            raise RedisConnectionError(f"Failed to connect to Redis: {str(e)}")
    
    def _update_stats(self, success: bool, error_type: str = None):
        """Update operation statistics"""
        self.stats["total_operations"] += 1
        if success:
            self.stats["successful_operations"] += 1
        else:
            self.stats["failed_operations"] += 1
            if error_type == "connection":
                self.stats["connection_errors"] += 1
            elif error_type == "timeout":
                self.stats["timeout_errors"] += 1
    
    @retry_on_redis_error(max_retries=3, delay=1.0)
    async def set_traffic_data(
        self, 
        intersection_id: str, 
        data: Dict[str, Any], 
        ttl: int = None
    ) -> bool:
        """
        Store traffic data with TTL and retry logic
        
        Args:
            intersection_id: Intersection identifier
            data: Traffic data to store
            ttl: Time to live in seconds (uses default if None)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            raise RedisConnectionError("Redis service is not connected")
        
        try:
            ttl = ttl or settings.traffic_data_ttl
            key = f"traffic:{intersection_id}"
            
            # Add metadata
            data_with_metadata = {
                **data,
                "stored_at": time.time(),
                "ttl": ttl
            }
            
            # Store with TTL
            result = self.redis_client.setex(key, ttl, json.dumps(data_with_metadata))
            
            if result:
                self._update_stats(True)
                logger.debug(f"Stored traffic data for intersection {intersection_id}")
                return True
            else:
                self._update_stats(False)
                logger.error(f"Failed to store traffic data for intersection {intersection_id}")
                return False
                
        except Exception as e:
            self._update_stats(False, "connection" if isinstance(e, ConnectionError) else "timeout")
            logger.error(f"Error storing traffic data: {e}")
            raise
    
    @retry_on_redis_error(max_retries=3, delay=1.0)
    async def get_traffic_data(self, intersection_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve traffic data with retry logic
        
        Args:
            intersection_id: Intersection identifier
        
        Returns:
            Traffic data dictionary or None if not found
        """
        if not self.connected:
            raise RedisConnectionError("Redis service is not connected")
        
        try:
            key = f"traffic:{intersection_id}"
            data = self.redis_client.get(key)
            
            if data:
                traffic_data = json.loads(data)
                self._update_stats(True)
                logger.debug(f"Retrieved traffic data for intersection {intersection_id}")
                return traffic_data
            else:
                self._update_stats(True)  # Not found is not an error
                logger.debug(f"No traffic data found for intersection {intersection_id}")
                return None
                
        except Exception as e:
            self._update_stats(False, "connection" if isinstance(e, ConnectionError) else "timeout")
            logger.error(f"Error retrieving traffic data: {e}")
            raise
    
    @retry_on_redis_error(max_retries=3, delay=1.0)
    async def set_signal_data(
        self, 
        intersection_id: str, 
        data: Dict[str, Any], 
        ttl: int = None
    ) -> bool:
        """
        Store signal optimization data with TTL and retry logic
        
        Args:
            intersection_id: Intersection identifier
            data: Signal data to store
            ttl: Time to live in seconds (uses default if None)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            raise RedisConnectionError("Redis service is not connected")
        
        try:
            ttl = ttl or settings.signal_data_ttl
            key = f"signal:{intersection_id}"
            
            # Add metadata
            data_with_metadata = {
                **data,
                "stored_at": time.time(),
                "ttl": ttl
            }
            
            # Store with TTL
            result = self.redis_client.setex(key, ttl, json.dumps(data_with_metadata))
            
            if result:
                self._update_stats(True)
                logger.debug(f"Stored signal data for intersection {intersection_id}")
                return True
            else:
                self._update_stats(False)
                logger.error(f"Failed to store signal data for intersection {intersection_id}")
                return False
                
        except Exception as e:
            self._update_stats(False, "connection" if isinstance(e, ConnectionError) else "timeout")
            logger.error(f"Error storing signal data: {e}")
            raise
    
    @retry_on_redis_error(max_retries=3, delay=1.0)
    async def get_signal_data(self, intersection_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve signal optimization data with retry logic
        
        Args:
            intersection_id: Intersection identifier
        
        Returns:
            Signal data dictionary or None if not found
        """
        if not self.connected:
            raise RedisConnectionError("Redis service is not connected")
        
        try:
            key = f"signal:{intersection_id}"
            data = self.redis_client.get(key)
            
            if data:
                signal_data = json.loads(data)
                self._update_stats(True)
                logger.debug(f"Retrieved signal data for intersection {intersection_id}")
                return signal_data
            else:
                self._update_stats(True)  # Not found is not an error
                logger.debug(f"No signal data found for intersection {intersection_id}")
                return None
                
        except Exception as e:
            self._update_stats(False, "connection" if isinstance(e, ConnectionError) else "timeout")
            logger.error(f"Error retrieving signal data: {e}")
            raise
    
    @retry_on_redis_error(max_retries=3, delay=1.0)
    async def get_all_intersections(self) -> List[str]:
        """
        Get all intersection IDs with retry logic
        
        Returns:
            List of intersection IDs
        """
        if not self.connected:
            raise RedisConnectionError("Redis service is not connected")
        
        try:
            # Get all traffic keys
            traffic_keys = self.redis_client.keys("traffic:*")
            intersections = [key.split(":")[1] for key in traffic_keys]
            
            self._update_stats(True)
            logger.debug(f"Retrieved {len(intersections)} intersections")
            return intersections
            
        except Exception as e:
            self._update_stats(False, "connection" if isinstance(e, ConnectionError) else "timeout")
            logger.error(f"Error retrieving intersections: {e}")
            raise
    
    @retry_on_redis_error(max_retries=3, delay=1.0)
    async def delete_data(self, key_pattern: str) -> int:
        """
        Delete data matching pattern with retry logic
        
        Args:
            key_pattern: Key pattern to match (e.g., "traffic:*")
        
        Returns:
            Number of keys deleted
        """
        if not self.connected:
            raise RedisConnectionError("Redis service is not connected")
        
        try:
            keys = self.redis_client.keys(key_pattern)
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                self._update_stats(True)
                logger.debug(f"Deleted {deleted_count} keys matching pattern {key_pattern}")
                return deleted_count
            else:
                self._update_stats(True)
                logger.debug(f"No keys found matching pattern {key_pattern}")
                return 0
                
        except Exception as e:
            self._update_stats(False, "connection" if isinstance(e, ConnectionError) else "timeout")
            logger.error(f"Error deleting data: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection
        
        Returns:
            Health check results
        """
        try:
            start_time = time.time()
            self.redis_client.ping()
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                "status": "healthy",
                "connected": True,
                "response_time_ms": round(response_time, 2),
                "stats": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "stats": self.stats.copy()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset service statistics"""
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "connection_errors": 0,
            "timeout_errors": 0,
            "last_connection_time": self.stats.get("last_connection_time")
        }
    
    async def close(self):
        """Close Redis connection and cleanup resources"""
        try:
            if self.redis_client:
                self.redis_client.close()
            if self.connection_pool:
                self.connection_pool.disconnect()
            self.connected = False
            logger.info("Redis service closed successfully")
        except Exception as e:
            logger.error(f"Error closing Redis service: {e}")


# Global enhanced Redis service instance
enhanced_redis_service = EnhancedRedisService()


"""
Enhanced Redis service with connection pooling, retry logic, and reliability features
"""

import redis
import redis.asyncio as aioredis
from redis.connection import ConnectionPool
from redis.exceptions import (
    ConnectionError, TimeoutError, RedisError, 
    BusyLoadingError, ResponseError
)
import json
import time
import asyncio
from typing import Any, Optional, Dict, List, Union
from contextlib import asynccontextmanager
import logging

from config.settings import settings
from config.logging_config import get_logger
from exceptions.custom_exceptions import RedisError as CustomRedisError

logger = get_logger("redis_service")


class RedisService:
    """
    Enhanced Redis service with connection pooling, retry logic, and error handling
    """
    
    def __init__(self):
        self.pool: Optional[ConnectionPool] = None
        self.async_pool: Optional[aioredis.ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self.async_client: Optional[aioredis.Redis] = None
        self.connected = False
        self.retry_attempts = 3
        self.retry_delay = 1.0
        self.max_retry_delay = 60.0
        
    def _create_connection_pool(self) -> ConnectionPool:
        """Create Redis connection pool"""
        return ConnectionPool(
            host=settings.redis.host,
            port=settings.redis.port,
            password=settings.redis.password,
            db=settings.redis.db,
            max_connections=settings.redis.max_connections,
            socket_timeout=settings.redis.socket_timeout,
            socket_connect_timeout=settings.redis.socket_connect_timeout,
            retry_on_timeout=settings.redis.retry_on_timeout,
            health_check_interval=settings.redis.health_check_interval,
            decode_responses=True
        )
    
    async def _create_async_connection_pool(self) -> aioredis.ConnectionPool:
        """Create async Redis connection pool"""
        return aioredis.ConnectionPool(
            host=settings.redis.host,
            port=settings.redis.port,
            password=settings.redis.password,
            db=settings.redis.db,
            max_connections=settings.redis.max_connections,
            socket_timeout=settings.redis.socket_timeout,
            socket_connect_timeout=settings.redis.socket_connect_timeout,
            retry_on_timeout=settings.redis.retry_on_timeout,
            health_check_interval=settings.redis.health_check_interval,
            decode_responses=True
        )
    
    def connect(self) -> bool:
        """Connect to Redis with connection pooling"""
        try:
            self.pool = self._create_connection_pool()
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.client.ping()
            self.connected = True
            
            logger.info(
                "Redis connected successfully",
                extra={
                    'host': settings.redis.host,
                    'port': settings.redis.port,
                    'db': settings.redis.db,
                    'max_connections': settings.redis.max_connections
                }
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            return False
    
    async def async_connect(self) -> bool:
        """Connect to Redis asynchronously with connection pooling"""
        try:
            self.async_pool = await self._create_async_connection_pool()
            self.async_client = aioredis.Redis(connection_pool=self.async_pool)
            
            # Test connection
            await self.async_client.ping()
            self.connected = True
            
            logger.info(
                "Async Redis connected successfully",
                extra={
                    'host': settings.redis.host,
                    'port': settings.redis.port,
                    'db': settings.redis.db,
                    'max_connections': settings.redis.max_connections
                }
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to async Redis: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            self.client.close()
        if self.pool:
            self.pool.disconnect()
        self.connected = False
        logger.info("Redis disconnected")
    
    async def async_disconnect(self):
        """Disconnect from async Redis"""
        if self.async_client:
            await self.async_client.close()
        if self.async_pool:
            await self.async_pool.disconnect()
        self.connected = False
        logger.info("Async Redis disconnected")
    
    def _retry_on_failure(self, func, *args, **kwargs):
        """Retry function on Redis failure with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, TimeoutError, BusyLoadingError) as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    delay = min(self.retry_delay * (2 ** attempt), self.max_retry_delay)
                    logger.warning(
                        f"Redis operation failed (attempt {attempt + 1}/{self.retry_attempts}): {e}. Retrying in {delay}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Redis operation failed after {self.retry_attempts} attempts: {e}")
            except ResponseError as e:
                # Don't retry on response errors (invalid commands, etc.)
                logger.error(f"Redis response error: {e}")
                raise CustomRedisError(f"Redis response error: {e}")
            except Exception as e:
                logger.error(f"Unexpected Redis error: {e}")
                raise CustomRedisError(f"Unexpected Redis error: {e}")
        
        raise CustomRedisError(f"Redis operation failed after {self.retry_attempts} attempts: {last_exception}")
    
    async def _async_retry_on_failure(self, func, *args, **kwargs):
        """Retry async function on Redis failure with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                return await func(*args, **kwargs)
            except (ConnectionError, TimeoutError, BusyLoadingError) as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    delay = min(self.retry_delay * (2 ** attempt), self.max_retry_delay)
                    logger.warning(
                        f"Async Redis operation failed (attempt {attempt + 1}/{self.retry_attempts}): {e}. Retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Async Redis operation failed after {self.retry_attempts} attempts: {e}")
            except ResponseError as e:
                logger.error(f"Async Redis response error: {e}")
                raise CustomRedisError(f"Async Redis response error: {e}")
            except Exception as e:
                logger.error(f"Unexpected async Redis error: {e}")
                raise CustomRedisError(f"Unexpected async Redis error: {e}")
        
        raise CustomRedisError(f"Async Redis operation failed after {self.retry_attempts} attempts: {last_exception}")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a key-value pair with optional TTL"""
        if not self.connected:
            raise CustomRedisError("Redis not connected")
        
        def _set():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            if ttl:
                return self.client.setex(key, ttl, value)
            else:
                return self.client.set(key, value)
        
        return self._retry_on_failure(_set)
    
    async def async_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a key-value pair asynchronously with optional TTL"""
        if not self.connected:
            raise CustomRedisError("Async Redis not connected")
        
        async def _async_set():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            if ttl:
                return await self.async_client.setex(key, ttl, value)
            else:
                return await self.async_client.set(key, value)
        
        return await self._async_retry_on_failure(_async_set)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value by key"""
        if not self.connected:
            raise CustomRedisError("Redis not connected")
        
        def _get():
            value = self.client.get(key)
            if value is None:
                return None
            
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        
        return self._retry_on_failure(_get)
    
    async def async_get(self, key: str) -> Optional[Any]:
        """Get a value by key asynchronously"""
        if not self.connected:
            raise CustomRedisError("Async Redis not connected")
        
        async def _async_get():
            value = await self.async_client.get(key)
            if value is None:
                return None
            
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        
        return await self._async_retry_on_failure(_async_get)
    
    def delete(self, key: str) -> bool:
        """Delete a key"""
        if not self.connected:
            raise CustomRedisError("Redis not connected")
        
        def _delete():
            return bool(self.client.delete(key))
        
        return self._retry_on_failure(_delete)
    
    async def async_delete(self, key: str) -> bool:
        """Delete a key asynchronously"""
        if not self.connected:
            raise CustomRedisError("Async Redis not connected")
        
        async def _async_delete():
            return bool(await self.async_client.delete(key))
        
        return await self._async_retry_on_failure(_async_delete)
    
    def exists(self, key: str) -> bool:
        """Check if a key exists"""
        if not self.connected:
            raise CustomRedisError("Redis not connected")
        
        def _exists():
            return bool(self.client.exists(key))
        
        return self._retry_on_failure(_exists)
    
    async def async_exists(self, key: str) -> bool:
        """Check if a key exists asynchronously"""
        if not self.connected:
            raise CustomRedisError("Async Redis not connected")
        
        async def _async_exists():
            return bool(await self.async_client.exists(key))
        
        return await self._async_retry_on_failure(_async_exists)
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        if not self.connected:
            raise CustomRedisError("Redis not connected")
        
        def _keys():
            return self.client.keys(pattern)
        
        return self._retry_on_failure(_keys)
    
    async def async_keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern asynchronously"""
        if not self.connected:
            raise CustomRedisError("Async Redis not connected")
        
        async def _async_keys():
            return await self.async_client.keys(pattern)
        
        return await self._async_retry_on_failure(_async_keys)
    
    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values by keys"""
        if not self.connected:
            raise CustomRedisError("Redis not connected")
        
        def _mget():
            values = self.client.mget(keys)
            result = []
            for value in values:
                if value is None:
                    result.append(None)
                else:
                    try:
                        result.append(json.loads(value))
                    except (json.JSONDecodeError, TypeError):
                        result.append(value)
            return result
        
        return self._retry_on_failure(_mget)
    
    async def async_mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values by keys asynchronously"""
        if not self.connected:
            raise CustomRedisError("Async Redis not connected")
        
        async def _async_mget():
            values = await self.async_client.mget(keys)
            result = []
            for value in values:
                if value is None:
                    result.append(None)
                else:
                    try:
                        result.append(json.loads(value))
                    except (json.JSONDecodeError, TypeError):
                        result.append(value)
            return result
        
        return await self._async_retry_on_failure(_async_mget)
    
    def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs"""
        if not self.connected:
            raise CustomRedisError("Redis not connected")
        
        def _mset():
            # Convert values to JSON if needed
            processed_mapping = {}
            for key, value in mapping.items():
                if isinstance(value, (dict, list)):
                    processed_mapping[key] = json.dumps(value)
                else:
                    processed_mapping[key] = value
            
            result = self.client.mset(processed_mapping)
            
            # Set TTL for all keys if specified
            if ttl and result:
                pipe = self.client.pipeline()
                for key in mapping.keys():
                    pipe.expire(key, ttl)
                pipe.execute()
            
            return result
        
        return self._retry_on_failure(_mset)
    
    async def async_mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs asynchronously"""
        if not self.connected:
            raise CustomRedisError("Async Redis not connected")
        
        async def _async_mset():
            # Convert values to JSON if needed
            processed_mapping = {}
            for key, value in mapping.items():
                if isinstance(value, (dict, list)):
                    processed_mapping[key] = json.dumps(value)
                else:
                    processed_mapping[key] = value
            
            result = await self.async_client.mset(processed_mapping)
            
            # Set TTL for all keys if specified
            if ttl and result:
                pipe = self.async_client.pipeline()
                for key in mapping.keys():
                    await pipe.expire(key, ttl)
                await pipe.execute()
            
            return result
        
        return await self._async_retry_on_failure(_async_mset)
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            if not self.connected:
                return {
                    "status": "disconnected",
                    "error": "Not connected to Redis"
                }
            
            # Test basic operations
            start_time = time.time()
            self.client.ping()
            ping_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get Redis info
            info = self.client.info()
            
            return {
                "status": "connected",
                "ping_time_ms": round(ping_time, 2),
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace": info.get("db0", {}).get("keys", 0)
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def async_health_check(self) -> Dict[str, Any]:
        """Check Redis health asynchronously"""
        try:
            if not self.connected:
                return {
                    "status": "disconnected",
                    "error": "Not connected to async Redis"
                }
            
            # Test basic operations
            start_time = time.time()
            await self.async_client.ping()
            ping_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get Redis info
            info = await self.async_client.info()
            
            return {
                "status": "connected",
                "ping_time_ms": round(ping_time, 2),
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace": info.get("db0", {}).get("keys", 0)
            }
            
        except Exception as e:
            logger.error(f"Async Redis health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


# Global Redis service instance
redis_service = RedisService()
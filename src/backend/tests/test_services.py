"""
Unit tests for enhanced services
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from services.enhanced_redis_service import EnhancedRedisService
from exceptions import RedisConnectionError, RedisOperationError


class TestEnhancedRedisService:
    """Test cases for EnhancedRedisService"""
    
    @pytest.fixture
    def redis_service(self):
        """Create Redis service instance for testing"""
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value.ping.return_value = True
            service = EnhancedRedisService(
                host="localhost",
                port=6379,
                db=0,
                max_connections=5
            )
            service.connected = True
            return service
    
    @pytest.fixture
    def disconnected_redis_service(self):
        """Create disconnected Redis service instance for testing"""
        with patch('redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")
            service = EnhancedRedisService(
                host="localhost",
                port=6379,
                db=0
            )
            service.connected = False
            return service
    
    def test_initialization_success(self):
        """Test successful Redis service initialization"""
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value.ping.return_value = True
            service = EnhancedRedisService()
            assert service.connected is True
            assert service.redis_client is not None
    
    def test_initialization_failure(self):
        """Test Redis service initialization failure"""
        with patch('redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")
            with pytest.raises(RedisConnectionError):
                EnhancedRedisService()
    
    @pytest.mark.asyncio
    async def test_set_traffic_data_success(self, redis_service):
        """Test successful traffic data storage"""
        with patch.object(redis_service.redis_client, 'setex', return_value=True):
            result = await redis_service.set_traffic_data(
                "junction-1",
                {"test": "data"},
                3600
            )
            assert result is True
            assert redis_service.stats["successful_operations"] == 1
    
    @pytest.mark.asyncio
    async def test_set_traffic_data_failure(self, redis_service):
        """Test traffic data storage failure"""
        with patch.object(redis_service.redis_client, 'setex', side_effect=Exception("Redis error")):
            with pytest.raises(RedisOperationError):
                await redis_service.set_traffic_data(
                    "junction-1",
                    {"test": "data"},
                    3600
                )
            assert redis_service.stats["failed_operations"] == 1
    
    @pytest.mark.asyncio
    async def test_get_traffic_data_success(self, redis_service):
        """Test successful traffic data retrieval"""
        test_data = {"test": "data", "timestamp": 1234567890}
        with patch.object(redis_service.redis_client, 'get', return_value='{"test": "data", "timestamp": 1234567890}'):
            result = await redis_service.get_traffic_data("junction-1")
            assert result == test_data
            assert redis_service.stats["successful_operations"] == 1
    
    @pytest.mark.asyncio
    async def test_get_traffic_data_not_found(self, redis_service):
        """Test traffic data retrieval when not found"""
        with patch.object(redis_service.redis_client, 'get', return_value=None):
            result = await redis_service.get_traffic_data("junction-1")
            assert result is None
            assert redis_service.stats["successful_operations"] == 1
    
    @pytest.mark.asyncio
    async def test_get_traffic_data_failure(self, redis_service):
        """Test traffic data retrieval failure"""
        with patch.object(redis_service.redis_client, 'get', side_effect=Exception("Redis error")):
            with pytest.raises(RedisOperationError):
                await redis_service.get_traffic_data("junction-1")
            assert redis_service.stats["failed_operations"] == 1
    
    @pytest.mark.asyncio
    async def test_disconnected_service(self, disconnected_redis_service):
        """Test operations on disconnected service"""
        with pytest.raises(RedisConnectionError):
            await disconnected_redis_service.set_traffic_data("junction-1", {"test": "data"})
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, redis_service):
        """Test health check when service is healthy"""
        with patch.object(redis_service.redis_client, 'ping', return_value=True):
            result = await redis_service.health_check()
            assert result["status"] == "healthy"
            assert result["connected"] is True
            assert "response_time_ms" in result
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, redis_service):
        """Test health check when service is unhealthy"""
        with patch.object(redis_service.redis_client, 'ping', side_effect=Exception("Connection failed")):
            result = await redis_service.health_check()
            assert result["status"] == "unhealthy"
            assert result["connected"] is False
            assert "error" in result
    
    def test_get_stats(self, redis_service):
        """Test getting service statistics"""
        stats = redis_service.get_stats()
        assert "total_operations" in stats
        assert "successful_operations" in stats
        assert "failed_operations" in stats
        assert "connection_errors" in stats
        assert "timeout_errors" in stats
    
    def test_reset_stats(self, redis_service):
        """Test resetting service statistics"""
        # Set some initial stats
        redis_service.stats["total_operations"] = 10
        redis_service.stats["successful_operations"] = 8
        
        # Reset stats
        redis_service.reset_stats()
        
        # Check stats are reset
        assert redis_service.stats["total_operations"] == 0
        assert redis_service.stats["successful_operations"] == 0
        assert redis_service.stats["last_connection_time"] is not None  # Should be preserved
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, redis_service):
        """Test retry mechanism for failed operations"""
        call_count = 0
        
        def mock_setex(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first two attempts
                raise Exception("Temporary error")
            return True
        
        with patch.object(redis_service.redis_client, 'setex', side_effect=mock_setex):
            result = await redis_service.set_traffic_data("junction-1", {"test": "data"})
            assert result is True
            assert call_count == 3  # Should have retried twice
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self, redis_service):
        """Test retry mechanism when all attempts are exhausted"""
        with patch.object(redis_service.redis_client, 'setex', side_effect=Exception("Persistent error")):
            with pytest.raises(RedisOperationError) as exc_info:
                await redis_service.set_traffic_data("junction-1", {"test": "data"})
            assert "Operation failed after 4 attempts" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_all_intersections(self, redis_service):
        """Test getting all intersections"""
        with patch.object(redis_service.redis_client, 'keys', return_value=['traffic:junction-1', 'traffic:junction-2']):
            intersections = await redis_service.get_all_intersections()
            assert intersections == ['junction-1', 'junction-2']
    
    @pytest.mark.asyncio
    async def test_delete_data(self, redis_service):
        """Test deleting data by pattern"""
        with patch.object(redis_service.redis_client, 'keys', return_value=['traffic:junction-1', 'traffic:junction-2']):
            with patch.object(redis_service.redis_client, 'delete', return_value=2):
                deleted_count = await redis_service.delete_data("traffic:*")
                assert deleted_count == 2
    
    @pytest.mark.asyncio
    async def test_close_service(self, redis_service):
        """Test closing Redis service"""
        with patch.object(redis_service.redis_client, 'close'):
            with patch.object(redis_service.connection_pool, 'disconnect'):
                await redis_service.close()
                assert redis_service.connected is False


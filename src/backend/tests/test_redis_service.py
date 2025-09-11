"""
Unit tests for Redis service functionality
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, MagicMock
from services.redis_service import RedisService
from exceptions.custom_exceptions import RedisError


class TestRedisService:
    """Test Redis service functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.redis_service = RedisService()
        self.redis_service.connected = True
        self.redis_service.client = Mock()
        self.redis_service.async_client = Mock()
    
    def test_redis_connection_success(self):
        """Test successful Redis connection"""
        with patch.object(self.redis_service, '_create_connection_pool') as mock_pool:
            with patch.object(self.redis_service.client, 'ping') as mock_ping:
                mock_ping.return_value = True
                
                result = self.redis_service.connect()
                
                assert result is True
                assert self.redis_service.connected is True
                mock_pool.assert_called_once()
    
    def test_redis_connection_failure(self):
        """Test Redis connection failure"""
        with patch.object(self.redis_service, '_create_connection_pool') as mock_pool:
            with patch.object(self.redis_service.client, 'ping') as mock_ping:
                mock_ping.side_effect = Exception("Connection failed")
                
                result = self.redis_service.connect()
                
                assert result is False
                assert self.redis_service.connected is False
    
    def test_set_operation_success(self):
        """Test successful set operation"""
        self.redis_service.client.setex.return_value = True
        
        result = self.redis_service.set("test_key", "test_value", ttl=60)
        
        assert result is True
        self.redis_service.client.setex.assert_called_once_with("test_key", 60, "test_value")
    
    def test_set_operation_with_json(self):
        """Test set operation with JSON data"""
        self.redis_service.client.setex.return_value = True
        test_data = {"key": "value", "number": 123}
        
        result = self.redis_service.set("test_key", test_data, ttl=60)
        
        assert result is True
        expected_json = json.dumps(test_data)
        self.redis_service.client.setex.assert_called_once_with("test_key", 60, expected_json)
    
    def test_set_operation_without_ttl(self):
        """Test set operation without TTL"""
        self.redis_service.client.set.return_value = True
        
        result = self.redis_service.set("test_key", "test_value")
        
        assert result is True
        self.redis_service.client.set.assert_called_once_with("test_key", "test_value")
    
    def test_get_operation_success(self):
        """Test successful get operation"""
        self.redis_service.client.get.return_value = "test_value"
        
        result = self.redis_service.get("test_key")
        
        assert result == "test_value"
        self.redis_service.client.get.assert_called_once_with("test_key")
    
    def test_get_operation_with_json(self):
        """Test get operation with JSON data"""
        test_data = {"key": "value", "number": 123}
        self.redis_service.client.get.return_value = json.dumps(test_data)
        
        result = self.redis_service.get("test_key")
        
        assert result == test_data
        self.redis_service.client.get.assert_called_once_with("test_key")
    
    def test_get_operation_none(self):
        """Test get operation when key doesn't exist"""
        self.redis_service.client.get.return_value = None
        
        result = self.redis_service.get("test_key")
        
        assert result is None
        self.redis_service.client.get.assert_called_once_with("test_key")
    
    def test_delete_operation_success(self):
        """Test successful delete operation"""
        self.redis_service.client.delete.return_value = 1
        
        result = self.redis_service.delete("test_key")
        
        assert result is True
        self.redis_service.client.delete.assert_called_once_with("test_key")
    
    def test_delete_operation_not_found(self):
        """Test delete operation when key doesn't exist"""
        self.redis_service.client.delete.return_value = 0
        
        result = self.redis_service.delete("test_key")
        
        assert result is False
        self.redis_service.client.delete.assert_called_once_with("test_key")
    
    def test_exists_operation_success(self):
        """Test exists operation"""
        self.redis_service.client.exists.return_value = 1
        
        result = self.redis_service.exists("test_key")
        
        assert result is True
        self.redis_service.client.exists.assert_called_once_with("test_key")
    
    def test_exists_operation_not_found(self):
        """Test exists operation when key doesn't exist"""
        self.redis_service.client.exists.return_value = 0
        
        result = self.redis_service.exists("test_key")
        
        assert result is False
        self.redis_service.client.exists.assert_called_once_with("test_key")
    
    def test_keys_operation(self):
        """Test keys operation"""
        expected_keys = ["key1", "key2", "key3"]
        self.redis_service.client.keys.return_value = expected_keys
        
        result = self.redis_service.keys("test_*")
        
        assert result == expected_keys
        self.redis_service.client.keys.assert_called_once_with("test_*")
    
    def test_mget_operation(self):
        """Test mget operation"""
        keys = ["key1", "key2", "key3"]
        values = ["value1", json.dumps({"data": "value2"}), "value3"]
        self.redis_service.client.mget.return_value = values
        
        result = self.redis_service.mget(keys)
        
        expected = ["value1", {"data": "value2"}, "value3"]
        assert result == expected
        self.redis_service.client.mget.assert_called_once_with(keys)
    
    def test_mset_operation(self):
        """Test mset operation"""
        mapping = {"key1": "value1", "key2": {"data": "value2"}}
        self.redis_service.client.mset.return_value = True
        self.redis_service.client.pipeline.return_value.__enter__.return_value = Mock()
        
        result = self.redis_service.mset(mapping, ttl=60)
        
        assert result is True
        self.redis_service.client.mset.assert_called_once()
    
    def test_health_check_success(self):
        """Test successful health check"""
        self.redis_service.client.ping.return_value = True
        self.redis_service.client.info.return_value = {
            "redis_version": "6.2.0",
            "used_memory_human": "1.2M",
            "connected_clients": 5,
            "total_commands_processed": 1000,
            "db0": {"keys": 50}
        }
        
        result = self.redis_service.health_check()
        
        assert result["status"] == "connected"
        assert "ping_time_ms" in result
        assert result["redis_version"] == "6.2.0"
        assert result["used_memory"] == "1.2M"
        assert result["connected_clients"] == 5
        assert result["total_commands_processed"] == 1000
        assert result["keyspace"] == 50
    
    def test_health_check_failure(self):
        """Test health check failure"""
        self.redis_service.client.ping.side_effect = Exception("Connection failed")
        
        result = self.redis_service.health_check()
        
        assert result["status"] == "error"
        assert "error" in result
    
    def test_health_check_disconnected(self):
        """Test health check when disconnected"""
        self.redis_service.connected = False
        
        result = self.redis_service.health_check()
        
        assert result["status"] == "disconnected"
        assert "error" in result
    
    def test_retry_logic_success(self):
        """Test retry logic on success"""
        self.redis_service.client.ping.return_value = True
        
        def test_func():
            return self.redis_service.client.ping()
        
        result = self.redis_service._retry_on_failure(test_func)
        
        assert result is True
        self.redis_service.client.ping.assert_called_once()
    
    def test_retry_logic_failure(self):
        """Test retry logic on failure"""
        self.redis_service.client.ping.side_effect = Exception("Connection failed")
        
        def test_func():
            return self.redis_service.client.ping()
        
        with pytest.raises(RedisError):
            self.redis_service._retry_on_failure(test_func)
    
    def test_operation_when_disconnected(self):
        """Test operations when Redis is disconnected"""
        self.redis_service.connected = False
        
        with pytest.raises(RedisError, match="Redis not connected"):
            self.redis_service.set("key", "value")
        
        with pytest.raises(RedisError, match="Redis not connected"):
            self.redis_service.get("key")
        
        with pytest.raises(RedisError, match="Redis not connected"):
            self.redis_service.delete("key")
    
    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async Redis operations"""
        self.redis_service.connected = True
        self.redis_service.async_client.setex.return_value = True
        self.redis_service.async_client.get.return_value = "test_value"
        self.redis_service.async_client.ping.return_value = True
        self.redis_service.async_client.info.return_value = {
            "redis_version": "6.2.0",
            "used_memory_human": "1.2M",
            "connected_clients": 5,
            "total_commands_processed": 1000,
            "db0": {"keys": 50}
        }
        
        # Test async set
        result = await self.redis_service.async_set("test_key", "test_value", ttl=60)
        assert result is True
        self.redis_service.async_client.setex.assert_called_once_with("test_key", 60, "test_value")
        
        # Test async get
        result = await self.redis_service.async_get("test_key")
        assert result == "test_value"
        self.redis_service.async_client.get.assert_called_once_with("test_key")
        
        # Test async health check
        health = await self.redis_service.async_health_check()
        assert health["status"] == "connected"
        assert health["redis_version"] == "6.2.0"
    
    def test_disconnect(self):
        """Test Redis disconnect"""
        self.redis_service.connected = True
        self.redis_service.client = Mock()
        self.redis_service.pool = Mock()
        
        self.redis_service.disconnect()
        
        assert self.redis_service.connected is False
        self.redis_service.client.close.assert_called_once()
        self.redis_service.pool.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_disconnect(self):
        """Test async Redis disconnect"""
        self.redis_service.connected = True
        self.redis_service.async_client = Mock()
        self.redis_service.async_pool = Mock()
        
        await self.redis_service.async_disconnect()
        
        assert self.redis_service.connected is False
        self.redis_service.async_client.close.assert_called_once()
        self.redis_service.async_pool.disconnect.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

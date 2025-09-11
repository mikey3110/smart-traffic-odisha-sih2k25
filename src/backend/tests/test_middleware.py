"""
Unit tests for middleware functionality
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi import Request, Response
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from middleware.rate_limiting import RateLimitingMiddleware, AdvancedRateLimitingMiddleware
from middleware.request_timing import RequestTimingMiddleware
from middleware.logging_middleware import LoggingMiddleware, SecurityLoggingMiddleware
from services.redis_service import redis_service


class TestRateLimitingMiddleware:
    """Test rate limiting middleware"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.app = Mock()
        self.middleware = RateLimitingMiddleware(
            app=self.app,
            requests_per_minute=10,
            window_size=60
        )
    
    def test_middleware_initialization(self):
        """Test middleware initialization"""
        assert self.middleware.max_requests == 10
        assert self.middleware.window_seconds == 60
        assert "/health" in self.middleware.skip_paths
    
    def test_default_key_func(self):
        """Test default key function"""
        request = Mock()
        request.client.host = "192.168.1.1"
        request.url.path = "/api/v1/traffic/status"
        
        key = self.middleware._default_key_func(request)
        
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length
    
    def test_get_rate_limit_key(self):
        """Test rate limit key generation"""
        request = Mock()
        request.client.host = "192.168.1.1"
        request.url.path = "/api/v1/traffic/status"
        
        key = self.middleware._get_rate_limit_key(request)
        
        assert key.startswith("rate_limit:")
        assert len(key) > 10
    
    @patch.object(redis_service, 'connected', True)
    @patch.object(redis_service, 'get')
    @patch.object(redis_service, 'set')
    def test_rate_limiting_not_limited(self, mock_set, mock_get):
        """Test rate limiting when not limited"""
        mock_get.return_value = {}
        
        request = Mock()
        request.client.host = "192.168.1.1"
        request.url.path = "/api/v1/traffic/status"
        
        is_limited, current_count, reset_time = self.middleware._is_rate_limited("test_key")
        
        assert is_limited is False
        assert current_count == 1
        assert reset_time > 0
        mock_set.assert_called_once()
    
    @patch.object(redis_service, 'connected', True)
    @patch.object(redis_service, 'get')
    @patch.object(redis_service, 'set')
    def test_rate_limiting_limited(self, mock_set, mock_get):
        """Test rate limiting when limited"""
        # Mock that we've already hit the limit
        current_time = int(time.time())
        window_data = {str(current_time - 30): 10}  # Already at limit
        mock_get.return_value = window_data
        
        request = Mock()
        request.client.host = "192.168.1.1"
        request.url.path = "/api/v1/traffic/status"
        
        is_limited, current_count, reset_time = self.middleware._is_rate_limited("test_key")
        
        assert is_limited is True
        assert current_count == 10
        assert reset_time > 0
    
    @patch.object(redis_service, 'connected', False)
    async def test_dispatch_redis_unavailable(self):
        """Test dispatch when Redis is unavailable"""
        request = Mock()
        request.url.path = "/api/v1/traffic/status"
        call_next = Mock()
        call_next.return_value = Response()
        
        response = await self.middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)
    
    @patch.object(redis_service, 'connected', True)
    @patch.object(redis_service, 'get')
    @patch.object(redis_service, 'set')
    async def test_dispatch_skip_paths(self, mock_set, mock_get):
        """Test dispatch for skip paths"""
        request = Mock()
        request.url.path = "/health"
        call_next = Mock()
        call_next.return_value = Response()
        
        response = await self.middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)
        mock_get.assert_not_called()


class TestAdvancedRateLimitingMiddleware:
    """Test advanced rate limiting middleware"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.app = Mock()
        self.rate_limits = {
            "/traffic/ingest": {"requests": 200, "window": 60},
            "/signal/optimize": {"requests": 10, "window": 60},
            "default": {"requests": 50, "window": 60}
        }
        self.middleware = AdvancedRateLimitingMiddleware(
            app=self.app,
            rate_limits=self.rate_limits
        )
    
    def test_get_rate_limit_config_exact_match(self):
        """Test getting rate limit config for exact path match"""
        config = self.middleware._get_rate_limit_config("/traffic/ingest")
        
        assert config["requests"] == 200
        assert config["window"] == 60
    
    def test_get_rate_limit_config_pattern_match(self):
        """Test getting rate limit config for pattern match"""
        config = self.middleware._get_rate_limit_config("/traffic/ingest/data")
        
        assert config["requests"] == 200
        assert config["window"] == 60
    
    def test_get_rate_limit_config_default(self):
        """Test getting default rate limit config"""
        config = self.middleware._get_rate_limit_config("/unknown/path")
        
        assert config["requests"] == 50
        assert config["window"] == 60
    
    def test_get_rate_limit_key(self):
        """Test rate limit key generation with config"""
        request = Mock()
        request.client.host = "192.168.1.1"
        request.url.path = "/traffic/ingest"
        config = {"requests": 200, "window": 60}
        
        key = self.middleware._get_rate_limit_key(request, config)
        
        assert key.startswith("rate_limit_advanced:")
        assert len(key) > 20


class TestRequestTimingMiddleware:
    """Test request timing middleware"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.app = Mock()
        self.middleware = RequestTimingMiddleware(app=self.app)
    
    def test_middleware_initialization(self):
        """Test middleware initialization"""
        assert self.middleware.skip_paths is not None
        assert "/health" in self.middleware.skip_paths
    
    async def test_dispatch_skip_paths(self):
        """Test dispatch for skip paths"""
        request = Mock()
        request.url.path = "/health"
        call_next = Mock()
        call_next.return_value = Response()
        
        response = await self.middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)
    
    async def test_dispatch_with_timing(self):
        """Test dispatch with timing"""
        request = Mock()
        request.url.path = "/api/v1/traffic/status"
        request.method = "GET"
        request.state = Mock()
        call_next = Mock()
        call_next.return_value = Response()
        
        with patch('middleware.request_timing.log_api_request') as mock_log_req:
            with patch('middleware.request_timing.log_api_response') as mock_log_resp:
                response = await self.middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)
        mock_log_req.assert_called_once()
        mock_log_resp.assert_called_once()
        assert hasattr(request.state, 'request_id')
    
    async def test_dispatch_with_error(self):
        """Test dispatch with error"""
        request = Mock()
        request.url.path = "/api/v1/traffic/status"
        request.method = "GET"
        request.state = Mock()
        call_next = Mock()
        call_next.side_effect = Exception("Test error")
        
        with patch('middleware.request_timing.logger') as mock_logger:
            with pytest.raises(Exception):
                await self.middleware.dispatch(request, call_next)
        
        mock_logger.error.assert_called_once()


class TestLoggingMiddleware:
    """Test logging middleware"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.app = Mock()
        self.middleware = LoggingMiddleware(
            app=self.app,
            log_requests=True,
            log_responses=True,
            log_request_body=False,
            log_response_body=False
        )
    
    def test_middleware_initialization(self):
        """Test middleware initialization"""
        assert self.middleware.log_requests is True
        assert self.middleware.log_responses is True
        assert self.middleware.log_request_body is False
        assert self.middleware.log_response_body is False
        assert self.middleware.max_body_size == 1024
    
    async def test_dispatch_skip_paths(self):
        """Test dispatch for skip paths"""
        request = Mock()
        request.url.path = "/health"
        call_next = Mock()
        call_next.return_value = Response()
        
        response = await self.middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)
    
    async def test_dispatch_with_logging(self):
        """Test dispatch with logging"""
        request = Mock()
        request.url.path = "/api/v1/traffic/status"
        request.method = "GET"
        request.state = Mock()
        request.state.request_id = "test-request-id"
        request.client.host = "192.168.1.1"
        request.headers = {"user-agent": "test-agent"}
        request.query_params = {}
        call_next = Mock()
        call_next.return_value = Response()
        
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await self.middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)
        assert mock_logger.info.call_count >= 1
    
    async def test_dispatch_with_error(self):
        """Test dispatch with error"""
        request = Mock()
        request.url.path = "/api/v1/traffic/status"
        request.method = "GET"
        request.state = Mock()
        request.state.request_id = "test-request-id"
        request.client.host = "192.168.1.1"
        call_next = Mock()
        call_next.side_effect = Exception("Test error")
        
        with patch('middleware.logging_middleware.logger') as mock_logger:
            with pytest.raises(Exception):
                await self.middleware.dispatch(request, call_next)
        
        mock_logger.error.assert_called_once()


class TestSecurityLoggingMiddleware:
    """Test security logging middleware"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.app = Mock()
        self.middleware = SecurityLoggingMiddleware(app=self.app)
    
    def test_middleware_initialization(self):
        """Test middleware initialization"""
        assert self.middleware.suspicious_patterns is not None
        assert "admin" in self.middleware.suspicious_patterns
        assert "script" in self.middleware.suspicious_patterns
    
    async def test_dispatch_normal_request(self):
        """Test dispatch with normal request"""
        request = Mock()
        request.url.path = "/api/v1/traffic/status"
        request.method = "GET"
        request.state = Mock()
        request.state.request_id = "test-request-id"
        request.client.host = "192.168.1.1"
        request.headers = {"user-agent": "test-agent"}
        request.query_params = {}
        call_next = Mock()
        call_next.return_value = Response()
        
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await self.middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)
        mock_logger.warning.assert_not_called()
    
    async def test_dispatch_suspicious_request(self):
        """Test dispatch with suspicious request"""
        request = Mock()
        request.url.path = "/admin/login"
        request.method = "GET"
        request.state = Mock()
        request.state.request_id = "test-request-id"
        request.client.host = "192.168.1.1"
        request.headers = {"user-agent": "test-agent"}
        request.query_params = {}
        call_next = Mock()
        call_next.return_value = Response()
        
        with patch('middleware.logging_middleware.logger') as mock_logger:
            response = await self.middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)
        mock_logger.warning.assert_called_once()
    
    async def test_dispatch_with_security_response(self):
        """Test dispatch with security-relevant response"""
        request = Mock()
        request.url.path = "/api/v1/traffic/status"
        request.method = "GET"
        request.state = Mock()
        request.state.request_id = "test-request-id"
        request.client.host = "192.168.1.1"
        call_next = Mock()
        response = Mock()
        response.status_code = 401  # Unauthorized
        call_next.return_value = response
        
        with patch('middleware.logging_middleware.logger') as mock_logger:
            await self.middleware.dispatch(request, call_next)
        
        mock_logger.warning.assert_called_once()


class TestMiddlewareIntegration:
    """Test middleware integration"""
    
    def test_middleware_chain(self):
        """Test middleware chain functionality"""
        app = Mock()
        
        # Create middleware chain
        timing_middleware = RequestTimingMiddleware(app)
        logging_middleware = LoggingMiddleware(app)
        rate_limiting_middleware = RateLimitingMiddleware(app)
        
        # Verify all middleware are properly initialized
        assert timing_middleware.skip_paths is not None
        assert logging_middleware.log_requests is True
        assert rate_limiting_middleware.max_requests > 0
    
    def test_middleware_configuration(self):
        """Test middleware configuration"""
        app = Mock()
        
        # Test different configurations
        middleware1 = LoggingMiddleware(
            app=app,
            log_requests=True,
            log_responses=False,
            log_request_body=True,
            max_body_size=2048
        )
        
        assert middleware1.log_requests is True
        assert middleware1.log_responses is False
        assert middleware1.log_request_body is True
        assert middleware1.max_body_size == 2048


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

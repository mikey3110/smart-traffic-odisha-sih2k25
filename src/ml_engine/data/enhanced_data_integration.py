"""
Enhanced Real-time Data Integration for ML Traffic Signal Optimization
Provides robust data fetching from backend API with error handling, retries, and fallback strategies
"""

import asyncio
import aiohttp
import requests
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
from enum import Enum
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.ml_config import DataIntegrationConfig, get_config
from prediction.traffic_predictor import TrafficPredictor


class DataSource(Enum):
    """Available data sources"""
    API = "api"
    MOCK = "mock"
    CACHE = "cache"
    PREDICTION = "prediction"


@dataclass
class TrafficDataPoint:
    """Single traffic data point with metadata"""
    intersection_id: str
    timestamp: datetime
    lane_counts: Dict[str, int]
    avg_speed: float
    weather_condition: str
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    visibility: Optional[float] = None
    source: DataSource = DataSource.API
    confidence: float = 1.0
    processing_time: float = 0.0


@dataclass
class DataIntegrationStats:
    """Statistics for data integration performance"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    prediction_requests: int = 0
    avg_response_time: float = 0.0
    last_successful_fetch: Optional[datetime] = None
    consecutive_failures: int = 0
    fallback_activations: int = 0


class DataIntegrationError(Exception):
    """Custom exception for data integration errors"""
    pass


class EnhancedDataIntegration:
    """
    Enhanced real-time data integration with multiple fallback strategies
    
    Features:
    - Robust API communication with retries and backoff
    - Intelligent caching with TTL
    - ML-based prediction fallback
    - Mock data generation for testing
    - Comprehensive error handling and logging
    - Performance monitoring and statistics
    """
    
    def __init__(self, config: Optional[DataIntegrationConfig] = None):
        self.config = config or get_config().data_integration
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.traffic_predictor = TrafficPredictor()
        self.cache: Dict[str, Tuple[TrafficDataPoint, float]] = {}
        self.stats = DataIntegrationStats()
        
        # Mock data generator for fallback
        self.mock_data_generator = MockDataGenerator()
        
        # Performance tracking
        self.response_times: List[float] = []
        self.last_cache_cleanup = time.time()
        
        # Initialize session for connection pooling
        self.session = None
        
        self.logger.info("Enhanced data integration initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.api_timeout),
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        if current_time - self.last_cache_cleanup < 60:  # Cleanup every minute
            return
        
        expired_keys = []
        for key, (data_point, timestamp) in self.cache.items():
            if current_time - timestamp > self.config.cache_duration:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        self.last_cache_cleanup = current_time
        self.logger.debug(f"Cache cleanup: removed {len(expired_keys)} expired entries")
    
    def _get_cache_key(self, intersection_id: str) -> str:
        """Generate cache key for intersection"""
        return f"traffic_data:{intersection_id}"
    
    def _is_cache_valid(self, cache_entry: Tuple[TrafficDataPoint, float]) -> bool:
        """Check if cache entry is still valid"""
        _, timestamp = cache_entry
        return time.time() - timestamp < self.config.cache_duration
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _fetch_from_api_async(self, intersection_id: str) -> TrafficDataPoint:
        """Fetch traffic data from API with retry logic"""
        if not self.session:
            raise DataIntegrationError("Session not initialized")
        
        start_time = time.time()
        url = f"{self.config.api_base_url}/api/v1/traffic/status/{intersection_id}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    processing_time = time.time() - start_time
                    
                    # Parse API response
                    traffic_data = self._parse_api_response(data, intersection_id, processing_time)
                    
                    # Update statistics
                    self.stats.successful_requests += 1
                    self.stats.last_successful_fetch = datetime.now()
                    self.stats.consecutive_failures = 0
                    
                    self.response_times.append(processing_time)
                    if len(self.response_times) > 100:
                        self.response_times = self.response_times[-100:]
                    
                    self.stats.avg_response_time = np.mean(self.response_times)
                    
                    return traffic_data
                else:
                    raise DataIntegrationError(f"API returned status {response.status}")
        
        except asyncio.TimeoutError:
            self.logger.warning(f"API timeout for intersection {intersection_id}")
            raise
        except aiohttp.ClientError as e:
            self.logger.warning(f"API client error for intersection {intersection_id}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error fetching data for {intersection_id}: {e}")
            raise DataIntegrationError(f"Unexpected error: {e}")
    
    def _fetch_from_api_sync(self, intersection_id: str) -> TrafficDataPoint:
        """Synchronous API fetch with retry logic"""
        start_time = time.time()
        url = f"{self.config.api_base_url}/api/v1/traffic/status/{intersection_id}"
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = requests.get(
                    url,
                    timeout=self.config.api_timeout,
                    headers={'User-Agent': 'ML-Traffic-Optimizer/1.0'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    processing_time = time.time() - start_time
                    
                    # Parse API response
                    traffic_data = self._parse_api_response(data, intersection_id, processing_time)
                    
                    # Update statistics
                    self.stats.successful_requests += 1
                    self.stats.last_successful_fetch = datetime.now()
                    self.stats.consecutive_failures = 0
                    
                    self.response_times.append(processing_time)
                    if len(self.response_times) > 100:
                        self.response_times = self.response_times[-100:]
                    
                    self.stats.avg_response_time = np.mean(self.response_times)
                    
                    return traffic_data
                else:
                    self.logger.warning(f"API returned status {response.status_code} (attempt {attempt + 1})")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"API timeout for intersection {intersection_id} (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"API connection error for intersection {intersection_id} (attempt {attempt + 1})")
            except Exception as e:
                self.logger.error(f"Unexpected error fetching data for {intersection_id} (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < self.config.retry_attempts - 1:
                wait_time = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                time.sleep(wait_time)
        
        # All retries failed
        self.stats.failed_requests += 1
        self.stats.consecutive_failures += 1
        raise DataIntegrationError(f"Failed to fetch data for {intersection_id} after {self.config.retry_attempts} attempts")
    
    def _parse_api_response(self, data: Dict[str, Any], intersection_id: str, processing_time: float) -> TrafficDataPoint:
        """Parse API response into TrafficDataPoint"""
        try:
            # Extract data from API response
            traffic_info = data.get('data', {})
            lane_counts = traffic_info.get('lane_counts', {})
            
            # Ensure all required lanes are present
            required_lanes = ['north_lane', 'south_lane', 'east_lane', 'west_lane']
            for lane in required_lanes:
                if lane not in lane_counts:
                    lane_counts[lane] = 0
            
            return TrafficDataPoint(
                intersection_id=intersection_id,
                timestamp=datetime.now(),
                lane_counts=lane_counts,
                avg_speed=traffic_info.get('avg_speed', 0.0),
                weather_condition=traffic_info.get('weather_condition', 'clear'),
                temperature=traffic_info.get('temperature'),
                humidity=traffic_info.get('humidity'),
                visibility=traffic_info.get('visibility'),
                source=DataSource.API,
                confidence=1.0,
                processing_time=processing_time
            )
        
        except Exception as e:
            self.logger.error(f"Error parsing API response: {e}")
            raise DataIntegrationError(f"Failed to parse API response: {e}")
    
    def _get_cached_data(self, intersection_id: str) -> Optional[TrafficDataPoint]:
        """Get data from cache if available and valid"""
        cache_key = self._get_cache_key(intersection_id)
        
        if cache_key in self.cache:
            data_point, timestamp = self.cache[cache_key]
            if self._is_cache_valid((data_point, timestamp)):
                self.stats.cache_hits += 1
                self.logger.debug(f"Cache hit for intersection {intersection_id}")
                return data_point
            else:
                # Remove expired entry
                del self.cache[cache_key]
        
        return None
    
    def _cache_data(self, data_point: TrafficDataPoint):
        """Cache data point with timestamp"""
        cache_key = self._get_cache_key(data_point.intersection_id)
        self.cache[cache_key] = (data_point, time.time())
        self.logger.debug(f"Cached data for intersection {data_point.intersection_id}")
    
    def _get_predicted_data(self, intersection_id: str) -> TrafficDataPoint:
        """Get predicted traffic data using ML model"""
        try:
            # Use traffic predictor to generate prediction
            prediction = self.traffic_predictor.predict_traffic_flow(
                intersection_id=intersection_id,
                prediction_horizon=5  # 5 minutes ahead
            )
            
            self.stats.prediction_requests += 1
            self.logger.info(f"Using predicted data for intersection {intersection_id}")
            
            return TrafficDataPoint(
                intersection_id=intersection_id,
                timestamp=datetime.now(),
                lane_counts=prediction.get('lane_counts', {}),
                avg_speed=prediction.get('avg_speed', 0.0),
                weather_condition=prediction.get('weather_condition', 'clear'),
                temperature=prediction.get('temperature'),
                humidity=prediction.get('humidity'),
                visibility=prediction.get('visibility'),
                source=DataSource.PREDICTION,
                confidence=prediction.get('confidence', 0.7),
                processing_time=0.0
            )
        
        except Exception as e:
            self.logger.error(f"Error generating prediction for {intersection_id}: {e}")
            raise DataIntegrationError(f"Failed to generate prediction: {e}")
    
    def _get_mock_data(self, intersection_id: str) -> TrafficDataPoint:
        """Generate mock data for testing/fallback"""
        mock_data = self.mock_data_generator.generate_traffic_data(intersection_id)
        self.stats.fallback_activations += 1
        self.logger.warning(f"Using mock data for intersection {intersection_id}")
        
        return TrafficDataPoint(
            intersection_id=intersection_id,
            timestamp=datetime.now(),
            lane_counts=mock_data['lane_counts'],
            avg_speed=mock_data['avg_speed'],
            weather_condition=mock_data['weather_condition'],
            temperature=mock_data.get('temperature'),
            humidity=mock_data.get('humidity'),
            visibility=mock_data.get('visibility'),
            source=DataSource.MOCK,
            confidence=0.5,
            processing_time=0.0
        )
    
    async def fetch_traffic_data_async(self, intersection_id: str) -> TrafficDataPoint:
        """
        Fetch traffic data with fallback strategy (async version)
        
        Fallback order:
        1. Cache (if valid)
        2. API (with retries)
        3. ML Prediction
        4. Mock data
        """
        self.stats.total_requests += 1
        self._cleanup_cache()
        
        # Try cache first
        cached_data = self._get_cached_data(intersection_id)
        if cached_data:
            return cached_data
        
        # Try API
        try:
            if self.config.fallback_to_mock:
                data = await self._fetch_from_api_async(intersection_id)
                self._cache_data(data)
                return data
        except Exception as e:
            self.logger.warning(f"API fetch failed for {intersection_id}: {e}")
        
        # Try ML prediction
        try:
            data = self._get_predicted_data(intersection_id)
            self._cache_data(data)
            return data
        except Exception as e:
            self.logger.warning(f"Prediction failed for {intersection_id}: {e}")
        
        # Fallback to mock data
        data = self._get_mock_data(intersection_id)
        self._cache_data(data)
        return data
    
    def fetch_traffic_data(self, intersection_id: str) -> TrafficDataPoint:
        """
        Fetch traffic data with fallback strategy (sync version)
        
        Fallback order:
        1. Cache (if valid)
        2. API (with retries)
        3. ML Prediction
        4. Mock data
        """
        self.stats.total_requests += 1
        self._cleanup_cache()
        
        # Try cache first
        cached_data = self._get_cached_data(intersection_id)
        if cached_data:
            return cached_data
        
        # Try API
        try:
            if self.config.fallback_to_mock:
                data = self._fetch_from_api_sync(intersection_id)
                self._cache_data(data)
                return data
        except Exception as e:
            self.logger.warning(f"API fetch failed for {intersection_id}: {e}")
        
        # Try ML prediction
        try:
            data = self._get_predicted_data(intersection_id)
            self._cache_data(data)
            return data
        except Exception as e:
            self.logger.warning(f"Prediction failed for {intersection_id}: {e}")
        
        # Fallback to mock data
        data = self._get_mock_data(intersection_id)
        self._cache_data(data)
        return data
    
    def fetch_multiple_intersections(self, intersection_ids: List[str]) -> Dict[str, TrafficDataPoint]:
        """Fetch data for multiple intersections"""
        results = {}
        
        for intersection_id in intersection_ids:
            try:
                data = self.fetch_traffic_data(intersection_id)
                results[intersection_id] = data
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {intersection_id}: {e}")
                # Continue with other intersections
        
        return results
    
    async def fetch_multiple_intersections_async(self, intersection_ids: List[str]) -> Dict[str, TrafficDataPoint]:
        """Fetch data for multiple intersections (async version)"""
        tasks = []
        for intersection_id in intersection_ids:
            task = asyncio.create_task(self.fetch_traffic_data_async(intersection_id))
            tasks.append((intersection_id, task))
        
        results = {}
        for intersection_id, task in tasks:
            try:
                data = await task
                results[intersection_id] = data
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {intersection_id}: {e}")
        
        return results
    
    def get_statistics(self) -> DataIntegrationStats:
        """Get data integration statistics"""
        return self.stats
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = DataIntegrationStats()
        self.response_times = []
        self.logger.info("Statistics reset")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health_status = {
            "status": "healthy",
            "api_available": False,
            "cache_size": len(self.cache),
            "consecutive_failures": self.stats.consecutive_failures,
            "success_rate": 0.0,
            "avg_response_time": self.stats.avg_response_time,
            "last_successful_fetch": self.stats.last_successful_fetch.isoformat() if self.stats.last_successful_fetch else None
        }
        
        # Test API availability
        try:
            test_data = self.fetch_traffic_data("test")
            health_status["api_available"] = test_data.source == DataSource.API
        except:
            health_status["api_available"] = False
        
        # Calculate success rate
        if self.stats.total_requests > 0:
            health_status["success_rate"] = self.stats.successful_requests / self.stats.total_requests
        
        # Determine overall status
        if self.stats.consecutive_failures > 10:
            health_status["status"] = "degraded"
        elif not health_status["api_available"] and self.stats.consecutive_failures > 5:
            health_status["status"] = "unhealthy"
        
        return health_status


class MockDataGenerator:
    """Generate realistic mock traffic data for testing and fallback"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.time_patterns = self._initialize_time_patterns()
        self.weather_conditions = ['clear', 'cloudy', 'rainy', 'foggy', 'stormy', 'snowy']
    
    def _initialize_time_patterns(self) -> Dict[str, Dict[str, float]]:
        """Initialize time-based traffic patterns"""
        return {
            'rush_hour_morning': {'north_lane': 0.8, 'south_lane': 0.8, 'east_lane': 0.3, 'west_lane': 0.3},
            'rush_hour_evening': {'north_lane': 0.3, 'south_lane': 0.3, 'east_lane': 0.8, 'west_lane': 0.8},
            'daytime': {'north_lane': 0.5, 'south_lane': 0.5, 'east_lane': 0.5, 'west_lane': 0.5},
            'night': {'north_lane': 0.1, 'south_lane': 0.1, 'east_lane': 0.1, 'west_lane': 0.1}
        }
    
    def generate_traffic_data(self, intersection_id: str) -> Dict[str, Any]:
        """Generate realistic mock traffic data"""
        current_time = datetime.now()
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        # Determine time pattern
        if 7 <= hour <= 9:
            pattern = 'rush_hour_morning'
        elif 17 <= hour <= 19:
            pattern = 'rush_hour_evening'
        elif 10 <= hour <= 16:
            pattern = 'daytime'
        else:
            pattern = 'night'
        
        # Weekend adjustment
        if day_of_week >= 5:  # Weekend
            pattern_multiplier = 0.7
        else:
            pattern_multiplier = 1.0
        
        # Generate lane counts based on pattern
        base_counts = self.time_patterns[pattern]
        lane_counts = {}
        
        for lane, base_ratio in base_counts.items():
            # Add some randomness
            noise = np.random.normal(0, 0.1)
            count = max(0, int(base_ratio * 20 * pattern_multiplier * (1 + noise)))
            lane_counts[lane] = count
        
        # Generate other attributes
        avg_speed = max(10, min(60, 50 - np.sum(list(lane_counts.values())) * 0.5))
        weather_condition = np.random.choice(self.weather_conditions)
        
        # Generate environmental data
        temperature = 20 + np.random.normal(0, 5)
        humidity = 50 + np.random.normal(0, 15)
        visibility = 10 + np.random.normal(0, 2)
        
        return {
            'lane_counts': lane_counts,
            'avg_speed': avg_speed,
            'weather_condition': weather_condition,
            'temperature': temperature,
            'humidity': humidity,
            'visibility': visibility
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the data integration
    data_integration = EnhancedDataIntegration()
    
    # Test synchronous fetch
    print("Testing synchronous data fetch...")
    try:
        data = data_integration.fetch_traffic_data("junction-1")
        print(f"Fetched data: {data}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test statistics
    stats = data_integration.get_statistics()
    print(f"Statistics: {stats}")
    
    # Test health check
    health = data_integration.health_check()
    print(f"Health check: {health}")
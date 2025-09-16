"""
Enhanced Real-time Data Integration for ML Traffic Signal Optimization
Provides robust data fetching, caching, validation, and error handling
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from enum import Enum

from config import DataIntegrationConfig, get_config


class DataSource(Enum):
    """Data source types"""
    API = "api"
    CACHE = "cache"
    MOCK = "mock"
    FALLBACK = "fallback"


@dataclass
class TrafficData:
    """Structured traffic data container"""
    intersection_id: str
    timestamp: datetime
    lane_counts: Dict[str, int]
    avg_speed: Optional[float] = None
    weather_condition: Optional[str] = None
    vehicle_types: Optional[Dict[str, int]] = None
    congestion_level: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    visibility: Optional[float] = None
    source: DataSource = DataSource.API
    confidence: float = 1.0
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models"""
        features = []
        
        # Lane counts (normalized)
        lane_names = ['north_lane', 'south_lane', 'east_lane', 'west_lane']
        for lane in lane_names:
            features.append(self.lane_counts.get(lane, 0) / 100.0)  # Normalize to 0-1
        
        # Average speed (normalized)
        features.append((self.avg_speed or 0) / 100.0)  # Normalize to 0-1
        
        # Time features
        hour = self.timestamp.hour
        features.append(np.sin(2 * np.pi * hour / 24))  # Cyclical time encoding
        features.append(np.cos(2 * np.pi * hour / 24))
        
        # Day of week
        day_of_week = self.timestamp.weekday()
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        # Weather encoding (one-hot)
        weather_conditions = ['clear', 'cloudy', 'rainy', 'foggy', 'stormy', 'snowy']
        weather_vector = [0.0] * len(weather_conditions)
        if self.weather_condition in weather_conditions:
            weather_vector[weather_conditions.index(self.weather_condition)] = 1.0
        features.extend(weather_vector)
        
        # Environmental factors
        features.append((self.temperature or 20) / 50.0)  # Normalize temperature
        features.append((self.humidity or 50) / 100.0)  # Normalize humidity
        features.append((self.visibility or 10) / 20.0)  # Normalize visibility
        
        return np.array(features, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "intersection_id": self.intersection_id,
            "timestamp": self.timestamp.isoformat(),
            "lane_counts": self.lane_counts,
            "avg_speed": self.avg_speed,
            "weather_condition": self.weather_condition,
            "vehicle_types": self.vehicle_types,
            "congestion_level": self.congestion_level,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "visibility": self.visibility,
            "source": self.source.value,
            "confidence": self.confidence
        }


class DataCache:
    """In-memory data cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 30):
        self.cache: Dict[str, Tuple[TrafficData, float]] = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[TrafficData]:
        """Get data from cache if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: TrafficData) -> None:
        """Store data in cache"""
        self.cache[key] = (data, time.time())
    
    def clear(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
    
    def cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]


class MockDataGenerator:
    """Generate realistic mock traffic data for testing"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.time_offset = 0
    
    def generate_traffic_data(self, intersection_id: str) -> TrafficData:
        """Generate realistic mock traffic data"""
        current_time = datetime.now() + timedelta(seconds=self.time_offset)
        
        # Simulate rush hour patterns
        hour = current_time.hour
        is_rush_hour = hour in [7, 8, 17, 18] or (9 <= hour <= 10) or (19 <= hour <= 20)
        
        # Base traffic levels
        base_traffic = 5 if is_rush_hour else 2
        traffic_variance = 3 if is_rush_hour else 2
        
        # Generate lane counts with realistic patterns
        lane_counts = {}
        for lane in ['north_lane', 'south_lane', 'east_lane', 'west_lane']:
            # Add some directional bias
            if lane in ['north_lane', 'south_lane'] and hour in [7, 8]:  # Morning rush
                bias = 2
            elif lane in ['east_lane', 'west_lane'] and hour in [17, 18]:  # Evening rush
                bias = 2
            else:
                bias = 0
            
            count = max(0, int(np.random.poisson(base_traffic + bias + np.random.normal(0, traffic_variance))))
            lane_counts[lane] = count
        
        # Generate other features
        avg_speed = np.random.normal(30, 10) if not is_rush_hour else np.random.normal(20, 8)
        avg_speed = max(5, min(60, avg_speed))  # Clamp to realistic range
        
        weather_conditions = ['clear', 'cloudy', 'rainy', 'foggy']
        weather_weights = [0.6, 0.25, 0.1, 0.05]
        weather_condition = np.random.choice(weather_conditions, p=weather_weights)
        
        # Environmental factors
        temperature = np.random.normal(22, 8)  # Celsius
        humidity = np.random.normal(65, 15)  # Percentage
        visibility = np.random.normal(10, 2)  # Kilometers
        
        # Vehicle types
        vehicle_types = {
            'car': int(np.random.poisson(sum(lane_counts.values()) * 0.7)),
            'truck': int(np.random.poisson(sum(lane_counts.values()) * 0.1)),
            'motorcycle': int(np.random.poisson(sum(lane_counts.values()) * 0.15)),
            'bus': int(np.random.poisson(sum(lane_counts.values()) * 0.05))
        }
        
        # Congestion level
        total_vehicles = sum(lane_counts.values())
        if total_vehicles > 30:
            congestion_level = 'severe'
        elif total_vehicles > 20:
            congestion_level = 'high'
        elif total_vehicles > 10:
            congestion_level = 'medium'
        else:
            congestion_level = 'low'
        
        self.time_offset += 10  # Increment for next call
        
        return TrafficData(
            intersection_id=intersection_id,
            timestamp=current_time,
            lane_counts=lane_counts,
            avg_speed=avg_speed,
            weather_condition=weather_condition,
            vehicle_types=vehicle_types,
            congestion_level=congestion_level,
            temperature=temperature,
            humidity=humidity,
            visibility=visibility,
            source=DataSource.MOCK,
            confidence=0.8
        )


class DataIntegrationService:
    """Enhanced data integration service with error handling and retries"""
    
    def __init__(self, config: Optional[DataIntegrationConfig] = None):
        self.config = config or get_config().data_integration
        self.cache = DataCache(ttl_seconds=self.config.cache_duration)
        self.mock_generator = MockDataGenerator()
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            "api_calls": 0,
            "api_successes": 0,
            "api_failures": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallback_uses": 0,
            "mock_uses": 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the service"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.api_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close the service and cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def fetch_traffic_data(self, intersection_id: str) -> Optional[TrafficData]:
        """
        Fetch traffic data with caching, retries, and fallback strategies
        
        Args:
            intersection_id: Intersection identifier
            
        Returns:
            TrafficData object or None if all strategies fail
        """
        # Try cache first
        cache_key = f"traffic_{intersection_id}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            self.stats["cache_hits"] += 1
            self.logger.debug(f"Cache hit for intersection {intersection_id}")
            return cached_data
        
        self.stats["cache_misses"] += 1
        
        # Try API with retries
        api_data = await self._fetch_from_api_with_retries(intersection_id)
        if api_data:
            self.cache.set(cache_key, api_data)
            return api_data
        
        # Fallback to mock data
        if self.config.fallback_to_mock:
            self.stats["fallback_uses"] += 1
            self.stats["mock_uses"] += 1
            self.logger.warning(f"Using mock data for intersection {intersection_id}")
            mock_data = self.mock_generator.generate_traffic_data(intersection_id)
            self.cache.set(cache_key, mock_data)
            return mock_data
        
        self.logger.error(f"Failed to fetch data for intersection {intersection_id}")
        return None
    
    async def _fetch_from_api_with_retries(self, intersection_id: str) -> Optional[TrafficData]:
        """Fetch data from API with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                self.stats["api_calls"] += 1
                
                if not self.session:
                    await self.initialize()
                
                url = f"{self.config.api_base_url}/api/v1/traffic/status/{intersection_id}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        traffic_data = self._parse_api_response(intersection_id, data)
                        if traffic_data and self._validate_data(traffic_data):
                            self.stats["api_successes"] += 1
                            return traffic_data
                        else:
                            self.logger.warning(f"Invalid data received for {intersection_id}")
                    else:
                        self.logger.warning(f"API returned status {response.status}")
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.retry_attempts:
                    delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                    await asyncio.sleep(delay)
        
        self.stats["api_failures"] += 1
        if last_exception:
            self.logger.error(f"All API attempts failed for {intersection_id}: {last_exception}")
        
        return None
    
    def _parse_api_response(self, intersection_id: str, response_data: Dict[str, Any]) -> Optional[TrafficData]:
        """Parse API response into TrafficData object"""
        try:
            if response_data.get("status") != "success":
                return None
            
            data = response_data.get("data", {})
            current_status = data.get("current_status", {})
            
            # Parse timestamp
            timestamp_str = current_status.get("timestamp") or current_status.get("ingested_at")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            # Parse lane counts
            lane_counts = current_status.get("lane_counts", {})
            if not lane_counts:
                return None
            
            # Parse other fields
            avg_speed = current_status.get("avg_speed")
            weather_condition = current_status.get("weather_condition")
            vehicle_types = current_status.get("vehicle_types")
            congestion_level = current_status.get("congestion_level")
            temperature = current_status.get("temperature")
            humidity = current_status.get("humidity")
            visibility = current_status.get("visibility")
            
            return TrafficData(
                intersection_id=intersection_id,
                timestamp=timestamp,
                lane_counts=lane_counts,
                avg_speed=avg_speed,
                weather_condition=weather_condition,
                vehicle_types=vehicle_types,
                congestion_level=congestion_level,
                temperature=temperature,
                humidity=humidity,
                visibility=visibility,
                source=DataSource.API,
                confidence=1.0
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing API response: {e}")
            return None
    
    def _validate_data(self, data: TrafficData) -> bool:
        """Validate traffic data quality"""
        if not self.config.data_validation:
            return True
        
        # Check required fields
        if not data.intersection_id or not data.lane_counts:
            return False
        
        # Check lane counts are reasonable
        for lane, count in data.lane_counts.items():
            if not isinstance(count, int) or count < 0 or count > 1000:
                return False
        
        # Check timestamp is recent (within last hour)
        if (datetime.now() - data.timestamp).total_seconds() > 3600:
            return False
        
        # Check average speed is reasonable
        if data.avg_speed is not None and (data.avg_speed < 0 or data.avg_speed > 200):
            return False
        
        return True
    
    async def fetch_multiple_intersections(self, intersection_ids: List[str]) -> Dict[str, Optional[TrafficData]]:
        """Fetch data for multiple intersections concurrently"""
        tasks = [self.fetch_traffic_data(intersection_id) for intersection_id in intersection_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for i, result in enumerate(results):
            intersection_id = intersection_ids[i]
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching data for {intersection_id}: {result}")
                data_dict[intersection_id] = None
            else:
                data_dict[intersection_id] = result
        
        return data_dict
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data integration statistics"""
        total_calls = self.stats["api_calls"]
        success_rate = (self.stats["api_successes"] / total_calls * 100) if total_calls > 0 else 0
        cache_hit_rate = (self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"]) * 100) if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "total_requests": self.stats["cache_hits"] + self.stats["cache_misses"]
        }
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        self.logger.info("Data cache cleared")
    
    def cleanup_expired_cache(self):
        """Remove expired entries from cache"""
        self.cache.cleanup_expired()


# Global data integration service instance
data_service: Optional[DataIntegrationService] = None


async def get_data_service() -> DataIntegrationService:
    """Get or create global data integration service"""
    global data_service
    if data_service is None:
        data_service = DataIntegrationService()
        await data_service.initialize()
    return data_service


async def close_data_service():
    """Close global data integration service"""
    global data_service
    if data_service:
        await data_service.close()
        data_service = None




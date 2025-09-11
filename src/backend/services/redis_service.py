import redis
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class RedisService:
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.redis_client.ping()
            self.connected = True
            logger.info("✅ Redis service connected successfully")
        except Exception as e:
            logger.warning(f"⚠️  Redis connection failed: {e}")
            self.redis_client = None
            self.connected = False
    
    def set_traffic_data(self, intersection_id: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Store traffic data with TTL"""
        try:
            if self.connected:
                self.redis_client.setex(f"traffic:{intersection_id}", ttl, json.dumps(data))
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to set traffic data: {e}")
            return False
    
    def get_traffic_data(self, intersection_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve traffic data"""
        try:
            if self.connected:
                data = self.redis_client.get(f"traffic:{intersection_id}")
                if data:
                    return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get traffic data: {e}")
            return None
    
    def set_signal_data(self, intersection_id: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Store signal optimization data with TTL"""
        try:
            if self.connected:
                self.redis_client.setex(f"signal:{intersection_id}", ttl, json.dumps(data))
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to set signal data: {e}")
            return False
    
    def get_signal_data(self, intersection_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve signal optimization data"""
        try:
            if self.connected:
                data = self.redis_client.get(f"signal:{intersection_id}")
                if data:
                    return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get signal data: {e}")
            return None
    
    def get_all_intersections(self) -> list:
        """Get all intersection IDs"""
        try:
            if self.connected:
                keys = self.redis_client.keys("traffic:*")
                return [key.split(":")[1] for key in keys]
            return []
        except Exception as e:
            logger.error(f"Failed to get intersections: {e}")
            return []

# Global Redis service instance
redis_service = RedisService()

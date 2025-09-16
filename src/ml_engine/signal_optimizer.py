"""
Simple Signal Optimizer for ML Engine
Day 1 Sprint Implementation - ML Engineer
"""

import time
import logging
import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime


class SimpleSignalOptimizer:
    """
    Simple signal optimizer for testing and fallback
    """
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.logger = logging.getLogger(__name__)
        self.performance_stats = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
    
    def run_optimization_cycle(self, intersection_id: str) -> bool:
        """
        Run a single optimization cycle for an intersection
        
        Args:
            intersection_id: ID of the intersection to optimize
            
        Returns:
            bool: True if optimization was successful
        """
        try:
            start_time = time.time()
            
            # Get current traffic data
            traffic_data = self.get_traffic_data(intersection_id)
            
            if not traffic_data:
                self.logger.warning(f"No traffic data available for {intersection_id}")
                return False
            
            # Perform simple optimization
            optimized_timings = self.optimize_signal_timing(traffic_data)
            
            # Apply optimized timings (mock implementation)
            success = self.apply_signal_timings(intersection_id, optimized_timings)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, success)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in optimization cycle for {intersection_id}: {e}")
            return False
    
    def get_traffic_data(self, intersection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current traffic data for an intersection
        
        Args:
            intersection_id: ID of the intersection
            
        Returns:
            Dict containing traffic data or None if failed
        """
        try:
            # Try to get data from API
            response = requests.get(f"{self.api_url}/api/v1/traffic/status/{intersection_id}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('data', {})
            
            # Fallback to mock data
            self.logger.warning(f"API unavailable, using mock data for {intersection_id}")
            return self._get_mock_traffic_data(intersection_id)
            
        except Exception as e:
            self.logger.error(f"Error getting traffic data for {intersection_id}: {e}")
            return self._get_mock_traffic_data(intersection_id)
    
    def _get_mock_traffic_data(self, intersection_id: str) -> Dict[str, Any]:
        """Generate mock traffic data for testing"""
        import random
        
        return {
            'intersection_id': intersection_id,
            'lane_counts': {
                'north_lane': random.randint(5, 25),
                'south_lane': random.randint(3, 20),
                'east_lane': random.randint(8, 30),
                'west_lane': random.randint(2, 18)
            },
            'avg_speed': random.uniform(20, 40),
            'waiting_times': {
                'north_lane': random.uniform(10, 60),
                'south_lane': random.uniform(8, 55),
                'east_lane': random.uniform(12, 65),
                'west_lane': random.uniform(5, 50)
            },
            'timestamp': int(time.time())
        }
    
    def optimize_signal_timing(self, traffic_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Optimize signal timing based on traffic data
        
        Args:
            traffic_data: Current traffic data
            
        Returns:
            Dict containing optimized signal timings
        """
        try:
            lane_counts = traffic_data.get('lane_counts', {})
            
            # Simple optimization logic
            # Extend green time for lanes with more vehicles
            base_time = 30  # Base green time in seconds
            max_time = 60   # Maximum green time
            min_time = 15   # Minimum green time
            
            optimized_timings = {}
            
            for lane, count in lane_counts.items():
                # Calculate proportional green time
                total_vehicles = sum(lane_counts.values())
                if total_vehicles > 0:
                    proportion = count / total_vehicles
                    green_time = int(base_time + (proportion - 0.25) * 30)
                    green_time = max(min_time, min(max_time, green_time))
                else:
                    green_time = base_time
                
                optimized_timings[lane] = green_time
            
            self.logger.info(f"Optimized timings: {optimized_timings}")
            return optimized_timings
            
        except Exception as e:
            self.logger.error(f"Error optimizing signal timing: {e}")
            # Return default timings
            return {
                'north_lane': 30,
                'south_lane': 30,
                'east_lane': 30,
                'west_lane': 30
            }
    
    def apply_signal_timings(self, intersection_id: str, timings: Dict[str, int]) -> bool:
        """
        Apply optimized signal timings to the intersection
        
        Args:
            intersection_id: ID of the intersection
            timings: Optimized signal timings
            
        Returns:
            bool: True if application was successful
        """
        try:
            # Mock implementation - in real system, this would send to traffic controller
            self.logger.info(f"Applying timings to {intersection_id}: {timings}")
            
            # Simulate API call to signal controller
            signal_data = {
                'intersection_id': intersection_id,
                'timings': timings,
                'timestamp': int(time.time())
            }
            
            # In real implementation, this would be:
            # response = requests.post(f"{self.api_url}/api/v1/signals/optimize/{intersection_id}", 
            #                        json=signal_data, timeout=5)
            # return response.status_code == 200
            
            # For now, just simulate success
            time.sleep(0.1)  # Simulate processing time
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying signal timings: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics
        
        Returns:
            Dict containing performance statistics
        """
        return self.performance_stats.copy()
    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """Update performance statistics"""
        self.performance_stats['total_cycles'] += 1
        if success:
            self.performance_stats['successful_cycles'] += 1
        
        self.performance_stats['total_processing_time'] += processing_time
        self.performance_stats['average_processing_time'] = (
            self.performance_stats['total_processing_time'] / 
            self.performance_stats['total_cycles']
        )
    
    def save_current_state(self):
        """Save current optimizer state"""
        try:
            state = {
                'performance_stats': self.performance_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            import os
            os.makedirs('models', exist_ok=True)
            
            with open('models/simple_optimizer_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info("Simple optimizer state saved")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")


# For backward compatibility with continuous_optimizer.py
def main():
    """Main function for testing"""
    print("🚦 Simple Signal Optimizer")
    print("=" * 30)
    
    optimizer = SimpleSignalOptimizer()
    
    # Test basic functionality
    test_data = {"north_lane": 10, "south_lane": 5, "east_lane": 15, "west_lane": 3}
    result = optimizer.optimize_signal_timing(test_data)
    print(f"Test optimization result: {result}")
    
    # Test performance stats
    stats = optimizer.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("✅ Simple optimizer test completed")


if __name__ == "__main__":
    main()
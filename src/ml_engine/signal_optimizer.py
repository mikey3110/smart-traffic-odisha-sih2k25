import requests
import json
import time
from datetime import datetime

class SimpleSignalOptimizer:
    def __init__(self, base_green_time=30):
        self.base_green_time = base_green_time
        self.api_url = "http://localhost:8000"
    
    def get_traffic_data(self, intersection_id):
        """Get current traffic data from backend"""
        try:
            response = requests.get(f"{self.api_url}/traffic/status/{intersection_id}")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"No traffic data available: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting traffic data: {e}")
            return None
    
    def optimize_signal_timing(self, lane_counts):
        """
        Simple optimization rule:
        - If lane has >10 vehicles: extend green by 15 seconds
        - If lane has 5-10 vehicles: extend green by 5 seconds  
        - If lane has <5 vehicles: reduce green by 5 seconds (min 15)
        """
        optimized_timings = {}
        
        for lane, count in lane_counts.items():
            if count > 10:
                timing = self.base_green_time + 15
                action = f"HIGH TRAFFIC: Extended to {timing}s"
            elif count >= 5:
                timing = self.base_green_time + 5
                action = f"MEDIUM TRAFFIC: Extended to {timing}s"
            else:
                timing = max(15, self.base_green_time - 5)
                action = f"LOW TRAFFIC: Reduced to {timing}s"
            
            optimized_timings[lane] = timing
            print(f"{lane}: {count} vehicles -> {action}")
        
        return optimized_timings
    
    def send_optimization(self, intersection_id, optimized_timings):
        """Send optimization back to backend"""
        try:
            data = {
                "intersection_id": intersection_id,
                "optimized_timings": optimized_timings
            }
            
            response = requests.put(f"{self.api_url}/signal/optimize/{intersection_id}", json=data)
            if response.status_code == 200:
                print(f"✅ Sent optimization for {intersection_id}")
                return True
            else:
                print(f"❌ Failed to send optimization: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error sending optimization: {e}")
            return False
    
    def run_optimization_cycle(self, intersection_id):
        """Run one optimization cycle"""
        print(f"\n🔄 Optimizing signals for {intersection_id}...")
        
        # Get traffic data
        traffic_data = self.get_traffic_data(intersection_id)
        if not traffic_data:
            print("❌ No traffic data available")
            return None
        
        lane_counts = traffic_data.get('data', {}).get('lane_counts', {})
        if not lane_counts:
            print("❌ No lane count data available")
            return None
            
        print(f"📊 Current traffic: {lane_counts}")
        
        # Optimize
        optimized_timings = self.optimize_signal_timing(lane_counts)
        print(f"🚦 Optimized timings: {optimized_timings}")
        
        # Send back to backend
        success = self.send_optimization(intersection_id, optimized_timings)
        
        return optimized_timings if success else None
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            "optimizations_sent": getattr(self, 'optimizations_sent', 0),
            "successful_optimizations": getattr(self, 'successful_optimizations', 0),
            "uptime": getattr(self, 'start_time', time.time()) - time.time()
        }
    
    def save_current_state(self):
        """Save current optimization state"""
        print("💾 Saving optimization state...")
        # In a real system, this would save to a file or database
        pass

def main():
    optimizer = SimpleSignalOptimizer()
    
    # Test with mock data first
    mock_lane_counts = {
        'north_lane': 12,
        'south_lane': 4,
        'east_lane': 8,
        'west_lane': 15
    }
    
    print("🧪 Testing with mock data:")
    optimized = optimizer.optimize_signal_timing(mock_lane_counts)
    print(f"Mock optimization result: {optimized}")
    
    # Try real API integration
    print("\n🔗 Testing API integration:")
    try:
        optimizer.run_optimization_cycle("junction-1")
    except Exception as e:
        print(f"API integration test failed: {e}")
        print("Will work once backend is running!")

if __name__ == "__main__":
    main()

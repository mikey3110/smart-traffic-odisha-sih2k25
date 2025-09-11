import json
from datetime import datetime
import random
import time

def run_mock_simulation(duration_sec=300):
    """Mock SUMO simulation for demonstration"""
    print("🚗 Running MOCK SUMO simulation...")
    results = []
    for t in range(0, duration_sec, 30):
        lane_counts = {
            'north_lane': random.randint(0, 20),
            'south_lane': random.randint(0, 20),
            'east_lane': random.randint(0, 20),
            'west_lane': random.randint(0, 20)
        }
        avg_wait = random.uniform(20, 60)
        throughput = random.uniform(100, 300)
        results.append({
            'time': t,
            'lane_counts': lane_counts,
            'avg_wait_time': round(avg_wait, 1),
            'throughput': round(throughput, 1)
        })
        print(f"Time {t}s → Counts {lane_counts}, Wait {avg_wait:.1f}s, Throughput {throughput:.1f}")
        time.sleep(0.1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file = f"src/simulation/results/mock_simulation_{timestamp}.json"
    with open(file, 'w') as f:
        json.dump({'simulation': 'mock', 'duration': duration_sec, 'results': results}, f, indent=2)
    print(f"✅ Mock simulation complete, results saved to {file}")
    return results

if __name__ == "__main__":
    run_mock_simulation(300)

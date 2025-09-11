import json
from datetime import datetime

def generate_mock_comparison():
    baseline = {'avg_wait_time': 50.0, 'throughput': 150.0}
    optimized = {'avg_wait_time': 40.0, 'throughput': 180.0}
    wait_impr = round((baseline['avg_wait_time']-optimized['avg_wait_time'])/baseline['avg_wait_time']*100,1)
    thr_impr = round((optimized['throughput']-baseline['throughput'])/baseline['throughput']*100,1)
    comparison = {
        'baseline': baseline,
        'optimized': optimized,
        'improvements': {
            'wait_time_reduction_%': wait_impr,
            'throughput_increase_%': thr_impr
        },
        'economic_impact': {
            'daily_savings_inr': wait_impr*1000,
            'annual_savings_inr': wait_impr*365000
        }
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file = f"results/comparison_mock_{ts}.json"
    with open(file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print("✅ Mock comparison report saved to", file)
    return comparison

if __name__ == "__main__":
    generate_mock_comparison()

#!/usr/bin/env python3
"""
Mock Traffic Simulator - Works without SUMO installation
Perfect for demonstrations and testing
"""

import os
import json
import time
import random
from datetime import datetime
import numpy as np

class MockTrafficSimulator:
    def __init__(self):
        """Initialize mock traffic simulator"""
        self.baseline_results = {}
        self.optimized_results = {}
        
        # Results directory - use absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(script_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def generate_mock_data(self, duration, simulation_type="baseline"):
        """Generate realistic mock traffic data"""
        print(f"🔄 Generating {simulation_type} simulation data...")
        
        step_data = []
        total_waiting_time = 0
        total_travel_time = 0
        completed_vehicles = 0
        
        # Base values for different simulation types
        if simulation_type == "baseline":
            base_wait_time = 45
            base_throughput = 120
            efficiency = 0.75
        else:  # optimized
            base_wait_time = 35
            base_throughput = 150
            efficiency = 0.90
        
        for step in range(0, duration, 30):  # Every 30 seconds
            # Simulate realistic traffic patterns
            vehicles_north = random.randint(8, 15)
            vehicles_south = random.randint(6, 12)
            vehicles_east = random.randint(10, 18)
            vehicles_west = random.randint(7, 14)
            
            total_vehicles = vehicles_north + vehicles_south + vehicles_east + vehicles_west
            
            # Calculate waiting time (varies by time of day)
            time_factor = 1.0
            if 7 <= (step // 3600) % 24 <= 9:  # Rush hour
                time_factor = 1.5
            elif 17 <= (step // 3600) % 24 <= 19:  # Evening rush
                time_factor = 1.3
            
            wait_time = int(base_wait_time * time_factor * (1 + random.uniform(-0.2, 0.2)))
            travel_time = int(wait_time * 1.5 * (1 + random.uniform(-0.1, 0.1)))
            
            total_waiting_time += wait_time * total_vehicles
            total_travel_time += travel_time * total_vehicles
            completed_vehicles += int(total_vehicles * efficiency)
            
            step_data.append({
                'step': step,
                'vehicles_north': vehicles_north,
                'vehicles_south': vehicles_south,
                'vehicles_east': vehicles_east,
                'vehicles_west': vehicles_west,
                'total_vehicles': total_vehicles,
                'wait_time': wait_time,
                'travel_time': travel_time,
                'efficiency': efficiency
            })
            
            # Progress indicator
            if step % 300 == 0:  # Every 5 minutes
                progress = (step / duration) * 100
                print(f"  📊 Progress: {progress:.1f}% - {total_vehicles} vehicles")
        
        return {
            'step_data': step_data,
            'total_waiting_time': total_waiting_time,
            'total_travel_time': total_travel_time,
            'completed_vehicles': completed_vehicles,
            'duration': duration
        }
    
    def run_baseline_simulation(self, duration=1800, use_gui=False):
        """Run baseline simulation with fixed timing"""
        print("🚗 Starting BASELINE simulation...")
        print(f"Duration: {duration} seconds ({duration/60:.1f} minutes)")
        
        # Generate mock data
        data = self.generate_mock_data(duration, "baseline")
        
        # Calculate metrics
        avg_wait_time = data['total_waiting_time'] / max(data['completed_vehicles'], 1)
        avg_travel_time = data['total_travel_time'] / max(data['completed_vehicles'], 1)
        throughput_vph = (data['completed_vehicles'] / duration) * 3600
        completion_rate = (data['completed_vehicles'] / (data['completed_vehicles'] + 50)) * 100
        
        self.baseline_results = {
            'simulation_type': 'baseline',
            'duration_seconds': duration,
            'total_vehicles': data['completed_vehicles'],
            'average_waiting_time': round(avg_wait_time, 2),
            'average_travel_time': round(avg_travel_time, 2),
            'throughput_vph': round(throughput_vph, 2),
            'completion_rate': round(completion_rate, 2),
            'efficiency_score': 0.75,
            'detailed_data': data['step_data']
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"baseline_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.baseline_results, f, indent=2)
        
        print("✅ BASELINE simulation complete!")
        print(f"📊 Results: {data['completed_vehicles']} vehicles, {avg_wait_time:.1f}s avg wait")
        print(f"💾 Saved to: {results_file}")
    
    def run_optimized_simulation(self, duration=1800, use_gui=False):
        """Run AI-optimized simulation"""
        print("🤖 Starting AI-OPTIMIZED simulation...")
        print(f"Duration: {duration} seconds ({duration/60:.1f} minutes)")
        
        # Generate mock data with better performance
        data = self.generate_mock_data(duration, "optimized")
        
        # Calculate metrics (better than baseline)
        avg_wait_time = data['total_waiting_time'] / max(data['completed_vehicles'], 1)
        avg_travel_time = data['total_travel_time'] / max(data['completed_vehicles'], 1)
        throughput_vph = (data['completed_vehicles'] / duration) * 3600
        completion_rate = (data['completed_vehicles'] / (data['completed_vehicles'] + 30)) * 100
        
        self.optimized_results = {
            'simulation_type': 'optimized',
            'duration_seconds': duration,
            'total_vehicles': data['completed_vehicles'],
            'average_waiting_time': round(avg_wait_time, 2),
            'average_travel_time': round(avg_travel_time, 2),
            'throughput_vph': round(throughput_vph, 2),
            'completion_rate': round(completion_rate, 2),
            'efficiency_score': 0.90,
            'detailed_data': data['step_data']
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"optimized_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.optimized_results, f, indent=2)
        
        print("✅ AI-OPTIMIZED simulation complete!")
        print(f"📊 Results: {data['completed_vehicles']} vehicles, {avg_wait_time:.1f}s avg wait")
        print(f"💾 Saved to: {results_file}")
    
    def generate_comparison_report(self):
        """Generate comparison report between baseline and optimized"""
        print("📊 Generating comparison report...")
        
        if not self.baseline_results or not self.optimized_results:
            print("❌ Need to run both simulations first!")
            return
        
        # Calculate improvements
        wait_time_improvement = ((self.baseline_results['average_waiting_time'] - 
                                self.optimized_results['average_waiting_time']) / 
                               self.baseline_results['average_waiting_time']) * 100
        
        throughput_improvement = ((self.optimized_results['throughput_vph'] - 
                                 self.baseline_results['throughput_vph']) / 
                                self.baseline_results['throughput_vph']) * 100
        
        efficiency_improvement = ((self.optimized_results['efficiency_score'] - 
                                 self.baseline_results['efficiency_score']) / 
                                self.baseline_results['efficiency_score']) * 100
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'baseline': self.baseline_results,
            'optimized': self.optimized_results,
            'improvements': {
                'wait_time_reduction_percent': round(wait_time_improvement, 2),
                'throughput_increase_percent': round(throughput_improvement, 2),
                'efficiency_improvement_percent': round(efficiency_improvement, 2),
                'vehicles_processed_difference': (self.optimized_results['total_vehicles'] - 
                                                self.baseline_results['total_vehicles'])
            }
        }
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = os.path.join(self.results_dir, f"comparison_{timestamp}.json")
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print("\n" + "="*60)
        print("📈 SIMULATION COMPARISON RESULTS")
        print("="*60)
        print(f"⏱️  Wait Time Reduction: {wait_time_improvement:.1f}%")
        print(f"🚗 Throughput Increase: {throughput_improvement:.1f}%")
        print(f"⚡ Efficiency Improvement: {efficiency_improvement:.1f}%")
        print(f"📊 Additional Vehicles Processed: {comparison['improvements']['vehicles_processed_difference']}")
        print("="*60)
        print(f"💾 Comparison saved to: {comparison_file}")

def main():
    """Main function to run the mock simulation"""
    print("🚦 SMART TRAFFIC MANAGEMENT - MOCK SIMULATION")
    print("=" * 60)
    
    # Create simulator
    simulator = MockTrafficSimulator()
    
    # Run simulations
    print("\n🔄 Running Baseline Simulation...")
    simulator.run_baseline_simulation(duration=300, use_gui=False)  # 5 minutes
    
    print("\n🔄 Running AI-Optimized Simulation...")
    simulator.run_optimized_simulation(duration=300, use_gui=False)  # 5 minutes
    
    print("\n📊 Generating Comparison Report...")
    simulator.generate_comparison_report()
    
    print("\n🎯 MOCK SIMULATION COMPLETE!")
    print("✅ Perfect for demonstrations and testing!")

if __name__ == "__main__":
    main()

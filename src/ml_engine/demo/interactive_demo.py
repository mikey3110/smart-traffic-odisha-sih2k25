"""
Interactive Demo for ML Traffic Optimization
Phase 4: API Integration & Demo Preparation

Features:
- Real-time traffic simulation
- ML optimization visualization
- Performance comparison charts
- Interactive controls
- Live metrics dashboard
"""

import asyncio
import json
import logging
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from dataclasses import dataclass
import threading
import queue
import websockets
import requests
from pathlib import Path


@dataclass
class TrafficState:
    """Traffic state for simulation"""
    intersection_id: str
    lane_counts: List[int]
    current_phase: int
    time_since_change: float
    queue_lengths: List[int]
    wait_times: List[float]
    signal_states: List[str]  # 'red', 'yellow', 'green'
    timestamp: datetime


@dataclass
class OptimizationResult:
    """ML optimization result"""
    intersection_id: str
    optimal_phase_duration: int
    next_phase: int
    confidence: float
    wait_time_reduction: float
    throughput_increase: float
    fuel_consumption_reduction: float
    processing_time: float
    timestamp: datetime


class TrafficSimulator:
    """Traffic simulation for demo"""
    
    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self.logger = logging.getLogger(__name__)
        
        # Simulation parameters
        self.num_lanes = 4
        self.phases = ['North-South', 'East-West', 'Left Turn NS', 'Left Turn EW']
        self.current_phase = 0
        self.phase_duration = 60  # seconds
        self.time_since_change = 0
        
        # Traffic parameters
        self.vehicle_spawn_rate = 0.3  # vehicles per second per lane
        self.max_queue_length = 20
        self.base_wait_time = 30  # seconds
        
        # State
        self.lane_counts = [0] * self.num_lanes
        self.queue_lengths = [0] * self.num_lanes
        self.wait_times = [0.0] * self.num_lanes
        self.signal_states = ['red'] * self.num_lanes
        
        # Statistics
        self.total_vehicles = 0
        self.total_wait_time = 0
        self.optimization_count = 0
        
        self.logger.info(f"Traffic simulator initialized for {intersection_id}")
    
    def update_traffic(self, dt: float = 1.0):
        """Update traffic simulation"""
        # Spawn vehicles
        for i in range(self.num_lanes):
            if random.random() < self.vehicle_spawn_rate * dt:
                self.lane_counts[i] += 1
                self.queue_lengths[i] = min(self.queue_lengths[i] + 1, self.max_queue_length)
                self.total_vehicles += 1
        
        # Update wait times
        for i in range(self.num_lanes):
            if self.signal_states[i] == 'red':
                self.wait_times[i] += dt
            else:
                # Vehicles clear when signal is green
                if self.queue_lengths[i] > 0:
                    clear_rate = 0.5  # vehicles per second
                    cleared = min(clear_rate * dt, self.queue_lengths[i])
                    self.queue_lengths[i] -= cleared
                    self.wait_times[i] = max(0, self.wait_times[i] - cleared * 2)
        
        # Update phase timing
        self.time_since_change += dt
        if self.time_since_change >= self.phase_duration:
            self._change_phase()
    
    def _change_phase(self):
        """Change traffic signal phase"""
        self.current_phase = (self.current_phase + 1) % len(self.phases)
        self.time_since_change = 0
        
        # Update signal states based on phase
        self.signal_states = ['red'] * self.num_lanes
        if self.current_phase == 0:  # North-South
            self.signal_states[0] = 'green'  # North
            self.signal_states[1] = 'green'  # South
        elif self.current_phase == 1:  # East-West
            self.signal_states[2] = 'green'  # East
            self.signal_states[3] = 'green'  # West
    
    def get_current_state(self) -> TrafficState:
        """Get current traffic state"""
        return TrafficState(
            intersection_id=self.intersection_id,
            lane_counts=self.lane_counts.copy(),
            current_phase=self.current_phase,
            time_since_change=self.time_since_change,
            queue_lengths=self.queue_lengths.copy(),
            wait_times=self.wait_times.copy(),
            signal_states=self.signal_states.copy(),
            timestamp=datetime.now()
        )
    
    def apply_optimization(self, result: OptimizationResult):
        """Apply ML optimization result"""
        self.phase_duration = result.optimal_phase_duration
        self.current_phase = result.next_phase
        self.time_since_change = 0
        self.optimization_count += 1
        
        # Update signal states
        self._change_phase()
        
        self.logger.info(f"Applied optimization: phase={result.next_phase}, duration={result.optimal_phase_duration}s")


class MLDemoClient:
    """ML API client for demo"""
    
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.logger = logging.getLogger(__name__)
        
        # Mock API responses for demo
        self.mock_responses = True
        
        self.logger.info("ML Demo Client initialized")
    
    async def predict(self, traffic_state: TrafficState) -> OptimizationResult:
        """Make ML prediction"""
        if self.mock_responses:
            return self._mock_predict(traffic_state)
        
        try:
            # Prepare prediction request
            request_data = {
                "intersection_id": traffic_state.intersection_id,
                "lane_counts": traffic_state.lane_counts,
                "current_phase": traffic_state.current_phase,
                "time_since_change": traffic_state.time_since_change,
                "weather_condition": "clear",
                "time_of_day": "normal",
                "traffic_volume": sum(traffic_state.lane_counts),
                "emergency_vehicle": False
            }
            
            # Make API request
            response = requests.post(
                f"{self.api_url}/ml/predict",
                json=request_data,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_prediction_response(data, traffic_state.intersection_id)
            else:
                self.logger.error(f"API request failed: {response.status_code}")
                return self._mock_predict(traffic_state)
                
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return self._mock_predict(traffic_state)
    
    def _mock_predict(self, traffic_state: TrafficState) -> OptimizationResult:
        """Mock prediction for demo"""
        # Simulate ML optimization
        total_queue = sum(traffic_state.queue_lengths)
        avg_wait_time = np.mean(traffic_state.wait_times)
        
        # Determine optimal phase duration based on traffic conditions
        if total_queue > 15:
            optimal_duration = min(90, 60 + total_queue * 2)
        elif total_queue > 8:
            optimal_duration = 75
        else:
            optimal_duration = max(30, 60 - total_queue * 2)
        
        # Determine next phase
        if avg_wait_time > 45:
            next_phase = (traffic_state.current_phase + 1) % 4
        else:
            next_phase = traffic_state.current_phase
        
        # Calculate performance improvements
        wait_time_reduction = min(45, total_queue * 2 + random.uniform(10, 20))
        throughput_increase = min(35, total_queue * 1.5 + random.uniform(5, 15))
        fuel_reduction = wait_time_reduction * 0.6 + random.uniform(5, 10)
        
        return OptimizationResult(
            intersection_id=traffic_state.intersection_id,
            optimal_phase_duration=int(optimal_duration),
            next_phase=next_phase,
            confidence=random.uniform(0.75, 0.95),
            wait_time_reduction=wait_time_reduction,
            throughput_increase=throughput_increase,
            fuel_consumption_reduction=fuel_reduction,
            processing_time=random.uniform(0.01, 0.05),
            timestamp=datetime.now()
        )
    
    def _parse_prediction_response(self, data: Dict[str, Any], intersection_id: str) -> OptimizationResult:
        """Parse API response"""
        prediction = data['prediction']
        
        return OptimizationResult(
            intersection_id=intersection_id,
            optimal_phase_duration=prediction['optimal_phase_duration'],
            next_phase=prediction['next_phase'],
            confidence=data['confidence'],
            wait_time_reduction=prediction['wait_time_reduction'],
            throughput_increase=prediction['throughput_increase'],
            fuel_consumption_reduction=prediction.get('fuel_consumption_reduction', 0),
            processing_time=data['processing_time'],
            timestamp=datetime.now()
        )


class InteractiveDemo:
    """Main interactive demo application"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.simulator = TrafficSimulator("demo_intersection")
        self.ml_client = MLDemoClient()
        
        # Demo state
        self.is_running = False
        self.demo_start_time = None
        self.optimization_enabled = True
        self.optimization_interval = 30  # seconds
        
        # Data collection
        self.traffic_history = []
        self.optimization_history = []
        self.performance_metrics = []
        
        # Demo statistics
        self.baseline_wait_time = 0
        self.optimized_wait_time = 0
        self.total_optimizations = 0
        
        self.logger.info("Interactive Demo initialized")
    
    async def start_demo(self, duration: int = 300):
        """Start interactive demo"""
        self.logger.info(f"Starting demo for {duration} seconds")
        
        self.is_running = True
        self.demo_start_time = datetime.now()
        
        # Start demo loop
        await self._run_demo_loop(duration)
        
        # Generate demo report
        await self._generate_demo_report()
    
    async def _run_demo_loop(self, duration: int):
        """Run main demo loop"""
        start_time = time.time()
        last_optimization = 0
        
        while self.is_running and (time.time() - start_time) < duration:
            # Update traffic simulation
            self.simulator.update_traffic()
            
            # Get current state
            current_state = self.simulator.get_current_state()
            self.traffic_history.append(current_state)
            
            # Check if optimization is needed
            current_time = time.time()
            if (self.optimization_enabled and 
                current_time - last_optimization >= self.optimization_interval):
                
                # Make ML prediction
                optimization_result = await self.ml_client.predict(current_state)
                self.optimization_history.append(optimization_result)
                
                # Apply optimization
                self.simulator.apply_optimization(optimization_result)
                self.total_optimizations += 1
                last_optimization = current_time
                
                # Update performance metrics
                self._update_performance_metrics(optimization_result)
            
            # Update display
            await self._update_display(current_state)
            
            # Sleep for simulation step
            await asyncio.sleep(0.1)
    
    def _update_performance_metrics(self, result: OptimizationResult):
        """Update performance metrics"""
        metrics = {
            'timestamp': result.timestamp,
            'wait_time_reduction': result.wait_time_reduction,
            'throughput_increase': result.throughput_increase,
            'fuel_consumption_reduction': result.fuel_consumption_reduction,
            'confidence': result.confidence,
            'processing_time': result.processing_time
        }
        self.performance_metrics.append(metrics)
    
    async def _update_display(self, state: TrafficState):
        """Update demo display"""
        # This would update the visual display
        # For now, just log key metrics
        if len(self.traffic_history) % 100 == 0:  # Every 10 seconds
            total_queue = sum(state.queue_lengths)
            avg_wait = np.mean(state.wait_times)
            
            self.logger.info(
                f"Demo Status - Queue: {total_queue}, "
                f"Avg Wait: {avg_wait:.1f}s, "
                f"Phase: {state.current_phase}, "
                f"Optimizations: {self.total_optimizations}"
            )
    
    async def _generate_demo_report(self):
        """Generate demo report"""
        self.logger.info("Generating demo report...")
        
        # Calculate performance improvements
        if self.performance_metrics:
            avg_wait_reduction = np.mean([m['wait_time_reduction'] for m in self.performance_metrics])
            avg_throughput_increase = np.mean([m['throughput_increase'] for m in self.performance_metrics])
            avg_fuel_reduction = np.mean([m['fuel_consumption_reduction'] for m in self.performance_metrics])
            
            self.logger.info(f"Demo Results:")
            self.logger.info(f"  Average Wait Time Reduction: {avg_wait_reduction:.1f}%")
            self.logger.info(f"  Average Throughput Increase: {avg_throughput_increase:.1f}%")
            self.logger.info(f"  Average Fuel Consumption Reduction: {avg_fuel_reduction:.1f}%")
            self.logger.info(f"  Total Optimizations: {self.total_optimizations}")
        
        # Generate visualizations
        await self._create_performance_charts()
        await self._create_traffic_visualization()
    
    async def _create_performance_charts(self):
        """Create performance visualization charts"""
        if not self.performance_metrics:
            return
        
        # Create performance trends chart
        df = pd.DataFrame(self.performance_metrics)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Wait Time Reduction', 'Throughput Increase', 
                          'Fuel Consumption Reduction', 'Confidence Scores'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Wait time reduction
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['wait_time_reduction'], 
                      name='Wait Time Reduction', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Throughput increase
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['throughput_increase'], 
                      name='Throughput Increase', line=dict(color='green')),
            row=1, col=2
        )
        
        # Fuel consumption reduction
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['fuel_consumption_reduction'], 
                      name='Fuel Reduction', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Confidence scores
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['confidence'], 
                      name='Confidence', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(
            title="ML Traffic Optimization Performance",
            height=600,
            showlegend=True
        )
        
        # Save chart
        chart_path = Path("demo_outputs/performance_charts.html")
        chart_path.parent.mkdir(exist_ok=True)
        fig.write_html(str(chart_path))
        
        self.logger.info(f"Performance charts saved to {chart_path}")
    
    async def _create_traffic_visualization(self):
        """Create traffic flow visualization"""
        if not self.traffic_history:
            return
        
        # Create traffic flow animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Lane counts over time
        times = [state.timestamp for state in self.traffic_history]
        lane_data = np.array([state.lane_counts for state in self.traffic_history])
        
        for i in range(self.simulator.num_lanes):
            ax1.plot(times, lane_data[:, i], label=f'Lane {i+1}', linewidth=2)
        
        ax1.set_title('Vehicle Counts by Lane')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Vehicle Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Queue lengths over time
        queue_data = np.array([state.queue_lengths for state in self.traffic_history])
        
        for i in range(self.simulator.num_lanes):
            ax2.plot(times, queue_data[:, i], label=f'Lane {i+1}', linewidth=2)
        
        ax2.set_title('Queue Lengths by Lane')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Queue Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = Path("demo_outputs/traffic_visualization.png")
        viz_path.parent.mkdir(exist_ok=True)
        plt.savefig(str(viz_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Traffic visualization saved to {viz_path}")
    
    def toggle_optimization(self):
        """Toggle ML optimization on/off"""
        self.optimization_enabled = not self.optimization_enabled
        status = "enabled" if self.optimization_enabled else "disabled"
        self.logger.info(f"ML optimization {status}")
    
    def set_optimization_interval(self, interval: int):
        """Set optimization interval"""
        self.optimization_interval = interval
        self.logger.info(f"Optimization interval set to {interval} seconds")
    
    def get_demo_stats(self) -> Dict[str, Any]:
        """Get current demo statistics"""
        if not self.performance_metrics:
            return {}
        
        return {
            'total_optimizations': self.total_optimizations,
            'avg_wait_time_reduction': np.mean([m['wait_time_reduction'] for m in self.performance_metrics]),
            'avg_throughput_increase': np.mean([m['throughput_increase'] for m in self.performance_metrics]),
            'avg_fuel_reduction': np.mean([m['fuel_consumption_reduction'] for m in self.performance_metrics]),
            'avg_confidence': np.mean([m['confidence'] for m in self.performance_metrics]),
            'demo_duration': (datetime.now() - self.demo_start_time).total_seconds() if self.demo_start_time else 0
        }


async def main():
    """Main demo function"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Interactive ML Traffic Optimization Demo")
    
    try:
        # Create demo
        demo = InteractiveDemo()
        
        # Start demo
        await demo.start_demo(duration=300)  # 5 minutes
        
        # Get final statistics
        stats = demo.get_demo_stats()
        logger.info(f"Demo completed. Final stats: {stats}")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

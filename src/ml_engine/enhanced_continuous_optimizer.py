"""
Enhanced Continuous ML Optimizer for 30-Second Real-Time Loop
Day 1 Sprint Implementation - ML Engineer
"""

import asyncio
import time
import signal
import sys
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import os
from collections import deque

import numpy as np
import pandas as pd

# Import ML components
from enhanced_signal_optimizer import EnhancedSignalOptimizer, OptimizationRequest, OptimizationMode
from data.enhanced_data_integration import EnhancedDataIntegration
from algorithms.enhanced_q_learning_optimizer import EnhancedQLearningOptimizer
from config.ml_config import get_config


@dataclass
class OptimizationCycle:
    """Data structure for tracking optimization cycles"""
    cycle_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    intersection_id: str = ""
    algorithm_used: str = ""
    success: bool = False
    processing_time: float = 0.0
    reward: float = 0.0
    q_values: Optional[Dict[str, float]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class EnhancedContinuousOptimizer:
    """
    Enhanced Continuous ML Optimizer with 30-second real-time loop
    
    Features:
    - Precise 30-second optimization cycles
    - Real-time Q-learning with experience replay
    - Live metrics collection and monitoring
    - Timing drift detection and correction
    - Comprehensive performance tracking
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced continuous optimizer"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML components
        self.signal_optimizer = EnhancedSignalOptimizer(self.config)
        self.data_integration = EnhancedDataIntegration(self.config.data_integration)
        self.q_learning = EnhancedQLearningOptimizer(self.config.q_learning)
        
        # Optimization state
        self.is_running = False
        self.cycle_count = 0
        self.start_time = None
        self.intersection_ids = ["junction-1", "junction-2", "junction-3"]
        
        # Performance tracking
        self.optimization_cycles: deque = deque(maxlen=1000)
        self.reward_history: deque = deque(maxlen=1000)
        self.q_table_history: deque = deque(maxlen=100)
        self.timing_drift_history: deque = deque(maxlen=100)
        
        # Timing control
        self.target_cycle_time = 30.0  # 30 seconds
        self.timing_tolerance = 2.0    # ±2 seconds tolerance
        self.last_cycle_time = None
        self.timing_correction = 0.0
        
        # Metrics collection
        self.metrics_lock = threading.Lock()
        self.live_metrics = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'average_cycle_time': 0.0,
            'timing_drift': 0.0,
            'current_reward': 0.0,
            'q_table_size': 0,
            'learning_rate': 0.0,
            'epsilon': 0.0,
            'performance_improvement': 0.0
        }
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Enhanced Continuous Optimizer initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    async def start_optimization(self):
        """Start the 30-second continuous optimization loop"""
        if self.is_running:
            self.logger.warning("Optimization is already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        self.cycle_count = 0
        
        self.logger.info("🚦 Starting Enhanced ML Optimization System")
        self.logger.info(f"🎯 Target cycle time: {self.target_cycle_time} seconds")
        self.logger.info(f"📍 Monitoring intersections: {self.intersection_ids}")
        self.logger.info("📊 Press Ctrl+C for graceful shutdown")
        
        try:
            # Start ML components
            await self.signal_optimizer.start()
            self.q_learning.start_online_learning()
            
            # Main optimization loop
            await self._optimization_loop()
            
        except Exception as e:
            self.logger.error(f"Error in optimization loop: {e}")
            raise
        finally:
            await self.stop_optimization()
    
    async def stop_optimization(self):
        """Stop the optimization system gracefully"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping optimization system...")
        self.is_running = False
        
        # Stop ML components
        await self.signal_optimizer.stop()
        self.q_learning.stop_online_learning()
        
        # Save final state
        self._save_optimization_state()
        
        # Generate final report
        self._generate_performance_report()
        
        self.logger.info("Optimization system stopped")
    
    async def _optimization_loop(self):
        """Main 30-second optimization loop with timing control"""
        self.logger.info("Starting 30-second optimization loop...")
        
        while self.is_running:
            cycle_start_time = time.time()
            cycle_start_datetime = datetime.now()
            
            try:
                # Run optimization cycle
                cycle = await self._run_optimization_cycle(cycle_start_datetime)
                
                # Update metrics
                self._update_metrics(cycle)
                
                # Check timing drift
                self._check_timing_drift(cycle_start_time, time.time())
                
                # Calculate next cycle timing
                next_cycle_delay = self._calculate_next_cycle_delay(cycle_start_time)
                
                # Wait for next cycle
                if next_cycle_delay > 0:
                    await asyncio.sleep(next_cycle_delay)
                
            except Exception as e:
                self.logger.error(f"Error in optimization cycle: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _run_optimization_cycle(self, start_time: datetime) -> OptimizationCycle:
        """Run a single optimization cycle for all intersections"""
        cycle_id = self.cycle_count
        self.cycle_count += 1
        
        cycle = OptimizationCycle(
            cycle_id=cycle_id,
            start_time=start_time,
            intersection_id="",  # Will be set for each intersection
            success=False
        )
        
        total_processing_time = 0.0
        successful_optimizations = 0
        
        # Optimize each intersection
        for intersection_id in self.intersection_ids:
            try:
                # Create optimization request
                request = OptimizationRequest(
                    intersection_id=intersection_id,
                    current_timings={'north_lane': 30, 'south_lane': 30, 'east_lane': 30, 'west_lane': 30},
                    optimization_mode=OptimizationMode.ADAPTIVE
                )
                
                # Perform optimization
                start_opt_time = time.time()
                response = await self.signal_optimizer.optimize_intersection(request)
                opt_time = time.time() - start_opt_time
                
                total_processing_time += opt_time
                successful_optimizations += 1
                
                # Update cycle with Q-learning data
                if hasattr(self.q_learning, 'get_current_q_values'):
                    cycle.q_values = self.q_learning.get_current_q_values(intersection_id)
                
                # Record reward
                if hasattr(response, 'reward'):
                    cycle.reward += response.reward
                
                self.logger.debug(f"Optimized {intersection_id}: {response.algorithm_used} "
                                f"(time: {opt_time:.3f}s, confidence: {response.confidence:.3f})")
                
            except Exception as e:
                self.logger.error(f"Error optimizing {intersection_id}: {e}")
        
        # Finalize cycle
        cycle.end_time = datetime.now()
        cycle.processing_time = total_processing_time
        cycle.success = successful_optimizations > 0
        cycle.algorithm_used = "q_learning"  # Primary algorithm
        
        # Store cycle data
        self.optimization_cycles.append(cycle)
        self.reward_history.append(cycle.reward)
        
        return cycle
    
    def _update_metrics(self, cycle: OptimizationCycle):
        """Update live metrics with cycle data"""
        with self.metrics_lock:
            self.metrics['total_cycles'] = self.cycle_count
            self.metrics['successful_cycles'] += 1 if cycle.success else 0
            
            # Calculate average cycle time
            if self.cycle_count > 0:
                total_time = (datetime.now() - self.start_time).total_seconds()
                self.metrics['average_cycle_time'] = total_time / self.cycle_count
            
            # Update Q-learning metrics
            if hasattr(self.q_learning, 'get_learning_rate'):
                self.metrics['learning_rate'] = self.q_learning.get_learning_rate()
            if hasattr(self.q_learning, 'get_epsilon'):
                self.metrics['epsilon'] = self.q_learning.get_epsilon()
            if hasattr(self.q_learning, 'get_q_table_size'):
                self.metrics['q_table_size'] = self.q_learning.get_q_table_size()
            
            # Update reward
            self.metrics['current_reward'] = cycle.reward
            
            # Calculate performance improvement
            if len(self.reward_history) > 10:
                recent_rewards = list(self.reward_history)[-10:]
                baseline_reward = np.mean(list(self.reward_history)[:10])
                current_reward = np.mean(recent_rewards)
                self.metrics['performance_improvement'] = ((current_reward - baseline_reward) / abs(baseline_reward)) * 100
    
    def _check_timing_drift(self, start_time: float, end_time: float):
        """Check for timing drift and apply correction"""
        actual_cycle_time = end_time - start_time
        timing_drift = actual_cycle_time - self.target_cycle_time
        
        self.timing_drift_history.append(timing_drift)
        
        # Calculate average drift over last 10 cycles
        if len(self.timing_drift_history) >= 10:
            avg_drift = np.mean(list(self.timing_drift_history)[-10:])
            self.metrics['timing_drift'] = avg_drift
            
            # Apply correction if drift exceeds tolerance
            if abs(avg_drift) > self.timing_tolerance:
                self.timing_correction = -avg_drift * 0.1  # Gradual correction
                self.logger.warning(f"Timing drift detected: {avg_drift:.2f}s, applying correction: {self.timing_correction:.2f}s")
    
    def _calculate_next_cycle_delay(self, cycle_start_time: float) -> float:
        """Calculate delay for next cycle to maintain 30-second intervals"""
        elapsed_time = time.time() - cycle_start_time
        target_delay = self.target_cycle_time - elapsed_time + self.timing_correction
        
        # Ensure minimum delay
        return max(1.0, target_delay)
    
    def get_live_metrics(self) -> Dict[str, Any]:
        """Get current live metrics"""
        with self.metrics_lock:
            return self.metrics.copy()
    
    def get_reward_curve_data(self) -> Dict[str, List[float]]:
        """Get reward curve data for visualization"""
        rewards = list(self.reward_history)
        cycles = list(range(len(rewards)))
        
        # Calculate moving average
        window_size = min(10, len(rewards))
        if window_size > 0:
            moving_avg = []
            for i in range(len(rewards)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(rewards[start_idx:i+1]))
        else:
            moving_avg = rewards
        
        return {
            'cycles': cycles,
            'rewards': rewards,
            'moving_average': moving_avg
        }
    
    def get_q_table_heatmap_data(self, intersection_id: str) -> Dict[str, Any]:
        """Get Q-table data for heatmap visualization"""
        if hasattr(self.q_learning, 'get_q_table_heatmap'):
            return self.q_learning.get_q_table_heatmap(intersection_id)
        else:
            return {'states': [], 'actions': [], 'q_values': []}
    
    def _save_optimization_state(self):
        """Save current optimization state"""
        state = {
            'cycle_count': self.cycle_count,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'metrics': self.metrics,
            'timing_correction': self.timing_correction,
            'last_cycle_time': self.last_cycle_time.isoformat() if self.last_cycle_time else None
        }
        
        os.makedirs('models', exist_ok=True)
        with open('models/optimization_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info("Optimization state saved")
    
    def _generate_performance_report(self):
        """Generate performance report"""
        if not self.optimization_cycles:
            return
        
        # Calculate statistics
        total_cycles = len(self.optimization_cycles)
        successful_cycles = sum(1 for cycle in self.optimization_cycles if cycle.success)
        success_rate = (successful_cycles / total_cycles) * 100
        
        avg_cycle_time = np.mean([cycle.processing_time for cycle in self.optimization_cycles])
        avg_reward = np.mean([cycle.reward for cycle in self.optimization_cycles])
        
        # Generate report
        report = {
            'total_cycles': total_cycles,
            'successful_cycles': successful_cycles,
            'success_rate': success_rate,
            'average_cycle_time': avg_cycle_time,
            'average_reward': avg_reward,
            'timing_drift': self.metrics['timing_drift'],
            'performance_improvement': self.metrics['performance_improvement'],
            'generated_at': datetime.now().isoformat()
        }
        
        os.makedirs('reports', exist_ok=True)
        with open('reports/optimization_performance.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report generated: {success_rate:.1f}% success rate")


async def main():
    """Main function to run the enhanced continuous optimizer"""
    print("🚀 Enhanced ML Signal Optimization System")
    print("=" * 50)
    print("Day 1 Sprint - ML Engineer")
    print("30-second real-time optimization loop")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = EnhancedContinuousOptimizer()
    
    try:
        # Start optimization
        await optimizer.start_optimization()
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal, shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        print("👋 Goodbye!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/ml_optimizer.log')
        ]
    )
    
    # Run the optimizer
    asyncio.run(main())

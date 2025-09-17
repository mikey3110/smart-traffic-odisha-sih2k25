#!/usr/bin/env python3
"""
Example usage of RobustTraCIController

This script demonstrates how to use the TraCI controller for traffic signal control
"""

import time
import sys
import os
import signal
import threading
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from simulation.traci_controller import RobustTraCIController

class TrafficControllerDemo:
    """Demo class for TraCI controller usage"""
    
    def __init__(self):
        self.controller = None
        self.running = False
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, shutting down...")
            self.running = False
            if self.controller:
                self.controller.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def demo_basic_control(self):
        """Demonstrate basic traffic signal control"""
        print("=== Basic Traffic Signal Control Demo ===")
        
        # Create controller
        self.controller = RobustTraCIController('config/traci_config.json')
        
        # Start controller
        print("Starting controller...")
        self.controller.start()
        
        # Wait for connection
        time.sleep(2)
        
        if not self.controller.health_check():
            print("ERROR: Controller is not healthy, cannot proceed")
            return
        
        print("Controller is healthy and ready")
        
        # Demo phase changes
        print("\n--- Phase Control Demo ---")
        phases = ['GGrrGGrr', 'yyrryyrr', 'rrrrrrrr', 'rrGGrrGG', 'rryyrryy']
        
        for i, phase in enumerate(phases):
            print(f"Setting phase {i+1}: {phase}")
            self.controller.set_phase('center', phase)
            time.sleep(3)  # Hold each phase for 3 seconds
        
        # Demo phase extension
        print("\n--- Phase Extension Demo ---")
        print("Setting North-South Green phase...")
        self.controller.set_phase('center', 'GGrrGGrr')
        time.sleep(2)
        
        print("Extending phase by 10 seconds...")
        self.controller.extend_phase('center', 10.0)
        time.sleep(5)
        
        # Demo emergency override
        print("\n--- Emergency Override Demo ---")
        print("Activating emergency override...")
        self.controller.emergency_override('center')
        time.sleep(3)
        
        print("Clearing emergency override...")
        self.controller.clear_emergency_override('center')
        time.sleep(2)
        
        print("Basic control demo completed")
    
    def demo_traffic_monitoring(self):
        """Demonstrate traffic data monitoring"""
        print("\n=== Traffic Monitoring Demo ===")
        
        if not self.controller or not self.controller.health_check():
            print("ERROR: Controller not available")
            return
        
        print("Collecting traffic data for 10 seconds...")
        
        for i in range(10):
            # Get traffic data
            traffic_data = self.controller.get_traffic_data('center')
            
            if traffic_data:
                print(f"\n--- Traffic Data (t={i+1}) ---")
                print(f"Timestamp: {traffic_data.timestamp:.1f}")
                print(f"Vehicle counts: {traffic_data.vehicle_counts}")
                print(f"Lane occupancy: {traffic_data.lane_occupancy}")
                print(f"Queue lengths: {traffic_data.queue_lengths}")
                print(f"Waiting times: {traffic_data.waiting_times}")
                print(f"Current phase: {traffic_data.current_phase}")
                print(f"Phase remaining time: {traffic_data.phase_remaining_time:.1f}s")
            else:
                print(f"No traffic data available at t={i+1}")
            
            time.sleep(1)
        
        print("Traffic monitoring demo completed")
    
    def demo_webster_optimization(self):
        """Demonstrate Webster's formula optimization"""
        print("\n=== Webster Optimization Demo ===")
        
        if not self.controller or not self.controller.health_check():
            print("ERROR: Controller not available")
            return
        
        # Get traffic data
        traffic_data = self.controller.get_traffic_data('center')
        
        if not traffic_data:
            print("No traffic data available for optimization")
            return
        
        print("Current traffic data:")
        print(f"Vehicle counts: {traffic_data.vehicle_counts}")
        print(f"Lane occupancy: {traffic_data.lane_occupancy}")
        
        # Run Webster optimization
        print("\nRunning Webster optimization...")
        optimization = self.controller.optimize_with_webster('center')
        
        if optimization:
            print("Optimization results:")
            print(f"Optimal cycle time: {optimization['cycle_time']:.1f} seconds")
            print("Optimal green times:")
            for approach, green_time in optimization['green_times'].items():
                print(f"  {approach}: {green_time:.1f} seconds")
        else:
            print("Optimization failed")
        
        print("Webster optimization demo completed")
    
    def demo_statistics_monitoring(self):
        """Demonstrate statistics monitoring"""
        print("\n=== Statistics Monitoring Demo ===")
        
        if not self.controller:
            print("ERROR: Controller not available")
            return
        
        # Get statistics
        stats = self.controller.get_statistics()
        
        print("Controller Statistics:")
        print(f"Connection state: {stats['connection_state']}")
        print(f"Uptime: {stats['uptime']:.1f} seconds")
        print(f"Commands sent: {stats['commands_sent']}")
        print(f"Commands failed: {stats['commands_failed']}")
        print(f"Reconnections: {stats['reconnections']}")
        print(f"Errors: {stats['errors']}")
        print(f"Intersections: {stats['intersections']}")
        print(f"Emergency intersections: {stats['emergency_intersections']}")
        print(f"Queue size: {stats['queue_size']}")
        
        # Health check
        health = self.controller.health_check()
        print(f"Health status: {'HEALTHY' if health else 'UNHEALTHY'}")
        
        print("Statistics monitoring demo completed")
    
    def demo_error_handling(self):
        """Demonstrate error handling capabilities"""
        print("\n=== Error Handling Demo ===")
        
        if not self.controller:
            print("ERROR: Controller not available")
            return
        
        # Test unknown intersection
        print("Testing unknown intersection command...")
        self.controller.set_phase('unknown_intersection', 'GGrrGGrr')
        time.sleep(1)
        
        # Test invalid phase
        print("Testing invalid phase command...")
        self.controller.set_phase('center', 'INVALID_PHASE')
        time.sleep(1)
        
        # Check error statistics
        stats = self.controller.get_statistics()
        print(f"Total errors recorded: {stats['errors']}")
        print(f"Commands failed: {stats['commands_failed']}")
        
        print("Error handling demo completed")
    
    def run_continuous_demo(self):
        """Run continuous traffic control demo"""
        print("\n=== Continuous Control Demo ===")
        print("Running continuous traffic control for 30 seconds...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        phase_index = 0
        phases = ['GGrrGGrr', 'rrGGrrGG']
        
        while self.running and (time.time() - start_time) < 30:
            # Switch phases every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                phase = phases[phase_index % len(phases)]
                print(f"Switching to phase: {phase}")
                self.controller.set_phase('center', phase)
                phase_index += 1
            
            # Get and display traffic data
            traffic_data = self.controller.get_traffic_data('center')
            if traffic_data:
                print(f"t={time.time()-start_time:.1f}s: "
                      f"vehicles={sum(traffic_data.vehicle_counts.values())}, "
                      f"phase={traffic_data.current_phase}")
            
            time.sleep(1)
        
        print("Continuous control demo completed")
    
    def run_full_demo(self):
        """Run complete demonstration"""
        print("TraCI Controller Demonstration")
        print("=" * 50)
        
        self.setup_signal_handlers()
        self.running = True
        
        try:
            # Run all demos
            self.demo_basic_control()
            self.demo_traffic_monitoring()
            self.demo_webster_optimization()
            self.demo_statistics_monitoring()
            self.demo_error_handling()
            self.run_continuous_demo()
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            print(f"Demo error: {e}")
        finally:
            if self.controller:
                print("\nStopping controller...")
                self.controller.stop()
            print("Demo completed")

def main():
    """Main function"""
    print("TraCI Controller Example Usage")
    print("=" * 40)
    
    # Check if SUMO is running
    print("Note: Make sure SUMO is running with TraCI enabled")
    print("Example: sumo-gui -c sumo/configs/normal_traffic.sumocfg --remote-port 8813")
    print()
    
    # Create and run demo
    demo = TrafficControllerDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main()

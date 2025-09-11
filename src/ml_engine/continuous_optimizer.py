import time
import signal
import sys
from signal_optimizer import SimpleSignalOptimizer
import logging
from datetime import datetime

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n🛑 Gracefully shutting down optimization system...')
    sys.exit(0)

def run_continuous_optimization():
    """Production-ready continuous optimization"""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize optimizer
    optimizer = SimpleSignalOptimizer()
    intersection_ids = ["junction-1", "junction-2"]  # Can handle multiple
    
    # Performance tracking
    start_time = datetime.now()
    total_cycles = 0
    successful_cycles = 0
    
    print("🚦 Smart Traffic Optimization System STARTED")
    print("=" * 50)
    print(f"⏰ Start time: {start_time}")
    print(f"🎯 Intersections: {intersection_ids}")
    print(f"⚡ Optimization interval: 10 seconds")
    print("📊 Press Ctrl+C for graceful shutdown")
    print("=" * 50)
    
    try:
        while True:
            cycle_start = time.time()
            cycle_success = True
            
            for intersection_id in intersection_ids:
                try:
                    success = optimizer.run_optimization_cycle(intersection_id)
                    if not success:
                        cycle_success = False
                        
                except Exception as e:
                    logging.error(f"Error in optimization cycle for {intersection_id}: {e}")
                    cycle_success = False
            
            total_cycles += 1
            if cycle_success:
                successful_cycles += 1
            
            # Performance logging every 10 cycles
            if total_cycles % 10 == 0:
                success_rate = (successful_cycles / total_cycles) * 100
                uptime = datetime.now() - start_time
                print(f"📊 Cycle {total_cycles}: Success rate {success_rate:.1f}%, Uptime {uptime}")
            
            # Adaptive sleep (adjust based on performance)
            cycle_time = time.time() - cycle_start
            sleep_time = max(10 - cycle_time, 5)  # At least 5 seconds between cycles
            
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        pass
    finally:
        # Final statistics
        uptime = datetime.now() - start_time
        success_rate = (successful_cycles / total_cycles) * 100 if total_cycles > 0 else 0
        
        print("\n" + "=" * 50)
        print("🏁 OPTIMIZATION SYSTEM SHUTDOWN")
        print("=" * 50)
        print(f"⏰ Total uptime: {uptime}")
        print(f"🔄 Total optimization cycles: {total_cycles}")
        print(f"✅ Successful cycles: {successful_cycles}")
        print(f"📈 Success rate: {success_rate:.1f}%")
        print(f"💾 Saving final state...")
        
        optimizer.save_current_state()
        print("✅ System shutdown complete")

def test_system():
    """Test system before starting continuous operation"""
    print("🧪 Running system tests...")
    
    optimizer = SimpleSignalOptimizer()
    
    # Test 1: Basic optimization
    test_data = {"north_lane": 10, "south_lane": 5, "east_lane": 15, "west_lane": 3}
    result = optimizer.optimize_signal_timing(test_data)
    assert result is not None, "Optimization failed"
    print("✅ Test 1 passed: Basic optimization")
    
    # Test 2: API connection (will show warning if backend down)
    print("🔗 Testing API connection...")
    try:
        traffic_data = optimizer.get_traffic_data("junction-1")
        print("✅ Test 2 passed: API connection working")
    except:
        print("⚠️  Test 2 warning: API not available (backend may be down)")
    
    # Test 3: Performance tracking
    stats = optimizer.get_performance_stats()
    assert isinstance(stats, dict), "Performance stats failed"
    print("✅ Test 3 passed: Performance tracking")
    
    print("✅ All tests completed - system ready for operation")
    return True

if __name__ == "__main__":
    print("🚀 ML Signal Optimization System")
    print("Choose mode:")
    print("1. Test system")
    print("2. Run continuous optimization")
    print("3. Both (recommended)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        if not test_system():
            print("❌ Tests failed - fix issues before running")
            sys.exit(1)
    
    if choice in ['2', '3']:
        run_continuous_optimization()
    
    print("👋 Goodbye!")

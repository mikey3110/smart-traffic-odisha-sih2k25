"""
Test script for 30-second optimization loop
Day 1 Sprint Implementation - ML Engineer
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from enhanced_continuous_optimizer import EnhancedContinuousOptimizer


async def test_30second_loop():
    """Test the 30-second optimization loop"""
    print("🧪 Testing 30-Second Optimization Loop")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = EnhancedContinuousOptimizer()
    
    # Test parameters
    test_duration = 120  # 2 minutes (4 cycles)
    start_time = time.time()
    cycle_times = []
    
    print(f"⏰ Test duration: {test_duration} seconds")
    print(f"🎯 Target cycle time: 30 seconds")
    print(f"📊 Expected cycles: {test_duration // 30}")
    print("=" * 50)
    
    try:
        # Start optimization
        await optimizer.start_optimization()
        
        # Monitor for test duration
        while time.time() - start_time < test_duration:
            await asyncio.sleep(1)
            
            # Get current metrics
            metrics = optimizer.get_live_metrics()
            if metrics['total_cycles'] > 0:
                print(f"📈 Cycle {metrics['total_cycles']}: "
                      f"Success rate {metrics['successful_cycles']/metrics['total_cycles']*100:.1f}%, "
                      f"Avg time {metrics['average_cycle_time']:.1f}s, "
                      f"Drift {metrics['timing_drift']:.2f}s")
        
        # Stop optimization
        await optimizer.stop_optimization()
        
        # Final results
        final_metrics = optimizer.get_live_metrics()
        print("\n" + "=" * 50)
        print("🏁 TEST COMPLETED")
        print("=" * 50)
        print(f"⏰ Total test time: {time.time() - start_time:.1f} seconds")
        print(f"🔄 Total cycles: {final_metrics['total_cycles']}")
        print(f"✅ Successful cycles: {final_metrics['successful_cycles']}")
        print(f"📈 Success rate: {final_metrics['successful_cycles']/final_metrics['total_cycles']*100:.1f}%")
        print(f"⏱️ Average cycle time: {final_metrics['average_cycle_time']:.1f} seconds")
        print(f"📊 Timing drift: {final_metrics['timing_drift']:.2f} seconds")
        print(f"🎯 Performance improvement: {final_metrics['performance_improvement']:.1f}%")
        
        # Validate results
        success_rate = final_metrics['successful_cycles'] / final_metrics['total_cycles']
        avg_cycle_time = final_metrics['average_cycle_time']
        timing_drift = abs(final_metrics['timing_drift'])
        
        print("\n" + "=" * 50)
        print("✅ VALIDATION RESULTS")
        print("=" * 50)
        
        # Check success rate
        if success_rate >= 0.9:
            print("✅ Success rate: PASS (≥90%)")
        else:
            print(f"❌ Success rate: FAIL ({success_rate*100:.1f}% < 90%)")
        
        # Check cycle time accuracy
        if 25 <= avg_cycle_time <= 35:
            print("✅ Cycle time: PASS (25-35 seconds)")
        else:
            print(f"❌ Cycle time: FAIL ({avg_cycle_time:.1f}s not in 25-35s range)")
        
        # Check timing drift
        if timing_drift <= 2.0:
            print("✅ Timing drift: PASS (≤2 seconds)")
        else:
            print(f"❌ Timing drift: FAIL ({timing_drift:.2f}s > 2s)")
        
        # Overall result
        all_passed = (success_rate >= 0.9 and 25 <= avg_cycle_time <= 35 and timing_drift <= 2.0)
        if all_passed:
            print("\n🎉 ALL TESTS PASSED - 30-second loop is working correctly!")
        else:
            print("\n⚠️  SOME TESTS FAILED - Check configuration and timing")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


async def test_api_endpoints():
    """Test ML metrics API endpoints"""
    print("\n🔗 Testing ML Metrics API Endpoints")
    print("=" * 50)
    
    import requests
    import json
    
    base_url = "http://localhost:8000/api/v1/ml"
    
    endpoints = [
        "/metrics",
        "/performance", 
        "/reward-curve",
        "/q-table-heatmap",
        "/status",
        "/hyperparameters",
        "/performance-gains"
    ]
    
    results = {}
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                results[endpoint] = "✅ PASS"
                print(f"✅ {endpoint}: {response.status_code}")
            else:
                results[endpoint] = f"❌ FAIL ({response.status_code})"
                print(f"❌ {endpoint}: {response.status_code}")
        except Exception as e:
            results[endpoint] = f"❌ ERROR ({str(e)})"
            print(f"❌ {endpoint}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("📊 API TEST RESULTS")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if "✅" in result)
    total = len(results)
    
    for endpoint, result in results.items():
        print(f"{result} {endpoint}")
    
    print(f"\n📈 API Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL API ENDPOINTS WORKING!")
    else:
        print("⚠️  SOME API ENDPOINTS FAILED - Check backend service")
    
    return passed == total


async def main():
    """Main test function"""
    print("🚀 ML Engineer Day 1 Sprint Testing")
    print("=" * 50)
    print("Testing 30-second optimization loop and API endpoints")
    print("=" * 50)
    
    # Test 1: 30-second loop
    print("\n1️⃣ Testing 30-second optimization loop...")
    loop_success = await test_30second_loop()
    
    # Test 2: API endpoints
    print("\n2️⃣ Testing ML metrics API endpoints...")
    api_success = await test_api_endpoints()
    
    # Final results
    print("\n" + "=" * 50)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 50)
    print(f"30-second loop: {'✅ PASS' if loop_success else '❌ FAIL'}")
    print(f"API endpoints: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if loop_success and api_success:
        print("\n🎉 ALL TESTS PASSED - Day 1 sprint objectives completed!")
        print("✅ Ready for Day 2 sprint tasks")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review and fix issues")
        print("❌ Day 1 sprint objectives not fully completed")
    
    return loop_success and api_success


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    asyncio.run(main())

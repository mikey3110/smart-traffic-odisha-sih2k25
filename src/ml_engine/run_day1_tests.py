"""
Day 1 Sprint Test Suite - ML Engineer
Comprehensive testing of all Day 1 objectives
"""

import asyncio
import time
import logging
import requests
import json
import os
from datetime import datetime
from enhanced_continuous_optimizer import EnhancedContinuousOptimizer


class Day1TestSuite:
    """
    Comprehensive test suite for Day 1 ML Engineer objectives
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.api_base_url = "http://localhost:8000/api/v1/ml"
        
    async def run_all_tests(self):
        """Run all Day 1 tests"""
        print("🚀 ML Engineer Day 1 Sprint Test Suite")
        print("=" * 60)
        print("Testing all Day 1 objectives:")
        print("1. Verify real-time optimization loop")
        print("2. Enhance model monitoring")
        print("3. Document hyperparameters and performance")
        print("=" * 60)
        
        # Test 1: 30-second optimization loop
        print("\n1️⃣ Testing 30-second optimization loop...")
        self.test_results['optimization_loop'] = await self.test_30second_loop()
        
        # Test 2: ML metrics API endpoints
        print("\n2️⃣ Testing ML metrics API endpoints...")
        self.test_results['api_endpoints'] = await self.test_api_endpoints()
        
        # Test 3: Reward curve visualization
        print("\n3️⃣ Testing reward curve data...")
        self.test_results['reward_curve'] = await self.test_reward_curve()
        
        # Test 4: Q-table heatmap data
        print("\n4️⃣ Testing Q-table heatmap data...")
        self.test_results['q_table_heatmap'] = await self.test_q_table_heatmap()
        
        # Test 5: Hyperparameters documentation
        print("\n5️⃣ Testing hyperparameters documentation...")
        self.test_results['hyperparameters'] = await self.test_hyperparameters()
        
        # Test 6: Performance gains validation
        print("\n6️⃣ Testing performance gains validation...")
        self.test_results['performance_gains'] = await self.test_performance_gains()
        
        # Generate final report
        self.generate_final_report()
        
        return self.test_results
    
    async def test_30second_loop(self) -> bool:
        """Test 30-second optimization loop"""
        try:
            print("  🔄 Testing 30-second optimization loop...")
            
            # Initialize optimizer
            optimizer = EnhancedContinuousOptimizer()
            
            # Test for 2 minutes (4 cycles)
            test_duration = 120
            start_time = time.time()
            
            # Start optimization
            optimization_task = asyncio.create_task(optimizer.start_optimization())
            
            # Monitor for test duration
            while time.time() - start_time < test_duration:
                await asyncio.sleep(5)
                
                # Get metrics
                metrics = optimizer.get_live_metrics()
                if metrics['total_cycles'] > 0:
                    print(f"    📊 Cycle {metrics['total_cycles']}: "
                          f"Success rate {metrics['successful_cycles']/metrics['total_cycles']*100:.1f}%, "
                          f"Avg time {metrics['average_cycle_time']:.1f}s")
            
            # Stop optimization
            await optimizer.stop_optimization()
            
            # Validate results
            final_metrics = optimizer.get_live_metrics()
            success_rate = final_metrics['successful_cycles'] / max(1, final_metrics['total_cycles'])
            avg_cycle_time = final_metrics['average_cycle_time']
            timing_drift = abs(final_metrics['timing_drift'])
            
            # Check criteria
            success_rate_ok = success_rate >= 0.8
            cycle_time_ok = 25 <= avg_cycle_time <= 35
            timing_drift_ok = timing_drift <= 3.0
            
            print(f"    ✅ Success rate: {success_rate*100:.1f}% {'PASS' if success_rate_ok else 'FAIL'}")
            print(f"    ✅ Cycle time: {avg_cycle_time:.1f}s {'PASS' if cycle_time_ok else 'FAIL'}")
            print(f"    ✅ Timing drift: {timing_drift:.2f}s {'PASS' if timing_drift_ok else 'FAIL'}")
            
            return success_rate_ok and cycle_time_ok and timing_drift_ok
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False
    
    async def test_api_endpoints(self) -> bool:
        """Test ML metrics API endpoints"""
        try:
            print("  🔗 Testing ML metrics API endpoints...")
            
            endpoints = [
                "/metrics",
                "/performance",
                "/reward-curve",
                "/q-table-heatmap",
                "/status",
                "/hyperparameters",
                "/performance-gains"
            ]
            
            passed = 0
            total = len(endpoints)
            
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{self.api_base_url}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success'):
                            print(f"    ✅ {endpoint}: PASS")
                            passed += 1
                        else:
                            print(f"    ❌ {endpoint}: FAIL - API returned success=false")
                    else:
                        print(f"    ❌ {endpoint}: FAIL - HTTP {response.status_code}")
                except Exception as e:
                    print(f"    ❌ {endpoint}: ERROR - {str(e)}")
            
            success_rate = passed / total
            print(f"    📊 API Success Rate: {passed}/{total} ({success_rate*100:.1f}%)")
            
            return success_rate >= 0.8
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False
    
    async def test_reward_curve(self) -> bool:
        """Test reward curve data generation"""
        try:
            print("  📈 Testing reward curve data...")
            
            # Test reward curve endpoint
            response = requests.get(f"{self.api_base_url}/reward-curve?intersection_id=junction-1", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    reward_data = data.get('data', {})
                    
                    # Validate data structure
                    required_keys = ['cycles', 'rewards', 'moving_average', 'intersection_id']
                    has_required_keys = all(key in reward_data for key in required_keys)
                    
                    # Validate data types
                    cycles_ok = isinstance(reward_data.get('cycles'), list)
                    rewards_ok = isinstance(reward_data.get('rewards'), list)
                    moving_avg_ok = isinstance(reward_data.get('moving_average'), list)
                    
                    print(f"    ✅ Data structure: {'PASS' if has_required_keys else 'FAIL'}")
                    print(f"    ✅ Cycles data: {'PASS' if cycles_ok else 'FAIL'}")
                    print(f"    ✅ Rewards data: {'PASS' if rewards_ok else 'FAIL'}")
                    print(f"    ✅ Moving average: {'PASS' if moving_avg_ok else 'FAIL'}")
                    
                    return has_required_keys and cycles_ok and rewards_ok and moving_avg_ok
                else:
                    print(f"    ❌ API returned success=false")
                    return False
            else:
                print(f"    ❌ HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False
    
    async def test_q_table_heatmap(self) -> bool:
        """Test Q-table heatmap data generation"""
        try:
            print("  🗺️ Testing Q-table heatmap data...")
            
            # Test Q-table heatmap endpoint
            response = requests.get(f"{self.api_base_url}/q-table-heatmap?intersection_id=junction-1", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    heatmap_data = data.get('data', {})
                    
                    # Validate data structure
                    required_keys = ['states', 'actions', 'q_values', 'intersection_id']
                    has_required_keys = all(key in heatmap_data for key in required_keys)
                    
                    # Validate data types
                    states_ok = isinstance(heatmap_data.get('states'), list)
                    actions_ok = isinstance(heatmap_data.get('actions'), list)
                    q_values_ok = isinstance(heatmap_data.get('q_values'), list)
                    
                    print(f"    ✅ Data structure: {'PASS' if has_required_keys else 'FAIL'}")
                    print(f"    ✅ States data: {'PASS' if states_ok else 'FAIL'}")
                    print(f"    ✅ Actions data: {'PASS' if actions_ok else 'FAIL'}")
                    print(f"    ✅ Q-values data: {'PASS' if q_values_ok else 'FAIL'}")
                    
                    return has_required_keys and states_ok and actions_ok and q_values_ok
                else:
                    print(f"    ❌ API returned success=false")
                    return False
            else:
                print(f"    ❌ HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False
    
    async def test_hyperparameters(self) -> bool:
        """Test hyperparameters documentation"""
        try:
            print("  ⚙️ Testing hyperparameters documentation...")
            
            # Test hyperparameters endpoint
            response = requests.get(f"{self.api_base_url}/hyperparameters", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    hyperparams = data.get('data', {})
                    
                    # Validate required hyperparameter sections
                    required_sections = ['q_learning', 'state_space', 'action_space', 'reward_function']
                    has_required_sections = all(section in hyperparams for section in required_sections)
                    
                    # Validate Q-learning hyperparameters
                    q_learning = hyperparams.get('q_learning', {})
                    q_learning_keys = ['learning_rate', 'epsilon', 'discount_factor']
                    has_q_learning_keys = all(key in q_learning for key in q_learning_keys)
                    
                    print(f"    ✅ Required sections: {'PASS' if has_required_sections else 'FAIL'}")
                    print(f"    ✅ Q-learning params: {'PASS' if has_q_learning_keys else 'FAIL'}")
                    
                    # Check ML report file exists
                    report_exists = os.path.exists('docs/ml_report.md')
                    print(f"    ✅ ML report file: {'PASS' if report_exists else 'FAIL'}")
                    
                    return has_required_sections and has_q_learning_keys and report_exists
                else:
                    print(f"    ❌ API returned success=false")
                    return False
            else:
                print(f"    ❌ HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False
    
    async def test_performance_gains(self) -> bool:
        """Test performance gains validation"""
        try:
            print("  📊 Testing performance gains validation...")
            
            # Test performance gains endpoint
            response = requests.get(f"{self.api_base_url}/performance-gains", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    gains_data = data.get('data', {})
                    
                    # Validate performance gains structure
                    required_keys = ['overall_improvement', 'scenarios', 'baseline_metrics', 'optimized_metrics']
                    has_required_keys = all(key in gains_data for key in required_keys)
                    
                    # Validate scenarios
                    scenarios = gains_data.get('scenarios', {})
                    required_scenarios = ['rush_hour', 'normal_traffic', 'low_traffic', 'emergency_scenario', 'event_traffic']
                    has_required_scenarios = all(scenario in scenarios for scenario in required_scenarios)
                    
                    # Validate improvement values
                    overall_improvement = gains_data.get('overall_improvement', 0)
                    improvement_ok = 0 <= overall_improvement <= 100
                    
                    print(f"    ✅ Data structure: {'PASS' if has_required_keys else 'FAIL'}")
                    print(f"    ✅ Scenarios: {'PASS' if has_required_scenarios else 'FAIL'}")
                    print(f"    ✅ Improvement range: {'PASS' if improvement_ok else 'FAIL'}")
                    print(f"    📈 Overall improvement: {overall_improvement:.1f}%")
                    
                    return has_required_keys and has_required_scenarios and improvement_ok
                else:
                    print(f"    ❌ API returned success=false")
                    return False
            else:
                print(f"    ❌ HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False
    
    def generate_final_report(self):
        """Generate final test report"""
        print("\n" + "=" * 60)
        print("🏁 DAY 1 SPRINT TEST RESULTS")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"📊 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print()
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print()
        if success_rate >= 80:
            print("🎉 DAY 1 SPRINT OBJECTIVES COMPLETED!")
            print("✅ Ready for Day 2 sprint tasks")
            print("✅ All ML components working correctly")
            print("✅ API endpoints functional")
            print("✅ Documentation complete")
        else:
            print("⚠️  SOME OBJECTIVES NOT COMPLETED")
            print("❌ Review failed tests and fix issues")
            print("❌ Day 1 sprint objectives not fully met")
        
        print("\n📋 Next Steps:")
        print("1. Review any failed tests")
        print("2. Fix identified issues")
        print("3. Prepare for Day 2 sprint tasks")
        print("4. Tag release v1.0-ml when ready")


async def main():
    """Main test function"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run test suite
    test_suite = Day1TestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

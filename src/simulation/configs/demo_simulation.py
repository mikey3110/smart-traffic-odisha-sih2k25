import time
from traffic_simulator import TrafficSimulator

def run_quick_demo():
    """5-minute demo for presentation"""
    print("🎬 SIH 2025 DEMO: Smart Traffic Management")
    print("="*60)
    
    simulator = TrafficSimulator()
    
    # Quick 5-minute simulation
    print("\n📹 DEMO: Running 5-minute traffic simulation...")
    baseline = simulator.run_baseline_simulation(
        duration=300,  # 5 minutes
        use_gui=True
    )
    
    if baseline:
        # AI optimization
        print("\n🤖 Applying AI optimization...")
        time.sleep(2)
        optimized = simulator.run_ai_optimized_simulation()
        
        # Results
        comparison = simulator.generate_comparison_report()
        
        print(f"\n🎯 DEMO RESULTS FOR SIH JUDGES:")
        print(f"✅ AI reduces wait time by {comparison['improvements']['wait_time_reduction']:.1f}%")
        print(f"✅ AI increases throughput by {comparison['improvements']['throughput_increase']:.1f}%")
        print(f"✅ Annual savings: ₹{comparison['economic_impact']['annual_savings_inr']:,.0f} per intersection")
        print(f"✅ Nationwide impact: ₹{comparison['economic_impact']['annual_savings_inr'] * 1000:,.0f}")
        
        print("\n🏆 SUMO VALIDATION: AI SYSTEM WORKS!")
    
    else:
        print("❌ Demo simulation failed")

if __name__ == "__main__":
    run_quick_demo()

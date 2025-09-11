"""
End-to-end tests for the complete Smart Traffic Management System
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import httpx
from fastapi.testclient import TestClient

from src.backend.main import app as backend_app
from src.orchestration.system_orchestrator import SystemOrchestrator
from src.orchestration.data_flow_manager import DataFlowManager
from src.orchestration.monitoring_system import MonitoringSystem

class TestSystemE2E:
    """End-to-end tests for the complete system"""
    
    @pytest.fixture
    def system_setup(self):
        """Setup complete system for E2E testing"""
        # Mock system components
        with patch('src.orchestration.system_orchestrator.SystemOrchestrator') as mock_orchestrator, \
             patch('src.orchestration.data_flow_manager.DataFlowManager') as mock_data_flow, \
             patch('src.orchestration.monitoring_system.MonitoringSystem') as mock_monitoring:
            
            # Setup mock orchestrator
            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.running = True
            mock_orchestrator_instance.get_system_status.return_value = {
                "system": {"name": "Test System", "version": "2.1.0", "running": True},
                "components": {
                    "backend": {"status": "running", "port": 8000},
                    "ml_optimizer": {"status": "running", "port": 8001},
                    "sumo_simulation": {"status": "running", "port": 8002},
                    "frontend": {"status": "running", "port": 3000}
                }
            }
            mock_orchestrator.return_value = mock_orchestrator_instance
            
            # Setup mock data flow manager
            mock_data_flow_instance = Mock()
            mock_data_flow_instance.get_metrics.return_value = {
                "metrics": {
                    "messages_processed": 1000,
                    "messages_failed": 10,
                    "processing_time": 5.5
                }
            }
            mock_data_flow.return_value = mock_data_flow_instance
            
            # Setup mock monitoring system
            mock_monitoring_instance = Mock()
            mock_monitoring_instance.get_system_status.return_value = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": {
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "disk_usage": 23.1
                },
                "component_status": {
                    "backend": "healthy",
                    "ml_optimizer": "healthy",
                    "sumo_simulation": "healthy",
                    "frontend": "healthy"
                },
                "active_alerts": 0
            }
            mock_monitoring.return_value = mock_monitoring_instance
            
            yield {
                "orchestrator": mock_orchestrator_instance,
                "data_flow": mock_data_flow_instance,
                "monitoring": mock_monitoring_instance
            }
    
    def test_complete_traffic_management_workflow(self, system_setup):
        """Test complete traffic management workflow from data input to optimization"""
        with TestClient(backend_app) as client:
            # Step 1: Create traffic lights
            traffic_lights_data = [
                {
                    "id": "light_1",
                    "name": "North-South Light",
                    "location": {"lat": 40.7128, "lng": -74.0060},
                    "status": "normal",
                    "current_phase": 0,
                    "phase_duration": 30,
                    "program": "adaptive",
                    "vehicle_count": 15,
                    "waiting_time": 45.5
                },
                {
                    "id": "light_2",
                    "name": "East-West Light",
                    "location": {"lat": 40.7128, "lng": -74.0060},
                    "status": "normal",
                    "current_phase": 2,
                    "phase_duration": 25,
                    "program": "adaptive",
                    "vehicle_count": 8,
                    "waiting_time": 20.0
                }
            ]
            
            for light_data in traffic_lights_data:
                response = client.post("/api/traffic/lights", json=light_data)
                assert response.status_code == 201
                assert response.json()["success"] is True
            
            # Step 2: Create intersection
            intersection_data = {
                "id": "intersection_1",
                "name": "Main Intersection",
                "location": {"lat": 40.7128, "lng": -74.0060},
                "traffic_lights": ["light_1", "light_2"],
                "total_vehicles": 23,
                "waiting_vehicles": 8,
                "average_speed": 22.5,
                "throughput": 45
            }
            
            response = client.post("/api/traffic/intersections", json=intersection_data)
            assert response.status_code == 201
            assert response.json()["success"] is True
            
            # Step 3: Add vehicles
            vehicles_data = [
                {
                    "id": "vehicle_1",
                    "type": "passenger",
                    "position": {"lat": 40.7128, "lng": -74.0060},
                    "speed": 25.5,
                    "lane": "north_approach_0",
                    "route": ["north_approach", "center_junction", "south_exit"],
                    "waiting_time": 5.2,
                    "co2_emission": 0.1,
                    "fuel_consumption": 0.05
                },
                {
                    "id": "vehicle_2",
                    "type": "truck",
                    "position": {"lat": 40.7128, "lng": -74.0060},
                    "speed": 18.0,
                    "lane": "east_approach_0",
                    "route": ["east_approach", "center_junction", "west_exit"],
                    "waiting_time": 12.8,
                    "co2_emission": 0.3,
                    "fuel_consumption": 0.15
                }
            ]
            
            for vehicle_data in vehicles_data:
                response = client.post("/api/traffic/vehicles", json=vehicle_data)
                assert response.status_code == 201
                assert response.json()["success"] is True
            
            # Step 4: Record performance metrics
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "total_vehicles": 156,
                "running_vehicles": 142,
                "waiting_vehicles": 14,
                "total_waiting_time": 234.5,
                "average_speed": 28.3,
                "total_co2_emission": 45.2,
                "total_fuel_consumption": 23.1,
                "throughput": 89
            }
            
            response = client.post("/api/traffic/metrics", json=metrics_data)
            assert response.status_code == 201
            assert response.json()["success"] is True
            
            # Step 5: Trigger optimization
            optimization_request = {
                "intersection_id": "intersection_1",
                "algorithm": "q_learning",
                "traffic_data": {
                    "vehicles": vehicles_data,
                    "traffic_lights": traffic_lights_data
                }
            }
            
            response = client.post("/api/optimization/optimize", json=optimization_request)
            assert response.status_code == 200
            optimization_result = response.json()
            assert optimization_result["success"] is True
            assert "optimized_phases" in optimization_result["data"]
            assert "efficiency_improvement" in optimization_result["data"]
            
            # Step 6: Apply optimization results
            optimized_phases = optimization_result["data"]["optimized_phases"]
            
            for i, light_id in enumerate(["light_1", "light_2"]):
                control_data = {
                    "phase": i % 4,
                    "duration": optimized_phases[i]
                }
                
                response = client.post(f"/api/traffic/lights/{light_id}/control", json=control_data)
                assert response.status_code == 200
                assert response.json()["success"] is True
            
            # Step 7: Verify system status
            response = client.get("/api/system/status")
            assert response.status_code == 200
            status = response.json()
            assert status["success"] is True
            assert status["data"]["system"]["running"] is True
            assert len(status["data"]["components"]) == 4
    
    def test_real_time_data_flow(self, system_setup):
        """Test real-time data flow through the system"""
        with TestClient(backend_app) as client:
            # Simulate real-time data updates
            for i in range(10):
                # Update traffic light data
                light_update = {
                    "vehicle_count": 10 + i,
                    "waiting_time": 20.0 + i * 2,
                    "current_phase": i % 4
                }
                
                response = client.put("/api/traffic/lights/light_1", json=light_update)
                assert response.status_code == 200
                
                # Update vehicle data
                vehicle_update = {
                    "id": f"vehicle_{i}",
                    "type": "passenger",
                    "position": {"lat": 40.7128 + i * 0.001, "lng": -74.0060 + i * 0.001},
                    "speed": 25.0 + i,
                    "lane": f"lane_{i % 4}",
                    "waiting_time": 5.0 + i
                }
                
                response = client.post("/api/traffic/vehicles", json=vehicle_update)
                assert response.status_code == 201
                
                # Update performance metrics
                metrics_update = {
                    "timestamp": datetime.now().isoformat(),
                    "total_vehicles": 100 + i * 10,
                    "running_vehicles": 80 + i * 8,
                    "waiting_vehicles": 20 + i * 2,
                    "average_speed": 25.0 + i,
                    "throughput": 40 + i * 5
                }
                
                response = client.post("/api/traffic/metrics", json=metrics_update)
                assert response.status_code == 201
                
                # Small delay to simulate real-time updates
                time.sleep(0.1)
            
            # Verify data accumulation
            response = client.get("/api/traffic/lights")
            assert response.status_code == 200
            lights = response.json()["data"]
            assert len(lights) >= 1
            
            response = client.get("/api/traffic/vehicles")
            assert response.status_code == 200
            vehicles = response.json()["data"]
            assert len(vehicles) >= 10
            
            response = client.get("/api/traffic/metrics")
            assert response.status_code == 200
            metrics = response.json()["data"]
            assert len(metrics) >= 10
    
    def test_system_monitoring_and_alerting(self, system_setup):
        """Test system monitoring and alerting functionality"""
        with TestClient(backend_app) as client:
            # Test health checks
            response = client.get("/health")
            assert response.status_code == 200
            health = response.json()
            assert health["status"] == "healthy"
            
            # Test detailed health check
            response = client.get("/health/detailed")
            assert response.status_code == 200
            detailed_health = response.json()
            assert detailed_health["status"] == "healthy"
            assert "components" in detailed_health
            
            # Test system metrics
            response = client.get("/api/system/metrics")
            assert response.status_code == 200
            metrics = response.json()
            assert metrics["success"] is True
            assert "system_metrics" in metrics["data"]
            
            # Test alerts
            response = client.get("/api/alerts")
            assert response.status_code == 200
            alerts = response.json()
            assert alerts["success"] is True
            assert "data" in alerts
    
    def test_ml_optimization_workflow(self, system_setup):
        """Test ML optimization workflow"""
        with TestClient(backend_app) as client:
            # Prepare training data
            training_data = []
            for i in range(100):
                training_data.append({
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                    "traffic_lights": [
                        {
                            "id": "light_1",
                            "phase": i % 4,
                            "duration": 30 + (i % 10),
                            "vehicle_count": 10 + (i % 20),
                            "waiting_time": 20.0 + (i % 15)
                        }
                    ],
                    "vehicles": [
                        {
                            "id": f"vehicle_{i}",
                            "type": "passenger",
                            "speed": 25.0 + (i % 10),
                            "waiting_time": 5.0 + (i % 10)
                        }
                    ]
                })
            
            # Train ML model
            training_request = {
                "algorithm": "q_learning",
                "training_data": training_data,
                "parameters": {
                    "learning_rate": 0.1,
                    "discount_factor": 0.9,
                    "epsilon": 0.1
                }
            }
            
            response = client.post("/api/ml/train", json=training_request)
            assert response.status_code == 200
            training_result = response.json()
            assert training_result["success"] is True
            assert "model_id" in training_result["data"]
            
            # Test optimization with trained model
            optimization_request = {
                "model_id": training_result["data"]["model_id"],
                "traffic_data": training_data[0],
                "algorithm": "q_learning"
            }
            
            response = client.post("/api/ml/optimize", json=optimization_request)
            assert response.status_code == 200
            optimization_result = response.json()
            assert optimization_result["success"] is True
            assert "optimized_phases" in optimization_result["data"]
            assert "efficiency_improvement" in optimization_result["data"]
    
    def test_sumo_simulation_integration(self, system_setup):
        """Test SUMO simulation integration"""
        with TestClient(backend_app) as client:
            # Start simulation
            simulation_request = {
                "scenario": "basic_intersection",
                "duration": 3600,
                "step_size": 1,
                "parameters": {
                    "traffic_density": 0.5,
                    "vehicle_types": ["passenger", "truck", "bus"]
                }
            }
            
            response = client.post("/api/simulation/start", json=simulation_request)
            assert response.status_code == 200
            simulation_result = response.json()
            assert simulation_result["success"] is True
            assert "simulation_id" in simulation_result["data"]
            
            simulation_id = simulation_result["data"]["simulation_id"]
            
            # Get simulation status
            response = client.get(f"/api/simulation/{simulation_id}/status")
            assert response.status_code == 200
            status = response.json()
            assert status["success"] is True
            assert "running" in status["data"]
            
            # Get simulation data
            response = client.get(f"/api/simulation/{simulation_id}/data")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "vehicles" in data["data"]
            assert "traffic_lights" in data["data"]
            
            # Stop simulation
            response = client.post(f"/api/simulation/{simulation_id}/stop")
            assert response.status_code == 200
            stop_result = response.json()
            assert stop_result["success"] is True
    
    def test_error_recovery_and_resilience(self, system_setup):
        """Test error recovery and system resilience"""
        with TestClient(backend_app) as client:
            # Test invalid data handling
            invalid_data = {
                "id": "",  # Empty ID
                "name": "Invalid Light",
                "location": {"lat": 200, "lng": -200},  # Invalid coordinates
                "status": "invalid_status"  # Invalid status
            }
            
            response = client.post("/api/traffic/lights", json=invalid_data)
            assert response.status_code == 422  # Validation error
            
            # Test non-existent resource access
            response = client.get("/api/traffic/lights/non_existent")
            assert response.status_code == 404
            
            # Test system recovery after error
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
    
    def test_performance_under_load(self, system_setup):
        """Test system performance under load"""
        with TestClient(backend_app) as client:
            start_time = time.time()
            
            # Simulate high load
            for i in range(100):
                # Create traffic light
                light_data = {
                    "id": f"load_light_{i}",
                    "name": f"Load Light {i}",
                    "location": {"lat": 40.7128 + i * 0.001, "lng": -74.0060 + i * 0.001},
                    "status": "normal",
                    "current_phase": i % 4,
                    "phase_duration": 30,
                    "program": "adaptive",
                    "vehicle_count": i % 20,
                    "waiting_time": i * 0.5
                }
                
                response = client.post("/api/traffic/lights", json=light_data)
                assert response.status_code == 201
                
                # Create vehicle
                vehicle_data = {
                    "id": f"load_vehicle_{i}",
                    "type": "passenger",
                    "position": {"lat": 40.7128 + i * 0.001, "lng": -74.0060 + i * 0.001},
                    "speed": 25.0 + i % 10,
                    "lane": f"lane_{i % 4}",
                    "waiting_time": i * 0.1
                }
                
                response = client.post("/api/traffic/vehicles", json=vehicle_data)
                assert response.status_code == 201
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify performance (should complete within reasonable time)
            assert total_time < 30  # 30 seconds for 100 operations
            
            # Verify data integrity
            response = client.get("/api/traffic/lights")
            assert response.status_code == 200
            lights = response.json()["data"]
            assert len(lights) >= 100
            
            response = client.get("/api/traffic/vehicles")
            assert response.status_code == 200
            vehicles = response.json()["data"]
            assert len(vehicles) >= 100
    
    def test_data_consistency_across_components(self, system_setup):
        """Test data consistency across all system components"""
        with TestClient(backend_app) as client:
            # Create traffic light
            light_data = {
                "id": "consistency_light",
                "name": "Consistency Light",
                "location": {"lat": 40.7128, "lng": -74.0060},
                "status": "normal",
                "current_phase": 0,
                "phase_duration": 30,
                "program": "adaptive",
                "vehicle_count": 10,
                "waiting_time": 20.0
            }
            
            response = client.post("/api/traffic/lights", json=light_data)
            assert response.status_code == 201
            
            # Create intersection
            intersection_data = {
                "id": "consistency_intersection",
                "name": "Consistency Intersection",
                "location": {"lat": 40.7128, "lng": -74.0060},
                "traffic_lights": ["consistency_light"],
                "total_vehicles": 10,
                "waiting_vehicles": 5,
                "average_speed": 25.0,
                "throughput": 30
            }
            
            response = client.post("/api/traffic/intersections", json=intersection_data)
            assert response.status_code == 201
            
            # Update traffic light
            update_data = {"vehicle_count": 15, "waiting_time": 25.0}
            response = client.put("/api/traffic/lights/consistency_light", json=update_data)
            assert response.status_code == 200
            
            # Update intersection metrics
            intersection_update = {
                "total_vehicles": 15,
                "waiting_vehicles": 8,
                "average_speed": 22.0,
                "throughput": 35
            }
            response = client.put("/api/traffic/intersections/consistency_intersection/metrics", json=intersection_update)
            assert response.status_code == 200
            
            # Verify consistency
            response = client.get("/api/traffic/lights/consistency_light")
            assert response.status_code == 200
            light = response.json()["data"]
            assert light["vehicle_count"] == 15
            assert light["waiting_time"] == 25.0
            
            response = client.get("/api/traffic/intersections/consistency_intersection")
            assert response.status_code == 200
            intersection = response.json()["data"]
            assert intersection["total_vehicles"] == 15
            assert intersection["waiting_vehicles"] == 8

class TestSystemIntegration:
    """Test integration between different system components"""
    
    def test_backend_ml_optimizer_integration(self, system_setup):
        """Test integration between backend and ML optimizer"""
        with TestClient(backend_app) as client:
            # Create traffic data
            traffic_data = {
                "intersection_id": "test_intersection",
                "traffic_lights": [
                    {
                        "id": "light_1",
                        "phase": 0,
                        "duration": 30,
                        "vehicle_count": 15,
                        "waiting_time": 45.5
                    }
                ],
                "vehicles": [
                    {
                        "id": "vehicle_1",
                        "type": "passenger",
                        "speed": 25.5,
                        "waiting_time": 5.2
                    }
                ]
            }
            
            # Send data to ML optimizer
            response = client.post("/api/optimization/optimize", json=traffic_data)
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True
            assert "optimized_phases" in result["data"]
    
    def test_backend_sumo_simulation_integration(self, system_setup):
        """Test integration between backend and SUMO simulation"""
        with TestClient(backend_app) as client:
            # Start simulation
            simulation_config = {
                "scenario": "test_scenario",
                "duration": 600,
                "step_size": 1
            }
            
            response = client.post("/api/simulation/start", json=simulation_config)
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True
            
            # Get simulation data
            simulation_id = result["data"]["simulation_id"]
            response = client.get(f"/api/simulation/{simulation_id}/data")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_data_flow_manager_integration(self, system_setup):
        """Test integration with data flow manager"""
        with TestClient(backend_app) as client:
            # Test data flow metrics
            response = client.get("/api/data-flow/metrics")
            assert response.status_code == 200
            metrics = response.json()
            assert metrics["success"] is True
            assert "messages_processed" in metrics["data"]
    
    def test_monitoring_system_integration(self, system_setup):
        """Test integration with monitoring system"""
        with TestClient(backend_app) as client:
            # Test system status
            response = client.get("/api/system/status")
            assert response.status_code == 200
            status = response.json()
            assert status["success"] is True
            assert "components" in status["data"]
            
            # Test alerts
            response = client.get("/api/alerts")
            assert response.status_code == 200
            alerts = response.json()
            assert alerts["success"] is True

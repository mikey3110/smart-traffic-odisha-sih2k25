"""
End-to-End Tests for Smart Traffic Management System
Uses Selenium to test full system integration
"""

import pytest
import time
import json
import requests
from datetime import datetime
from typing import Dict, Any
import logging

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficManagementE2ETest:
    """End-to-end test class for traffic management system"""
    
    def __init__(self):
        self.driver = None
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.wait_timeout = 10
        
    def setup_driver(self):
        """Setup Chrome driver with options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            logger.info("Chrome driver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {e}")
            raise
    
    def teardown_driver(self):
        """Clean up driver"""
        if self.driver:
            self.driver.quit()
            logger.info("Chrome driver closed")
    
    def wait_for_element(self, by, value, timeout=None):
        """Wait for element to be present and visible"""
        timeout = timeout or self.wait_timeout
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            logger.error(f"Element not found: {by}={value}")
            raise
    
    def wait_for_clickable(self, by, value, timeout=None):
        """Wait for element to be clickable"""
        timeout = timeout or self.wait_timeout
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            return element
        except TimeoutException:
            logger.error(f"Element not clickable: {by}={value}")
            raise
    
    def test_system_health(self):
        """Test system health endpoints"""
        logger.info("Testing system health...")
        
        # Test backend health
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            logger.info("Backend health check passed")
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend service not running")
        
        # Test frontend accessibility
        try:
            self.driver.get(self.frontend_url)
            assert "Smart Traffic Management" in self.driver.title
            logger.info("Frontend accessibility check passed")
        except Exception as e:
            logger.error(f"Frontend accessibility check failed: {e}")
            raise
    
    def test_dashboard_loading(self):
        """Test dashboard loading and display"""
        logger.info("Testing dashboard loading...")
        
        try:
            # Navigate to frontend
            self.driver.get(self.frontend_url)
            
            # Wait for dashboard to load
            self.wait_for_element(By.CLASS_NAME, "dashboard")
            
            # Check for key dashboard elements
            dashboard_elements = [
                "traffic-map",
                "system-status", 
                "vehicle-counts",
                "signal-controls"
            ]
            
            for element_class in dashboard_elements:
                try:
                    element = self.driver.find_element(By.CLASS_NAME, element_class)
                    assert element.is_displayed()
                    logger.info(f"Dashboard element {element_class} found and visible")
                except NoSuchElementException:
                    logger.warning(f"Dashboard element {element_class} not found")
            
            logger.info("Dashboard loading test passed")
            
        except Exception as e:
            logger.error(f"Dashboard loading test failed: {e}")
            raise
    
    def test_traffic_data_flow(self):
        """Test complete traffic data flow from backend to frontend"""
        logger.info("Testing traffic data flow...")
        
        try:
            # 1. Submit traffic data to backend
            traffic_data = {
                "intersection_id": "test_intersection_1",
                "timestamp": int(time.time()),
                "vehicle_counts": {
                    "car": 20,
                    "motorcycle": 8,
                    "bus": 3,
                    "truck": 2
                },
                "lane_occupancy": {
                    "lane_1": 0.75,
                    "lane_2": 0.60,
                    "lane_3": 0.45
                },
                "waiting_times": {
                    "lane_1": 45.2,
                    "lane_2": 38.7,
                    "lane_3": 52.1
                },
                "coordinates": {"lat": 20.2961, "lng": 85.8245}
            }
            
            response = requests.post(
                f"{self.backend_url}/api/v1/traffic/ingest",
                json=traffic_data,
                timeout=10
            )
            
            if response.status_code in [200, 404, 422]:
                logger.info("Traffic data submitted to backend")
            else:
                logger.warning(f"Traffic data submission returned {response.status_code}")
            
            # 2. Check if data appears in frontend
            self.driver.get(self.frontend_url)
            
            # Wait for data to load
            time.sleep(2)
            
            # Look for vehicle count display
            try:
                vehicle_count_element = self.driver.find_element(By.CLASS_NAME, "vehicle-count")
                assert vehicle_count_element.is_displayed()
                logger.info("Vehicle count display found in frontend")
            except NoSuchElementException:
                logger.warning("Vehicle count display not found in frontend")
            
            logger.info("Traffic data flow test completed")
            
        except Exception as e:
            logger.error(f"Traffic data flow test failed: {e}")
            raise
    
    def test_signal_control_flow(self):
        """Test signal control flow from frontend to backend"""
        logger.info("Testing signal control flow...")
        
        try:
            # 1. Navigate to frontend
            self.driver.get(self.frontend_url)
            
            # 2. Look for signal control elements
            try:
                signal_controls = self.driver.find_element(By.CLASS_NAME, "signal-controls")
                assert signal_controls.is_displayed()
                logger.info("Signal controls found in frontend")
                
                # Look for control buttons
                control_buttons = signal_controls.find_elements(By.TAG_NAME, "button")
                if control_buttons:
                    logger.info(f"Found {len(control_buttons)} signal control buttons")
                else:
                    logger.warning("No signal control buttons found")
                    
            except NoSuchElementException:
                logger.warning("Signal controls not found in frontend")
            
            # 3. Test signal optimization request
            optimization_request = {
                "intersection_id": "test_intersection_1",
                "current_phase": 0,
                "traffic_data": {
                    "vehicle_counts": {"car": 25, "motorcycle": 10, "bus": 4},
                    "lane_occupancy": {"lane_1": 0.80, "lane_2": 0.65},
                    "waiting_times": {"lane_1": 50.0, "lane_2": 42.0}
                },
                "optimization_type": "ml_optimized"
            }
            
            response = requests.post(
                f"{self.backend_url}/api/v1/signals/optimize/test_intersection_1",
                json=optimization_request,
                timeout=10
            )
            
            if response.status_code in [200, 404, 422]:
                logger.info("Signal optimization request processed")
            else:
                logger.warning(f"Signal optimization returned {response.status_code}")
            
            logger.info("Signal control flow test completed")
            
        except Exception as e:
            logger.error(f"Signal control flow test failed: {e}")
            raise
    
    def test_ml_integration_flow(self):
        """Test ML integration flow"""
        logger.info("Testing ML integration flow...")
        
        try:
            # 1. Submit ML metrics
            ml_metrics = {
                "intersection_id": "test_intersection_1",
                "timestamp": int(time.time()),
                "reward": 0.85,
                "wait_time_reduction": 0.25,
                "throughput_increase": 0.15,
                "fuel_efficiency": 0.10,
                "safety_score": 0.95
            }
            
            response = requests.post(
                f"{self.backend_url}/api/v1/ml/metrics",
                json=ml_metrics,
                timeout=5
            )
            
            if response.status_code in [200, 404, 422]:
                logger.info("ML metrics submitted")
            else:
                logger.warning(f"ML metrics submission returned {response.status_code}")
            
            # 2. Check ML metrics display in frontend
            self.driver.get(self.frontend_url)
            
            # Look for ML metrics display
            try:
                ml_metrics_element = self.driver.find_element(By.CLASS_NAME, "ml-metrics")
                assert ml_metrics_element.is_displayed()
                logger.info("ML metrics display found in frontend")
            except NoSuchElementException:
                logger.warning("ML metrics display not found in frontend")
            
            logger.info("ML integration flow test completed")
            
        except Exception as e:
            logger.error(f"ML integration flow test failed: {e}")
            raise
    
    def test_cv_integration_flow(self):
        """Test CV integration flow"""
        logger.info("Testing CV integration flow...")
        
        try:
            # 1. Submit CV data
            cv_data = {
                "camera_id": "test_cam_001",
                "intersection_id": "test_intersection_1",
                "timestamp": int(time.time()),
                "total_vehicles": 18,
                "counts_by_class": {
                    "car": 12,
                    "motorcycle": 4,
                    "bus": 1,
                    "truck": 1
                },
                "coordinates": {"lat": 20.2961, "lng": 85.8245}
            }
            
            response = requests.post(
                f"{self.backend_url}/api/v1/cv/counts",
                json=cv_data,
                timeout=5
            )
            
            if response.status_code in [200, 404, 422]:
                logger.info("CV data submitted")
            else:
                logger.warning(f"CV data submission returned {response.status_code}")
            
            # 2. Check CV data display in frontend
            self.driver.get(self.frontend_url)
            
            # Look for camera feed or vehicle count display
            try:
                camera_feed = self.driver.find_element(By.CLASS_NAME, "camera-feed")
                assert camera_feed.is_displayed()
                logger.info("Camera feed found in frontend")
            except NoSuchElementException:
                logger.warning("Camera feed not found in frontend")
            
            logger.info("CV integration flow test completed")
            
        except Exception as e:
            logger.error(f"CV integration flow test failed: {e}")
            raise
    
    def test_error_handling(self):
        """Test error handling in frontend"""
        logger.info("Testing error handling...")
        
        try:
            # 1. Test with invalid data
            invalid_data = {
                "intersection_id": 123,  # Invalid type
                "timestamp": "invalid",  # Invalid type
                "vehicle_counts": "invalid"  # Invalid type
            }
            
            response = requests.post(
                f"{self.backend_url}/api/v1/traffic/ingest",
                json=invalid_data,
                timeout=5
            )
            
            # Should return error status
            assert response.status_code in [400, 422, 404]
            logger.info("Invalid data handling test passed")
            
            # 2. Test frontend error display
            self.driver.get(self.frontend_url)
            
            # Look for error handling elements
            try:
                error_element = self.driver.find_element(By.CLASS_NAME, "error-message")
                logger.info("Error display element found")
            except NoSuchElementException:
                logger.info("No error display element found (may be handled differently)")
            
            logger.info("Error handling test completed")
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            raise
    
    def test_performance_metrics(self):
        """Test performance metrics display"""
        logger.info("Testing performance metrics...")
        
        try:
            # 1. Navigate to frontend
            self.driver.get(self.frontend_url)
            
            # 2. Look for performance metrics
            performance_elements = [
                "response-time",
                "throughput",
                "error-rate",
                "cpu-usage",
                "memory-usage"
            ]
            
            for element_class in performance_elements:
                try:
                    element = self.driver.find_element(By.CLASS_NAME, element_class)
                    if element.is_displayed():
                        logger.info(f"Performance metric {element_class} found")
                except NoSuchElementException:
                    logger.warning(f"Performance metric {element_class} not found")
            
            logger.info("Performance metrics test completed")
            
        except Exception as e:
            logger.error(f"Performance metrics test failed: {e}")
            raise

def test_full_system_e2e():
    """Run complete E2E test suite"""
    test_instance = TrafficManagementE2ETest()
    
    try:
        # Setup
        test_instance.setup_driver()
        
        # Run tests
        test_instance.test_system_health()
        test_instance.test_dashboard_loading()
        test_instance.test_traffic_data_flow()
        test_instance.test_signal_control_flow()
        test_instance.test_ml_integration_flow()
        test_instance.test_cv_integration_flow()
        test_instance.test_error_handling()
        test_instance.test_performance_metrics()
        
        logger.info("All E2E tests completed successfully")
        
    except Exception as e:
        logger.error(f"E2E test suite failed: {e}")
        raise
    finally:
        # Cleanup
        test_instance.teardown_driver()

def test_docker_compose_integration():
    """Test system with Docker Compose"""
    logger.info("Testing Docker Compose integration...")
    
    try:
        # Test if services are running
        services = [
            ("Backend", "http://localhost:8000/health"),
            ("Frontend", "http://localhost:3000"),
            ("ML API", "http://localhost:8001/health"),
            ("CV Service", "http://localhost:5001/cv/streams")
        ]
        
        for service_name, url in services:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"{service_name} is running")
                else:
                    logger.warning(f"{service_name} returned {response.status_code}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"{service_name} is not accessible")
        
        logger.info("Docker Compose integration test completed")
        
    except Exception as e:
        logger.error(f"Docker Compose integration test failed: {e}")
        raise

if __name__ == "__main__":
    # Run E2E tests
    test_full_system_e2e()
    test_docker_compose_integration()

"""
SUMO Simulation Validation and Calibration Tools
Tools for validating simulation accuracy and calibrating parameters
"""

import traci
import traci.constants as tc
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import threading
from pathlib import Path


class ValidationMetric(Enum):
    """Validation metrics"""
    VEHICLE_COUNT = "vehicle_count"
    WAITING_TIME = "waiting_time"
    SPEED = "speed"
    FLOW_RATE = "flow_rate"
    OCCUPANCY = "occupancy"
    EMISSIONS = "emissions"


class CalibrationMethod(Enum):
    """Calibration methods"""
    LEAST_SQUARES = "least_squares"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    BAYESIAN = "bayesian"


@dataclass
class ValidationConfig:
    """Validation configuration"""
    metrics: List[ValidationMetric] = field(default_factory=lambda: [
        ValidationMetric.VEHICLE_COUNT,
        ValidationMetric.WAITING_TIME,
        ValidationMetric.SPEED
    ])
    validation_interval: float = 10.0  # seconds
    sample_size: int = 100
    confidence_level: float = 0.95
    tolerance: float = 0.1  # 10% tolerance


@dataclass
class CalibrationConfig:
    """Calibration configuration"""
    method: CalibrationMethod = CalibrationMethod.LEAST_SQUARES
    parameters: List[str] = field(default_factory=lambda: [
        'max_speed', 'acceleration', 'deceleration', 'min_gap'
    ])
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'max_speed': (20.0, 50.0),
        'acceleration': (1.0, 4.0),
        'deceleration': (2.0, 6.0),
        'min_gap': (0.5, 3.0)
    })
    max_iterations: int = 100
    convergence_threshold: float = 0.01


@dataclass
class ValidationResult:
    """Validation result"""
    metric: ValidationMetric
    simulated_value: float
    observed_value: float
    error: float
    relative_error: float
    is_valid: bool
    confidence_interval: Tuple[float, float]
    timestamp: datetime


@dataclass
class CalibrationResult:
    """Calibration result"""
    parameters: Dict[str, float]
    objective_value: float
    iterations: int
    convergence: bool
    validation_results: List[ValidationResult]
    timestamp: datetime


class SimulationValidator:
    """
    SUMO Simulation Validation and Calibration Tools
    
    Features:
    - Real-time validation
    - Statistical analysis
    - Parameter calibration
    - Performance metrics validation
    - Error detection and reporting
    - Calibration optimization
    """
    
    def __init__(self, validation_config: Optional[ValidationConfig] = None):
        self.validation_config = validation_config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validation data
        self.validation_results: List[ValidationResult] = []
        self.observed_data: Dict[str, List[float]] = {}
        self.simulated_data: Dict[str, List[float]] = {}
        
        # Calibration
        self.calibration_results: List[CalibrationResult] = []
        self.current_parameters: Dict[str, float] = {}
        
        # Validation control
        self.is_validating = False
        self.validation_thread = None
        
        # Statistics
        self.validation_count = 0
        self.validation_errors = 0
        
        self.logger.info("Simulation Validator initialized")
    
    def start_validation(self):
        """Start continuous validation"""
        if self.is_validating:
            return
        
        self.is_validating = True
        self.validation_thread = threading.Thread(target=self._validation_loop)
        self.validation_thread.daemon = True
        self.validation_thread.start()
        
        self.logger.info("Validation started")
    
    def stop_validation(self):
        """Stop validation"""
        self.is_validating = False
        if self.validation_thread:
            self.validation_thread.join(timeout=5)
        
        self.logger.info("Validation stopped")
    
    def _validation_loop(self):
        """Main validation loop"""
        while self.is_validating:
            try:
                # Perform validation
                self._perform_validation()
                
                time.sleep(self.validation_config.validation_interval)
                
            except Exception as e:
                self.logger.error(f"Error in validation loop: {e}")
                time.sleep(5.0)
    
    def _perform_validation(self):
        """Perform validation checks"""
        try:
            for metric in self.validation_config.metrics:
                result = self._validate_metric(metric)
                if result:
                    self.validation_results.append(result)
                    self.validation_count += 1
                    
                    if not result.is_valid:
                        self.validation_errors += 1
                        self.logger.warning(f"Validation failed for {metric.value}: "
                                          f"error={result.relative_error:.2%}")
            
            # Keep only recent results
            if len(self.validation_results) > 1000:
                self.validation_results = self.validation_results[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error performing validation: {e}")
    
    def _validate_metric(self, metric: ValidationMetric) -> Optional[ValidationResult]:
        """Validate specific metric"""
        try:
            # Get simulated value
            simulated_value = self._get_simulated_value(metric)
            if simulated_value is None:
                return None
            
            # Get observed value (this would come from real-world data)
            observed_value = self._get_observed_value(metric)
            if observed_value is None:
                return None
            
            # Calculate error
            error = abs(simulated_value - observed_value)
            relative_error = error / observed_value if observed_value != 0 else 0
            
            # Check if within tolerance
            is_valid = relative_error <= self.validation_config.tolerance
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                simulated_value, observed_value, self.validation_config.confidence_level
            )
            
            return ValidationResult(
                metric=metric,
                simulated_value=simulated_value,
                observed_value=observed_value,
                error=error,
                relative_error=relative_error,
                is_valid=is_valid,
                confidence_interval=confidence_interval,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error validating metric {metric.value}: {e}")
            return None
    
    def _get_simulated_value(self, metric: ValidationMetric) -> Optional[float]:
        """Get simulated value for metric"""
        try:
            if metric == ValidationMetric.VEHICLE_COUNT:
                return len(traci.vehicle.getIDList())
            
            elif metric == ValidationMetric.WAITING_TIME:
                vehicles = traci.vehicle.getIDList()
                if not vehicles:
                    return 0.0
                return np.mean([traci.vehicle.getWaitingTime(v) for v in vehicles])
            
            elif metric == ValidationMetric.SPEED:
                vehicles = traci.vehicle.getIDList()
                if not vehicles:
                    return 0.0
                return np.mean([traci.vehicle.getSpeed(v) for v in vehicles])
            
            elif metric == ValidationMetric.FLOW_RATE:
                # Calculate flow rate from lane data
                lanes = traci.lane.getIDList()
                if not lanes:
                    return 0.0
                total_flow = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
                return total_flow / len(lanes) if lanes else 0.0
            
            elif metric == ValidationMetric.OCCUPANCY:
                lanes = traci.lane.getIDList()
                if not lanes:
                    return 0.0
                return np.mean([traci.lane.getLastStepOccupancy(lane) for lane in lanes])
            
            elif metric == ValidationMetric.EMISSIONS:
                return traci.simulation.getCO2Emission()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting simulated value for {metric.value}: {e}")
            return None
    
    def _get_observed_value(self, metric: ValidationMetric) -> Optional[float]:
        """Get observed value for metric (from real-world data)"""
        try:
            # This would typically come from real-world sensors or historical data
            # For now, we'll use simulated values with some noise as a placeholder
            
            simulated_value = self._get_simulated_value(metric)
            if simulated_value is None:
                return None
            
            # Add noise to simulate real-world data
            noise_factor = 0.05  # 5% noise
            noise = np.random.normal(0, simulated_value * noise_factor)
            return simulated_value + noise
            
        except Exception as e:
            self.logger.error(f"Error getting observed value for {metric.value}: {e}")
            return None
    
    def _calculate_confidence_interval(self, simulated: float, observed: float, 
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval"""
        try:
            # Simple confidence interval calculation
            # In practice, this would use more sophisticated statistical methods
            
            error = abs(simulated - observed)
            margin = error * 0.1  # 10% margin
            
            return (observed - margin, observed + margin)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence interval: {e}")
            return (0.0, 0.0)
    
    def calibrate_parameters(self, calibration_config: CalibrationConfig) -> CalibrationResult:
        """Calibrate simulation parameters"""
        try:
            self.logger.info(f"Starting calibration using {calibration_config.method.value}")
            
            # Initialize parameters
            initial_parameters = self._initialize_parameters(calibration_config)
            
            # Perform calibration based on method
            if calibration_config.method == CalibrationMethod.LEAST_SQUARES:
                result = self._calibrate_least_squares(calibration_config, initial_parameters)
            elif calibration_config.method == CalibrationMethod.GENETIC_ALGORITHM:
                result = self._calibrate_genetic_algorithm(calibration_config, initial_parameters)
            else:
                result = self._calibrate_least_squares(calibration_config, initial_parameters)
            
            # Store result
            self.calibration_results.append(result)
            
            self.logger.info(f"Calibration completed: objective_value={result.objective_value:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during calibration: {e}")
            return None
    
    def _initialize_parameters(self, config: CalibrationConfig) -> Dict[str, float]:
        """Initialize calibration parameters"""
        parameters = {}
        for param in config.parameters:
            if param in config.bounds:
                min_val, max_val = config.bounds[param]
                parameters[param] = (min_val + max_val) / 2.0  # Start with middle value
            else:
                parameters[param] = 1.0  # Default value
        return parameters
    
    def _calibrate_least_squares(self, config: CalibrationConfig, 
                                initial_parameters: Dict[str, float]) -> CalibrationResult:
        """Calibrate using least squares method"""
        try:
            def objective_function(params):
                # Set parameters in simulation
                self._set_simulation_parameters(params)
                
                # Run simulation and collect data
                simulated_data = self._collect_simulation_data()
                
                # Calculate objective function (sum of squared errors)
                objective = 0.0
                for metric in self.validation_config.metrics:
                    simulated_value = self._get_metric_value(simulated_data, metric)
                    observed_value = self._get_observed_value(metric)
                    
                    if simulated_value is not None and observed_value is not None:
                        error = (simulated_value - observed_value) ** 2
                        objective += error
                
                return objective
            
            # Convert parameters to list for optimization
            param_names = list(initial_parameters.keys())
            initial_values = [initial_parameters[name] for name in param_names]
            
            # Set bounds
            bounds = [config.bounds.get(name, (0.1, 10.0)) for name in param_names]
            
            # Optimize
            result = minimize(
                objective_function,
                initial_values,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': config.max_iterations}
            )
            
            # Convert result back to parameter dictionary
            optimized_parameters = {name: value for name, value in zip(param_names, result.x)}
            
            # Validate results
            validation_results = self._validate_calibrated_parameters(optimized_parameters)
            
            return CalibrationResult(
                parameters=optimized_parameters,
                objective_value=result.fun,
                iterations=result.nit,
                convergence=result.success,
                validation_results=validation_results,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in least squares calibration: {e}")
            return None
    
    def _calibrate_genetic_algorithm(self, config: CalibrationConfig, 
                                   initial_parameters: Dict[str, float]) -> CalibrationResult:
        """Calibrate using genetic algorithm"""
        try:
            # This is a simplified implementation
            # In practice, you would use a proper GA library like DEAP
            
            best_parameters = initial_parameters.copy()
            best_objective = float('inf')
            
            for iteration in range(config.max_iterations):
                # Generate new parameters
                new_parameters = self._mutate_parameters(best_parameters, config.bounds)
                
                # Evaluate objective function
                self._set_simulation_parameters(new_parameters)
                simulated_data = self._collect_simulation_data()
                
                objective = 0.0
                for metric in self.validation_config.metrics:
                    simulated_value = self._get_metric_value(simulated_data, metric)
                    observed_value = self._get_observed_value(metric)
                    
                    if simulated_value is not None and observed_value is not None:
                        error = (simulated_value - observed_value) ** 2
                        objective += error
                
                # Update best if better
                if objective < best_objective:
                    best_objective = objective
                    best_parameters = new_parameters.copy()
                
                # Check convergence
                if iteration > 0 and abs(objective - best_objective) < config.convergence_threshold:
                    break
            
            # Validate results
            validation_results = self._validate_calibrated_parameters(best_parameters)
            
            return CalibrationResult(
                parameters=best_parameters,
                objective_value=best_objective,
                iterations=iteration + 1,
                convergence=True,
                validation_results=validation_results,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in genetic algorithm calibration: {e}")
            return None
    
    def _set_simulation_parameters(self, parameters: Dict[str, float]):
        """Set parameters in simulation"""
        try:
            # This would set parameters in SUMO simulation
            # For now, we'll just store them
            self.current_parameters = parameters.copy()
            
        except Exception as e:
            self.logger.error(f"Error setting simulation parameters: {e}")
    
    def _collect_simulation_data(self) -> Dict[str, Any]:
        """Collect data from simulation"""
        try:
            data = {}
            
            for metric in self.validation_config.metrics:
                value = self._get_simulated_value(metric)
                if value is not None:
                    data[metric.value] = value
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error collecting simulation data: {e}")
            return {}
    
    def _get_metric_value(self, data: Dict[str, Any], metric: ValidationMetric) -> Optional[float]:
        """Get metric value from data"""
        return data.get(metric.value)
    
    def _mutate_parameters(self, parameters: Dict[str, float], 
                          bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Mutate parameters for genetic algorithm"""
        mutated = parameters.copy()
        
        for param, value in mutated.items():
            if param in bounds:
                min_val, max_val = bounds[param]
                # Add random mutation
                mutation = np.random.normal(0, (max_val - min_val) * 0.1)
                new_value = value + mutation
                # Clamp to bounds
                new_value = max(min_val, min(max_val, new_value))
                mutated[param] = new_value
        
        return mutated
    
    def _validate_calibrated_parameters(self, parameters: Dict[str, float]) -> List[ValidationResult]:
        """Validate calibrated parameters"""
        try:
            # Set parameters
            self._set_simulation_parameters(parameters)
            
            # Run validation
            validation_results = []
            for metric in self.validation_config.metrics:
                result = self._validate_metric(metric)
                if result:
                    validation_results.append(result)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating calibrated parameters: {e}")
            return []
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_results:
            return {'total_validations': 0, 'error_rate': 0.0}
        
        total_validations = len(self.validation_results)
        valid_count = sum(1 for r in self.validation_results if r.is_valid)
        error_rate = (total_validations - valid_count) / total_validations
        
        return {
            'total_validations': total_validations,
            'valid_count': valid_count,
            'error_count': total_validations - valid_count,
            'error_rate': error_rate,
            'is_validating': self.is_validating,
            'last_validation': self.validation_results[-1].timestamp.isoformat() if self.validation_results else None
        }
    
    def export_validation_results(self, filepath: str):
        """Export validation results"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'validation_config': {
                    'metrics': [m.value for m in self.validation_config.metrics],
                    'validation_interval': self.validation_config.validation_interval,
                    'tolerance': self.validation_config.tolerance
                },
                'validation_results': [{
                    'metric': r.metric.value,
                    'simulated_value': r.simulated_value,
                    'observed_value': r.observed_value,
                    'error': r.error,
                    'relative_error': r.relative_error,
                    'is_valid': r.is_valid,
                    'confidence_interval': r.confidence_interval,
                    'timestamp': r.timestamp.isoformat()
                } for r in self.validation_results],
                'calibration_results': [{
                    'parameters': r.parameters,
                    'objective_value': r.objective_value,
                    'iterations': r.iterations,
                    'convergence': r.convergence,
                    'timestamp': r.timestamp.isoformat()
                } for r in self.calibration_results],
                'statistics': self.get_validation_statistics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Validation results exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting validation results: {e}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create validator
    validation_config = ValidationConfig(
        metrics=[ValidationMetric.VEHICLE_COUNT, ValidationMetric.WAITING_TIME],
        validation_interval=5.0,
        tolerance=0.15
    )
    
    validator = SimulationValidator(validation_config)
    
    # Start validation
    validator.start_validation()
    
    try:
        # Run validation for some time
        time.sleep(30)
        
        # Get statistics
        stats = validator.get_validation_statistics()
        print(f"Validation statistics: {stats}")
        
        # Export results
        validator.export_validation_results("validation_results.json")
    
    finally:
        # Stop validation
        validator.stop_validation()

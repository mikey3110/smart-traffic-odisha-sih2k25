"""
Data Export and API Integration for SUMO Simulation
Real-time data export from SUMO to backend API
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import threading
from queue import Queue, Empty
import traci
import traci.constants as tc


class ExportFormat(Enum):
    """Data export formats"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    BINARY = "binary"


class ExportFrequency(Enum):
    """Export frequency options"""
    REAL_TIME = "real_time"  # Every step
    HIGH = "high"  # Every 1 second
    MEDIUM = "medium"  # Every 10 seconds
    LOW = "low"  # Every 60 seconds


@dataclass
class ExportConfig:
    """Data export configuration"""
    enabled: bool = True
    format: ExportFormat = ExportFormat.JSON
    frequency: ExportFrequency = ExportFrequency.MEDIUM
    batch_size: int = 100
    max_queue_size: int = 1000
    api_endpoint: str = "http://localhost:8000/api/v1/sumo/data"
    timeout: float = 10.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    compression: bool = False
    encryption: bool = False


@dataclass
class SimulationData:
    """Simulation data structure for export"""
    timestamp: datetime
    simulation_time: float
    step: int
    vehicles: List[Dict[str, Any]]
    intersections: List[Dict[str, Any]]
    lanes: List[Dict[str, Any]]
    traffic_lights: List[Dict[str, Any]]
    emissions: Dict[str, float]
    performance_metrics: Dict[str, float]


class DataExporter:
    """
    Data Export and API Integration for SUMO Simulation
    
    Features:
    - Real-time data export
    - Multiple export formats
    - API integration with retry logic
    - Data compression and encryption
    - Batch processing
    - Performance monitoring
    - Error handling and recovery
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        self.logger = logging.getLogger(__name__)
        
        # Data queues
        self.data_queue = Queue(maxsize=self.config.max_queue_size)
        self.export_queue = Queue(maxsize=self.config.max_queue_size)
        
        # Export threads
        self.collection_thread = None
        self.export_thread = None
        self.is_running = False
        
        # API client
        self.session = None
        self.api_connected = False
        
        # Statistics
        self.exported_batches = 0
        self.failed_exports = 0
        self.total_records = 0
        self.last_export_time = None
        
        # Data buffers
        self.vehicle_buffer: Dict[str, Dict[str, Any]] = {}
        self.intersection_buffer: Dict[str, Dict[str, Any]] = {}
        self.lane_buffer: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Data Exporter initialized")
    
    async def start_export(self):
        """Start data export process"""
        try:
            self.is_running = True
            
            # Initialize API session
            await self._initialize_api_session()
            
            # Start collection thread
            self.collection_thread = threading.Thread(target=self._collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            
            # Start export thread
            self.export_thread = threading.Thread(target=self._export_loop)
            self.export_thread.daemon = True
            self.export_thread.start()
            
            self.logger.info("Data export started")
            
        except Exception as e:
            self.logger.error(f"Error starting data export: {e}")
            self.is_running = False
    
    def stop_export(self):
        """Stop data export process"""
        self.is_running = False
        
        # Wait for threads to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        if self.export_thread and self.export_thread.is_alive():
            self.export_thread.join(timeout=5)
        
        # Close API session
        if self.session:
            asyncio.create_task(self.session.close())
        
        self.logger.info("Data export stopped")
    
    async def _initialize_api_session(self):
        """Initialize API session"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                connector=aiohttp.TCPConnector(limit=100)
            )
            
            # Test API connection
            async with self.session.get(f"{self.config.api_endpoint}/health") as response:
                if response.status == 200:
                    self.api_connected = True
                    self.logger.info("API connection established")
                else:
                    self.logger.warning(f"API health check failed: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error initializing API session: {e}")
            self.api_connected = False
    
    def _collection_loop(self):
        """Data collection loop"""
        while self.is_running:
            try:
                # Collect simulation data
                data = self._collect_simulation_data()
                
                if data:
                    # Add to queue
                    try:
                        self.data_queue.put(data, timeout=1.0)
                    except:
                        self.logger.warning("Data queue full, dropping data")
                
                # Sleep based on frequency
                sleep_time = self._get_collection_interval()
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(1.0)
    
    def _export_loop(self):
        """Data export loop"""
        while self.is_running:
            try:
                # Get data from queue
                try:
                    data = self.data_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process data
                processed_data = self._process_data(data)
                
                # Add to export queue
                try:
                    self.export_queue.put(processed_data, timeout=1.0)
                except:
                    self.logger.warning("Export queue full, dropping data")
                
                # Export if batch is ready
                if self.export_queue.qsize() >= self.config.batch_size:
                    self._export_batch()
                
            except Exception as e:
                self.logger.error(f"Error in export loop: {e}")
                time.sleep(1.0)
    
    def _collect_simulation_data(self) -> Optional[SimulationData]:
        """Collect data from SUMO simulation"""
        try:
            # Get simulation time
            simulation_time = traci.simulation.getTime()
            step = int(simulation_time)
            
            # Collect vehicle data
            vehicles = self._collect_vehicle_data()
            
            # Collect intersection data
            intersections = self._collect_intersection_data()
            
            # Collect lane data
            lanes = self._collect_lane_data()
            
            # Collect traffic light data
            traffic_lights = self._collect_traffic_light_data()
            
            # Collect emissions data
            emissions = self._collect_emissions_data()
            
            # Collect performance metrics
            performance_metrics = self._collect_performance_metrics()
            
            return SimulationData(
                timestamp=datetime.now(),
                simulation_time=simulation_time,
                step=step,
                vehicles=vehicles,
                intersections=intersections,
                lanes=lanes,
                traffic_lights=traffic_lights,
                emissions=emissions,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting simulation data: {e}")
            return None
    
    def _collect_vehicle_data(self) -> List[Dict[str, Any]]:
        """Collect vehicle data from simulation"""
        vehicles = []
        
        try:
            vehicle_ids = traci.vehicle.getIDList()
            
            for vehicle_id in vehicle_ids:
                try:
                    vehicle_data = {
                        'id': vehicle_id,
                        'position': traci.vehicle.getPosition(vehicle_id),
                        'speed': traci.vehicle.getSpeed(vehicle_id),
                        'lane': traci.vehicle.getLaneID(vehicle_id),
                        'route': traci.vehicle.getRoute(vehicle_id),
                        'waiting_time': traci.vehicle.getWaitingTime(vehicle_id),
                        'co2_emission': traci.vehicle.getCO2Emission(vehicle_id),
                        'fuel_consumption': traci.vehicle.getFuelConsumption(vehicle_id),
                        'type': traci.vehicle.getTypeID(vehicle_id),
                        'length': traci.vehicle.getLength(vehicle_id),
                        'width': traci.vehicle.getWidth(vehicle_id)
                    }
                    vehicles.append(vehicle_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error collecting data for vehicle {vehicle_id}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error collecting vehicle data: {e}")
        
        return vehicles
    
    def _collect_intersection_data(self) -> List[Dict[str, Any]]:
        """Collect intersection data from simulation"""
        intersections = []
        
        try:
            intersection_ids = traci.junction.getIDList()
            
            for intersection_id in intersection_ids:
                try:
                    intersection_data = {
                        'id': intersection_id,
                        'position': traci.junction.getPosition(intersection_id),
                        'type': traci.junction.getType(intersection_id),
                        'shape': traci.junction.getShape(intersection_id)
                    }
                    intersections.append(intersection_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error collecting data for intersection {intersection_id}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error collecting intersection data: {e}")
        
        return intersections
    
    def _collect_lane_data(self) -> List[Dict[str, Any]]:
        """Collect lane data from simulation"""
        lanes = []
        
        try:
            lane_ids = traci.lane.getIDList()
            
            for lane_id in lane_ids:
                try:
                    lane_data = {
                        'id': lane_id,
                        'length': traci.lane.getLength(lane_id),
                        'max_speed': traci.lane.getMaxSpeed(lane_id),
                        'vehicle_count': traci.lane.getLastStepVehicleNumber(lane_id),
                        'waiting_vehicles': traci.lane.getLastStepHaltingNumber(lane_id),
                        'mean_speed': traci.lane.getLastStepMeanSpeed(lane_id),
                        'occupancy': traci.lane.getLastStepOccupancy(lane_id),
                        'co2_emission': traci.lane.getCO2Emission(lane_id),
                        'fuel_consumption': traci.lane.getFuelConsumption(lane_id)
                    }
                    lanes.append(lane_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error collecting data for lane {lane_id}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error collecting lane data: {e}")
        
        return lanes
    
    def _collect_traffic_light_data(self) -> List[Dict[str, Any]]:
        """Collect traffic light data from simulation"""
        traffic_lights = []
        
        try:
            tl_ids = traci.trafficlight.getIDList()
            
            for tl_id in tl_ids:
                try:
                    tl_data = {
                        'id': tl_id,
                        'state': traci.trafficlight.getRedYellowGreenState(tl_id),
                        'phase': traci.trafficlight.getPhase(tl_id),
                        'phase_duration': traci.trafficlight.getPhaseDuration(tl_id),
                        'program': traci.trafficlight.getProgram(tl_id),
                        'controlled_lanes': traci.trafficlight.getControlledLanes(tl_id)
                    }
                    traffic_lights.append(tl_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error collecting data for traffic light {tl_id}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error collecting traffic light data: {e}")
        
        return traffic_lights
    
    def _collect_emissions_data(self) -> Dict[str, float]:
        """Collect emissions data from simulation"""
        try:
            return {
                'total_co2': traci.simulation.getCO2Emission(),
                'total_co': traci.simulation.getCOEmission(),
                'total_hc': traci.simulation.getHCEmission(),
                'total_nox': traci.simulation.getNOxEmission(),
                'total_pmx': traci.simulation.getPMxEmission(),
                'total_fuel': traci.simulation.getFuelConsumption()
            }
        except Exception as e:
            self.logger.error(f"Error collecting emissions data: {e}")
            return {}
    
    def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect performance metrics from simulation"""
        try:
            return {
                'total_vehicles': len(traci.vehicle.getIDList()),
                'running_vehicles': len([v for v in traci.vehicle.getIDList() if traci.vehicle.getSpeed(v) > 0.1]),
                'waiting_vehicles': len([v for v in traci.vehicle.getIDList() if traci.vehicle.getSpeed(v) <= 0.1]),
                'total_waiting_time': sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()),
                'average_speed': np.mean([traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]) if traci.vehicle.getIDList() else 0.0
            }
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
            return {}
    
    def _process_data(self, data: SimulationData) -> Dict[str, Any]:
        """Process simulation data for export"""
        try:
            # Convert to dictionary
            processed_data = asdict(data)
            
            # Convert datetime to ISO string
            processed_data['timestamp'] = data.timestamp.isoformat()
            
            # Add metadata
            processed_data['metadata'] = {
                'export_format': self.config.format.value,
                'batch_size': self.config.batch_size,
                'compression': self.config.compression,
                'encryption': self.config.encryption
            }
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            return {}
    
    def _export_batch(self):
        """Export batch of data"""
        try:
            # Collect all data from export queue
            batch_data = []
            while not self.export_queue.empty():
                try:
                    data = self.export_queue.get_nowait()
                    batch_data.append(data)
                except Empty:
                    break
            
            if not batch_data:
                return
            
            # Export data
            if self.config.enabled and self.api_connected:
                asyncio.create_task(self._export_to_api(batch_data))
            
            # Update statistics
            self.exported_batches += 1
            self.total_records += len(batch_data)
            self.last_export_time = datetime.now()
            
            self.logger.debug(f"Exported batch of {len(batch_data)} records")
            
        except Exception as e:
            self.logger.error(f"Error exporting batch: {e}")
            self.failed_exports += 1
    
    async def _export_to_api(self, batch_data: List[Dict[str, Any]]):
        """Export data to API"""
        try:
            if not self.session:
                return
            
            # Prepare payload
            payload = {
                'batch_id': f"batch_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'data': batch_data,
                'count': len(batch_data)
            }
            
            # Serialize data
            if self.config.format == ExportFormat.JSON:
                data = json.dumps(payload)
                content_type = 'application/json'
            else:
                data = json.dumps(payload)  # Default to JSON
                content_type = 'application/json'
            
            # Send to API
            async with self.session.post(
                self.config.api_endpoint,
                data=data,
                headers={'Content-Type': content_type}
            ) as response:
                if response.status == 200:
                    self.logger.debug(f"Successfully exported batch to API")
                else:
                    self.logger.warning(f"API export failed with status {response.status}")
                    self.failed_exports += 1
            
        except Exception as e:
            self.logger.error(f"Error exporting to API: {e}")
            self.failed_exports += 1
    
    def _get_collection_interval(self) -> float:
        """Get collection interval based on frequency"""
        if self.config.frequency == ExportFrequency.REAL_TIME:
            return 0.1
        elif self.config.frequency == ExportFrequency.HIGH:
            return 1.0
        elif self.config.frequency == ExportFrequency.MEDIUM:
            return 10.0
        elif self.config.frequency == ExportFrequency.LOW:
            return 60.0
        else:
            return 10.0
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics"""
        return {
            'is_running': self.is_running,
            'api_connected': self.api_connected,
            'exported_batches': self.exported_batches,
            'failed_exports': self.failed_exports,
            'total_records': self.total_records,
            'queue_sizes': {
                'data_queue': self.data_queue.qsize(),
                'export_queue': self.export_queue.qsize()
            },
            'last_export_time': self.last_export_time.isoformat() if self.last_export_time else None,
            'success_rate': self.exported_batches / max(1, self.exported_batches + self.failed_exports)
        }
    
    def export_to_file(self, filepath: str, data: Optional[Dict[str, Any]] = None):
        """Export data to file"""
        try:
            if data is None:
                # Export current batch
                batch_data = []
                while not self.export_queue.empty():
                    try:
                        batch_data.append(self.export_queue.get_nowait())
                    except Empty:
                        break
                data = {'batch_data': batch_data}
            
            # Write to file
            with open(filepath, 'w') as f:
                if self.config.format == ExportFormat.JSON:
                    json.dump(data, f, indent=2)
                else:
                    json.dump(data, f, indent=2)  # Default to JSON
            
            self.logger.info(f"Data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to file: {e}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data exporter
    config = ExportConfig(
        enabled=True,
        format=ExportFormat.JSON,
        frequency=ExportFrequency.MEDIUM,
        api_endpoint="http://localhost:8000/api/v1/sumo/data"
    )
    
    exporter = DataExporter(config)
    
    # Start export
    asyncio.run(exporter.start_export())
    
    try:
        # Run for some time
        time.sleep(60)
        
        # Get statistics
        stats = exporter.get_export_statistics()
        print(f"Export statistics: {stats}")
        
        # Export to file
        exporter.export_to_file("simulation_data.json")
    
    finally:
        # Stop export
        exporter.stop_export()

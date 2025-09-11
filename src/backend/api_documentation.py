"""
OpenAPI/Swagger documentation for Smart Traffic Management System API
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# API Models for Documentation
class TrafficLightModel(BaseModel):
    """Traffic Light Model"""
    id: str = Field(..., description="Unique identifier for the traffic light")
    name: str = Field(..., description="Human-readable name for the traffic light")
    location: Dict[str, float] = Field(..., description="Geographic coordinates (lat, lng)")
    status: str = Field(..., description="Current status: normal, maintenance, error")
    current_phase: int = Field(..., ge=0, le=3, description="Current traffic light phase (0-3)")
    phase_duration: int = Field(..., gt=0, description="Duration of current phase in seconds")
    program: str = Field(..., description="Control program: adaptive, fixed, manual")
    vehicle_count: int = Field(..., ge=0, description="Number of vehicles detected")
    waiting_time: float = Field(..., ge=0, description="Average waiting time in seconds")
    last_update: datetime = Field(..., description="Last update timestamp")

class VehicleModel(BaseModel):
    """Vehicle Model"""
    id: str = Field(..., description="Unique identifier for the vehicle")
    type: str = Field(..., description="Vehicle type: passenger, truck, bus, emergency")
    position: Dict[str, float] = Field(..., description="Current position (lat, lng)")
    speed: float = Field(..., ge=0, description="Current speed in km/h")
    lane: str = Field(..., description="Current lane identifier")
    route: List[str] = Field(..., description="Planned route through the intersection")
    waiting_time: float = Field(..., ge=0, description="Time spent waiting in seconds")
    co2_emission: float = Field(..., ge=0, description="CO2 emission in kg")
    fuel_consumption: float = Field(..., ge=0, description="Fuel consumption in liters")
    timestamp: datetime = Field(..., description="Data timestamp")

class IntersectionModel(BaseModel):
    """Intersection Model"""
    id: str = Field(..., description="Unique identifier for the intersection")
    name: str = Field(..., description="Human-readable name for the intersection")
    location: Dict[str, float] = Field(..., description="Geographic coordinates (lat, lng)")
    traffic_lights: List[str] = Field(..., description="List of traffic light IDs")
    total_vehicles: int = Field(..., ge=0, description="Total number of vehicles")
    waiting_vehicles: int = Field(..., ge=0, description="Number of waiting vehicles")
    average_speed: float = Field(..., ge=0, description="Average speed in km/h")
    throughput: int = Field(..., ge=0, description="Vehicles per hour")
    last_update: datetime = Field(..., description="Last update timestamp")

class PerformanceMetricsModel(BaseModel):
    """Performance Metrics Model"""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    total_vehicles: int = Field(..., ge=0, description="Total number of vehicles")
    running_vehicles: int = Field(..., ge=0, description="Number of running vehicles")
    waiting_vehicles: int = Field(..., ge=0, description="Number of waiting vehicles")
    total_waiting_time: float = Field(..., ge=0, description="Total waiting time in seconds")
    average_speed: float = Field(..., ge=0, description="Average speed in km/h")
    total_co2_emission: float = Field(..., ge=0, description="Total CO2 emission in kg")
    total_fuel_consumption: float = Field(..., ge=0, description="Total fuel consumption in liters")
    throughput: int = Field(..., ge=0, description="Vehicles per hour")

class OptimizationRequestModel(BaseModel):
    """Optimization Request Model"""
    intersection_id: str = Field(..., description="Intersection ID to optimize")
    algorithm: str = Field(..., description="Optimization algorithm: q_learning, dynamic_programming, websters_formula")
    traffic_data: Dict[str, Any] = Field(..., description="Current traffic data")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Algorithm-specific parameters")

class OptimizationResultModel(BaseModel):
    """Optimization Result Model"""
    optimized_phases: List[int] = Field(..., description="Optimized phase durations")
    efficiency_improvement: float = Field(..., description="Expected efficiency improvement (0-1)")
    waiting_time_reduction: float = Field(..., description="Expected waiting time reduction in seconds")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the optimization result")
    algorithm_used: str = Field(..., description="Algorithm used for optimization")
    processing_time: float = Field(..., description="Time taken to process optimization in seconds")

class SimulationRequestModel(BaseModel):
    """Simulation Request Model"""
    scenario: str = Field(..., description="Simulation scenario name")
    duration: int = Field(..., gt=0, description="Simulation duration in seconds")
    step_size: int = Field(..., gt=0, description="Simulation step size in seconds")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Simulation parameters")

class SimulationResultModel(BaseModel):
    """Simulation Result Model"""
    simulation_id: str = Field(..., description="Unique simulation identifier")
    status: str = Field(..., description="Simulation status: running, completed, error")
    start_time: datetime = Field(..., description="Simulation start time")
    end_time: Optional[datetime] = Field(None, description="Simulation end time")
    duration: int = Field(..., description="Actual simulation duration in seconds")
    results: Optional[Dict[str, Any]] = Field(None, description="Simulation results")

class ErrorResponseModel(BaseModel):
    """Error Response Model"""
    success: bool = Field(False, description="Request success status")
    error: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class SuccessResponseModel(BaseModel):
    """Success Response Model"""
    success: bool = Field(True, description="Request success status")
    data: Any = Field(..., description="Response data")
    timestamp: datetime = Field(..., description="Response timestamp")
    message: Optional[str] = Field(None, description="Optional success message")

class HealthCheckModel(BaseModel):
    """Health Check Model"""
    status: str = Field(..., description="System status: healthy, unhealthy, degraded")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")

class DetailedHealthCheckModel(BaseModel):
    """Detailed Health Check Model"""
    status: str = Field(..., description="System status: healthy, unhealthy, degraded")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health status")
    database: Dict[str, Any] = Field(..., description="Database health status")
    redis: Dict[str, Any] = Field(..., description="Redis health status")

def create_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """Create comprehensive OpenAPI schema for the Smart Traffic Management System"""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Smart Traffic Management System API",
        version="2.1.0",
        description="""
        # Smart Traffic Management System API
        
        A comprehensive API for intelligent traffic control and optimization.
        
        ## Features
        
        - **Real-time Traffic Control**: Dynamic traffic light optimization based on live traffic data
        - **Machine Learning Optimization**: AI-powered traffic flow optimization using multiple algorithms
        - **SUMO Simulation**: Realistic traffic simulation with configurable scenarios
        - **Performance Monitoring**: Comprehensive metrics and analytics
        - **System Management**: Health monitoring and system administration
        
        ## Authentication
        
        The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:
        
        ```
        Authorization: Bearer <your-jwt-token>
        ```
        
        ## Rate Limiting
        
        API requests are rate limited to prevent abuse:
        - **General endpoints**: 100 requests per minute
        - **Optimization endpoints**: 10 requests per minute
        - **Simulation endpoints**: 5 requests per minute
        
        ## Error Handling
        
        The API returns consistent error responses with the following structure:
        
        ```json
        {
            "success": false,
            "error": "Error message",
            "timestamp": "2024-01-01T00:00:00Z",
            "details": {
                "field": "Additional error details"
            }
        }
        ```
        
        ## Status Codes
        
        - **200**: Success
        - **201**: Created
        - **400**: Bad Request
        - **401**: Unauthorized
        - **403**: Forbidden
        - **404**: Not Found
        - **409**: Conflict
        - **422**: Validation Error
        - **429**: Rate Limit Exceeded
        - **500**: Internal Server Error
        
        ## Data Models
        
        ### Traffic Light
        Represents a traffic light at an intersection with current state and control parameters.
        
        ### Vehicle
        Represents a vehicle in the traffic system with position, speed, and environmental impact data.
        
        ### Intersection
        Represents a traffic intersection with multiple traffic lights and aggregate metrics.
        
        ### Performance Metrics
        Represents system-wide performance metrics including efficiency and environmental impact.
        
        ## Optimization Algorithms
        
        ### Q-Learning
        Reinforcement learning algorithm that learns optimal traffic light timing through trial and error.
        
        ### Dynamic Programming
        Mathematical optimization technique that finds optimal solutions by breaking down complex problems.
        
        ### Webster's Formula
        Classical traffic engineering formula for calculating optimal cycle times and phase splits.
        
        ## Simulation Scenarios
        
        ### Basic Intersection
        Simple four-way intersection with standard traffic patterns.
        
        ### Complex Intersection
        Multi-lane intersection with advanced traffic patterns and pedestrian crossings.
        
        ### Rush Hour
        High-density traffic scenario simulating peak commuting hours.
        
        ### Emergency Vehicle
        Scenario with emergency vehicle priority and special traffic patterns.
        
        ## WebSocket Events
        
        Real-time updates are available via WebSocket connections:
        
        - **traffic_update**: Traffic light state changes
        - **vehicle_update**: Vehicle position and status updates
        - **metrics_update**: Performance metrics updates
        - **optimization_result**: Optimization completion notifications
        - **alert**: System alerts and notifications
        
        ## Examples
        
        ### Basic Traffic Light Control
        
        ```python
        import requests
        
        # Get traffic light status
        response = requests.get("https://api.traffic-management.com/api/traffic/lights/light_1")
        light = response.json()["data"]
        
        # Control traffic light
        control_data = {
            "phase": 2,
            "duration": 30
        }
        response = requests.post(
            "https://api.traffic-management.com/api/traffic/lights/light_1/control",
            json=control_data
        )
        ```
        
        ### Optimization Request
        
        ```python
        optimization_data = {
            "intersection_id": "intersection_1",
            "algorithm": "q_learning",
            "traffic_data": {
                "vehicles": [...],
                "traffic_lights": [...]
            }
        }
        
        response = requests.post(
            "https://api.traffic-management.com/api/optimization/optimize",
            json=optimization_data
        )
        result = response.json()["data"]
        ```
        
        ### WebSocket Connection
        
        ```javascript
        const ws = new WebSocket('wss://api.traffic-management.com/ws');
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'traffic_update') {
                updateTrafficLight(data.data);
            }
        };
        ```
        """,
        routes=app.routes,
    )
    
    # Add custom OpenAPI extensions
    openapi_schema["info"]["contact"] = {
        "name": "Traffic Management Team",
        "email": "support@traffic-management.com",
        "url": "https://traffic-management.com/support"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    openapi_schema["info"]["x-logo"] = {
        "url": "https://traffic-management.com/logo.png",
        "altText": "Smart Traffic Management System"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "https://api.traffic-management.com",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.traffic-management.com",
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token for API authentication"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service-to-service authentication"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [
        {"BearerAuth": []},
        {"ApiKeyAuth": []}
    ]
    
    # Add tags for better organization
    openapi_schema["tags"] = [
        {
            "name": "Traffic Lights",
            "description": "Traffic light control and monitoring"
        },
        {
            "name": "Vehicles",
            "description": "Vehicle tracking and management"
        },
        {
            "name": "Intersections",
            "description": "Intersection management and metrics"
        },
        {
            "name": "Performance",
            "description": "Performance metrics and analytics"
        },
        {
            "name": "Optimization",
            "description": "Traffic optimization algorithms"
        },
        {
            "name": "Simulation",
            "description": "SUMO traffic simulation"
        },
        {
            "name": "System",
            "description": "System health and administration"
        },
        {
            "name": "WebSocket",
            "description": "Real-time WebSocket events"
        }
    ]
    
    # Add examples for common responses
    openapi_schema["components"]["examples"] = {
        "TrafficLightExample": {
            "summary": "Traffic Light Example",
            "value": {
                "id": "light_1",
                "name": "Main Street & First Avenue",
                "location": {"lat": 40.7128, "lng": -74.0060},
                "status": "normal",
                "current_phase": 0,
                "phase_duration": 30,
                "program": "adaptive",
                "vehicle_count": 15,
                "waiting_time": 45.5,
                "last_update": "2024-01-01T12:00:00Z"
            }
        },
        "VehicleExample": {
            "summary": "Vehicle Example",
            "value": {
                "id": "vehicle_1",
                "type": "passenger",
                "position": {"lat": 40.7128, "lng": -74.0060},
                "speed": 25.5,
                "lane": "north_approach_0",
                "route": ["north_approach", "center_junction", "south_exit"],
                "waiting_time": 5.2,
                "co2_emission": 0.1,
                "fuel_consumption": 0.05,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        },
        "OptimizationResultExample": {
            "summary": "Optimization Result Example",
            "value": {
                "optimized_phases": [30, 25, 35, 20],
                "efficiency_improvement": 0.15,
                "waiting_time_reduction": 12.5,
                "confidence": 0.85,
                "algorithm_used": "q_learning",
                "processing_time": 2.3
            }
        }
    }
    
    # Add response schemas
    openapi_schema["components"]["schemas"].update({
        "TrafficLight": TrafficLightModel.schema(),
        "Vehicle": VehicleModel.schema(),
        "Intersection": IntersectionModel.schema(),
        "PerformanceMetrics": PerformanceMetricsModel.schema(),
        "OptimizationRequest": OptimizationRequestModel.schema(),
        "OptimizationResult": OptimizationResultModel.schema(),
        "SimulationRequest": SimulationRequestModel.schema(),
        "SimulationResult": SimulationResultModel.schema(),
        "ErrorResponse": ErrorResponseModel.schema(),
        "SuccessResponse": SuccessResponseModel.schema(),
        "HealthCheck": HealthCheckModel.schema(),
        "DetailedHealthCheck": DetailedHealthCheckModel.schema()
    })
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def create_api_documentation_routes(app: FastAPI):
    """Create additional API documentation routes"""
    
    @app.get("/api/docs/health", 
             summary="API Health Check",
             description="Check the health status of the API documentation system",
             tags=["System"])
    async def api_docs_health():
        """Check API documentation health"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0",
            "endpoints": len(app.routes),
            "models": len(app.openapi_schema.get("components", {}).get("schemas", {}))
        }
    
    @app.get("/api/docs/endpoints",
             summary="List All Endpoints",
             description="Get a list of all available API endpoints",
             tags=["System"])
    async def list_endpoints():
        """List all available endpoints"""
        endpoints = []
        for route in app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                endpoints.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": getattr(route, 'name', None),
                    "summary": getattr(route, 'summary', None)
                })
        return {
            "success": True,
            "data": endpoints,
            "timestamp": datetime.now().isoformat(),
            "total": len(endpoints)
        }
    
    @app.get("/api/docs/models",
             summary="List All Models",
             description="Get a list of all available data models",
             tags=["System"])
    async def list_models():
        """List all available data models"""
        schemas = app.openapi_schema.get("components", {}).get("schemas", {})
        models = []
        for name, schema in schemas.items():
            models.append({
                "name": name,
                "type": schema.get("type", "object"),
                "properties": list(schema.get("properties", {}).keys()),
                "required": schema.get("required", [])
            })
        return {
            "success": True,
            "data": models,
            "timestamp": datetime.now().isoformat(),
            "total": len(models)
        }
    
    @app.get("/api/docs/examples/{model_name}",
             summary="Get Model Examples",
             description="Get example data for a specific model",
             tags=["System"])
    async def get_model_examples(model_name: str):
        """Get examples for a specific model"""
        examples = app.openapi_schema.get("components", {}).get("examples", {})
        model_examples = {}
        
        for example_name, example_data in examples.items():
            if model_name.lower() in example_name.lower():
                model_examples[example_name] = example_data
        
        if not model_examples:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"No examples found for model: {model_name}",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        return {
            "success": True,
            "data": model_examples,
            "timestamp": datetime.now().isoformat()
        }

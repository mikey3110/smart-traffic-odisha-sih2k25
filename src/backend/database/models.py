"""
SQLAlchemy models for the Smart Traffic Management System
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any

from .connection import Base


class Intersection(Base):
    """
    Intersection model for storing intersection information
    """
    __tablename__ = "intersections"
    
    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(200), nullable=True)
    location_lat = Column(Float, nullable=True)
    location_lng = Column(Float, nullable=True)
    lanes = Column(JSON, nullable=False)  # List of lane identifiers
    status = Column(String(20), default="operational", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    traffic_data = relationship("TrafficData", back_populates="intersection", cascade="all, delete-orphan")
    signal_timings = relationship("SignalTiming", back_populates="intersection", cascade="all, delete-orphan")
    optimization_results = relationship("OptimizationResult", back_populates="intersection", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_intersection_status', 'status'),
        Index('idx_intersection_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Intersection(id='{self.id}', name='{self.name}', status='{self.status}')>"


class TrafficData(Base):
    """
    Traffic data model for storing vehicle counts and traffic information
    """
    __tablename__ = "traffic_data"
    
    id = Column(Integer, primary_key=True, index=True)
    intersection_id = Column(String(50), ForeignKey("intersections.id"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    lane_counts = Column(JSON, nullable=False)  # Dict of lane -> count
    avg_speed = Column(Float, nullable=True)
    weather_condition = Column(String(20), nullable=True)
    vehicle_types = Column(JSON, nullable=True)  # Dict of vehicle_type -> count
    confidence_score = Column(Float, nullable=True)
    data_source = Column(String(50), default="camera", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    intersection = relationship("Intersection", back_populates="traffic_data")
    
    # Indexes
    __table_args__ = (
        Index('idx_traffic_intersection_timestamp', 'intersection_id', 'timestamp'),
        Index('idx_traffic_timestamp', 'timestamp'),
        Index('idx_traffic_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<TrafficData(id={self.id}, intersection_id='{self.intersection_id}', timestamp='{self.timestamp}')>"


class SignalTiming(Base):
    """
    Signal timing model for storing traffic light configurations
    """
    __tablename__ = "signal_timings"
    
    id = Column(Integer, primary_key=True, index=True)
    intersection_id = Column(String(50), ForeignKey("intersections.id"), nullable=False, index=True)
    lane = Column(String(50), nullable=False)
    duration = Column(Integer, nullable=False)  # Duration in seconds
    state = Column(String(20), nullable=False)  # red, yellow, green, etc.
    priority = Column(Integer, default=2, nullable=False)  # Priority level
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    intersection = relationship("Intersection", back_populates="signal_timings")
    
    # Indexes
    __table_args__ = (
        Index('idx_signal_intersection_lane', 'intersection_id', 'lane'),
        Index('idx_signal_active', 'is_active'),
        Index('idx_signal_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<SignalTiming(id={self.id}, intersection_id='{self.intersection_id}', lane='{self.lane}', state='{self.state}')>"


class OptimizationResult(Base):
    """
    Optimization result model for storing AI optimization outcomes
    """
    __tablename__ = "optimization_results"
    
    id = Column(Integer, primary_key=True, index=True)
    intersection_id = Column(String(50), ForeignKey("intersections.id"), nullable=False, index=True)
    algorithm_used = Column(String(100), nullable=True)
    confidence_score = Column(Float, nullable=False)
    expected_improvement = Column(Float, nullable=True)
    optimization_time = Column(Float, nullable=True)  # Time taken in seconds
    optimized_timings = Column(JSON, nullable=False)  # Dict of lane -> timing
    status = Column(String(20), default="pending", nullable=False)  # pending, completed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    applied_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    intersection = relationship("Intersection", back_populates="optimization_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_optimization_intersection_status', 'intersection_id', 'status'),
        Index('idx_optimization_created', 'created_at'),
        Index('idx_optimization_confidence', 'confidence_score'),
    )
    
    def __repr__(self):
        return f"<OptimizationResult(id={self.id}, intersection_id='{self.intersection_id}', status='{self.status}')>"


class SystemEvent(Base):
    """
    System event model for logging and monitoring
    """
    __tablename__ = "system_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    intersection_id = Column(String(50), nullable=True, index=True)
    message = Column(Text, nullable=False)
    level = Column(String(20), default="INFO", nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    details = Column(JSON, nullable=True)  # Additional event details
    request_id = Column(String(100), nullable=True, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_event_type_created', 'event_type', 'created_at'),
        Index('idx_event_intersection_created', 'intersection_id', 'created_at'),
        Index('idx_event_level_created', 'level', 'created_at'),
        Index('idx_event_request', 'request_id'),
    )
    
    def __repr__(self):
        return f"<SystemEvent(id={self.id}, event_type='{self.event_type}', level='{self.level}')>"


class APILog(Base):
    """
    API log model for tracking API requests and responses
    """
    __tablename__ = "api_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    path = Column(String(500), nullable=False)
    query_params = Column(JSON, nullable=True)
    headers = Column(JSON, nullable=True)
    client_ip = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    user_id = Column(String(100), nullable=True, index=True)
    status_code = Column(Integer, nullable=False, index=True)
    response_time = Column(Float, nullable=True)  # Response time in seconds
    request_body = Column(Text, nullable=True)
    response_body = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_api_request_id', 'request_id'),
        Index('idx_api_method_path', 'method', 'path'),
        Index('idx_api_status_created', 'status_code', 'created_at'),
        Index('idx_api_user_created', 'user_id', 'created_at'),
        Index('idx_api_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<APILog(id={self.id}, method='{self.method}', path='{self.path}', status_code={self.status_code})>"


class HealthCheck(Base):
    """
    Health check model for system monitoring
    """
    __tablename__ = "health_checks"
    
    id = Column(Integer, primary_key=True, index=True)
    component = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # healthy, degraded, unhealthy
    message = Column(Text, nullable=True)
    response_time = Column(Float, nullable=True)  # Response time in seconds
    details = Column(JSON, nullable=True)  # Component-specific details
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_health_component_created', 'component', 'created_at'),
        Index('idx_health_status_created', 'status', 'created_at'),
        Index('idx_health_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<HealthCheck(id={self.id}, component='{self.component}', status='{self.status}')>"


class Configuration(Base):
    """
    Configuration model for storing system settings
    """
    __tablename__ = "configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    value_type = Column(String(20), default="string", nullable=False)  # string, int, float, bool, json
    description = Column(Text, nullable=True)
    is_sensitive = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_config_key', 'key'),
        Index('idx_config_sensitive', 'is_sensitive'),
    )
    
    def __repr__(self):
        return f"<Configuration(id={self.id}, key='{self.key}', type='{self.value_type}')>"
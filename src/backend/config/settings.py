"""
Enhanced configuration management using Pydantic Settings
Handles environment variables, validation, and type conversion
"""

from pydantic import BaseSettings, Field, validator
from typing import Optional, List
import os
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    url: str = Field(
        default="sqlite:///./traffic_management.db",
        description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Enable SQLAlchemy query logging")
    pool_size: int = Field(default=10, description="Database connection pool size")
    max_overflow: int = Field(default=20, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Connection pool timeout")
    pool_recycle: int = Field(default=3600, description="Connection recycle time")
    
    class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis configuration settings"""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    max_connections: int = Field(default=20, description="Redis connection pool size")
    socket_timeout: int = Field(default=5, description="Redis socket timeout")
    socket_connect_timeout: int = Field(default=5, description="Redis connection timeout")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    health_check_interval: int = Field(default=30, description="Health check interval")
    
    @property
    def url(self) -> str:
        """Generate Redis URL from settings"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"


class APISettings(BaseSettings):
    """API configuration settings"""
    title: str = Field(default="Smart Traffic Management API", description="API title")
    description: str = Field(
        default="AI-based traffic signal optimization system for SIH 2025",
        description="API description"
    )
    version: str = Field(default="1.0.0", description="API version")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins"
    )
    cors_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    class Config:
        env_prefix = "API_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings"""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file_path: Optional[str] = Field(default="logs/traffic_api.log", description="Log file path")
    max_file_size: int = Field(default=10485760, description="Max log file size (10MB)")
    backup_count: int = Field(default=5, description="Number of backup log files")
    enable_console: bool = Field(default=True, description="Enable console logging")
    enable_file: bool = Field(default=True, description="Enable file logging")
    
    @validator('level')
    def validate_log_level(cls, v):
        """Validate logging level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    class Config:
        env_prefix = "LOG_"


class SecuritySettings(BaseSettings):
    """Security configuration settings"""
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time"
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration time"
    )
    
    class Config:
        env_prefix = "SECURITY_"


class Settings(BaseSettings):
    """Main application settings"""
    # Environment
    environment: str = Field(default="development", description="Environment name")
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    security: SecuritySettings = SecuritySettings()
    
    # Application settings
    app_name: str = Field(default="Smart Traffic Management", description="Application name")
    timezone: str = Field(default="UTC", description="Application timezone")
    
    # Feature flags
    enable_redis: bool = Field(default=True, description="Enable Redis caching")
    enable_database: bool = Field(default=True, description="Enable database storage")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment name"""
        valid_envs = ['development', 'staging', 'production', 'testing']
        if v.lower() not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v.lower()
    
    def create_directories(self):
        """Create necessary directories"""
        if self.logging.file_path:
            log_dir = Path(self.logging.file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Create necessary directories
settings.create_directories()
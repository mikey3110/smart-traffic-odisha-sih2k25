"""
Database connection management with SQLAlchemy
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from contextlib import contextmanager
import logging
from typing import Generator, Optional

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger("database_connection")

# Create declarative base
Base = declarative_base()

# Global database engine and session factory
engine: Optional[Engine] = None
SessionLocal: Optional[sessionmaker] = None


def create_database_engine() -> Engine:
    """
    Create and configure SQLAlchemy engine with connection pooling
    """
    global engine
    
    if engine is not None:
        return engine
    
    try:
        # Create engine with connection pooling
        engine = create_engine(
            settings.database.url,
            echo=settings.database.echo,
            poolclass=QueuePool,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            pool_recycle=settings.database.pool_recycle,
            pool_pre_ping=True,  # Verify connections before use
            connect_args={
                "check_same_thread": False  # For SQLite compatibility
            } if "sqlite" in settings.database.url else {}
        )
        
        logger.info(
            "Database engine created successfully",
            extra={
                'database_url': settings.database.url.split('@')[-1] if '@' in settings.database.url else settings.database.url,
                'pool_size': settings.database.pool_size,
                'max_overflow': settings.database.max_overflow,
                'echo': settings.database.echo
            }
        )
        
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


def create_session_factory() -> sessionmaker:
    """
    Create SQLAlchemy session factory
    """
    global SessionLocal
    
    if SessionLocal is not None:
        return SessionLocal
    
    if engine is None:
        create_database_engine()
    
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
    
    logger.info("Database session factory created")
    return SessionLocal


def get_database_session() -> Generator[Session, None, None]:
    """
    Get database session with automatic cleanup
    """
    if SessionLocal is None:
        create_session_factory()
    
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_database_session_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions
    """
    if SessionLocal is None:
        create_session_factory()
    
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def init_database():
    """
    Initialize database tables
    """
    try:
        if engine is None:
            create_database_engine()
        
        # Import all models to ensure they are registered
        from .models import (
            Intersection, TrafficData, SignalTiming, 
            OptimizationResult, SystemEvent, APILog
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def health_check() -> dict:
    """
    Check database health
    """
    try:
        if engine is None:
            return {
                "status": "disconnected",
                "error": "Database engine not initialized"
            }
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            result.fetchone()
        
        # Get connection pool info
        pool = engine.pool
        pool_info = {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
        
        return {
            "status": "connected",
            "pool_info": pool_info,
            "database_url": settings.database.url.split('@')[-1] if '@' in settings.database.url else settings.database.url
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def close_database():
    """
    Close database connections
    """
    global engine, SessionLocal
    
    if engine:
        engine.dispose()
        engine = None
    
    SessionLocal = None
    logger.info("Database connections closed")


# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions
    """
    return get_database_session()


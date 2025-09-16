"""
Enhanced database configuration and connection management
for Smart Traffic Management System using SQLAlchemy
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config.logging_config import get_logger
from config.settings import settings
from exceptions import DatabaseError

logger = get_logger(__name__)

# Create declarative base
Base = declarative_base()

# Metadata for table management
metadata = MetaData()


class DatabaseManager:
    """
    Enhanced database manager with connection pooling and error handling
    """
    
    def __init__(self, database_url: str = None, echo: bool = None):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection URL
            echo: Whether to echo SQL statements
        """
        self.database_url = database_url or settings.database_url
        self.echo = echo if echo is not None else settings.database_echo
        
        # Engine and session factory
        self.engine: Optional[Engine] = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        
        # Connection pool settings
        self.pool_settings = {
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True,
            "pool_recycle": 3600,  # 1 hour
            "pool_timeout": 30
        }
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "queries_executed": 0,
            "last_connection_time": None
        }
    
    def initialize(self):
        """Initialize database engine and session factory"""
        try:
            # Create synchronous engine
            self.engine = create_engine(
                self.database_url,
                echo=self.echo,
                poolclass=QueuePool,
                **self.pool_settings
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False
            )
            
            # Add connection event listeners
            self._add_connection_listeners()
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            self.stats["last_connection_time"] = asyncio.get_event_loop().time()
            logger.info(f"✅ Database initialized successfully: {self.database_url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            self.stats["failed_connections"] += 1
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    async def initialize_async(self):
        """Initialize async database engine and session factory"""
        try:
            # Convert sync URL to async URL
            async_url = self._convert_to_async_url(self.database_url)
            
            # Create async engine
            self.async_engine = create_async_engine(
                async_url,
                echo=self.echo,
                poolclass=QueuePool,
                **self.pool_settings
            )
            
            # Create async session factory
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test async connection
            async with self.async_engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            self.stats["last_connection_time"] = asyncio.get_event_loop().time()
            logger.info(f"✅ Async database initialized successfully: {async_url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize async database: {e}")
            self.stats["failed_connections"] += 1
            raise DatabaseError(f"Failed to initialize async database: {str(e)}")
    
    def _convert_to_async_url(self, sync_url: str) -> str:
        """
        Convert synchronous database URL to asynchronous URL
        
        Args:
            sync_url: Synchronous database URL
        
        Returns:
            Asynchronous database URL
        """
        if sync_url.startswith("sqlite:///"):
            return sync_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        elif sync_url.startswith("postgresql://"):
            return sync_url.replace("postgresql://", "postgresql+asyncpg://")
        elif sync_url.startswith("mysql://"):
            return sync_url.replace("mysql://", "mysql+aiomysql://")
        else:
            # Default to sqlite+aiosqlite for unsupported databases
            return sync_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    
    def _add_connection_listeners(self):
        """Add connection event listeners for monitoring"""
        
        @event.listens_for(self.engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Called when a new connection is created"""
            self.stats["total_connections"] += 1
            self.stats["active_connections"] += 1
            logger.debug("New database connection created")
        
        @event.listens_for(self.engine, "close")
        def on_close(dbapi_connection, connection_record):
            """Called when a connection is closed"""
            self.stats["active_connections"] -= 1
            logger.debug("Database connection closed")
        
        @event.listens_for(self.engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Called when a connection is checked out from the pool"""
            logger.debug("Database connection checked out from pool")
        
        @event.listens_for(self.engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Called when a connection is checked in to the pool"""
            logger.debug("Database connection checked in to pool")
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with automatic cleanup
        
        Yields:
            AsyncSession: Database session
        """
        if not self.async_session_factory:
            raise DatabaseError("Async database not initialized")
        
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            await session.close()
    
    def get_session(self):
        """
        Get synchronous database session
        
        Returns:
            Database session
        """
        if not self.session_factory:
            raise DatabaseError("Database not initialized")
        
        return self.session_factory()
    
    async def create_tables(self):
        """Create all database tables"""
        try:
            if self.async_engine:
                async with self.async_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created successfully")
            else:
                Base.metadata.create_all(bind=self.engine)
                logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise DatabaseError(f"Failed to create database tables: {str(e)}")
    
    async def drop_tables(self):
        """Drop all database tables"""
        try:
            if self.async_engine:
                async with self.async_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.drop_all)
                logger.info("Database tables dropped successfully")
            else:
                Base.metadata.drop_all(bind=self.engine)
                logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise DatabaseError(f"Failed to drop database tables: {str(e)}")
    
    async def health_check(self) -> dict:
        """
        Perform database health check
        
        Returns:
            Health check results
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            if self.async_engine:
                async with self.async_engine.begin() as conn:
                    await conn.execute("SELECT 1")
            else:
                with self.engine.connect() as conn:
                    conn.execute("SELECT 1")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "connected": True,
                "response_time_ms": round(response_time, 2),
                "stats": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "stats": self.stats.copy()
            }
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        return self.stats.copy()
    
    async def close(self):
        """Close database connections"""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
            if self.engine:
                self.engine.dispose()
            logger.info("Database connections closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Global database manager instance
db_manager = DatabaseManager()




"""Database connection management for API key system."""

import os
import logging
from typing import Generator, Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..config import get_config
from .models import Base

logger = logging.getLogger(__name__)

# Global database engine and session maker
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_database_url() -> str:
    """Get database URL from configuration."""
    config = get_config()
    
    # Check for API key database URL first
    db_url = os.getenv("API_KEY_DATABASE_URL")
    if db_url:
        return db_url
    
    # Fallback based on deployment mode
    if config.rate_limit.deployment_mode.value == "local":
        # Use SQLite for local development
        return "sqlite:///./api_keys.db"
    else:
        # Use PostgreSQL for production
        # Format: postgresql://user:password@host:port/database
        return os.getenv(
            "DATABASE_URL", 
            "postgresql://kokoro:password@localhost:5432/kokorotts"
        )


def create_database_engine() -> Engine:
    """Create database engine with appropriate configuration."""
    database_url = get_database_url()
    
    if database_url.startswith("sqlite"):
        # SQLite configuration for local development
        engine = create_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            },
            echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
        )
    else:
        # PostgreSQL configuration for production
        engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
        )
    
    logger.info(f"Created database engine for: {database_url.split('://')[0]}")
    return engine


def initialize_database() -> None:
    """Initialize database connection and create tables."""
    global _engine, _SessionLocal
    
    try:
        _engine = create_database_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=_engine)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # In local development, we should continue without API key features
        config = get_config()
        if config.rate_limit.deployment_mode.value == "local":
            logger.warning("API key database unavailable - continuing without API key features")
        else:
            raise


def get_session() -> Generator[Session, None, None]:
    """Get database session with automatic cleanup."""
    if _SessionLocal is None:
        initialize_database()
    
    if _SessionLocal is None:
        # Database initialization failed - return None to disable API key features
        yield None
        return
    
    session = _SessionLocal()
    try:
        yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def close_database() -> None:
    """Close database connection."""
    global _engine
    if _engine:
        _engine.dispose()
        _engine = None
        logger.info("Database connection closed")


def is_database_available() -> bool:
    """Check if database is available and functioning."""
    try:
        with next(get_session()) as session:
            if session is None:
                return False
            # Simple query to test connection
            session.execute("SELECT 1")
            return True
    except Exception as e:
        logger.debug(f"Database availability check failed: {e}")
        return False


def ensure_database_initialized() -> bool:
    """Ensure database is initialized, initialize if needed."""
    global _engine, _SessionLocal
    
    if _engine is None or _SessionLocal is None:
        try:
            initialize_database()
            return _engine is not None and _SessionLocal is not None
        except Exception as e:
            logger.error(f"Failed to ensure database initialization: {e}")
            return False
    
    return True


# Database health check for monitoring
def get_database_health() -> dict:
    """Get database health status for monitoring endpoints."""
    try:
        if not ensure_database_initialized():
            return {
                "status": "unavailable",
                "error": "Database not initialized",
                "api_key_features": False
            }
        
        with next(get_session()) as session:
            if session is None:
                return {
                    "status": "unavailable", 
                    "error": "Session creation failed",
                    "api_key_features": False
                }
            
            # Test query
            result = session.execute("SELECT 1").scalar()
            
            return {
                "status": "healthy",
                "database_type": get_database_url().split("://")[0],
                "api_key_features": True,
                "query_test": result == 1
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "api_key_features": False
        }
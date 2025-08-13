"""
Database connection and session management for EduPulse Analytics.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
import os
from typing import Generator

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://edupulse_user:development_password@localhost:5432/edupulse"
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

@contextmanager
def get_db() -> Generator:
    """
    Context manager for database sessions.
    Ensures proper cleanup of database connections.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize database tables.
    This should be called once during application startup.
    """
    Base.metadata.create_all(bind=engine)
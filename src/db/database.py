"""
@fileoverview Database connection management with SQLAlchemy and session handling
@lastmodified 2025-08-13T00:50:05-05:00

Features: Engine creation, session factory, context manager, connection pooling
Main APIs: get_db(), init_db(), SessionLocal, Base, engine
Constraints: Requires PostgreSQL URL, SQLAlchemy, connection pool settings
Patterns: Context manager for sessions, declarative base, pool pre-ping for health
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
    Context manager for database sessions with automatic cleanup.
    
    Provides a database session that is automatically closed after use,
    ensuring proper connection pool management and preventing connection
    leaks. Should be used for all database operations.
    
    Yields:
        Session: SQLAlchemy database session for queries and transactions
        
    Examples:
        >>> with get_db() as db:
        ...     students = db.query(Student).all()
        ...     print(f"Found {len(students)} students")
        
        >>> # Or as dependency injection in FastAPI
        >>> def get_students(db: Session = Depends(get_db)):
        ...     return db.query(Student).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize database schema by creating all tables.
    
    Creates all database tables defined in the SQLAlchemy models using
    the declarative base metadata. Should be called once during application
    startup or when setting up a new database instance.
    
    Note:
        This function is idempotent - it will only create tables that don't
        already exist. Existing tables and data are not affected.
        
    Examples:
        >>> # During application startup
        >>> init_db()
        >>> print("Database tables created successfully")
        
        >>> # In tests or development setup
        >>> from src.db.database import init_db
        >>> init_db()  # Ensures all tables exist
    """
    Base.metadata.create_all(bind=engine)
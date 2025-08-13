"""
@fileoverview Health check and readiness endpoints for monitoring infrastructure
@lastmodified 2025-08-13T00:50:05-05:00

Features: Basic health check, readiness probe, database/Redis connectivity checks
Main APIs: health_check(), readiness_check()
Constraints: Requires FastAPI router, database session, Redis connection, settings
Patterns: Kubernetes health/readiness probes, dependency injection, exception handling
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import redis

from src.db.database import get_db
from src.config.settings import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint for infrastructure monitoring.
    
    Provides a lightweight endpoint to verify the API service is running
    and responding to requests. Used by load balancers and monitoring systems.
    
    Returns:
        dict: Health status response containing status, timestamp, and environment
            - status: Always "healthy" if the service is responding
            - timestamp: UTC timestamp of the health check
            - environment: Current deployment environment
        
    Examples:
        >>> response = await health_check()
        >>> print(response["status"])
        healthy
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment
    }


@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness check for Kubernetes deployments and service dependencies.
    
    Performs health checks on critical service dependencies (database and Redis)
    to determine if the service is ready to handle traffic. Used by Kubernetes
    readiness probes to control traffic routing.
    
    Args:
        db: Database session dependency for connectivity testing
        
    Returns:
        dict: Readiness status response containing:
            - status: "ready" if all checks pass, "not ready" otherwise
            - checks: Individual check results for database and redis
            - timestamp: UTC timestamp of the readiness check
            
    Examples:
        >>> response = await readiness_check(db_session)
        >>> print(response["status"])
        ready
        >>> print(response["checks"]["database"])
        True
    """
    checks = {
        "database": False,
        "redis": False,
        "ready": False
    }
    
    # Check database
    try:
        db.execute("SELECT 1")
        checks["database"] = True
    except Exception as e:
        pass
    
    # Check Redis
    try:
        r = redis.from_url(str(settings.redis_url))
        r.ping()
        checks["redis"] = True
    except Exception as e:
        pass
    
    checks["ready"] = checks["database"] and checks["redis"]
    
    return {
        "status": "ready" if checks["ready"] else "not ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
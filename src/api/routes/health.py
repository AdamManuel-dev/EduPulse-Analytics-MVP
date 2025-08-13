"""
Health check endpoints for monitoring and readiness.
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
    Basic health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment
    }


@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness check for Kubernetes deployments.
    Checks database and Redis connectivity.
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
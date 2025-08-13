"""
@fileoverview Main FastAPI application for EduPulse Analytics MVP
@lastmodified 2025-08-13T00:50:05-05:00

Features: API initialization, CORS setup, middleware, route mounting, metrics endpoint
Main APIs: lifespan(), root(), app instance configuration
Constraints: Requires DATABASE_URL, REDIS_URL environment variables
Patterns: Uses lifespan context for startup/shutdown, structured logging throughout
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
from prometheus_client import make_asgi_app

from src.config.settings import get_settings
from src.api.routes import health, predictions, students, training
from src.db.database import engine, Base

settings = get_settings()


# Configure structured logging
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.
    """
    # Startup
    logger.info("Starting EduPulse API", environment=settings.environment)

    # Initialize database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down EduPulse API")


# Create FastAPI application
app = FastAPI(
    title="EduPulse Analytics API",
    description="Temporal ML system for K-12 student success monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(students.router, prefix="/api/v1/students", tags=["students"])
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(training.router, prefix="/api/v1/train", tags=["training"])


@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {"message": "EduPulse Analytics API", "version": "1.0.0", "docs": "/docs"}

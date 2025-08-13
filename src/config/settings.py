"""
@fileoverview Application configuration management with Pydantic validation
@lastmodified 2025-08-13T00:50:05-05:00

Features: Environment-based config, database/Redis/Celery settings, ML parameters, validation
Main APIs: Settings(), get_settings(), parse_cors_origins(), validate_environment()
Constraints: Requires .env file, PostgresDsn, RedisDsn, SECRET_KEY, JWT_SECRET_KEY
Patterns: LRU cached singleton, field validators, environment mode properties
"""

from functools import lru_cache
from typing import List, Annotated

from pydantic import Field, PostgresDsn, RedisDsn, field_validator, BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors_origins(v):
    """Parse CORS origins from comma-separated string or list."""
    if v is None or v == "":
        return ["http://localhost:3000"]  # Default fallback
    if isinstance(v, str):
        # Handle comma-separated string
        origins = [origin.strip() for origin in v.split(",") if origin.strip()]
        return origins if origins else ["http://localhost:3000"]
    if isinstance(v, list):
        return v
    # If it's some other type, try to convert to string first
    try:
        return [str(v).strip()] if str(v).strip() else ["http://localhost:3000"]
    except Exception:
        return ["http://localhost:3000"]


CorsOrigins = Annotated[List[str], BeforeValidator(parse_cors_origins)]


class Settings(BaseSettings):
    """
    Main application configuration with environment-based validation.

    Provides comprehensive configuration management for the EduPulse application
    using Pydantic for validation and type safety. Settings are loaded from
    environment variables with fallback defaults.

    Configuration categories include:
        - Application: Environment, debug mode, logging
        - API: Host, port, CORS, rate limiting
        - Database: PostgreSQL connection and pooling
        - ML: Model parameters and training settings
        - Infrastructure: Redis, Celery, monitoring

    Examples:
        >>> settings = Settings()
        >>> print(f"Running in {settings.environment} mode")
        Running in development mode
        >>> print(f"Debug enabled: {settings.debug}")
        Debug enabled: False
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from environment
        protected_namespaces=("settings_",),  # Avoid model_ namespace warnings
    )

    # Application
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    secret_key: str = Field(..., description="Application secret key")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_version: str = Field(default="v1", description="API version")
    api_prefix: str = Field(default="/api", description="API prefix")
    cors_origins_raw: str = Field(
        default="http://localhost:3000",
        description="CORS allowed origins (comma-separated)",
        alias="CORS_ORIGINS"
    )
    api_rate_limit: int = Field(default=100, description="API rate limit per minute")

    # Database
    database_url: PostgresDsn
    db_pool_size: int = Field(default=20, description="Database connection pool size")
    db_max_overflow: int = Field(default=40, description="Maximum overflow connections")
    db_pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    db_echo: bool = Field(default=False, description="Echo SQL statements")

    # TimescaleDB
    timescale_chunk_time_interval: str = Field(
        default="7d", description="TimescaleDB chunk time interval"
    )
    timescale_compression_after: str = Field(
        default="30d", description="Compress data after this period"
    )

    # Redis
    redis_url: RedisDsn
    redis_max_connections: int = Field(default=50, description="Redis max connections")
    cache_ttl: int = Field(default=3600, description="Default cache TTL in seconds")

    # Celery
    celery_broker_url: RedisDsn
    celery_result_backend: RedisDsn
    celery_task_time_limit: int = Field(default=3600, description="Task time limit in seconds")
    celery_task_soft_time_limit: int = Field(default=3300, description="Soft time limit in seconds")

    # ML Model
    model_path: str = Field(default="/app/models", description="Model storage path")
    model_version: str = Field(default="latest", description="Model version to use")
    model_device: str = Field(default="cpu", description="Device for model inference")
    model_batch_size: int = Field(default=32, description="Batch size for inference")
    model_max_sequence_length: int = Field(default=365, description="Maximum sequence length")
    model_learning_rate: float = Field(default=0.001, description="Learning rate")
    model_epochs: int = Field(default=100, description="Training epochs")
    model_early_stopping_patience: int = Field(default=10, description="Early stopping patience")

    # Feature Engineering
    feature_window_days: int = Field(default=90, description="Feature calculation window in days")
    feature_lag_days: int = Field(default=7, description="Feature lag in days")
    feature_cache_enabled: bool = Field(default=True, description="Enable feature caching")
    feature_cache_ttl: int = Field(default=86400, description="Feature cache TTL in seconds")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", description="MLflow tracking URI"
    )
    mlflow_experiment_name: str = Field(
        default="edupulse-experiments", description="MLflow experiment name"
    )

    # JWT
    jwt_secret_key: str = Field(..., description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(
        default=30, description="Access token expiry in minutes"
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7, description="Refresh token expiry in days"
    )

    # Data Ingestion
    data_upload_max_size_mb: int = Field(default=100, description="Maximum upload size in MB")
    data_batch_size: int = Field(default=1000, description="Data processing batch size")
    data_validation_enabled: bool = Field(default=True, description="Enable data validation")

    # Monitoring
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")

    # Logging
    log_format: str = Field(default="json", description="Log format")
    log_file_path: str = Field(default="/app/logs/edupulse.log", description="Log file path")
    log_max_bytes: int = Field(default=10485760, description="Max log file size")
    log_backup_count: int = Field(default=5, description="Number of log backups")

    # Performance
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    worker_connections: int = Field(default=1000, description="Worker connections")

    # Feature Flags
    enable_async_predictions: bool = Field(default=True, description="Enable async predictions")
    enable_continuous_learning: bool = Field(default=True, description="Enable continuous learning")
    enable_model_interpretability: bool = Field(
        default=True, description="Enable model interpretability"
    )

    # Resource Limits
    max_prediction_batch_size: int = Field(default=100, description="Maximum prediction batch size")
    max_concurrent_tasks: int = Field(default=10, description="Maximum concurrent tasks")


    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        allowed = ["development", "staging", "production", "testing"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from the raw string."""
        return parse_cors_origins(self.cors_origins_raw)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance using LRU cache for performance.

    Creates a singleton-like pattern for settings access while maintaining
    the ability to refresh configuration if needed. The LRU cache ensures
    settings are only parsed once per application lifecycle.

    Returns:
        Settings: Validated and cached settings instance

    Examples:
        >>> settings = get_settings()
        >>> settings2 = get_settings()  # Returns same cached instance
        >>> print(settings is settings2)
        True
    """
    return Settings()


# Global settings instance
settings = get_settings()

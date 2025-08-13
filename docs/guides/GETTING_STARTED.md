# Getting Started Guide

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** - [Download Python](https://www.python.org/downloads/)
- **Docker Desktop** - [Download Docker](https://www.docker.com/products/docker-desktop)
- **Git** - [Download Git](https://git-scm.com/downloads)
- **PostgreSQL Client** (optional) - For database debugging

### System Requirements

- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: 4+ cores recommended

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/edupulse.git
cd edupulse
```

### 2. Set Up Python Environment

#### Using venv (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Using Conda

```bash
# Create conda environment
conda create -n edupulse python=3.11
conda activate edupulse
```

### 3. Install Dependencies

```bash
# Install all dependencies including dev tools
pip install -r requirements-dev.txt

# Or install production dependencies only
pip install -r requirements.txt
```

### 4. Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
# Required variables:
# - DATABASE_URL
# - REDIS_URL
# - SECRET_KEY
# - ML_MODEL_PATH
```

Example `.env` file:
```env
# Database
DATABASE_URL=postgresql://edupulse:password@localhost:5432/edupulse_db
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=60

# ML Configuration
ML_MODEL_PATH=models/gru_model.pt
MODEL_VERSION=1.0.0
FEATURE_CACHE_TTL=3600

# API Configuration
API_V1_PREFIX=/api/v1
CORS_ORIGINS=["http://localhost:3000"]
ENVIRONMENT=development

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

### 5. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks for code quality
pre-commit install

# Run hooks manually (optional)
pre-commit run --all-files
```

## Running the Application

### Option 1: Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

This will start:
- PostgreSQL with TimescaleDB (port 5432)
- Redis (port 6379)
- API Server (port 8000)
- Celery Worker
- Celery Beat
- Flower (port 5555)

### Option 2: Local Development

#### Start Infrastructure

```bash
# Start PostgreSQL
docker run -d \
  --name edupulse-postgres \
  -e POSTGRES_USER=edupulse \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=edupulse_db \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg15

# Start Redis
docker run -d \
  --name edupulse-redis \
  -p 6379:6379 \
  redis:7-alpine
```

#### Initialize Database

```bash
# Run database migrations
python -m src.db.database init

# Or use the SQL script
psql $DATABASE_URL < scripts/db/init.sql
```

#### Start API Server

```bash
# Development mode with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

#### Start Background Workers

```bash
# In separate terminals:

# Start Celery worker
celery -A src.tasks.worker worker --loglevel=info

# Start Celery beat scheduler
celery -A src.tasks.worker beat --loglevel=info

# Start Flower monitoring (optional)
celery -A src.tasks.worker flower
```

## Verification

### 1. Check Health Endpoints

```bash
# API health check
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "1.0.0"}

# Readiness check
curl http://localhost:8000/ready

# Expected response:
# {"status": "ready", "database": "connected", "redis": "connected"}
```

### 2. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Flower** (Celery monitoring): http://localhost:5555

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

## Quick Start Examples

### 1. Create a Student

```bash
curl -X POST http://localhost:8000/api/v1/students \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "STU001",
    "first_name": "John",
    "last_name": "Doe",
    "grade_level": 10,
    "enrollment_date": "2024-09-01"
  }'
```

### 2. Make a Prediction

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"student_id": "STU001"}
)

result = response.json()
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Category: {result['risk_category']}")
```

### 3. Train the Model

```python
# Trigger model training
response = requests.post(
    "http://localhost:8000/api/v1/train/update",
    json={
        "training_config": {
            "epochs": 50,
            "batch_size": 32
        }
    }
)

training_id = response.json()["training_id"]
print(f"Training started: {training_id}")
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

```bash
# Edit files
# Run tests locally
pytest tests/

# Check code quality
black src tests
flake8 src tests
mypy src
```

### 3. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
# Create pull request on GitHub
```

## Common Tasks

### Adding a New API Endpoint

1. Create route handler in `src/api/routes/`
2. Add Pydantic models in `src/models/schemas.py`
3. Update router includes in `src/api/main.py`
4. Add tests in `tests/api/`

### Adding a New Feature Extractor

1. Create extractor class in `src/features/`
2. Inherit from `BaseFeatureExtractor`
3. Implement `extract()` method
4. Register in `FeaturePipeline`
5. Add unit tests

### Updating the Model

1. Modify architecture in `src/models/gru_model.py`
2. Update training config in `src/training/trainer.py`
3. Retrain model with new data
4. Version and save model
5. Update model path in `.env`

## Debugging

### Enable Debug Logging

```python
# In .env
LOG_LEVEL=DEBUG

# Or programmatically
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Database Debugging

```bash
# Connect to database
psql $DATABASE_URL

# Check tables
\dt

# View recent predictions
SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;
```

### Redis Debugging

```bash
# Connect to Redis
redis-cli

# Check keys
KEYS *

# Get cached feature
GET features:STU001:2024-12-01
```

## IDE Setup

### VS Code

Install recommended extensions:
- Python
- Pylance
- Black Formatter
- Docker
- Thunder Client (API testing)

Settings (`settings.json`):
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

### PyCharm

1. Set Python interpreter to virtual environment
2. Enable Django support (for similar features)
3. Configure code style to use Black
4. Set up run configurations for API and tests

## Deployment

### Local Deployment Checklist

- [ ] All tests passing
- [ ] Code formatted and linted
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] Model file present
- [ ] Docker containers healthy

### Production Deployment

See [Deployment Guide](./DEPLOYMENT.md) for production deployment instructions.

## Getting Help

### Resources

- **Documentation**: `/docs` directory
- **API Docs**: http://localhost:8000/docs
- **Architecture**: [ARCHITECTURE.md](../ARCHITECTURE.md)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

### Common Issues

**Issue**: `ModuleNotFoundError`
- **Solution**: Ensure virtual environment is activated and dependencies installed

**Issue**: Database connection error
- **Solution**: Check DATABASE_URL and ensure PostgreSQL is running

**Issue**: Model file not found
- **Solution**: Train model first or download pre-trained model

### Support Channels

- GitHub Issues: Report bugs and feature requests
- Discussions: Ask questions and share ideas
- Email: support@edupulse.com

---

*Next Steps: Check out the [API Reference](../API_REFERENCE.md) to start building!*
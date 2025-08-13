# EduPulse Analytics MVP

An AI-powered early warning system for student success, leveraging machine learning to identify at-risk students and provide actionable interventions.

## Features

- **Predictive Analytics**: GRU-based neural network with attention mechanisms for risk prediction
- **Real-time Processing**: Async task processing with Celery for scalable predictions
- **Time-Series Database**: TimescaleDB for efficient temporal data storage
- **RESTful API**: FastAPI-based API with JWT authentication
- **Continuous Learning**: Feedback loop for model improvement
- **Monitoring**: Prometheus metrics and structured logging

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL (via Docker)
- Redis (via Docker)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EduPulse
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Copy environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Set up pre-commit hooks:
```bash
pre-commit install
```

### Running with Docker

Start all services:
```bash
docker-compose up -d
```

This will start:
- PostgreSQL with TimescaleDB (port 5432)
- Redis (port 6379)
- API Server (port 8000)
- Celery Worker
- Celery Beat Scheduler
- Flower (Celery monitoring, port 5555)

### Development

Run tests:
```bash
pytest
```

Format code:
```bash
black src tests
isort src tests
```

Lint code:
```bash
flake8 src tests
mypy src
```

### API Documentation

Once the API is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Project Structure

```
EduPulse/
├── src/
│   ├── api/          # FastAPI application and endpoints
│   ├── models/       # Pydantic and SQLAlchemy models
│   ├── data/         # Data processing and ingestion
│   ├── features/     # Feature engineering
│   ├── training/     # ML model training
│   ├── db/           # Database utilities
│   └── utils/        # Common utilities
├── tests/            # Test suite
├── config/           # Configuration files
├── docker/           # Docker configurations
├── scripts/          # Utility scripts
├── data/             # Data storage
├── logs/             # Application logs
├── models/           # Trained models
└── notebooks/        # Jupyter notebooks
```

## License

[Your License Here]

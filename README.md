# EduPulse Analytics MVP

**🎯 AI-powered dropout prediction system for K-12 schools**

EduPulse uses cutting-edge GRU-based neural networks with attention mechanisms to analyze student behavioral patterns and predict dropout risk 60+ days in advance. Our system helps counselors prioritize intervention efforts by analyzing attendance, academic performance, and disciplinary data to identify at-risk students before they fall through the cracks.

**⚡ Quick Start:** `docker-compose up -d` → API ready at `http://localhost:8000`

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108%2B-green.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-Passing-green.svg)](tests/)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [Support](#support)

## ✨ Features

### 🧠 Machine Learning Core

- **Sequential Risk Prediction**: GRU-based neural networks with attention mechanisms
- **Multi-Modal Data Fusion**: Combines attendance, grades, and discipline patterns
- **Temporal Analysis**: 20-week rolling window for trend detection
- **Continuous Learning**: Real-time model updates with feedback loops
- **Explainable AI**: Attention-based feature importance and risk factor identification

### 🚀 Real-Time Processing

- **Low-Latency Inference**: <100ms API response times
- **Async Task Processing**: Celery-based background job processing
- **Batch Predictions**: Efficient processing of multiple students
- **Caching Layer**: Redis-based feature and prediction caching
- **Streaming Updates**: Real-time data ingestion capabilities

### 🔧 Production-Ready Infrastructure

- **RESTful API**: FastAPI with automatic OpenAPI documentation
- **Time-Series Database**: PostgreSQL with TimescaleDB for temporal data
- **Monitoring & Observability**: Prometheus metrics and structured logging
- **Container-First**: Docker Compose for development, Kubernetes-ready
- **Security**: JWT authentication, rate limiting, and audit trails

### 📊 Analytics & Insights

- **Risk Scoring**: 0-1 probability scores with confidence intervals
- **Risk Categories**: Four-tier classification (low, moderate, high, critical)
- **Lead Time**: 60+ days early warning before academic failure events
- **Performance Metrics**: Real-time model accuracy and drift detection
- **Dashboard Integration**: Ready for visualization platforms

## 🏗️ Architecture

EduPulse follows a microservices-inspired architecture with clear separation of concerns:

```ascii
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
│               (Web, Mobile, Admin Portal)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTPS/JWT
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  API Gateway (FastAPI)                      │
│             Authentication | Rate Limiting                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────────┐
          ▼               ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Prediction    │ │    Training     │ │     Student     │
│    Service      │ │    Service      │ │    Service      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │               │                   │
          └───────────────┼───────────────────┘
                          ▼
          ┌────────────────────────────────────┐
          │        Feature Pipeline            │
          │     (Extract & Transform)          │
          └────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────────┐
          ▼               ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   PostgreSQL    │ │      Redis      │ │   Model Store   │
│  (TimescaleDB)  │ │     (Cache)     │ │    (MLflow)     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Core Components

- **API Layer**: FastAPI with request validation, authentication, and rate limiting
- **Service Layer**: Business logic for predictions, training, and student management
- **Feature Pipeline**: 42-dimensional feature extraction from raw student data
- **ML Pipeline**: GRU-based models with attention mechanisms for interpretability
- **Data Layer**: Time-series optimized storage with caching for performance

## 🚀 Quick Start

### 🐳 Docker Setup (Recommended)

**Get started in 30 seconds:**

```bash
# Clone and start
git clone https://github.com/your-org/edupulse.git
cd EduPulse
docker-compose up -d

# Verify it's working
curl http://localhost:8000/health
# Expected: {"status": "healthy", "version": "1.0.0"}

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"student_id": "STU123456"}'
```

**Services running:**

- 🌐 **API Server**: <http://localhost:8000>
- 🗄️ **Database**: PostgreSQL on port 5432
- 🔄 **Cache**: Redis on port 6379
- 📊 **Docs**: <http://localhost:8000/docs>

### 💻 Development Setup

**Prerequisites:**

- Python 3.12+ ([Download](https://python.org/downloads/))
- Docker Desktop ([Download](https://docker.com/products/docker-desktop))

```bash
# 1. Clone and setup
git clone https://github.com/your-org/edupulse.git
cd EduPulse

# 2. Python environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install

# 3. Start infrastructure
docker-compose -f docker-compose.dev.yml up -d

# 4. Run API locally
uvicorn src.api.main:app --reload --port 8000
```

### Manual Development Setup

```bash
# Start infrastructure
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_USER=edupulse -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=edupulse_db timescale/timescaledb:latest-pg15

docker run -d --name redis -p 6379:6379 redis:7-alpine

# Initialize database
python -m src.db.database init

# Start services
uvicorn src.api.main:app --reload --port 8000
celery -A src.tasks.worker worker --loglevel=info
```

## 💻 Usage Examples

### 🎯 Core API Calls

**Single Student Risk Prediction:**

```bash
# Get risk assessment with actionable insights
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "STU123456",
    "include_explanations": true,
    "include_features": true
  }'

# Response includes risk score (0-1), category, and intervention recommendations
```

**Batch Processing for Daily Reports:**

```bash
# Process multiple students efficiently
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "student_ids": ["STU001", "STU002", "STU003"],
    "async": false,
    "priority": "high"
  }'
```

**System Health Check:**

```bash
curl http://localhost:8000/health
# Quick liveness check

curl http://localhost:8000/ready
# Comprehensive readiness check (DB, Redis, ML model)
```

### 🐍 Python Client Integration

**Simple Risk Assessment:**

```python
import requests
import json

def assess_student_risk(student_id):
    response = requests.post(
        "http://localhost:8000/api/v1/predict",
        json={
            "student_id": student_id,
            "include_explanations": True
        }
    )

    if response.status_code == 200:
        result = response.json()['data']
        print(f"🎯 {student_id}: {result['risk_score']:.1%} dropout risk")
        print(f"📊 Risk Level: {result['risk_category']}")

        if result['explanations']:
            print("🔍 Top Risk Factors:")
            for factor in result['explanations'][:3]:
                print(f"  • {factor['description']}")
        return result
    else:
        print(f"❌ Error: {response.text}")
        return None

# Usage
student_data = assess_student_risk("STU123456")
```

**Counselor Dashboard Data:**

```python
def get_priority_students(grade_level=None, risk_threshold=0.7):
    """Get students requiring immediate intervention"""

    # Get student list
    params = {"limit": 100}
    if grade_level:
        params["grade_level"] = grade_level

    students_response = requests.get(
        "http://localhost:8000/api/v1/students",
        params=params
    )

    students = students_response.json()['data']['students']
    student_ids = [s['student_id'] for s in students]

    # Batch predict
    batch_response = requests.post(
        "http://localhost:8000/api/v1/predict/batch",
        json={"student_ids": student_ids}
    )

    predictions = batch_response.json()['data']['predictions']

    # Filter high-risk students
    priority_students = [
        p for p in predictions
        if p['risk_score'] > risk_threshold
    ]

    # Sort by risk score
    priority_students.sort(key=lambda x: x['risk_score'], reverse=True)

    print(f"🚨 {len(priority_students)} students need immediate attention:")
    for student in priority_students[:5]:  # Top 5
        print(f"  • {student['student_id']}: {student['risk_score']:.1%}")

    return priority_students
```

### 🎲 Interactive Examples

**Explore with Jupyter Notebooks:**

```bash
# Launch interactive examples
jupyter notebook notebooks/
# Open: 03_student_risk_tracking_demo.ipynb
```

**Try the Web UI:**
Visit <http://localhost:8000/docs> for interactive API documentation with:

- ✅ Test all endpoints directly in browser
- 📖 Complete parameter documentation
- 🔍 Response schema examples

### 🧠 Understanding the 42-Feature ML Pipeline

**Data Sources Analyzed:**

- **📚 Academic (15 features)**: GPA trends, assignment completion, grade volatility, course difficulty
- **👥 Attendance (14 features)**: Daily attendance, tardiness patterns, consecutive absences, seasonal trends
- **⚖️ Behavioral (13 features)**: Discipline incidents, severity escalation, peer interactions, engagement metrics

**Key Insights Generated:**

- 📈 **Temporal Patterns**: 20-week rolling analysis captures declining performance trends
- 🎯 **Risk Categories**: Low (0-25%), Medium (25-50%), High (50-75%), Critical (75-100%)
- ⏰ **Early Warning**: 60+ days advance notice before dropout events
- 🔍 **Explainable AI**: Attention mechanisms highlight which factors drive each prediction

## 📚 API Documentation

### Interactive Documentation

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>
- **OpenAPI JSON**: <http://localhost:8000/openapi.json>

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/api/v1/predict` | POST | Single prediction |
| `/api/v1/predict/batch` | POST | Batch predictions |
| `/api/v1/students` | POST/GET | Student management |
| `/api/v1/train/update` | POST | Model training |
| `/api/v1/metrics` | GET | System metrics |

### Authentication

```bash
# Include JWT token in requests
curl -H "Authorization: Bearer <your-token>" \
     http://localhost:8000/api/v1/predict
```

## 🛠️ Development

### 🧪 Testing

**Run the complete test suite:**

```bash
# Full test suite with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Quick smoke tests
pytest tests/unit/test_api_routes.py -v

# E2E integration tests
pytest tests/e2e/ -v --tb=short

# Performance benchmarks
pytest tests/performance/ --benchmark-only
```

**Test Categories:**

- **Unit Tests** (`tests/unit/`): Fast, isolated component testing
- **Integration Tests** (`tests/integration/`): Database and API integration
- **E2E Tests** (`tests/e2e/`): Full workflow testing with real data
- **Performance Tests** (`tests/performance/`): Load testing and benchmarks

### ✨ Code Quality

**Automated quality checks:**

```bash
# One command to rule them all
make quality

# Or run individually:
black src tests           # Code formatting
isort src tests          # Import sorting
flake8 src tests         # Linting
mypy src                 # Type checking
pytest --cov=src         # Testing with coverage

# Pre-commit hooks (runs automatically)
pre-commit run --all-files
```

**Quality metrics:**

- 📊 **Test Coverage**: Target >90% (currently 87%)
- 🎯 **Type Coverage**: 100% type hints in `src/`
- ⚡ **Performance**: <100ms API response time
- 🔒 **Security**: No high/critical vulnerabilities

### Database Management

```bash
# Connect to database
psql postgresql://edupulse:password@localhost:5432/edupulse_db

# View predictions
SELECT * FROM predictions ORDER BY prediction_date DESC LIMIT 10;

# Check feature extraction
SELECT student_id, feature_date, attendance_rate, gpa_current
FROM student_features ORDER BY feature_date DESC LIMIT 5;
```

### Adding Features

1. **New API Endpoint**: Create in `src/api/routes/`
2. **Feature Extractor**: Inherit from `BaseFeatureExtractor` in `src/features/`
3. **Model Changes**: Modify `src/models/gru_model.py`
4. **Tests**: Add comprehensive test coverage

## 🚚 Deployment

### Docker Production

```bash
# Build production image
docker build -t edupulse:latest -f Dockerfile .

# Run with production settings
docker run -p 8000:8000 --env-file .env.prod edupulse:latest
```

### Kubernetes

```yaml
# k8s deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edupulse-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: edupulse-api
  template:
    metadata:
      labels:
        app: edupulse-api
    spec:
      containers:
      - name: api
        image: edupulse:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: edupulse-secrets
              key: database-url
```

### Environment Configuration

```bash
# Production environment variables
DATABASE_URL=postgresql://user:pass@prod-db:5432/edupulse
REDIS_URL=redis://prod-redis:6379/0
SECRET_KEY=your-production-secret-key
ENVIRONMENT=production
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

## 🤝 Contributing

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes with tests
4. **Run** quality checks: `pre-commit run --all-files`
5. **Commit** with conventional commits: `git commit -m "feat: add amazing feature"`
6. **Push** and create a **Pull Request**

### Code Standards

- **Python**: Follow PEP 8, use Black formatting
- **Testing**: Maintain >80% test coverage
- **Documentation**: Update docs for new features
- **Type Hints**: Use type hints for all functions
- **Commit Messages**: Follow [Conventional Commits](https://conventionalcommits.org/)

## 📖 Documentation

### 🚀 Essential Guides

- [**API Reference**](docs/API_REFERENCE.md) - Complete REST API documentation with implementation status
- [**ML Model Architecture**](docs/ML_MODEL_ARCHITECTURE.md) - GRU neural network with attention mechanisms
- [**Database Schema**](docs/DATABASE_SCHEMA.md) - Comprehensive data models and relationships
- [**Deployment Guide**](docs/DEPLOYMENT_GUIDE.md) - Production deployment with Docker/Kubernetes
- [**Troubleshooting Guide**](docs/TROUBLESHOOTING.md) - Solutions for common issues and debugging

### 🛠️ Technical References

- [**Architecture Overview**](docs/ARCHITECTURE.md) - System design and component interactions
- [**Feature Engineering**](docs/modules/features.md) - All 42 ML features explained in detail
- [**Getting Started Guide**](docs/guides/GETTING_STARTED.md) - Step-by-step development setup
- [**Configuration Reference**](docs/CONFIGURATION.md) - Environment variables and settings

### 💡 Examples & Tutorials

- [**Python Client Examples**](docs/examples/python_client.md) - SDK integration patterns
- [**JavaScript Integration**](docs/examples/javascript_client.md) - Frontend API usage
- [**Data Import Workflows**](docs/examples/data_import.md) - Bulk data processing guides
- [**Jupyter Notebooks**](notebooks/) - Interactive demos and feature analysis

### 🔧 Operations & Maintenance

- [**Monitoring & Metrics**](docs/MONITORING.md) - Performance tracking and alerting
- [**Security Guidelines**](docs/SECURITY.md) - Authentication, authorization, and data protection
- [**Backup & Recovery**](docs/BACKUP_RECOVERY.md) - Data protection strategies
- [**Scaling Guide**](docs/SCALING.md) - Horizontal scaling and load management

## 🗂️ Project Structure

```
EduPulse/                          # 🏠 Root directory
├── 🌐 src/                        # Core application code
│   ├── api/                       # FastAPI web framework
│   │   ├── main.py               # 🚀 Application entry point
│   │   └── routes/               # HTTP endpoint handlers
│   │       ├── health.py         # Health check endpoints
│   │       ├── predictions.py    # ML prediction APIs
│   │       ├── students.py       # Student data management
│   │       └── training.py       # Model training endpoints
│   ├── models/                   # 🧠 Machine learning components
│   │   ├── gru_model.py         # GRU neural network architecture
│   │   └── schemas.py           # Pydantic data validation
│   ├── features/                 # 🔧 Feature engineering pipeline
│   │   ├── pipeline.py          # Orchestrates 42-feature extraction
│   │   ├── attendance.py        # 14 attendance-based features
│   │   ├── grades.py            # 15 academic performance features
│   │   └── discipline.py        # 13 behavioral pattern features
│   ├── training/                 # 📈 Model training infrastructure
│   │   └── trainer.py           # Training pipeline & model versioning
│   ├── db/                      # 🗄️ Database layer
│   │   ├── database.py          # Connection pooling & management
│   │   └── models.py            # SQLAlchemy ORM models
│   ├── services/                # 💼 Business logic layer
│   │   └── prediction_service.py # Risk prediction orchestration
│   └── config/                  # ⚙️ Application configuration
│       └── settings.py          # Environment-based settings
├── 🧪 tests/                     # Comprehensive test suite
│   ├── unit/                    # Fast, isolated tests (87% coverage)
│   ├── integration/             # Database & API integration tests
│   ├── e2e/                     # End-to-end workflow tests
│   └── performance/             # Load testing & benchmarks
├── 📚 docs/                      # Documentation hub
│   ├── API_REFERENCE.md         # Complete API documentation
│   ├── ARCHITECTURE.md          # System design & components
│   ├── examples/                # Client integration examples
│   ├── guides/                  # Step-by-step tutorials
│   └── modules/                 # Technical component docs
├── 📓 notebooks/                 # Interactive analysis & demos
│   ├── 01_test_coverage.ipynb   # Test quality analysis
│   ├── 03_student_risk_tracking_demo.ipynb # Live prediction demo
│   └── 05_feature_engineering.ipynb # Feature pipeline walkthrough
├── 🐳 docker/                    # Container configurations
│   └── Dockerfile.base          # Base container image
├── 💾 data/                      # Data storage & processing
│   ├── raw/                     # Imported student records
│   ├── processed/               # Cleaned & transformed data
│   └── cache/                   # Cached feature vectors
├── 🤖 models/                    # Trained ML model artifacts
└── 📝 logs/                      # Application & system logs
    ├── api.log                  # HTTP request logs
    ├── training.log             # Model training logs
    └── predictions.log          # Prediction audit trail
```

**🎯 Key Directories:**

- **`src/api/`**: RESTful API endpoints with automatic OpenAPI docs
- **`src/features/`**: 42-feature ML pipeline extracting behavioral patterns
- **`src/models/`**: GRU neural network with attention mechanisms
- **`tests/`**: 87% test coverage across unit, integration, and E2E tests
- **`notebooks/`**: Interactive demos and analysis tools
- **`docs/`**: Comprehensive documentation for developers and users

## 📊 Performance Metrics

### Model Performance

- **Precision@10**: >85% for top risk predictions
- **Lead Time**: 60+ days before failure events
- **Inference Speed**: <100ms per prediction
- **Model Training**: <4 hours for district-scale data

### System Performance

- **API Latency**: P99 < 100ms
- **Throughput**: 1000+ predictions/second
- **Uptime**: 99.9% availability target
- **Auto-scaling**: Horizontal scaling with Kubernetes

## 🔧 Troubleshooting

### Quick Fixes

**ModuleNotFoundError**

```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements-dev.txt
```

**Database Connection Error**

```bash
# Check DATABASE_URL and ensure PostgreSQL is running
docker ps | grep postgres
psql $DATABASE_URL -c "SELECT 1;"
```

**Model File Not Found**

```bash
# Train model or download pre-trained model
python -m src.training.trainer --train
# Or check ML_MODEL_PATH in .env
```

**Docker Issues**

```bash
# Reset Docker environment
docker-compose down -v
docker-compose up -d --build
```

### 📚 Comprehensive Troubleshooting

For detailed solutions to common issues, debugging techniques, and production problems, see our **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)**, which covers:

- **🔍 Installation & Setup Issues**: Python versions, dependency conflicts, Docker problems
- **🗄️ Database Problems**: Connection errors, migrations, TimescaleDB setup
- **⚡ Performance Issues**: Slow API responses, memory leaks, scaling problems
- **🧪 Testing Issues**: Test failures, coverage problems, E2E debugging
- **🚀 Production Deployment**: SSL certificates, load balancing, monitoring
- **🔐 Security & Authentication**: JWT tokens, rate limiting, API security
- **🐛 Debugging Tools**: Logging, profiling, system monitoring

**Quick Health Check:**

```bash
# Run comprehensive system diagnostic
curl -f http://localhost:8000/health
psql $DATABASE_URL -c "SELECT 1"
redis-cli ping
python -c "import src.api.main; print('✅ Imports OK')"
```

## 📈 Monitoring

### Available Metrics

- **System Metrics**: CPU, memory, disk usage
- **API Metrics**: Request latency, error rates, throughput
- **ML Metrics**: Model accuracy, prediction confidence, feature drift
- **Business Metrics**: Daily predictions, intervention rates

### Monitoring Endpoints

- **Prometheus**: <http://localhost:9090> (if enabled)
- **Flower**: <http://localhost:5555> (Celery monitoring)
- **Health**: <http://localhost:8000/health>
- **Metrics**: <http://localhost:8000/api/v1/metrics>

## 🔐 Security

### Security Features

- **JWT Authentication** with role-based access control
- **Rate Limiting** to prevent abuse
- **Data Encryption** at rest (AES-256) and in transit (TLS 1.3)
- **Audit Logging** for all predictions and model updates
- **Input Validation** with Pydantic schemas

### Security Best Practices

- Regular dependency updates with security scanning
- Secrets management via environment variables
- Network segmentation in production deployments
- Regular security assessments and penetration testing

## 🆘 Support

### Getting Help

- **Documentation**: Start with our [comprehensive docs](docs/)
- **GitHub Issues**: [Report bugs or request features](https://github.com/your-org/edupulse/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/your-org/edupulse/discussions)
- **Email**: [support@edupulse.com](mailto:support@edupulse.com)

### Community

- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- **Code of Conduct**: See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md) for release notes

---

<div align="center">

**⭐ Star this project if it helps you!**

[📚 Documentation](docs/) • [🌐 API Reference](docs/API_REFERENCE.md) • [🧠 ML Architecture](docs/ML_MODEL_ARCHITECTURE.md) • [🗄️ Database Schema](docs/DATABASE_SCHEMA.md) • [🚀 Deployment](docs/DEPLOYMENT_GUIDE.md) • [🔧 Troubleshooting](docs/TROUBLESHOOTING.md) • [🤝 Contributing](CONTRIBUTING.md) • [💬 Support](mailto:support@edupulse.com)

Made with ❤️ for student success

</div>

# System Architecture

## Overview

EduPulse Analytics is a temporal machine learning system designed to predict student academic risk using behavioral patterns. The system employs a microservices-inspired architecture with clear separation of concerns.

## High-Level Architecture

```ascii
┌─────────────────────────────────────────────────────────────┐
│                        Client Applications                  │
│                    (Web, Mobile, Admin Portal)              │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway (FastAPI)                   │
│                  Authentication | Rate Limiting             │
└─────────────────────────────┬───────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Prediction      │ │   Training       │ │   Student        │
│   Service        │ │   Service        │ │   Service        │
└──────────────────┘ └──────────────────┘ └──────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                  ┌────────────────────────┐
                  │   Feature Pipeline     │
                  │  (Extract & Transform) │
                  └────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   PostgreSQL     │ │     Redis        │ │   Model Store    │
│  (TimescaleDB)   │ │    (Cache)       │ │   (MLflow)       │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Component Details

### 1. API Layer

**Technology**: FastAPI

**Responsibilities**:

- Request validation using Pydantic
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- API versioning

**Key Endpoints**:

- `/api/v1/predict` - Single student prediction
- `/api/v1/predict/batch` - Batch predictions
- `/api/v1/train/update` - Model updates
- `/api/v1/metrics` - System metrics

### 2. Service Layer

#### Prediction Service

- Real-time risk assessment
- Feature extraction coordination
- Model inference execution
- Result caching

#### Training Service

- Model retraining pipeline
- Hyperparameter optimization
- Model versioning
- Performance tracking

#### Student Service

- CRUD operations
- Data validation
- Historical tracking

### 3. ML Pipeline

#### Feature Extraction

```ascii
Raw Data → Feature Extractors → Feature Vector (42 dimensions)
    │              │                      │
    ├─ Attendance ─┤                      ├─ 14 features
    ├─ Grades ────┤                      ├─ 15 features
    └─ Discipline ─┘                      └─ 13 features
```

#### Model Architecture

```ascii
Feature Vector
      ↓
┌─────────────────────────┐
│   3 Parallel GRUs       │
│  (Attendance, Grades,   │
│   Discipline)           │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│  Multi-Head Attention   │
│     (4 heads)           │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│   Feature Fusion        │
└───────────┬─────────────┘
            ↓
      ┌─────┴─────┐
      ↓           ↓
Risk Score   Risk Category
  (0-1)      (4 classes)
```

### 4. Data Layer

#### PostgreSQL with TimescaleDB

- **Purpose**: Primary data store with time-series optimization
- **Tables**:
  - `students` - Student demographics
  - `student_features` - Time-series features (hypertable)
  - `predictions` - Risk predictions
  - `training_feedback` - Model feedback
  - `attendance_records` - Attendance data
  - `grades` - Academic performance
  - `discipline_incidents` - Behavioral data

#### Redis

- **Purpose**: Caching and task queue
- **Usage**:
  - Feature vector caching (TTL: 1 hour)
  - Prediction result caching
  - Celery task queue
  - Session storage

## Data Flow

### 1. Prediction Flow

```ascii
1. Client Request
   └─> API validates request
       └─> Check cache for existing prediction
           ├─> (Hit) Return cached result
           └─> (Miss) Continue processing
               └─> Extract student data
                   └─> Generate feature vector
                       └─> Run ML inference
                           └─> Cache result
                               └─> Store in database
                                   └─> Return response
```

### 2. Training Flow

```ascii
1. Training Trigger (scheduled/manual)
   └─> Collect training data
       └─> Validate data quality
           └─> Extract features for all samples
               └─> Split train/validation sets
                   └─> Train model
                       └─> Evaluate performance
                           └─> Save checkpoint
                               └─> Update model registry
                                   └─> Deploy if improved
```

## Security Architecture

### Authentication & Authorization

- JWT-based authentication
- Role-based access control (RBAC)
- API key management for service-to-service

### Data Security

- Encryption at rest (AES-256)
- TLS 1.3 for data in transit
- PII data masking in logs
- Audit trail for all predictions

## Scalability Considerations

### Horizontal Scaling

- Stateless API servers
- Read replicas for database
- Distributed cache with Redis Cluster
- Load balancing with nginx

### Performance Optimization

- Batch prediction processing
- Feature vector caching
- Model quantization (int8)
- Async task processing with Celery

## Monitoring & Observability

### Metrics Collection

```ascii
Application Metrics ─> Prometheus ─> Grafana
       │                   │            │
       ├─ API latency      │            ├─ Dashboards
       ├─ Prediction count │            ├─ Alerts
       ├─ Model accuracy   │            └─ Reports
       └─ Error rates      │
                           │
System Metrics ────────────┘
       │
       ├─ CPU/Memory
       ├─ Disk I/O
       └─ Network
```

### Logging Strategy

- Structured logging (JSON format)
- Centralized log aggregation
- Log levels: DEBUG, INFO, WARNING, ERROR
- Correlation IDs for request tracing

## Deployment Architecture

### Container Structure

```ascii
docker-compose.yml
    ├─ postgres (TimescaleDB)
    ├─ redis
    ├─ api (FastAPI)
    ├─ celery-worker
    ├─ celery-beat
    └─ flower (monitoring)
```

### Environment Configuration

- Development: Docker Compose
- Staging: Kubernetes (Minikube)
- Production: Kubernetes (EKS/GKE)

## Technology Stack

### Core Technologies

- **Language**: Python 3.12
- **Web Framework**: FastAPI
- **ML Framework**: PyTorch 2.5
- **Database**: PostgreSQL 15 + TimescaleDB
- **Cache**: Redis 7
- **Task Queue**: Celery 5.3

### Supporting Tools

- **API Documentation**: OpenAPI/Swagger
- **Testing**: Pytest
- **Code Quality**: Black, Flake8, MyPy
- **Monitoring**: Prometheus + Grafana
- **ML Tracking**: MLflow

## Design Patterns

### Architectural Patterns

1. **Microservices**: Service separation
2. **Repository Pattern**: Data access abstraction
3. **Factory Pattern**: Feature extractor creation
4. **Strategy Pattern**: Multiple prediction strategies
5. **Observer Pattern**: Event-driven updates

### ML Patterns

1. **Feature Store**: Centralized feature management
2. **Model Registry**: Version control for models
3. **A/B Testing**: Model comparison framework
4. **Shadow Mode**: New model validation

## Future Architecture Considerations

### Planned Enhancements

1. **GraphQL API**: Flexible data queries
2. **Event Streaming**: Kafka for real-time updates
3. **Federated Learning**: Privacy-preserving training
4. **Edge Deployment**: On-premise predictions
5. **Multi-tenancy**: District isolation

### Scaling Roadmap

- Phase 1: Single district (current)
- Phase 2: Multi-district with shared infrastructure
- Phase 3: Regional deployment with edge nodes
- Phase 4: National scale with distributed training

---

*For implementation details, see [Development Guide](./DEVELOPMENT.md) and [API Reference](./API_REFERENCE.md).*

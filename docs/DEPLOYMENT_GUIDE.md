# EduPulse Deployment Guide

> **Last Updated**: 2025-08-13 02:56:19 CDT
> **Version**: v1.0
> **Target**: Production and Development Environments

## Overview

This guide covers complete deployment of EduPulse, from development setup to production scaling. EduPulse is designed as a cloud-native application with containerized services, automated scaling, and comprehensive monitoring.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Load Balancer (ALB/NGINX)                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                  FastAPI Application Layer                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐   │
│  │   API Pod   │ │   API Pod   │ │   API Pod   │ │  Worker  │   │
│  │   (Main)    │ │   (Main)    │ │   (Main)    │ │   Pod    │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                     Data Layer                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐   │
│  │ PostgreSQL  │ │    Redis    │ │   S3/Minio  │ │ MLflow   │   │
│  │ (Primary)   │ │  (Cache)    │ │  (Models)   │ │(Registry)│   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (Development)

### Prerequisites

```bash
# Required software
- Python 3.11+
- Docker & Docker Compose
- Git
- Node.js 18+ (for frontend, if applicable)

# Recommended tools
- kubectl (for Kubernetes deployment)
- helm (for Kubernetes package management)
- awscli (for AWS deployment)
```

### Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/edupulse.git
cd edupulse

# 2. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-dev.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your local configuration

# 5. Start infrastructure services
docker-compose -f docker-compose.dev.yml up -d

# 6. Run database migrations
alembic upgrade head

# 7. Load sample data (optional)
python scripts/load_sample_data.py

# 8. Start the development server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check API documentation
open http://localhost:8000/docs

# Run tests to verify setup
python -m pytest tests/ -v
```

---

## Environment Configuration

### Environment Variables

Create `.env` files for each environment with these required variables:

```bash
# .env.production
# Application Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-super-secure-secret-key-min-32-chars
JWT_SECRET_KEY=your-jwt-secret-key-min-32-chars

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_VERSION=v1
CORS_ORIGINS=https://your-frontend-domain.com,https://dashboard.school.edu

# Database Configuration
DATABASE_URL=postgresql://edupulse_user:secure_password@postgres.internal:5432/edupulse_prod
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30
DB_ECHO=false

# Redis Configuration
REDIS_URL=redis://redis.internal:6379/0
REDIS_MAX_CONNECTIONS=50
CACHE_TTL=3600

# Celery Configuration (for async tasks)
CELERY_BROKER_URL=redis://redis.internal:6379/1
CELERY_RESULT_BACKEND=redis://redis.internal:6379/2
CELERY_TASK_TIME_LIMIT=3600
CELERY_TASK_SOFT_TIME_LIMIT=3300

# ML Model Configuration
MODEL_PATH=/app/models
MODEL_VERSION=latest
MODEL_DEVICE=cpu
MODEL_BATCH_SIZE=32
MODEL_MAX_SEQUENCE_LENGTH=365

# Feature Engineering
FEATURE_WINDOW_DAYS=90
FEATURE_LAG_DAYS=7
FEATURE_CACHE_ENABLED=true
FEATURE_CACHE_TTL=86400

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow.internal:5000
MLFLOW_EXPERIMENT_NAME=edupulse-production

# Monitoring Configuration
PROMETHEUS_PORT=9090
METRICS_ENABLED=true

# Performance Configuration
REQUEST_TIMEOUT=30
WORKER_CONNECTIONS=1000
MAX_PREDICTION_BATCH_SIZE=100
MAX_CONCURRENT_TASKS=10

# Feature Flags
ENABLE_ASYNC_PREDICTIONS=true
ENABLE_CONTINUOUS_LEARNING=true
ENABLE_MODEL_INTERPRETABILITY=true
```

### Configuration by Environment

#### Development

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
DATABASE_URL=postgresql://edupulse:dev_password@localhost:5432/edupulse_dev
REDIS_URL=redis://localhost:6379/0
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

#### Staging

```bash
# .env.staging
ENVIRONMENT=staging
DEBUG=false
DATABASE_URL=postgresql://edupulse:staging_password@postgres-staging:5432/edupulse_staging
REDIS_URL=redis://redis-staging:6379/0
CORS_ORIGINS=https://staging.edupulse.com
```

#### Production

```bash
# .env.production (stored in secure secret management)
ENVIRONMENT=production
DEBUG=false
# Use AWS Secrets Manager, HashiCorp Vault, or Kubernetes secrets
# Never store production secrets in plain text files
```

---

## Docker Deployment

### Dockerfile Structure

```dockerfile
# Multi-stage Dockerfile for optimized production builds
FROM python:3.11-slim as base

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
CMD ["uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

# Production stage
FROM base as production
COPY . .

# Create non-root user
RUN groupadd -r edupulse && useradd -r -g edupulse edupulse
RUN chown -R edupulse:edupulse /app
USER edupulse

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Docker Compose for Local Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  api:
    build:
      context: .
      target: development
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://edupulse:dev_password@postgres:5432/edupulse_dev
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - .:/app
      - ./models:/app/models
    depends_on:
      - postgres
      - redis
    networks:
      - edupulse

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: edupulse
      POSTGRES_PASSWORD: dev_password
      POSTGRES_DB: edupulse_dev
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./scripts/db/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - edupulse

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_dev_data:/data
    networks:
      - edupulse

  # Optional: pgAdmin for database management
  pgadmin:
    image: dpage/pgadmin4
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@edupulse.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgres
    networks:
      - edupulse

volumes:
  postgres_dev_data:
  redis_dev_data:

networks:
  edupulse:
    driver: bridge
```

### Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    build:
      context: .
      target: production
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env.production
    restart: unless-stopped
    depends_on:
      - postgres
      - redis
    networks:
      - edupulse
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  # Background worker for async tasks
  worker:
    build:
      context: .
      target: production
    command: celery -A src.tasks.celery_app worker --loglevel=info
    env_file:
      - .env.production
    depends_on:
      - postgres
      - redis
    networks:
      - edupulse
    deploy:
      replicas: 2

  # Task scheduler
  scheduler:
    build:
      context: .
      target: production
    command: celery -A src.tasks.celery_app beat --loglevel=info
    env_file:
      - .env.production
    depends_on:
      - redis
    networks:
      - edupulse

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - edupulse
    deploy:
      placement:
        constraints:
          - node.role == manager

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_prod_data:/data
    networks:
      - edupulse

  # Reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - api
    networks:
      - edupulse

volumes:
  postgres_prod_data:
  redis_prod_data:

networks:
  edupulse:
    driver: overlay
    attachable: true
```

---

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: edupulse-prod
  labels:
    name: edupulse-prod
    environment: production

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: edupulse-config
  namespace: edupulse-prod
data:
  ENVIRONMENT: "production"
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_VERSION: "v1"
  MODEL_DEVICE: "cpu"
  MODEL_BATCH_SIZE: "32"
  FEATURE_WINDOW_DAYS: "90"
  FEATURE_LAG_DAYS: "7"
  PROMETHEUS_PORT: "9090"
  METRICS_ENABLED: "true"
```

### Secrets Management

```yaml
# k8s/secrets.yaml (apply with kubectl, never commit to git)
apiVersion: v1
kind: Secret
metadata:
  name: edupulse-secrets
  namespace: edupulse-prod
type: Opaque
stringData:
  SECRET_KEY: "your-super-secure-secret-key-min-32-chars"
  JWT_SECRET_KEY: "your-jwt-secret-key-min-32-chars"
  DATABASE_URL: "postgresql://edupulse_user:secure_password@postgres-service:5432/edupulse_prod"
  REDIS_URL: "redis://redis-service:6379/0"
  POSTGRES_USER: "edupulse_user"
  POSTGRES_PASSWORD: "secure_password"
  POSTGRES_DB: "edupulse_prod"
```

### Application Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edupulse-api
  namespace: edupulse-prod
  labels:
    app: edupulse-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: edupulse-api
  template:
    metadata:
      labels:
        app: edupulse-api
    spec:
      containers:
      - name: edupulse-api
        image: your-registry/edupulse:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: edupulse-config
              key: ENVIRONMENT
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: edupulse-secrets
              key: DATABASE_URL
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: edupulse-secrets
              key: SECRET_KEY
        envFrom:
        - configMapRef:
            name: edupulse-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models-volume
        configMap:
          name: ml-models-config
```

### Services and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: edupulse-api-service
  namespace: edupulse-prod
spec:
  selector:
    app: edupulse-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: edupulse-ingress
  namespace: edupulse-prod
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.edupulse.com
    secretName: edupulse-tls
  rules:
  - host: api.edupulse.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: edupulse-api-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: edupulse-api-hpa
  namespace: edupulse-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: edupulse-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

---

## Database Setup

### PostgreSQL Configuration

```sql
-- Create production database and user
CREATE USER edupulse_user WITH PASSWORD 'secure_password';
CREATE DATABASE edupulse_prod OWNER edupulse_user;
GRANT ALL PRIVILEGES ON DATABASE edupulse_prod TO edupulse_user;

-- Enable required extensions
\c edupulse_prod
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Performance tuning (adjust based on your hardware)
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.7;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Reload configuration
SELECT pg_reload_conf();
```

### Database Migrations

```bash
# Install Alembic (included in requirements.txt)
pip install alembic

# Initialize migrations (only needed once)
alembic init migrations

# Generate migration for schema changes
alembic revision --autogenerate -m "Create initial tables"

# Apply migrations to database
alembic upgrade head

# Production deployment migration script
#!/bin/bash
# migrate_production.sh

set -e

echo "Starting database migration for production..."

# Backup database before migration
pg_dump $DATABASE_URL > "backup_before_migration_$(date +%Y%m%d_%H%M%S).sql"

# Run migrations
alembic upgrade head

echo "Migration completed successfully!"
```

### Database Monitoring

```sql
-- Monitor database performance
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY tablename, attname;

-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- Monitor slow queries
SELECT
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
WHERE query LIKE '%students%'
ORDER BY mean_time DESC
LIMIT 10;
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy EduPulse

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run linting
      run: |
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
        black --check src/
        isort --check-only src/

    - name: Run type checking
      run: mypy src/

    - name: Run tests
      env:
        DATABASE_URL: postgresql://test:test@localhost:5432/test
        REDIS_URL: redis://localhost:6379/0
      run: |
        pytest tests/ -v --cov=src --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        target: production
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment..."
        # Add your staging deployment commands here
        # kubectl set image deployment/edupulse-api edupulse-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} -n edupulse-staging

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production environment..."
        # Add your production deployment commands here
        # kubectl set image deployment/edupulse-api edupulse-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} -n edupulse-prod
```

### Deployment Scripts

```bash
#!/bin/bash
# scripts/deploy.sh - Production deployment script

set -e

ENVIRONMENT=${1:-production}
IMAGE_TAG=${2:-latest}

echo "🚀 Starting deployment to $ENVIRONMENT..."

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    echo "❌ Invalid environment. Use 'staging' or 'production'"
    exit 1
fi

# Set kubectl context
kubectl config use-context $ENVIRONMENT

# Create namespace if it doesn't exist
kubectl create namespace edupulse-$ENVIRONMENT --dry-run=client -o yaml | kubectl apply -f -

# Apply configurations
echo "📋 Applying configurations..."
kubectl apply -f k8s/configmap.yaml -n edupulse-$ENVIRONMENT
kubectl apply -f k8s/secrets.yaml -n edupulse-$ENVIRONMENT

# Database migration
echo "🗄️  Running database migrations..."
kubectl run migration-job-$(date +%s) \
  --image=your-registry/edupulse:$IMAGE_TAG \
  --restart=Never \
  --env="DATABASE_URL=$(kubectl get secret edupulse-secrets -n edupulse-$ENVIRONMENT -o jsonpath='{.data.DATABASE_URL}' | base64 -d)" \
  --command -- alembic upgrade head \
  -n edupulse-$ENVIRONMENT

# Deploy application
echo "🚢 Deploying application..."
envsubst < k8s/deployment.yaml | kubectl apply -f - -n edupulse-$ENVIRONMENT

# Apply services and ingress
kubectl apply -f k8s/service.yaml -n edupulse-$ENVIRONMENT
kubectl apply -f k8s/ingress.yaml -n edupulse-$ENVIRONMENT

# Apply autoscaling
kubectl apply -f k8s/hpa.yaml -n edupulse-$ENVIRONMENT

# Wait for deployment
echo "⏳ Waiting for deployment to complete..."
kubectl rollout status deployment/edupulse-api -n edupulse-$ENVIRONMENT --timeout=600s

# Verify deployment
echo "✅ Verifying deployment..."
ENDPOINT=$(kubectl get ingress edupulse-ingress -n edupulse-$ENVIRONMENT -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -f http://$ENDPOINT/health || (echo "❌ Health check failed" && exit 1)

echo "🎉 Deployment to $ENVIRONMENT completed successfully!"
```

---

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'edupulse-api'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - edupulse-prod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: edupulse-api
    - source_labels: [__meta_kubernetes_pod_ip]
      target_label: __address__
      replacement: ${1}:9090

  - job_name: 'postgres'
    static_configs:
    - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
    - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093
```

### Application Metrics

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import functools

# Define metrics
REQUEST_COUNT = Counter(
    'edupulse_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'edupulse_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'edupulse_predictions_total',
    'Total predictions made',
    ['risk_category']
)

ACTIVE_USERS = Gauge(
    'edupulse_active_users',
    'Currently active users'
)

MODEL_INFERENCE_TIME = Histogram(
    'edupulse_model_inference_seconds',
    'ML model inference time',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

def track_requests(f):
    """Decorator to track request metrics."""
    @functools.wraps(f)
    async def wrapper(request, *args, **kwargs):
        method = request.method
        endpoint = request.url.path

        start_time = time.time()

        try:
            response = await f(request, *args, **kwargs)
            status_code = response.status_code

            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()

            return response

        except Exception as e:
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            raise
        finally:
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(time.time() - start_time)

    return wrapper
```

### Health Checks and Alerts

```yaml
# monitoring/alert_rules.yml
groups:
- name: edupulse_alerts
  rules:
  - alert: HighErrorRate
    expr: |
      (
        rate(edupulse_requests_total{status_code=~"5.."}[5m]) /
        rate(edupulse_requests_total[5m])
      ) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

  - alert: HighLatency
    expr: |
      histogram_quantile(0.95, rate(edupulse_request_duration_seconds_bucket[5m])) > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High request latency"
      description: "95th percentile latency is {{ $value }}s"

  - alert: DatabaseDown
    expr: up{job="postgres"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "PostgreSQL database is down"
      description: "Database has been down for more than 1 minute"

  - alert: HighMemoryUsage
    expr: |
      (
        container_memory_usage_bytes{pod=~"edupulse-api-.*"} /
        container_spec_memory_limit_bytes{pod=~"edupulse-api-.*"}
      ) > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Pod {{ $labels.pod }} memory usage is {{ $value | humanizePercentage }}"

  - alert: PodRestartingTooOften
    expr: |
      rate(kube_pod_container_status_restarts_total[15m]) > 0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Pod restarting frequently"
      description: "Pod {{ $labels.pod }} has restarted {{ $value }} times in the last 15 minutes"
```

---

## Security Considerations

### Application Security

```python
# src/security/middleware.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
import hashlib

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for production deployment."""

    async def dispatch(self, request: Request, call_next):
        # Rate limiting per IP
        client_ip = self._get_client_ip(request)
        if not self._check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Security headers
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response

    def _get_client_ip(self, request: Request) -> str:
        # Handle reverse proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
```

### Infrastructure Security

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: edupulse-network-policy
  namespace: edupulse-prod
spec:
  podSelector:
    matchLabels:
      app: edupulse-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS outbound
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### Secrets Management

```bash
# Use sealed-secrets for GitOps-friendly secret management
kubectl create secret generic edupulse-secrets \
  --from-literal=SECRET_KEY="your-secret-key" \
  --from-literal=DATABASE_URL="postgresql://..." \
  --dry-run=client -o yaml | \
  kubeseal -o yaml > sealed-secrets.yaml

# Or use external secret management
# AWS Secrets Manager
aws secretsmanager create-secret \
  --name "edupulse/production/api-keys" \
  --secret-string '{"SECRET_KEY":"...","DATABASE_URL":"..."}'

# Use External Secrets Operator to sync
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: edupulse-external-secrets
  namespace: edupulse-prod
spec:
  refreshInterval: 15s
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: edupulse-secrets
    creationPolicy: Owner
  data:
  - secretKey: SECRET_KEY
    remoteRef:
      key: edupulse/production/api-keys
      property: SECRET_KEY
```

---

## Performance Optimization

### Application Performance

```python
# src/performance/optimization.py
import asyncio
from functools import lru_cache
import aioredis

class PerformanceOptimizer:
    """Performance optimization utilities."""

    def __init__(self):
        self.redis = None
        self.connection_pool = None

    async def initialize_redis_pool(self):
        """Initialize Redis connection pool."""
        self.redis = aioredis.from_url(
            "redis://localhost",
            encoding="utf-8",
            decode_responses=True,
            max_connections=20
        )

    @lru_cache(maxsize=128)
    def get_cached_prediction(self, student_id: str, date: str):
        """Cache prediction results."""
        cache_key = f"prediction:{student_id}:{date}"
        # Implementation with Redis caching
        pass

    async def batch_database_queries(self, student_ids: list):
        """Optimize database queries with batching."""
        # Use SQLAlchemy's bulk operations
        async with self.db_session() as session:
            students = await session.execute(
                select(Student).where(Student.id.in_(student_ids))
            )
            return students.scalars().all()
```

### Database Optimization

```sql
-- Create optimized indexes
CREATE INDEX CONCURRENTLY idx_students_grade_enrollment
ON students(grade_level, enrollment_date);

CREATE INDEX CONCURRENTLY idx_attendance_student_date_status
ON attendance_records(student_id, date, status);

CREATE INDEX CONCURRENTLY idx_predictions_student_date
ON predictions(student_id, prediction_date DESC);

-- Partitioning for large tables
CREATE TABLE attendance_records_2025_q1 PARTITION OF attendance_records
FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

-- Connection pooling configuration
-- postgresql.conf
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
work_mem = 4MB
```

---

## Backup and Disaster Recovery

### Automated Backups

```bash
#!/bin/bash
# scripts/backup_database.sh

set -e

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/edupulse_backup_$TIMESTAMP.sql"

echo "Starting database backup..."

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Create database dump
pg_dump $DATABASE_URL > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to S3 (optional)
if [[ ! -z "$AWS_S3_BACKUP_BUCKET" ]]; then
    aws s3 cp "$BACKUP_FILE.gz" "s3://$AWS_S3_BACKUP_BUCKET/database-backups/"
fi

# Clean up old backups (keep last 7 days)
find $BACKUP_DIR -name "edupulse_backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

### Disaster Recovery Plan

```yaml
# k8s/disaster-recovery.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: edupulse-prod
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: edupulse-secrets
                  key: DATABASE_URL
            command:
            - /bin/bash
            - -c
            - |
              pg_dump $DATABASE_URL | gzip > /backups/backup_$(date +%Y%m%d_%H%M%S).sql.gz
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Pod Startup Failures

```bash
# Check pod status
kubectl get pods -n edupulse-prod

# Describe pod for events
kubectl describe pod <pod-name> -n edupulse-prod

# Check logs
kubectl logs <pod-name> -n edupulse-prod --previous

# Common fixes:
# - Resource limits too low
# - Missing environment variables
# - Database connection issues
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- bash
psql $DATABASE_URL

# Check connection pool
kubectl exec -it <api-pod> -- python -c "
from src.db.database import engine
print(engine.pool.status())
"

# Monitor active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';
```

#### 3. Performance Issues

```bash
# Check resource usage
kubectl top pods -n edupulse-prod

# Monitor request latency
curl -w "@curl-format.txt" -o /dev/null -s "http://api.edupulse.com/health"

# Check database performance
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC LIMIT 10;
```

### Log Analysis

```bash
# Centralized logging with ELK/EFK stack
# Parse application logs
kubectl logs -f deployment/edupulse-api -n edupulse-prod | jq '.level="ERROR"'

# Search for specific errors
kubectl logs deployment/edupulse-api -n edupulse-prod --since=1h | grep "prediction failed"

# Monitor error patterns
kubectl logs deployment/edupulse-api -n edupulse-prod |
  grep ERROR |
  awk '{print $4}' |
  sort | uniq -c | sort -rn
```

---

This comprehensive deployment guide provides everything needed to successfully deploy and maintain EduPulse in production environments. Regular updates and monitoring ensure optimal performance and reliability.

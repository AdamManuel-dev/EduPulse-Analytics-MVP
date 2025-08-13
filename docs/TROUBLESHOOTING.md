# EduPulse Troubleshooting Guide

> **Last Updated**: 2025-08-13 02:56:19 CDT  
> **Version**: 1.0  

This comprehensive troubleshooting guide covers common issues, debugging techniques, and solutions encountered during EduPulse development, testing, and deployment.

## Quick Diagnosis Checklist

Before diving into specific issues, run this quick health check:

```bash
# 1. Check service health
curl -f http://localhost:8000/health || echo "❌ API unreachable"

# 2. Verify database connection  
psql $DATABASE_URL -c "SELECT 1" > /dev/null && echo "✅ Database OK" || echo "❌ Database issue"

# 3. Test Redis connectivity
redis-cli ping | grep -q PONG && echo "✅ Redis OK" || echo "❌ Redis issue"

# 4. Check Python environment
python -c "import src.api.main; print('✅ Python imports OK')" || echo "❌ Import issue"

# 5. Verify test database
python -m pytest tests/conftest.py::test_db_connection -v || echo "❌ Test DB issue"
```

## Common Issues and Solutions

### Installation Issues

#### Python Version Errors

**Problem**: `ERROR: This package requires Python 3.11+`

**Solution**:
```bash
# Check Python version
python --version

# Install Python 3.11+ using pyenv
pyenv install 3.11.7
pyenv local 3.11.7

# Or use conda
conda create -n edupulse python=3.11
conda activate edupulse
```

#### Dependency Installation Failures

**Problem**: `ERROR: Failed building wheel for psycopg2`

**Solution**:
```bash
# Install system dependencies first

# On macOS
brew install postgresql

# On Ubuntu/Debian
sudo apt-get install postgresql-dev python3-dev

# On RHEL/CentOS
sudo yum install postgresql-devel python3-devel

# Then retry
pip install psycopg2-binary  # Or use binary version
```

**Problem**: `ERROR: torch installation failed`

**Solution**:
```bash
# Install PyTorch separately based on your system
# CPU only
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu

# With CUDA 11.8
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

### Database Issues

#### Connection Errors

**Problem**: `psycopg2.OperationalError: could not connect to server`

**Solution**:
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# If not running, start it
docker-compose up -d postgres

# Verify connection string
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1"
```

#### Migration Errors

**Problem**: `ERROR: relation "students" does not exist`

**Solution**:
```bash
# Run database initialization
python -m src.db.database init

# Or manually run SQL
psql $DATABASE_URL < scripts/db/init.sql

# Verify tables exist
psql $DATABASE_URL -c "\dt"
```

#### TimescaleDB Extension Missing

**Problem**: `ERROR: extension "timescaledb" does not exist`

**Solution**:
```sql
-- Connect to database
psql $DATABASE_URL

-- Create extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Verify installation
SELECT * FROM timescaledb_information.hypertables;
```

### Redis Issues

#### Connection Refused

**Problem**: `redis.exceptions.ConnectionError: Connection refused`

**Solution**:
```bash
# Check if Redis is running
docker ps | grep redis

# Start Redis
docker-compose up -d redis

# Test connection
redis-cli ping
# Should return: PONG
```

#### Memory Issues

**Problem**: `OOM command not allowed when used memory > 'maxmemory'`

**Solution**:
```bash
# Connect to Redis
redis-cli

# Check memory usage
INFO memory

# Clear cache (development only!)
FLUSHDB

# Or increase memory limit in docker-compose.yml
# redis:
#   command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

### API Issues

#### Server Won't Start

**Problem**: `ERROR: [Errno 48] Address already in use`

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uvicorn src.api.main:app --port 8001
```

#### CORS Errors

**Problem**: `Access to fetch at 'http://localhost:8000' from origin 'http://localhost:3000' has been blocked by CORS`

**Solution**:
```python
# Update .env file
CORS_ORIGINS=["http://localhost:3000", "http://localhost:3001"]

# Or allow all origins (development only!)
CORS_ORIGINS=["*"]

# Restart the API server
```

#### Authentication Failures

**Problem**: `401 Unauthorized: Invalid token`

**Solution**:
```python
# Generate a new token
from src.auth import create_access_token

token = create_access_token(data={"sub": "user@example.com"})
print(f"Bearer {token}")

# Verify token is not expired
import jwt
decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
print(decoded["exp"])  # Expiration timestamp
```

### Model Issues

#### Model File Not Found

**Problem**: `FileNotFoundError: Model file not found at models/gru_model.pt`

**Solution**:
```bash
# Create models directory
mkdir -p models

# Train a new model
python -m src.training.trainer train --epochs 50

# Or download pre-trained model
wget https://example.com/models/gru_model.pt -O models/gru_model.pt

# Update .env
ML_MODEL_PATH=models/gru_model.pt
```

#### Out of Memory During Training

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce batch size in training config
config = {
    'batch_size': 16,  # Reduced from 32
    'gradient_accumulation_steps': 2  # Compensate with accumulation
}

# Or use CPU
device = torch.device('cpu')
model = model.to(device)

# Clear GPU cache
torch.cuda.empty_cache()
```

#### Poor Model Performance

**Problem**: Model accuracy below expected threshold

**Solution**:
```python
# Check data quality
from src.data import validate_data
issues = validate_data(training_data)
print(f"Data issues: {issues}")

# Adjust hyperparameters
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'hidden_size': [64, 128, 256],
    'dropout': [0.2, 0.3, 0.4]
}

# Use more training data
min_samples = 1000  # Increase minimum

# Check for class imbalance
from collections import Counter
print(Counter(labels))
```

### Feature Extraction Issues

#### Missing Data Errors

**Problem**: `ValueError: Cannot extract features: insufficient data`

**Solution**:
```python
# Check data availability
from src.features import check_data_availability

availability = check_data_availability(student_id)
print(f"Attendance data: {availability['attendance']} days")
print(f"Grade data: {availability['grades']} records")

# Use imputation for missing values
from src.features import impute_missing
data = impute_missing(data, strategy='forward_fill')
```

#### Slow Feature Extraction

**Problem**: Feature extraction taking > 5 seconds per student

**Solution**:
```python
# Enable caching
FEATURE_CACHE_TTL=3600  # in .env

# Use batch extraction
from src.features.pipeline import FeaturePipeline
pipeline = FeaturePipeline()
features = pipeline.batch_extract(student_ids, use_cache=True)

# Optimize database queries
# Add indexes to frequently queried columns
CREATE INDEX idx_student_date ON student_features(student_id, date);
```

### Celery/Background Task Issues

#### Tasks Not Executing

**Problem**: Celery tasks stuck in pending state

**Solution**:
```bash
# Check if workers are running
celery -A src.tasks.worker inspect active

# Check for errors in worker logs
docker-compose logs celery-worker

# Restart workers
docker-compose restart celery-worker celery-beat

# Clear task queue (development only!)
celery -A src.tasks.worker purge
```

#### Task Timeouts

**Problem**: `SoftTimeLimitExceeded: Task timed out`

**Solution**:
```python
# Increase task timeout in celery config
CELERY_TASK_SOFT_TIME_LIMIT = 300  # 5 minutes
CELERY_TASK_TIME_LIMIT = 600  # 10 minutes

# Or per-task basis
@celery.task(soft_time_limit=300, time_limit=600)
def long_running_task():
    pass
```

### Docker Issues

#### Container Fails to Start

**Problem**: `docker-compose up` fails with error

**Solution**:
```bash
# Check logs
docker-compose logs <service-name>

# Rebuild containers
docker-compose build --no-cache

# Remove old containers and volumes
docker-compose down -v
docker-compose up -d

# Check disk space
df -h
docker system prune -a  # Clean up unused images
```

#### Network Issues Between Containers

**Problem**: Containers can't communicate

**Solution**:
```yaml
# In docker-compose.yml, use service names for internal communication
environment:
  DATABASE_URL: postgresql://user:pass@postgres:5432/db  # Not localhost
  REDIS_URL: redis://redis:6379/0  # Not localhost
```

### Performance Issues

#### Slow API Response Times

**Problem**: API latency > 1 second

**Diagnosis**:
```python
# Enable profiling
from src.utils.profiling import profile_endpoint

@profile_endpoint
async def slow_endpoint():
    pass

# Check slow queries
EXPLAIN ANALYZE SELECT * FROM large_table;

# Monitor with Prometheus
curl http://localhost:8000/metrics | grep http_request_duration
```

**Solutions**:
1. Enable caching
2. Add database indexes
3. Use pagination for large results
4. Implement query optimization
5. Use async processing for heavy operations

#### High Memory Usage

**Problem**: Application consuming excessive memory

**Solution**:
```python
# Profile memory usage
from memory_profiler import profile

@profile
def memory_intensive_function():
    pass

# Use generators for large datasets
def process_large_dataset():
    for chunk in pd.read_csv('large_file.csv', chunksize=1000):
        yield process_chunk(chunk)

# Clear caches periodically
import gc
gc.collect()
```

### Testing Issues

#### Tests Failing Locally

**Problem**: Tests pass in CI but fail locally

**Solution**:
```bash
# Ensure clean test environment
pytest --cache-clear

# Use test database
export DATABASE_URL=postgresql://test:test@localhost:5432/test_db

# Reset test data
pytest --fixtures tests/fixtures/

# Run tests in isolation
pytest -x tests/unit/test_specific.py::test_function
```

#### Coverage Reports Not Generated

**Problem**: No coverage report after running tests

**Solution**:
```bash
# Install coverage tools
pip install pytest-cov

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

#### E2E Tests Failing

**Problem**: End-to-end tests fail with 404 errors or import issues

**Common Causes & Solutions**:

1. **Non-existent endpoints**:
```bash
# Check if endpoint exists
curl -I http://localhost:8000/api/v1/nonexistent

# Update tests to use only implemented endpoints:
# - /health (health check)
# - /api/v1/students/* (student CRUD)
# - /api/v1/predict (risk prediction)
# - /api/v1/train/* (model training)
```

2. **Missing data_generation module**:
```bash
# Check if module exists
ls src/data/generation.py

# If missing, either:
# - Create the module, or
# - Update imports in tests to use existing modules
```

3. **Database connection issues in tests**:
```python
# Ensure test fixtures are properly configured
# Check tests/conftest.py for proper database setup
# Verify test database URL in test settings
```

#### Feature Extractor Tests Failing

**Problem**: Tests fail after model schema changes

**Common Issues**:
- Field name mismatches (e.g., `student_id` vs `district_id`)
- Missing Grade model fields
- Incorrect foreign key relationships

**Solution**:
```python
# Check Grade model schema
from src.db.models import Grade
print(Grade.__table__.columns.keys())

# Update test fixtures to match current schema
# Verify field mappings in feature extraction code
```

#### Database Integration Tests

**Problem**: `FOREIGN KEY constraint failed` in SQLite tests

**Solution**:
```python
# Ensure foreign key constraints are enabled (already done in conftest.py)
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

# Use proper UUIDs for related records
student = Student(district_id=f"STU{uuid4()[:8]}")
attendance = AttendanceRecord(student_id=student.id, ...)
```

#### Parallel Test Execution Issues

**Problem**: Tests interfere with each other when run in parallel

**Solution**:
```bash
# Run tests sequentially for debugging
pytest -x --disable-warnings tests/

# Use test isolation with unique identifiers
# Ensure proper cleanup in test fixtures
```

#### Mock vs Real Database Tests

**Problem**: Mixing mocked and real database calls causes inconsistencies

**Solution**:
```python
# Use consistent testing approach:
# Option 1: Pure unit tests with mocks
@patch('src.db.database.get_db')
def test_with_mock_db(mock_db):
    pass

# Option 2: Integration tests with test database
def test_with_real_db(db_session):
    # Use actual database operations
    pass

# Avoid mixing approaches in the same test
```

## Debugging Tools

### Logging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add custom logging
logger = logging.getLogger(__name__)
logger.debug(f"Variable value: {variable}")
```

### Interactive Debugging

```python
# Use pdb for debugging
import pdb; pdb.set_trace()

# Or use ipdb (better interface)
import ipdb; ipdb.set_trace()

# In async code
import asyncio
asyncio.create_task(debug_function())
```

### API Debugging

```bash
# Use curl with verbose output
curl -v http://localhost:8000/api/v1/predict

# Use httpie for better formatting
http POST localhost:8000/api/v1/predict student_id=STU001

# Use Postman or Thunder Client in VS Code
```

### Database Debugging

```sql
-- Check query performance
EXPLAIN ANALYZE <your-query>;

-- View active queries
SELECT * FROM pg_stat_activity WHERE state = 'active';

-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(tablename::regclass) DESC;
```

## Performance Monitoring

### Application Metrics

```python
# Add custom metrics
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@prediction_duration.time()
def make_prediction():
    prediction_counter.inc()
    # ... prediction logic
```

### System Monitoring

```bash
# Monitor resource usage
htop  # Interactive process viewer
iotop  # I/O usage
nethogs  # Network usage by process

# Docker stats
docker stats

# Database monitoring
pg_stat_statements  # Query performance
pg_stat_user_tables  # Table statistics
```

### CI/CD Pipeline Issues

#### GitHub Actions Failing

**Problem**: CI pipeline fails on pull requests or pushes

**Common Causes**:
1. **Environment variable missing**:
```yaml
# Check .github/workflows/test.yml
env:
  DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
  REDIS_URL: redis://localhost:6379/0
```

2. **Docker service not starting**:
```yaml
# Ensure services are properly configured
services:
  postgres:
    image: postgres:15
    env:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: test_db
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5
```

3. **Test dependencies not installed**:
```bash
# Verify all test dependencies in requirements.txt
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.20.0
httpx>=0.24.0
```

#### Docker Build Failures

**Problem**: Docker image fails to build

**Solution**:
```bash
# Build with verbose output
docker build --progress=plain --no-cache -t edupulse .

# Check for common issues:
# - Base image availability
# - Copy paths in Dockerfile
# - Missing system dependencies
# - File permissions

# Test build locally first
docker-compose build --no-cache api
```

### Production Environment Issues

#### High Latency Responses

**Problem**: API responses taking >5 seconds

**Diagnosis Steps**:
```bash
# 1. Check application metrics
curl http://localhost:8000/metrics | grep http_request_duration

# 2. Profile database queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;

# 3. Monitor system resources
docker exec -it edupulse-api htop

# 4. Check for memory leaks
python -m memory_profiler src/api/main.py
```

**Solutions**:
- Add database indexes for frequent queries
- Enable query result caching with Redis
- Implement request/response compression
- Use async/await for I/O operations
- Add connection pooling

#### Memory Leaks

**Problem**: Container memory usage continuously increasing

**Diagnosis**:
```python
# Add memory tracking to critical functions
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function implementation
    pass

# Monitor garbage collection
import gc
print(f"Garbage collector stats: {gc.get_stats()}")
```

**Solutions**:
- Use generators for large data processing
- Clear caches periodically
- Close database connections properly
- Monitor tensor memory in PyTorch models

#### SSL/TLS Certificate Issues

**Problem**: HTTPS certificate errors in production

**Solution**:
```bash
# Check certificate expiration
openssl x509 -in /path/to/cert.pem -text -noout | grep "Not After"

# Verify certificate chain
openssl s_client -connect your-domain.com:443 -showcerts

# Let's Encrypt renewal (if using certbot)
sudo certbot renew --dry-run

# Update certificate in load balancer/ingress
kubectl get certificate -n edupulse
kubectl describe certificate edupulse-tls -n edupulse
```

### Load Balancing and Scaling Issues

#### High CPU Usage Under Load

**Problem**: CPU usage spikes during peak hours

**Diagnosis**:
```bash
# Monitor CPU usage patterns
kubectl top pods -n edupulse

# Check horizontal pod autoscaling
kubectl get hpa -n edupulse

# Profile application performance
python -m cProfile -o profile.stats src/api/main.py
```

**Solutions**:
```yaml
# Increase replicas and configure HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: edupulse-hpa
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
```

#### Database Connection Pool Exhaustion

**Problem**: "Connection pool exhausted" errors

**Solution**:
```python
# Adjust connection pool settings
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30

# Monitor active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

# Close idle connections
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' AND query_start < now() - interval '1 hour';
```

### Security Issues

#### API Rate Limiting

**Problem**: API being overwhelmed by requests

**Solution**:
```python
# Implement rate limiting middleware
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/predict")
@limiter.limit("100/minute")
async def predict_risk(request: Request, ...):
    pass
```

#### Authentication Token Issues

**Problem**: JWT tokens expiring unexpectedly

**Diagnosis**:
```python
# Check token validity
import jwt
from datetime import datetime

def debug_token(token):
    try:
        decoded = jwt.decode(token, verify=False)  # Don't verify for debugging
        exp = datetime.fromtimestamp(decoded['exp'])
        print(f"Token expires: {exp}")
        print(f"Current time: {datetime.now()}")
        print(f"Time remaining: {exp - datetime.now()}")
    except Exception as e:
        print(f"Token decode error: {e}")
```

**Solution**:
```python
# Adjust token expiration times
ACCESS_TOKEN_EXPIRE_MINUTES=60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS=30   # 30 days

# Implement token refresh endpoint
@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    # Validate and issue new access token
    pass
```

## Getting Help

### Log Locations

- API logs: `logs/api.log`
- Celery logs: `logs/celery.log`
- Database logs: `docker-compose logs postgres`
- Redis logs: `docker-compose logs redis`

### Diagnostic Commands

```bash
# System health check
curl http://localhost:8000/health

# Database connection test
psql $DATABASE_URL -c "SELECT version()"

# Redis connection test
redis-cli ping

# Python package versions
pip list | grep -E "(fastapi|torch|celery|redis|psycopg2)"
```

### Reporting Issues

When reporting issues, include:

1. Error message and stack trace
2. Steps to reproduce
3. Environment details:
   ```bash
   python --version
   pip freeze > requirements-freeze.txt
   docker version
   ```
4. Relevant logs
5. Configuration files (sanitized)

### Support Channels

- GitHub Issues: [github.com/your-org/edupulse/issues](https://github.com/your-org/edupulse/issues)
- Documentation: This guide and `/docs` directory
- Community Forum: [forum.edupulse.com](https://forum.edupulse.com)
- Email Support: support@edupulse.com

---

*For development setup, see [Getting Started](./guides/GETTING_STARTED.md). For API issues, consult the [API Reference](./API_REFERENCE.md).*
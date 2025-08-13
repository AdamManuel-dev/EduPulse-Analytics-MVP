# Database Module Documentation

## Overview

The database module provides the data persistence layer for the EduPulse Analytics system, managing student records, predictions, and system data using PostgreSQL with SQLAlchemy ORM.

## Architecture

```
┌─────────────────────────────────────────┐
│           Database Module               │
├─────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐     │
│  │   Models    │   │   Session   │     │
│  └─────────────┘   └─────────────┘     │
│         ▼                 ▼             │
│  ┌─────────────────────────────────┐   │
│  │       SQLAlchemy ORM            │   │
│  └─────────────────────────────────┘   │
│         ▼                              │
│  ┌─────────────────────────────────┐   │
│  │       PostgreSQL Database       │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Schema

### Core Tables

#### Students Table
```sql
CREATE TABLE students (
    student_id VARCHAR PRIMARY KEY,
    first_name VARCHAR NOT NULL,
    last_name VARCHAR NOT NULL,
    grade_level INTEGER,
    enrollment_date DATE,
    date_of_birth DATE,
    gender VARCHAR(10),
    ethnicity VARCHAR(50),
    special_ed_status BOOLEAN DEFAULT FALSE,
    english_learner_status BOOLEAN DEFAULT FALSE,
    socioeconomic_status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Attendance Records Table
```sql
CREATE TABLE attendance_records (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR REFERENCES students(student_id),
    date DATE NOT NULL,
    status VARCHAR(20) NOT NULL, -- present, absent, tardy, excused
    period INTEGER,
    minutes_late INTEGER DEFAULT 0,
    reason VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Grades Table
```sql
CREATE TABLE grades (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR REFERENCES students(student_id),
    course_id VARCHAR NOT NULL,
    assignment_id VARCHAR,
    assignment_type VARCHAR(50), -- homework, quiz, test, project
    grade_value FLOAT NOT NULL,
    max_grade_value FLOAT DEFAULT 100,
    submission_date DATE,
    semester VARCHAR(20),
    academic_year VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Discipline Incidents Table
```sql
CREATE TABLE discipline_incidents (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR REFERENCES students(student_id),
    incident_date DATE NOT NULL,
    incident_type VARCHAR(50),
    severity_level INTEGER CHECK (severity_level BETWEEN 1 AND 5),
    description TEXT,
    action_taken VARCHAR(255),
    reported_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Predictions Table
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR REFERENCES students(student_id),
    risk_score FLOAT NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    risk_category VARCHAR(20), -- low, medium, high, critical
    confidence FLOAT,
    risk_factors JSONB,
    model_version VARCHAR(50),
    prediction_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes

```sql
-- Performance indexes
CREATE INDEX idx_students_grade_level ON students(grade_level);
CREATE INDEX idx_attendance_student_date ON attendance_records(student_id, date);
CREATE INDEX idx_grades_student_course ON grades(student_id, course_id);
CREATE INDEX idx_predictions_student_date ON predictions(student_id, prediction_date);
CREATE INDEX idx_predictions_risk_category ON predictions(risk_category);

-- Full-text search indexes
CREATE INDEX idx_students_name ON students(first_name, last_name);
```

## Key Components

### Database Connection (`src/db/database.py`)

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.config.settings import get_settings

# Create engine
engine = create_engine(
    get_settings().database_url,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Session:
    """Get database session for dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Models (`src/db/models.py`)

The ORM models that map to database tables:

- `Student` - Student demographic and enrollment data
- `AttendanceRecord` - Daily attendance tracking
- `Grade` - Academic performance records
- `DisciplineIncident` - Behavioral incidents
- `Prediction` - ML model predictions

## Usage Examples

### Basic CRUD Operations

```python
from src.db.database import get_db
from src.db.models import Student, Prediction

# Create a student
with get_db() as db:
    student = Student(
        student_id="STU001",
        first_name="John",
        last_name="Doe",
        grade_level=10
    )
    db.add(student)
    db.commit()

# Query students at risk
with get_db() as db:
    at_risk = db.query(Student).join(Prediction).filter(
        Prediction.risk_category.in_(['high', 'critical'])
    ).all()
```

### Complex Queries

```python
# Get student with recent attendance issues
from datetime import datetime, timedelta
from sqlalchemy import and_

with get_db() as db:
    cutoff_date = datetime.now() - timedelta(days=30)
    
    students_with_issues = db.query(Student).join(AttendanceRecord).filter(
        and_(
            AttendanceRecord.date >= cutoff_date,
            AttendanceRecord.status.in_(['absent', 'tardy'])
        )
    ).group_by(Student.student_id).having(
        func.count(AttendanceRecord.id) > 5
    ).all()
```

### Batch Operations

```python
# Bulk insert attendance records
attendance_data = [
    {"student_id": "STU001", "date": "2024-01-01", "status": "present"},
    {"student_id": "STU002", "date": "2024-01-01", "status": "absent"},
    # ... more records
]

with get_db() as db:
    db.bulk_insert_mappings(AttendanceRecord, attendance_data)
    db.commit()
```

## Redis Integration

The system uses Redis for caching frequently accessed data:

### Cache Configuration

```python
import redis
from src.config.settings import get_settings

redis_client = redis.Redis.from_url(
    get_settings().redis_url,
    decode_responses=True
)
```

### Caching Patterns

```python
# Cache prediction results
def cache_prediction(student_id: str, prediction: dict):
    key = f"prediction:{student_id}"
    redis_client.setex(
        key, 
        3600,  # 1 hour TTL
        json.dumps(prediction)
    )

# Get cached prediction
def get_cached_prediction(student_id: str):
    key = f"prediction:{student_id}"
    data = redis_client.get(key)
    return json.loads(data) if data else None
```

## Database Migrations

### Using Alembic

```bash
# Initialize migrations
alembic init alembic

# Create a new migration
alembic revision --autogenerate -m "Add new column"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1
```

### Migration Best Practices

1. Always review auto-generated migrations
2. Test migrations on a copy of production data
3. Include both upgrade and downgrade scripts
4. Use batch operations for large data modifications

## Performance Optimization

### Query Optimization

1. **Use eager loading for relationships**:
```python
students = db.query(Student).options(
    joinedload(Student.attendance_records)
).all()
```

2. **Batch queries**:
```python
# Instead of N+1 queries
for student_id in student_ids:
    student = db.query(Student).get(student_id)
    
# Use IN clause
students = db.query(Student).filter(
    Student.student_id.in_(student_ids)
).all()
```

3. **Use database views for complex queries**:
```sql
CREATE VIEW student_risk_summary AS
SELECT 
    s.student_id,
    s.first_name,
    s.last_name,
    p.risk_score,
    p.risk_category
FROM students s
JOIN predictions p ON s.student_id = p.student_id
WHERE p.prediction_date = (
    SELECT MAX(prediction_date) 
    FROM predictions 
    WHERE student_id = s.student_id
);
```

### Connection Pooling

```python
# Configure connection pool
engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # Number of connections to maintain
    max_overflow=40,        # Maximum overflow connections
    pool_timeout=30,        # Timeout for getting connection
    pool_recycle=1800,      # Recycle connections after 30 minutes
    pool_pre_ping=True      # Test connections before using
)
```

## Monitoring

### Health Checks

```python
async def check_database_health():
    """Check database connectivity and performance"""
    try:
        with get_db() as db:
            # Simple connectivity check
            result = db.execute("SELECT 1")
            
            # Check table accessibility
            count = db.query(Student).count()
            
            # Check response time
            start = time.time()
            db.query(Prediction).limit(100).all()
            query_time = time.time() - start
            
            return {
                "status": "healthy",
                "student_count": count,
                "query_time_ms": query_time * 1000
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

### Metrics Collection

```python
# Track query performance
from prometheus_client import Histogram

query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['operation', 'table']
)

@query_duration.time()
def execute_query(operation, table, query):
    return db.execute(query)
```

## Backup and Recovery

### Backup Strategy

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump $DATABASE_URL > backup_$DATE.sql

# Compress and upload to S3
gzip backup_$DATE.sql
aws s3 cp backup_$DATE.sql.gz s3://backups/edupulse/
```

### Point-in-Time Recovery

```bash
# Enable WAL archiving in postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'

# Restore to specific time
pg_basebackup -D /var/lib/postgresql/data_restore
recovery_target_time = '2024-01-15 14:30:00'
```

## Security Considerations

### Access Control

1. **Use role-based permissions**:
```sql
-- Create read-only role
CREATE ROLE readonly;
GRANT CONNECT ON DATABASE edupulse TO readonly;
GRANT USAGE ON SCHEMA public TO readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;

-- Create application role
CREATE ROLE app_user;
GRANT CONNECT ON DATABASE edupulse TO app_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_user;
```

2. **Encrypt sensitive data**:
```python
from cryptography.fernet import Fernet

def encrypt_field(value: str) -> str:
    cipher = Fernet(settings.encryption_key)
    return cipher.encrypt(value.encode()).decode()

def decrypt_field(encrypted_value: str) -> str:
    cipher = Fernet(settings.encryption_key)
    return cipher.decrypt(encrypted_value.encode()).decode()
```

3. **SQL injection prevention**:
```python
# Always use parameterized queries
# Good
student = db.query(Student).filter(
    Student.student_id == student_id
).first()

# Bad - vulnerable to SQL injection
query = f"SELECT * FROM students WHERE student_id = '{student_id}'"
db.execute(query)
```

## Troubleshooting

### Common Issues

1. **Connection pool exhaustion**:
   - Symptom: `TimeoutError: QueuePool limit exceeded`
   - Solution: Increase pool_size or check for connection leaks

2. **Slow queries**:
   - Use `EXPLAIN ANALYZE` to identify bottlenecks
   - Add appropriate indexes
   - Consider query restructuring

3. **Lock contention**:
   - Monitor with `pg_locks` view
   - Use `SELECT FOR UPDATE SKIP LOCKED` for queue-like operations

### Debug Logging

```python
# Enable SQLAlchemy query logging
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Log slow queries
from sqlalchemy import event

@event.listens_for(Engine, "after_execute")
def receive_after_execute(conn, clauseelement, multiparams, params, result):
    duration = time.time() - conn.info.get('query_start_time', time.time())
    if duration > 1.0:  # Log queries taking more than 1 second
        logger.warning(f"Slow query ({duration:.2f}s): {clauseelement}")
```

## Best Practices

1. **Always use transactions for multi-step operations**
2. **Close connections properly using context managers**
3. **Implement retry logic for transient failures**
4. **Monitor connection pool metrics**
5. **Regular vacuum and analyze operations**
6. **Keep statistics up to date**
7. **Use prepared statements for repeated queries**
8. **Implement proper indexing strategy**

## Related Documentation

- [API Routes](./API_ROUTES.md) - How the API uses the database
- [Feature Extraction](./features.md) - Data retrieval for ML features
- [Architecture](../ARCHITECTURE.md) - System-wide data flow
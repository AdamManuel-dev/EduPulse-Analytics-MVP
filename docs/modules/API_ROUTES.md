# API Routes Module Documentation

## Overview

The API Routes module implements RESTful endpoints for the EduPulse Analytics system using FastAPI. It provides interfaces for predictions, student management, model training, and system health monitoring.

## Architecture

```
┌──────────────────────────────────────────┐
│            FastAPI Application           │
├──────────────────────────────────────────┤
│  ┌──────────────────────────────────┐   │
│  │         Route Handlers           │   │
│  ├──────────────────────────────────┤   │
│  │ • /predictions  • /students      │   │
│  │ • /training     • /health        │   │
│  └────────────┬─────────────────────┘   │
│               ▼                          │
│  ┌──────────────────────────────────┐   │
│  │      Dependency Injection        │   │
│  ├──────────────────────────────────┤   │
│  │ • Database Sessions              │   │
│  │ • Authentication                 │   │
│  │ • Settings Configuration         │   │
│  └──────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

## Route Modules

### Health Routes (`src/api/routes/health.py`)

Health check and monitoring endpoints.

#### GET /health
```python
@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    
    Returns:
        dict: Health status and timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow()
    }
```

#### GET /health/detailed
```python
@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Comprehensive health check with component status
    
    Checks:
        - Database connectivity
        - Model availability
        - Redis connection
        - System resources
    
    Returns:
        dict: Detailed health status of all components
    """
```

### Prediction Routes (`src/api/routes/predictions.py`)

ML prediction endpoints for risk assessment.

#### POST /predictions/predict
```python
@router.post("/predictions/predict", response_model=PredictResponse)
async def predict_single(
    request: PredictRequest,
    db: Session = Depends(get_db)
):
    """
    Generate risk prediction for a single student
    
    Args:
        request: Contains student_id and optional parameters
        
    Returns:
        PredictResponse: Risk score, category, and contributing factors
        
    Example:
        POST /predictions/predict
        {
            "student_id": "STU001",
            "include_factors": true
        }
    """
```

#### POST /predictions/predict-batch
```python
@router.post("/predictions/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(
    request: BatchPredictRequest,
    db: Session = Depends(get_db)
):
    """
    Generate predictions for multiple students
    
    Args:
        request: List of student IDs and parameters
        
    Returns:
        BatchPredictResponse: Predictions for all students
        
    Example:
        POST /predictions/predict-batch
        {
            "student_ids": ["STU001", "STU002", "STU003"],
            "top_k": 10
        }
    """
```

#### GET /predictions/history/{student_id}
```python
@router.get("/predictions/history/{student_id}")
async def get_prediction_history(
    student_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Retrieve historical predictions for a student
    
    Args:
        student_id: Unique student identifier
        limit: Maximum number of predictions to return
        
    Returns:
        List of historical predictions with timestamps
    """
```

### Student Routes (`src/api/routes/students.py`)

Student data management endpoints.

#### GET /students
```python
@router.get("/students", response_model=List[StudentResponse])
async def list_students(
    skip: int = 0,
    limit: int = 100,
    grade_level: Optional[int] = None,
    risk_category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List students with optional filtering
    
    Query Parameters:
        skip: Pagination offset
        limit: Maximum results
        grade_level: Filter by grade
        risk_category: Filter by risk level
        
    Returns:
        List of students matching criteria
    """
```

#### GET /students/{student_id}
```python
@router.get("/students/{student_id}", response_model=StudentDetailResponse)
async def get_student(
    student_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed information for a specific student
    
    Returns:
        Complete student profile with recent metrics
    """
```

#### POST /students
```python
@router.post("/students", response_model=StudentResponse)
async def create_student(
    student: CreateStudentRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new student record
    
    Args:
        student: Student demographic and enrollment data
        
    Returns:
        Created student record
    """
```

#### PUT /students/{student_id}
```python
@router.put("/students/{student_id}", response_model=StudentResponse)
async def update_student(
    student_id: str,
    updates: UpdateStudentRequest,
    db: Session = Depends(get_db)
):
    """
    Update student information
    
    Args:
        student_id: Student to update
        updates: Fields to update
        
    Returns:
        Updated student record
    """
```

### Training Routes (`src/api/routes/training.py`)

Model training and management endpoints.

#### POST /training/train
```python
@router.post("/training/train")
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Initiate model training process
    
    Args:
        request: Training configuration
        
    Returns:
        Training job ID and status
        
    Example:
        POST /training/train
        {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "early_stopping": true
        }
    """
```

#### GET /training/status/{job_id}
```python
@router.get("/training/status/{job_id}")
async def get_training_status(job_id: str):
    """
    Check status of a training job
    
    Returns:
        Current status, metrics, and progress
    """
```

#### GET /training/metrics
```python
@router.get("/training/metrics")
async def get_training_metrics():
    """
    Get performance metrics of current model
    
    Returns:
        Accuracy, F1 score, confusion matrix, etc.
    """
```

## Request/Response Models

### Prediction Models

```python
class PredictRequest(BaseModel):
    student_id: str
    reference_date: Optional[date] = None
    include_factors: bool = True

class PredictResponse(BaseModel):
    student_id: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_category: str
    confidence: float
    contributing_factors: Optional[List[RiskFactor]]
    timestamp: datetime

class RiskFactor(BaseModel):
    factor: str
    weight: float
    details: str
```

### Student Models

```python
class StudentResponse(BaseModel):
    student_id: str
    first_name: str
    last_name: str
    grade_level: int
    enrollment_date: date
    risk_category: Optional[str]
    last_updated: datetime

class CreateStudentRequest(BaseModel):
    student_id: str
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    grade_level: int = Field(..., ge=1, le=12)
    date_of_birth: date
    enrollment_date: Optional[date]
```

## Dependency Injection

### Database Session

```python
def get_db():
    """Provide database session for request"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Authentication

```python
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Validate JWT token and return current user"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception
```

### Settings

```python
def get_settings():
    """Get application settings"""
    return Settings()
```

## Middleware

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Request Logging

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"completed in {duration:.3f}s "
        f"with status {response.status_code}"
    )
    return response
```

### Error Handling

```python
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
```

## API Versioning

```python
# Version 1 routes
v1 = APIRouter(prefix="/api/v1")
v1.include_router(health.router, tags=["health"])
v1.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
v1.include_router(students.router, prefix="/students", tags=["students"])
v1.include_router(training.router, prefix="/training", tags=["training"])

app.include_router(v1)
```

## Authentication & Security

### JWT Token Generation

```python
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

### Protected Routes

```python
@router.post("/protected")
async def protected_route(
    current_user: str = Depends(get_current_user)
):
    """Route that requires authentication"""
    return {"user": current_user}
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/limited")
@limiter.limit("5/minute")
async def rate_limited_route(request: Request):
    return {"message": "This route is rate limited"}
```

## Testing

### Unit Tests

```python
def test_predict_single():
    """Test single prediction endpoint"""
    response = client.post(
        "/api/v1/predictions/predict",
        json={"student_id": "STU001"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert 0 <= data["risk_score"] <= 1
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_prediction_flow():
    """Test complete prediction workflow"""
    # Create student
    student_response = await client.post(
        "/api/v1/students",
        json=student_data
    )
    
    # Generate prediction
    predict_response = await client.post(
        "/api/v1/predictions/predict",
        json={"student_id": student_response.json()["student_id"]}
    )
    
    assert predict_response.status_code == 200
```

## Performance Optimization

### Response Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def get_cached_prediction(student_id: str, date_str: str):
    """Cache predictions for repeated requests"""
    return prediction_service.predict_risk(student_id, date_str)
```

### Async Processing

```python
@router.post("/async-process")
async def async_process(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """Queue long-running tasks"""
    task_id = str(uuid4())
    background_tasks.add_task(process_data, task_id, request)
    return {"task_id": task_id, "status": "processing"}
```

### Database Query Optimization

```python
# Use select_related for foreign keys
students = db.query(Student).options(
    selectinload(Student.predictions)
).all()

# Batch queries
student_ids = ["STU001", "STU002", "STU003"]
students = db.query(Student).filter(
    Student.student_id.in_(student_ids)
).all()
```

## Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)
```

### Health Metrics

```python
@router.get("/metrics")
async def get_metrics():
    """Expose application metrics"""
    return {
        "requests_total": request_count._value.sum(),
        "avg_response_time": request_duration._sum.sum() / request_duration._count.sum(),
        "active_connections": len(app.state.connections),
        "cache_hit_rate": cache.hit_rate()
    }
```

## Error Handling

### Custom Exceptions

```python
class StudentNotFoundError(HTTPException):
    def __init__(self, student_id: str):
        super().__init__(
            status_code=404,
            detail=f"Student {student_id} not found"
        )

class PredictionError(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=500,
            detail=f"Prediction failed: {message}"
        )
```

### Error Responses

```python
@app.exception_handler(StudentNotFoundError)
async def student_not_found_handler(request: Request, exc: StudentNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error": "STUDENT_NOT_FOUND",
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

## API Documentation

### OpenAPI/Swagger

```python
app = FastAPI(
    title="EduPulse Analytics API",
    description="Student dropout risk prediction system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="EduPulse Analytics API",
        version="1.0.0",
        description="Complete API documentation",
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
```

## Best Practices

1. **Use proper HTTP status codes**
   - 200: Success
   - 201: Created
   - 400: Bad Request
   - 401: Unauthorized
   - 404: Not Found
   - 500: Internal Server Error

2. **Implement request validation**
   - Use Pydantic models
   - Validate field types and ranges
   - Provide clear error messages

3. **Follow RESTful conventions**
   - Use nouns for resources
   - Use HTTP verbs appropriately
   - Maintain consistent URL structure

4. **Version your API**
   - Use URL path versioning (/api/v1/)
   - Maintain backward compatibility
   - Document breaking changes

5. **Implement proper logging**
   - Log all requests and responses
   - Include correlation IDs
   - Use structured logging

## Related Documentation

- [API Reference](../API_REFERENCE.md) - Complete endpoint documentation
- [Database Module](./DATABASE.md) - Data persistence layer
- [Architecture](../ARCHITECTURE.md) - System design overview
# EduPulse API Reference

**🎯 Complete API documentation for dropout risk prediction system**

## Overview

The EduPulse Analytics API is a RESTful service that provides AI-powered student dropout risk predictions for K-12 schools. Built with FastAPI, it offers automatic OpenAPI documentation, comprehensive error handling, and production-ready performance.

**🚀 Key Features:**
- **Machine Learning Predictions**: GRU-based neural networks with 60+ day early warning
- **Batch Processing**: Efficient processing of 100+ students simultaneously  
- **Real-time Analysis**: <100ms response times for individual predictions
- **Explainable AI**: Detailed risk factor analysis with intervention recommendations
- **Production Ready**: Rate limiting, authentication, monitoring, and scaling support

**🔗 Quick Links:**
- 📊 **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- 📖 **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- 🔍 **OpenAPI Schema**: http://localhost:8000/openapi.json

## Base URL

The API base URL varies by environment. Use the appropriate URL for your deployment:

```text
# Local Development
http://localhost:8000/api/v1

# Staging Environment
https://staging-api.edupulse.com/api/v1

# Production Environment
https://api.edupulse.com/api/v1
```

**Environment Configuration:**
- **Local Development**: Used for testing and development. No HTTPS required.
- **Staging**: Mirror of production for testing before deployment. Requires authentication.
- **Production**: Live system serving real school data. Requires HTTPS and strong authentication.

## Authentication

The API uses JWT (JSON Web Token) authentication for secure access. All endpoints except `/health` require a valid JWT token.

### Authentication Header Format

Include the JWT token in the Authorization header of every API request:

```text
Authorization: Bearer <your-token>
```

**Example with actual token:**
```bash
# Example API call with authentication
curl -X GET https://api.edupulse.com/api/v1/students/STU123456 \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -H "Content-Type: application/json"
```

### Obtaining Authentication Tokens

**For Development:**
```bash
# Login to get tokens
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "counselor@school.edu",
    "password": "your-password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Token Management

- **Access Tokens**: Valid for 1 hour (3600 seconds)
- **Refresh Tokens**: Valid for 30 days, use to get new access tokens
- **Token Refresh**: Use the refresh token before the access token expires

**Token Refresh Example:**
```bash
curl -X POST http://localhost:8000/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4..."
  }'
```

### Authentication Errors

Common authentication issues and solutions:

- **401 Unauthorized**: Token is missing, expired, or invalid
- **403 Forbidden**: Token is valid but user lacks permission for this resource
- **Token Expired**: Use refresh token to get a new access token

## Response Format

All API responses follow a consistent structure to provide predictable data handling and error management.

### Standard Response Structure

```json
{
  "status": "success|error",
  "data": {},
  "message": "Response message", 
  "timestamp": "2025-08-13T00:00:00Z"
}
```

**Field Explanations:**

- **`status`**: Indicates if the request was successful (`"success"`) or failed (`"error"`)
- **`data`**: Contains the actual response data (empty object `{}` for errors)
- **`message`**: Human-readable message describing the result or error
- **`timestamp`**: ISO 8601 timestamp when the response was generated

### Success Response Examples

**Single Student Prediction:**
```json
{
  "status": "success",
  "data": {
    "student_id": "STU123456",
    "risk_score": 0.75,
    "risk_category": "high",
    "confidence": 0.92
  },
  "message": "Prediction generated successfully",
  "timestamp": "2025-08-13T10:30:45Z"
}
```

**Student List Response:**
```json
{
  "status": "success", 
  "data": {
    "students": [
      {"student_id": "STU001", "first_name": "John", "risk_score": 0.45},
      {"student_id": "STU002", "first_name": "Jane", "risk_score": 0.23}
    ],
    "total": 2,
    "pagination": {
      "limit": 20,
      "offset": 0,
      "has_more": false
    }
  },
  "message": "Retrieved 2 students successfully",
  "timestamp": "2025-08-13T10:30:45Z"
}
```

### Error Response Examples

**Student Not Found (404):**
```json
{
  "status": "error",
  "data": {},
  "message": "Student with ID 'STU999999' not found in the system",
  "timestamp": "2025-08-13T10:30:45Z"
}
```

**Validation Error (400):**
```json
{
  "status": "error",
  "data": {
    "validation_errors": [
      {"field": "grade_level", "message": "Must be between 1 and 12"},
      {"field": "enrollment_date", "message": "Must be a valid date in YYYY-MM-DD format"}
    ]
  },
  "message": "Request validation failed",
  "timestamp": "2025-08-13T10:30:45Z"
}
```

### Response Headers

All responses include additional metadata in headers:

```text
Content-Type: application/json
X-Response-Time: 45ms
X-Request-ID: req_123456789
X-API-Version: v1.0.0
X-RateLimit-Remaining: 95
```

**Header Explanations:**
- **`X-Response-Time`**: Server processing time in milliseconds
- **`X-Request-ID`**: Unique identifier for request tracking and debugging
- **`X-API-Version`**: API version that processed the request
- **`X-RateLimit-Remaining`**: Number of requests remaining in current rate limit window

## Endpoints

### Health & System Status

Critical endpoints for monitoring system health, ensuring reliable operations, and troubleshooting deployment issues.

#### GET /health

**🎯 Purpose**: Quick liveness check - "Is the API server responding?"

**🔒 Authentication**: ❌ Not required (public endpoint)

**⚡ Performance**: ~1ms response time, minimal server load

**🎛️ Use Cases**:
- **Load Balancer Health Checks**: Configure F5, HAProxy, ALB health checks
- **Kubernetes Liveness Probes**: Restart pods if API becomes unresponsive  
- **Uptime Monitoring**: Services like Pingdom, DataDog, New Relic
- **CI/CD Pipeline Verification**: Ensure deployment succeeded

**Example Request:**
```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0", 
  "timestamp": "2025-08-13T10:30:45Z",
  "uptime_seconds": 86400
}
```

**Response Fields:**
- **`status`**: Always `"healthy"` if server is running (otherwise no response)
- **`version`**: Current API version for compatibility checking
- **`timestamp`**: Current server time (useful for clock drift detection)
- **`uptime_seconds`**: How long the server has been running (helps identify recent restarts)

#### GET /ready

**Purpose**: Comprehensive readiness check to verify the API can handle actual requests.

**Authentication**: Not required (public endpoint)

**Use Cases**:
- Kubernetes readiness probes (don't send traffic until ready)
- Pre-deployment verification
- Dependency health monitoring
- Troubleshooting service issues

**Example Request:**
```bash
curl -X GET http://localhost:8000/ready
```

**Healthy Response (200 OK):**
```json
{
  "status": "ready",
  "database": "connected",
  "redis": "connected", 
  "model": "loaded",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12,
      "connection_pool": {
        "active": 5,
        "idle": 15,
        "max": 20
      }
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 3,
      "memory_usage": "45MB"
    },
    "ml_model": {
      "status": "loaded",
      "version": "gru_v1.2.0",
      "load_time_ms": 1250,
      "memory_usage": "512MB"
    }
  },
  "timestamp": "2025-08-13T10:30:45Z"
}
```

**Unhealthy Response (503 Service Unavailable):**
```json
{
  "status": "not_ready",
  "database": "disconnected",
  "redis": "connected",
  "model": "loading",
  "errors": [
    "Database connection failed: timeout after 5s",
    "ML model still loading, estimated 30s remaining"
  ],
  "timestamp": "2025-08-13T10:30:45Z"
}
```

**Response Fields:**
- **`status`**: `"ready"` if all checks pass, `"not_ready"` if any fail
- **`database`**: Connection status to PostgreSQL database
- **`redis`**: Connection status to Redis cache server
- **`model`**: Machine learning model loading status
- **`checks`**: Detailed health information for each dependency
- **`errors`**: Array of error messages if any checks fail

**Status Codes:**
- **200 OK**: All systems ready to handle requests
- **503 Service Unavailable**: One or more dependencies are unhealthy

### Predictions

The core functionality of EduPulse - generating dropout risk predictions using machine learning models trained on student behavioral data.

#### POST /predict

**🎯 Purpose**: Generate AI-powered dropout risk prediction for a single student

**🔒 Authentication**: ✅ Required (JWT Bearer token)

**🏫 Business Context**: 
Analyzes 20 weeks of student behavioral data across 42 features to predict dropout probability within the next academic year. Counselors use these insights to prioritize intervention efforts and allocate limited resources effectively.

**⚡ Performance**: 
- **Response Time**: 50-100ms typical, 200ms max
- **Caching**: Results cached for 1 hour to improve performance
- **Rate Limits**: 100 requests/minute per user, 10 requests/second burst
- **ML Model**: Latest GRU v1.2.0 with 89% accuracy on validation data

**🧠 What It Analyzes**:
- 📚 **Academic Trends**: GPA changes, assignment completion, grade patterns
- 🏫 **Attendance Patterns**: Daily attendance, tardiness, absence streaks  
- ⚖️ **Behavioral Indicators**: Discipline incidents, engagement metrics, social factors

**Request Body:**
```json
{
  "student_id": "STU123456",
  "include_features": false,
  "include_explanations": false,
  "reference_date": "2025-08-13"
}
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `student_id` | string | ✅ | Unique student identifier (format: STU + digits) |
| `include_features` | boolean | ❌ | Include raw feature values used by ML model (default: false) |
| `include_explanations` | boolean | ❌ | Include interpretable explanations of risk factors (default: false) |
| `reference_date` | string | ❌ | Date to calculate prediction from (ISO date, default: today) |

**Example Request:**
```bash
curl -X POST https://api.edupulse.com/api/v1/predict \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "STU123456",
    "include_features": true,
    "include_explanations": true
  }'
```

**Successful Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "student_id": "STU123456",
    "risk_score": 0.75,
    "risk_category": "high",
    "confidence": 0.92,
    "timestamp": "2025-08-13T10:30:45Z",
    "model_version": "gru_v1.2.0",
    "features": {
      "attendance_rate_30d": 0.65,
      "current_gpa": 2.1,
      "consecutive_absences": 5,
      "discipline_incidents_semester": 2,
      "assignment_completion_rate": 0.68
    },
    "explanations": [
      {
        "factor": "attendance_rate_30d",
        "weight": 0.34,
        "value": 0.65,
        "description": "Student has attended only 65% of classes in the last 30 days",
        "impact": "negative",
        "recommendation": "Contact family to discuss attendance barriers"
      },
      {
        "factor": "current_gpa",
        "weight": 0.28,
        "value": 2.1,
        "description": "Current GPA of 2.1 is below the 2.5 threshold associated with dropout risk",
        "impact": "negative", 
        "recommendation": "Academic support and tutoring intervention needed"
      }
    ],
    "historical_trend": "declining",
    "next_review_date": "2025-08-20"
  },
  "message": "Risk prediction generated successfully"
}
```

**Response Field Details:**

- **`risk_score`** (0.0-1.0): Probability of dropout within the next academic year
- **`risk_category`**: Human-readable risk level:
  - `"low"` (0.0-0.25): Minimal intervention needed
  - `"medium"` (0.25-0.50): Monitor closely
  - `"high"` (0.50-0.75): Active intervention recommended
  - `"critical"` (0.75-1.0): Immediate intensive intervention required
- **`confidence`** (0.0-1.0): Model's confidence in this prediction (higher = more reliable)
- **`features`**: Raw numerical features from the ML model (if requested)
- **`explanations`**: Human-interpretable factors driving the prediction (if requested)
- **`historical_trend`**: How risk has changed over time (`"improving"`, `"stable"`, `"declining"`)
- **`next_review_date`**: Recommended date for next prediction review

**Error Responses:**

**Student Not Found (404):**
```json
{
  "status": "error",
  "data": {},
  "message": "Student with ID 'STU123456' not found. Verify the student ID is correct and the student is enrolled in the current academic year.",
  "timestamp": "2025-08-13T10:30:45Z"
}
```

**Insufficient Data (400):**
```json
{
  "status": "error",
  "data": {
    "student_id": "STU123456",
    "missing_data": ["attendance_records", "grade_records"],
    "minimum_required": "4 weeks of data"
  },
  "message": "Insufficient data to generate reliable prediction. Student needs at least 4 weeks of attendance and grade data.",
  "timestamp": "2025-08-13T10:30:45Z"
}
```

**Model Unavailable (503):**
```json
{
  "status": "error", 
  "data": {},
  "message": "Prediction model is currently unavailable. Model is being updated - try again in 5 minutes.",
  "timestamp": "2025-08-13T10:30:45Z"
}
```

**Status Codes:**
- **200 OK**: Prediction generated successfully
- **400 Bad Request**: Invalid student ID or insufficient data
- **401 Unauthorized**: Missing or invalid authentication token
- **404 Not Found**: Student not found in system
- **429 Too Many Requests**: Rate limit exceeded
- **503 Service Unavailable**: Model temporarily unavailable
- **500 Internal Server Error**: Unexpected server error

#### POST /predict/batch

Predict risk for multiple students.

**Request Body:**

```json
{
  "student_ids": ["STU123456", "STU789012"],
  "include_features": false,
  "async": false
}
```

**Parameters:**

- `student_ids` (array, required): List of student identifiers (max 100)
- `include_features` (boolean, optional): Include extracted features
- `async` (boolean, optional): Process asynchronously

**Response (Synchronous):**

```json
{
  "predictions": [
    {
      "student_id": "STU123456",
      "risk_score": 0.75,
      "risk_category": "high",
      "confidence": 0.92
    }
  ],
  "failed": [],
  "processing_time": 1.234
}
```

**Response (Asynchronous):**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "estimated_completion": "2025-08-13T00:05:00Z"
}
```

**Status Codes:**

- 200: Success
- 202: Accepted (async)
- 400: Invalid request (e.g., too many students)
- 500: Server error

#### GET /predict/task/{task_id}

Check status of async batch prediction.

**Response:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "result": {
    "predictions": [],
    "failed": []
  }
}
```

### Students

#### POST /students

Create a new student record.

**Request Body:**

```json
{
  "student_id": "STU123456",
  "first_name": "John",
  "last_name": "Doe",
  "grade_level": 10,
  "enrollment_date": "2024-09-01",
  "demographic_info": {
    "gender": "M",
    "ethnicity": "Hispanic",
    "socioeconomic_status": "low"
  }
}
```

**Response:**

```json
{
  "student_id": "STU123456",
  "created_at": "2025-08-13T00:00:00Z",
  "status": "active"
}
```

**Status Codes:**

- 201: Created
- 409: Student already exists
- 400: Invalid data
- 500: Server error

#### GET /students/{student_id}

Retrieve student information.

**Response:**

```json
{
  "student_id": "STU123456",
  "first_name": "John",
  "last_name": "Doe",
  "grade_level": 10,
  "enrollment_date": "2024-09-01",
  "current_risk": {
    "score": 0.75,
    "category": "high",
    "last_updated": "2025-08-13T00:00:00Z"
  },
  "attendance_summary": {
    "rate": 0.85,
    "absences": 15,
    "tardies": 5
  },
  "academic_summary": {
    "gpa": 2.5,
    "failing_courses": 1
  }
}
```

**Status Codes:**

- 200: Success
- 404: Student not found
- 500: Server error

#### GET /students

List students with filtering.

**Query Parameters:**

- `grade_level` (integer): Filter by grade
- `risk_category` (string): Filter by risk category
- `limit` (integer): Results per page (default: 20, max: 100)
- `offset` (integer): Pagination offset

**Response:**

```json
{
  "students": [],
  "total": 500,
  "limit": 20,
  "offset": 0
}
```

#### PUT /students/{student_id}

Update student information.

**Request Body:**

```json
{
  "grade_level": 11,
  "status": "active"
}
```

**Response:**

```json
{
  "student_id": "STU123456",
  "updated_at": "2025-08-13T00:00:00Z",
  "changes": ["grade_level", "status"]
}
```

### Training

#### POST /train/update

Trigger model update with new data.

**Request Body:**

```json
{
  "training_config": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "data_filters": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "min_samples": 100
  }
}
```

**Response:**

```json
{
  "training_id": "train_20250813_001",
  "status": "initiated",
  "estimated_duration": 3600
}
```

**Status Codes:**

- 202: Accepted
- 400: Invalid configuration
- 409: Training already in progress
- 500: Server error

#### GET /train/status/{training_id}

Get training job status.

**Response:**

```json
{
  "training_id": "train_20250813_001",
  "status": "in_progress",
  "progress": 45,
  "current_epoch": 23,
  "total_epochs": 50,
  "metrics": {
    "loss": 0.234,
    "accuracy": 0.89,
    "val_loss": 0.256,
    "val_accuracy": 0.87
  },
  "estimated_completion": "2025-08-13T01:00:00Z"
}
```

#### POST /train/feedback

Submit prediction feedback for continuous learning.

**Request Body:**

```json
{
  "prediction_id": "pred_123456",
  "actual_outcome": "high",
  "feedback_type": "correction",
  "comments": "Student showed significant improvement"
}
```

**Response:**

```json
{
  "feedback_id": "fb_789012",
  "received_at": "2025-08-13T00:00:00Z",
  "will_use_for_training": true
}
```

### Metrics

#### GET /metrics

Retrieve system and model metrics.

**Query Parameters:**

- `type` (string): "system" | "model" | "all"
- `period` (string): "hour" | "day" | "week" | "month"

**Response:**

```json
{
  "model_metrics": {
    "accuracy": 0.89,
    "precision": 0.87,
    "recall": 0.91,
    "f1_score": 0.89,
    "auc_roc": 0.92,
    "predictions_today": 1234,
    "avg_latency_ms": 45
  },
  "system_metrics": {
    "uptime_seconds": 86400,
    "cpu_usage": 0.45,
    "memory_usage": 0.62,
    "active_connections": 15,
    "request_rate": 10.5
  },
  "timestamp": "2025-08-13T00:00:00Z"
}
```

## Error Handling

### Error Response Format

```json
{
  "status": "error",
  "error": {
    "code": "STUDENT_NOT_FOUND",
    "message": "Student with ID STU123456 not found",
    "details": {},
    "timestamp": "2025-08-13T00:00:00Z"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| UNAUTHORIZED | 401 | Missing or invalid authentication |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| VALIDATION_ERROR | 400 | Invalid request data |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| SERVER_ERROR | 500 | Internal server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

## Rate Limiting

- Default rate limit: 100 requests per minute per API key
- Batch endpoints: 10 requests per minute
- Training endpoints: 1 request per hour

Rate limit headers:

```text
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1629843600
```

## Webhooks

Configure webhooks to receive notifications:

### Events

- `prediction.completed`: Batch prediction finished
- `training.completed`: Model training completed
- `risk.critical`: Critical risk detected
- `system.alert`: System alert triggered

### Webhook Payload

```json
{
  "event": "prediction.completed",
  "data": {},
  "timestamp": "2025-08-13T00:00:00Z",
  "signature": "sha256=..."
}
```

## SDK Examples

EduPulse provides official SDKs and comprehensive examples for integrating dropout risk prediction into your school's existing systems.

### Python SDK

The Python SDK provides a Pythonic interface with built-in error handling, retry logic, and response validation.

**Installation:**
```bash
pip install edupulse-sdk
```

**Basic Usage:**
```python
from edupulse import Client
import os

# Initialize client with API key (get from EduPulse admin dashboard)
client = Client(
    api_key=os.getenv('EDUPULSE_API_KEY'),
    base_url='https://api.edupulse.com'  # Optional: defaults to production
)

# Single student prediction with error handling
try:
    result = client.predict(
        student_id="STU123456",
        include_explanations=True  # Get actionable insights
    )
    
    print(f"Student: {result.student_id}")
    print(f"Risk Score: {result.risk_score:.1%}")  # Format as percentage
    print(f"Risk Level: {result.risk_category}")
    print(f"Confidence: {result.confidence:.1%}")
    
    # Display top risk factors for counselors
    if result.explanations:
        print("\nTop Risk Factors:")
        for explanation in result.explanations[:3]:
            print(f"  • {explanation.description}")
            print(f"    Recommendation: {explanation.recommendation}")
            
except client.StudentNotFound:
    print("Student not found - check ID format and enrollment status")
except client.InsufficientData:
    print("Not enough data for prediction - student needs 4+ weeks of records")
except client.RateLimitError:
    print("Rate limit exceeded - please wait before making more requests")
except Exception as e:
    print(f"Unexpected error: {e}")
```

**Batch Prediction for Daily Reports:**
```python
# Process entire grade level for daily counselor reports
grade_9_students = [
    "STU001234", "STU001235", "STU001236", "STU001237", "STU001238"
]

try:
    # Batch prediction with async processing for large groups
    batch_result = client.predict_batch(
        student_ids=grade_9_students,
        async_mode=True,  # Use for >20 students
        include_features=False  # Faster response, smaller payload
    )
    
    if batch_result.is_async:
        print(f"Batch job started: {batch_result.task_id}")
        
        # Wait for completion with progress updates
        final_result = client.wait_for_batch(batch_result.task_id, timeout=300)
    else:
        final_result = batch_result
    
    # Process results for counselor dashboard
    high_risk_students = [
        r for r in final_result.predictions 
        if r.risk_score >= 0.7
    ]
    
    print(f"\n📊 Grade 9 Risk Assessment Summary:")
    print(f"Total students: {len(final_result.predictions)}")
    print(f"High risk students: {len(high_risk_students)}")
    print(f"Average risk score: {sum(r.risk_score for r in final_result.predictions) / len(final_result.predictions):.1%}")
    
    if high_risk_students:
        print(f"\n🚨 Students requiring immediate attention:")
        for student in sorted(high_risk_students, key=lambda x: x.risk_score, reverse=True):
            print(f"  • {student.student_id}: {student.risk_score:.1%} risk")
            
except Exception as e:
    print(f"Batch prediction failed: {e}")
```

### JavaScript/Node.js SDK

Perfect for integrating EduPulse into web applications, student information systems, or Node.js backend services.

**Installation:**
```bash
npm install edupulse-sdk
# or
yarn add edupulse-sdk
```

**Basic Usage:**
```javascript
const EduPulse = require('edupulse-sdk');

// Initialize client with configuration
const client = new EduPulse.Client({
  apiKey: process.env.EDUPULSE_API_KEY,
  baseURL: 'https://api.edupulse.com',
  timeout: 30000,  // 30 second timeout
  retries: 3       // Automatic retry on failure
});

// Single prediction with comprehensive error handling
async function assessStudentRisk(studentId) {
  try {
    const result = await client.predict(studentId, {
      includeExplanations: true,
      includeFeatures: false
    });
    
    console.log(`📊 Risk Assessment for ${result.studentId}`);
    console.log(`Risk Score: ${(result.riskScore * 100).toFixed(1)}%`);
    console.log(`Risk Level: ${result.riskCategory}`);
    console.log(`Model Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    
    // Display actionable insights
    if (result.explanations && result.explanations.length > 0) {
      console.log('\n🔍 Key Risk Factors:');
      result.explanations.slice(0, 3).forEach((explanation, index) => {
        console.log(`  ${index + 1}. ${explanation.description}`);
        if (explanation.recommendation) {
          console.log(`     💡 Recommended Action: ${explanation.recommendation}`);
        }
      });
    }
    
    return {
      success: true,
      data: result
    };
    
  } catch (error) {
    if (error instanceof EduPulse.StudentNotFoundError) {
      console.error(`❌ Student ${studentId} not found`);
      return { success: false, error: 'student_not_found' };
    } else if (error instanceof EduPulse.InsufficientDataError) {
      console.error(`⚠️ Not enough data for ${studentId}`);
      return { success: false, error: 'insufficient_data' };
    } else if (error instanceof EduPulse.RateLimitError) {
      console.error('🚦 Rate limit exceeded - please wait');
      return { success: false, error: 'rate_limit' };
    } else {
      console.error(`💥 Unexpected error for ${studentId}:`, error.message);
      return { success: false, error: 'unknown' };
    }
  }
}

// Example: Daily risk assessment for counselor dashboard
async function generateDailyReport(gradeLevel) {
  console.log(`🎯 Generating daily risk report for Grade ${gradeLevel}`);
  
  try {
    // Get list of students from your SIS
    const students = await getStudentsByGrade(gradeLevel); // Your SIS function
    const studentIds = students.map(s => s.student_id);
    
    // Batch prediction for efficiency
    const batchResult = await client.predictBatch({
      studentIds: studentIds,
      asyncMode: studentIds.length > 20,
      includeFeatures: false
    });
    
    let predictions;
    if (batchResult.isAsync) {
      console.log(`⏳ Processing ${studentIds.length} students asynchronously...`);
      predictions = await client.waitForBatch(batchResult.taskId, {
        timeout: 300000,  // 5 minutes
        pollInterval: 5000  // Check every 5 seconds
      });
    } else {
      predictions = batchResult.predictions;
    }
    
    // Analyze results
    const summary = {
      total: predictions.length,
      critical: predictions.filter(p => p.riskScore >= 0.75).length,
      high: predictions.filter(p => p.riskScore >= 0.5 && p.riskScore < 0.75).length,
      medium: predictions.filter(p => p.riskScore >= 0.25 && p.riskScore < 0.5).length,
      low: predictions.filter(p => p.riskScore < 0.25).length,
      averageRisk: predictions.reduce((sum, p) => sum + p.riskScore, 0) / predictions.length
    };
    
    console.log(`\n📈 Grade ${gradeLevel} Risk Summary:`);
    console.log(`  Total Students: ${summary.total}`);
    console.log(`  🔴 Critical Risk: ${summary.critical}`);
    console.log(`  🟠 High Risk: ${summary.high}`);
    console.log(`  🟡 Medium Risk: ${summary.medium}`);
    console.log(`  🟢 Low Risk: ${summary.low}`);
    console.log(`  📊 Average Risk: ${(summary.averageRisk * 100).toFixed(1)}%`);
    
    // Return prioritized list for counselor review
    const priorityStudents = predictions
      .filter(p => p.riskScore >= 0.5)  // High and critical risk
      .sort((a, b) => b.riskScore - a.riskScore)  // Highest risk first
      .slice(0, 10);  // Top 10 for immediate attention
    
    return {
      summary,
      priorityStudents,
      allPredictions: predictions
    };
    
  } catch (error) {
    console.error('Failed to generate daily report:', error);
    throw error;
  }
}

// Usage
(async () => {
  const report = await generateDailyReport(9);
  
  if (report.priorityStudents.length > 0) {
    console.log('\n🚨 Priority Students for Intervention:');
    report.priorityStudents.forEach((student, index) => {
      console.log(`  ${index + 1}. ${student.studentId}: ${(student.riskScore * 100).toFixed(1)}% risk`);
    });
  }
})();
```

### cURL Examples

For direct API testing, automation scripts, or integration with non-JavaScript/Python systems.

**Single Student Prediction:**
```bash
# Basic prediction request
curl -X POST https://api.edupulse.com/api/v1/predict \
  -H "Authorization: Bearer your-jwt-token-here" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "student_id": "STU123456",
    "include_explanations": true
  }'

# Example response (formatted for readability)
{
  "status": "success",
  "data": {
    "student_id": "STU123456",
    "risk_score": 0.67,
    "risk_category": "high",
    "confidence": 0.89,
    "explanations": [
      {
        "factor": "attendance_rate_30d",
        "description": "Student has attended only 72% of classes in the last 30 days",
        "recommendation": "Contact family to discuss attendance barriers"
      }
    ]
  },
  "message": "Risk prediction generated successfully"
}
```

**Batch Prediction with Async Processing:**
```bash
# Start batch job for multiple students
curl -X POST https://api.edupulse.com/api/v1/predict/batch \
  -H "Authorization: Bearer your-jwt-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "student_ids": ["STU123456", "STU789012", "STU345678", "STU901234"],
    "async": true,
    "include_features": false
  }'

# Response with task ID for async job
{
  "status": "success", 
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "processing",
    "estimated_completion": "2025-08-13T10:35:00Z"
  },
  "message": "Batch prediction job started"
}

# Check job status (repeat until status is "completed")
curl -X GET https://api.edupulse.com/api/v1/predict/task/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer your-jwt-token-here"

# Final result when job completes
{
  "status": "success",
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "progress": 100,
    "result": {
      "predictions": [
        {"student_id": "STU123456", "risk_score": 0.67, "risk_category": "high"},
        {"student_id": "STU789012", "risk_score": 0.23, "risk_category": "low"}
      ],
      "failed": []
    }
  }
}
```

**Authentication Flow:**
```bash
# Step 1: Login to get tokens
curl -X POST https://api.edupulse.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "counselor@school.edu",
    "password": "your-secure-password"
  }'

# Response with tokens
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4...",
  "token_type": "bearer",
  "expires_in": 3600
}

# Step 2: Use access token for API calls (as shown above)

# Step 3: Refresh token when it expires (before 1 hour)
curl -X POST https://api.edupulse.com/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4..."
  }'
```

### Integration Examples

**Webhook Setup for Real-time Alerts:**
```bash
# Configure webhook for critical risk alerts
curl -X POST https://api.edupulse.com/api/v1/webhooks \
  -H "Authorization: Bearer your-jwt-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-school-system.com/edupulse-webhook",
    "events": ["risk.critical", "prediction.completed"],
    "config": {
      "risk_threshold": 0.8,
      "include_recommendations": true
    }
  }'
```

**Health Check for Monitoring:**
```bash
# Check if API is healthy (for load balancer health checks)
curl -X GET https://api.edupulse.com/health

# Check if API is ready to handle requests (for deployment verification)
curl -X GET https://api.edupulse.com/ready
```

## OpenAPI Specification

The complete OpenAPI specification is available at:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- JSON: `/openapi.json`

## Versioning

The API uses URL versioning. The current version is `v1`.

Future versions will maintain backward compatibility for at least 6 months after a new version is released.

## Support

For API support:

- Documentation: <https://docs.edupulse.com>
- Status Page: <https://status.edupulse.com>
- Support Email: <api-support@edupulse.com>

---

### Last Updated

2025-08-13 02:56:19 CDT

## Implementation Notes

This API reference reflects the current implementation in EduPulse v1.0. The actual endpoints are built using:

- **FastAPI**: Modern Python web framework with automatic OpenAPI generation
- **SQLAlchemy**: Database ORM with PostgreSQL in production, SQLite for testing  
- **PyTorch**: Deep learning framework for GRU-based neural network models
- **Redis**: Caching layer for prediction results and session management
- **Pydantic**: Data validation and serialization with automatic schema generation

## Current Endpoint Status

| Endpoint | Implementation Status | Notes |
|----------|----------------------|-------|
| `GET /health` | ✅ Fully Implemented | Basic health check with environment info |
| `GET /ready` | ✅ Fully Implemented | Database and Redis connectivity checks |
| `POST /students/` | ✅ Fully Implemented | Create with validation and unique district_id |
| `GET /students/{id}` | ✅ Fully Implemented | Retrieve by UUID with full record |
| `GET /students/` | ✅ Fully Implemented | List with pagination (skip/limit) |
| `PATCH /students/{id}` | ✅ Fully Implemented | Partial update with field validation |  
| `DELETE /students/{id}` | ✅ Fully Implemented | Cascade deletion of related records |
| `POST /predictions/predict` | ✅ Fully Implemented | Single prediction with ML model |
| `POST /predictions/predict/batch` | ✅ Fully Implemented | Batch processing with size limits |
| `GET /predictions/metrics` | ⚠️ Mock Implementation | Returns static metrics for now |
| `POST /training/update` | 🚧 Partial Implementation | Training triggers implemented |
| `GET /training/status/{id}` | 🚧 Partial Implementation | Basic status tracking |

**Legend**: ✅ Complete | ⚠️ Mock/Limited | 🚧 In Progress | ❌ Not Implemented

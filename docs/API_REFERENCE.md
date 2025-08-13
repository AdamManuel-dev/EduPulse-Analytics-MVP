# API Reference

## Overview

The EduPulse Analytics API provides RESTful endpoints for student risk prediction, model training, and data management. All endpoints require authentication unless otherwise specified.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-token>
```

## Response Format

All responses follow this structure:

```json
{
  "status": "success|error",
  "data": {},
  "message": "Response message",
  "timestamp": "2025-08-13T00:00:00Z"
}
```

## Endpoints

### Health Check

#### GET /health

Check if the API is running.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-08-13T00:00:00Z"
}
```

#### GET /ready

Check if the API is ready to handle requests.

**Response:**

```json
{
  "status": "ready",
  "database": "connected",
  "redis": "connected",
  "model": "loaded"
}
```

### Predictions

#### POST /predict

Predict risk for a single student.

**Request Body:**

```json
{
  "student_id": "STU123456",
  "include_features": false,
  "include_explanations": false
}
```

**Parameters:**

- `student_id` (string, required): Unique student identifier
- `include_features` (boolean, optional): Include extracted features in response
- `include_explanations` (boolean, optional): Include model explanations

**Response:**

```json
{
  "student_id": "STU123456",
  "risk_score": 0.75,
  "risk_category": "high",
  "confidence": 0.92,
  "timestamp": "2025-08-13T00:00:00Z",
  "features": {},  // If requested
  "explanations": {}  // If requested
}
```

**Status Codes:**

- 200: Success
- 404: Student not found
- 400: Invalid request
- 500: Server error

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

```
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

### Python

```python
from edupulse import Client

client = Client(api_key="your_api_key")

# Single prediction
result = client.predict(student_id="STU123456")
print(f"Risk score: {result.risk_score}")

# Batch prediction
results = client.predict_batch(
    student_ids=["STU123456", "STU789012"],
    async_mode=True
)
```

### JavaScript

```javascript
const EduPulse = require('edupulse-sdk');

const client = new EduPulse.Client({
  apiKey: 'your_api_key'
});

// Single prediction
const result = await client.predict('STU123456');
console.log(`Risk score: ${result.riskScore}`);
```

### cURL

```bash
# Single prediction
curl -X POST https://api.edupulse.com/v1/predict \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{"student_id": "STU123456"}'
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

*Last updated: 2025-08-13*

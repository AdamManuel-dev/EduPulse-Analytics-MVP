"""
Unit tests for API routes.
"""

from datetime import date, datetime
from unittest.mock import patch


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_readiness_check(self, client, db_session):
        """Test readiness check endpoint."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["database"] == "connected"

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["docs"] == "/docs"


class TestStudentEndpoints:
    """Test student CRUD endpoints."""

    def test_create_student(self, client, db_session):
        """Test creating a new student."""
        student_data = {
            "student_id": "STU99999",
            "first_name": "New",
            "last_name": "Student",
            "grade_level": 11,
            "enrollment_date": str(date.today()),
            "date_of_birth": "2007-05-15",
            "gender": "F",
            "ethnicity": "Asian",
            "special_ed_status": False,
            "english_learner_status": True,
            "socioeconomic_status": "low",
        }

        response = client.post("/api/v1/students", json=student_data)
        assert response.status_code == 201
        data = response.json()
        assert data["student_id"] == "STU99999"
        assert data["status"] == "created"

    def test_get_student(self, client, sample_student):
        """Test retrieving a student."""
        response = client.get(f"/api/v1/students/{sample_student.student_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["student_id"] == sample_student.student_id
        assert data["first_name"] == sample_student.first_name
        assert data["last_name"] == sample_student.last_name

    def test_get_nonexistent_student(self, client):
        """Test retrieving a non-existent student."""
        response = client.get("/api/v1/students/NONEXISTENT")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data or "detail" in data

    def test_list_students(self, client, batch_students):
        """Test listing students with pagination."""
        response = client.get("/api/v1/students?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert "students" in data
        assert len(data["students"]) <= 5
        assert "total" in data
        assert data["total"] == len(batch_students)

    def test_list_students_with_filters(self, client, batch_students):
        """Test listing students with grade filter."""
        response = client.get("/api/v1/students?grade_level=10")
        assert response.status_code == 200
        data = response.json()
        assert "students" in data
        # Check all returned students are in grade 10
        for student in data["students"]:
            assert student["grade_level"] == 10

    def test_update_student(self, client, sample_student):
        """Test updating a student."""
        update_data = {"grade_level": 11, "special_ed_status": True}

        response = client.put(f"/api/v1/students/{sample_student.student_id}", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["student_id"] == sample_student.student_id
        assert "updated" in data

    def test_delete_student(self, client, sample_student):
        """Test deleting a student."""
        response = client.delete(f"/api/v1/students/{sample_student.student_id}")
        assert response.status_code in [200, 204]

        # Verify student is deleted
        response = client.get(f"/api/v1/students/{sample_student.student_id}")
        assert response.status_code == 404


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    @patch("src.services.prediction_service.PredictionService.predict_risk")
    def test_single_prediction(self, mock_predict, client, sample_student):
        """Test single student prediction."""
        mock_predict.return_value = {
            "student_id": sample_student.student_id,
            "risk_score": 0.75,
            "risk_category": "high",
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat(),
        }

        request_data = {
            "student_id": sample_student.student_id,
            "include_features": False,
            "include_explanations": False,
        }

        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["student_id"] == sample_student.student_id
        assert data["risk_score"] == 0.75
        assert data["risk_category"] == "high"
        assert data["confidence"] == 0.85

    @patch("src.services.prediction_service.PredictionService.predict_risk")
    def test_prediction_with_features(self, mock_predict, client, sample_student):
        """Test prediction with feature extraction."""
        mock_predict.return_value = {
            "student_id": sample_student.student_id,
            "risk_score": 0.45,
            "risk_category": "medium",
            "confidence": 0.90,
            "features": {"attendance_rate": 0.85, "gpa_current": 3.2, "incident_count": 1},
            "timestamp": datetime.now().isoformat(),
        }

        request_data = {
            "student_id": sample_student.student_id,
            "include_features": True,
            "include_explanations": False,
        }

        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert data["features"]["attendance_rate"] == 0.85

    @patch("src.services.prediction_service.PredictionService.batch_predict")
    def test_batch_prediction(self, mock_batch_predict, client, batch_students):
        """Test batch prediction for multiple students."""
        student_ids = [s.student_id for s in batch_students[:5]]

        mock_predictions = [
            {"student_id": sid, "risk_score": 0.5, "risk_category": "medium", "confidence": 0.8}
            for sid in student_ids
        ]
        mock_batch_predict.return_value = {
            "predictions": mock_predictions,
            "failed": [],
            "processing_time": 1.234,
        }

        request_data = {"student_ids": student_ids, "include_features": False, "async": False}

        response = client.post("/api/v1/predict/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 5
        assert data["failed"] == []

    def test_batch_prediction_limit(self, client):
        """Test batch prediction with too many students."""
        student_ids = [f"STU{i:05d}" for i in range(101)]  # Exceeds limit

        request_data = {"student_ids": student_ids, "include_features": False, "async": False}

        response = client.post("/api/v1/predict/batch", json=request_data)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data or "detail" in data


class TestTrainingEndpoints:
    """Test model training endpoints."""

    @patch("src.training.trainer.ModelTrainer.fit")
    def test_trigger_training(self, mock_fit, client):
        """Test triggering model training."""
        mock_fit.return_value = {
            "training_id": "train_20250813_001",
            "status": "initiated",
            "estimated_duration": 3600,
        }

        training_config = {
            "training_config": {"epochs": 50, "batch_size": 32, "learning_rate": 0.001},
            "data_filters": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "min_samples": 100,
            },
        }

        response = client.post("/api/v1/train/update", json=training_config)
        assert response.status_code in [200, 202]
        data = response.json()
        assert "training_id" in data
        assert data["status"] == "initiated"

    @patch("src.training.trainer.get_training_status")
    def test_get_training_status(self, mock_status, client):
        """Test getting training job status."""
        mock_status.return_value = {
            "training_id": "train_20250813_001",
            "status": "in_progress",
            "progress": 45,
            "current_epoch": 23,
            "total_epochs": 50,
            "metrics": {"loss": 0.234, "accuracy": 0.89, "val_loss": 0.256, "val_accuracy": 0.87},
        }

        response = client.get("/api/v1/train/status/train_20250813_001")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "in_progress"
        assert data["progress"] == 45
        assert data["current_epoch"] == 23

    def test_submit_feedback(self, client):
        """Test submitting prediction feedback."""
        feedback_data = {
            "prediction_id": "pred_123456",
            "actual_outcome": "high",
            "feedback_type": "correction",
            "comments": "Student showed significant improvement",
        }

        response = client.post("/api/v1/train/feedback", json=feedback_data)
        assert response.status_code in [200, 201]
        data = response.json()
        assert "feedback_id" in data
        assert data["received"] is True


class TestMetricsEndpoint:
    """Test metrics endpoint."""

    def test_get_metrics(self, client):
        """Test retrieving system metrics."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "model_metrics" in data or "system_metrics" in data

    def test_get_model_metrics(self, client):
        """Test retrieving model-specific metrics."""
        response = client.get("/api/v1/metrics?type=model")
        assert response.status_code == 200
        data = response.json()
        if "model_metrics" in data:
            metrics = data["model_metrics"]
            assert "accuracy" in metrics or "predictions_today" in metrics

    def test_get_system_metrics(self, client):
        """Test retrieving system metrics."""
        response = client.get("/api/v1/metrics?type=system")
        assert response.status_code == 200
        data = response.json()
        if "system_metrics" in data:
            metrics = data["system_metrics"]
            assert "uptime_seconds" in metrics or "cpu_usage" in metrics

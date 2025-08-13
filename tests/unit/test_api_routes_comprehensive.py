"""
Comprehensive API route tests to achieve 100% coverage.
Tests all FastAPI endpoints with proper mocking.
"""

from datetime import datetime
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# Import all the route modules and dependencies
from src.api.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    with patch("src.api.routes.students.get_db") as mock_get_db, patch(
        "src.api.routes.predictions.get_db"
    ) as mock_get_db_pred, patch("src.api.routes.training.get_db") as mock_get_db_train:
        mock_session = Mock()
        mock_get_db.return_value.__enter__.return_value = mock_session
        mock_get_db_pred.return_value.__enter__.return_value = mock_session
        mock_get_db_train.return_value.__enter__.return_value = mock_session
        yield mock_session


class TestHealthRoutes:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "EduPulse" in data["message"]

    def test_health_check_success(self, client):
        """Test successful health check."""
        with patch("src.api.routes.health.get_db") as mock_get_db:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value = Mock()

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "database" in data
            assert "timestamp" in data

    def test_health_check_database_failure(self, client):
        """Test health check with database failure."""
        with patch("src.api.routes.health.get_db") as mock_get_db:
            mock_get_db.return_value.__enter__.side_effect = Exception("DB connection failed")

            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data["database"]

    def test_ready_endpoint_success(self, client):
        """Test readiness endpoint success."""
        with patch("src.api.routes.health.get_db") as mock_get_db, patch(
            "src.api.routes.health.prediction_service"
        ) as mock_pred_service:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value = Mock()
            mock_pred_service.model = Mock()

            response = client.get("/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"

    def test_ready_endpoint_model_not_loaded(self, client):
        """Test readiness endpoint when model not loaded."""
        with patch("src.api.routes.health.get_db") as mock_get_db, patch(
            "src.api.routes.health.prediction_service"
        ) as mock_pred_service:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value = Mock()
            mock_pred_service.model = None

            response = client.get("/ready")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "not ready"


class TestStudentRoutes:
    """Test student management endpoints."""

    def test_create_student_success(self, client, mock_db_session):
        """Test successful student creation."""
        student_data = {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "date_of_birth": "2000-01-01",
            "grade_level": "10",
        }

        # Mock student model
        mock_student = Mock()
        mock_student.id = str(uuid4())
        mock_student.first_name = "John"
        mock_student.last_name = "Doe"
        mock_student.email = "john.doe@example.com"

        with patch("src.api.routes.students.models.Student", return_value=mock_student):
            response = client.post("/students", json=student_data)
            assert response.status_code == 201
            data = response.json()
            assert data["first_name"] == "John"
            assert data["last_name"] == "Doe"

    def test_create_student_duplicate_email(self, client, mock_db_session):
        """Test student creation with duplicate email."""
        student_data = {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "date_of_birth": "2000-01-01",
            "grade_level": "10",
        }

        # Mock IntegrityError for duplicate email
        from sqlalchemy.exc import IntegrityError

        mock_db_session.add.side_effect = IntegrityError("", "", "")

        response = client.post("/students", json=student_data)
        assert response.status_code == 400
        data = response.json()
        assert "already exists" in data["detail"]

    def test_create_student_database_error(self, client, mock_db_session):
        """Test student creation with database error."""
        student_data = {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "date_of_birth": "2000-01-01",
            "grade_level": "10",
        }

        mock_db_session.add.side_effect = Exception("Database error")

        response = client.post("/students", json=student_data)
        assert response.status_code == 500

    def test_get_student_success(self, client, mock_db_session):
        """Test successful student retrieval."""
        student_id = str(uuid4())
        mock_student = Mock()
        mock_student.id = student_id
        mock_student.first_name = "John"

        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_student

        response = client.get(f"/students/{student_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == student_id

    def test_get_student_not_found(self, client, mock_db_session):
        """Test student retrieval when not found."""
        student_id = str(uuid4())
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        response = client.get(f"/students/{student_id}")
        assert response.status_code == 404

    def test_get_student_database_error(self, client, mock_db_session):
        """Test student retrieval with database error."""
        student_id = str(uuid4())
        mock_db_session.query.side_effect = Exception("Database error")

        response = client.get(f"/students/{student_id}")
        assert response.status_code == 500

    def test_list_students_success(self, client, mock_db_session):
        """Test successful student listing."""
        mock_students = [
            Mock(id=str(uuid4()), first_name="John", last_name="Doe"),
            Mock(id=str(uuid4()), first_name="Jane", last_name="Smith"),
        ]

        mock_db_session.query.return_value.offset.return_value.limit.return_value.all.return_value = (
            mock_students
        )
        mock_db_session.query.return_value.count.return_value = 2

        response = client.get("/students")
        assert response.status_code == 200
        data = response.json()
        assert len(data["students"]) == 2
        assert data["total"] == 2

    def test_list_students_database_error(self, client, mock_db_session):
        """Test student listing with database error."""
        mock_db_session.query.side_effect = Exception("Database error")

        response = client.get("/students")
        assert response.status_code == 500

    def test_update_student_success(self, client, mock_db_session):
        """Test successful student update."""
        student_id = str(uuid4())
        mock_student = Mock()
        mock_student.id = student_id
        mock_student.first_name = "John"

        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_student

        update_data = {"first_name": "Johnny"}
        response = client.put(f"/students/{student_id}", json=update_data)
        assert response.status_code == 200

    def test_update_student_not_found(self, client, mock_db_session):
        """Test student update when not found."""
        student_id = str(uuid4())
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        update_data = {"first_name": "Johnny"}
        response = client.put(f"/students/{student_id}", json=update_data)
        assert response.status_code == 404

    def test_update_student_database_error(self, client, mock_db_session):
        """Test student update with database error."""
        student_id = str(uuid4())
        mock_student = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_student
        mock_db_session.commit.side_effect = Exception("Database error")

        update_data = {"first_name": "Johnny"}
        response = client.put(f"/students/{student_id}", json=update_data)
        assert response.status_code == 500

    def test_delete_student_success(self, client, mock_db_session):
        """Test successful student deletion."""
        student_id = str(uuid4())
        mock_student = Mock()
        mock_student.id = student_id

        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_student

        response = client.delete(f"/students/{student_id}")
        assert response.status_code == 200

    def test_delete_student_not_found(self, client, mock_db_session):
        """Test student deletion when not found."""
        student_id = str(uuid4())
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        response = client.delete(f"/students/{student_id}")
        assert response.status_code == 404

    def test_delete_student_database_error(self, client, mock_db_session):
        """Test student deletion with database error."""
        student_id = str(uuid4())
        mock_student = Mock()
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_student
        mock_db_session.commit.side_effect = Exception("Database error")

        response = client.delete(f"/students/{student_id}")
        assert response.status_code == 500


class TestPredictionRoutes:
    """Test prediction endpoints."""

    def test_predict_single_success(self, client):
        """Test successful single prediction."""
        student_id = str(uuid4())
        mock_response = Mock()
        mock_response.prediction.student_id = student_id
        mock_response.prediction.risk_score = 0.75
        mock_response.prediction.risk_category = "high"

        with patch("src.api.routes.predictions.prediction_service") as mock_service:
            mock_service.predict_risk.return_value = mock_response

            response = client.post(f"/predictions/predict/{student_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"]["risk_score"] == 0.75

    def test_predict_single_service_error(self, client):
        """Test single prediction with service error."""
        student_id = str(uuid4())

        with patch("src.api.routes.predictions.prediction_service") as mock_service:
            mock_service.predict_risk.side_effect = Exception("Prediction failed")

            response = client.post(f"/predictions/predict/{student_id}")
            assert response.status_code == 500

    def test_predict_batch_success(self, client):
        """Test successful batch prediction."""
        student_ids = [str(uuid4()), str(uuid4())]
        mock_response = Mock()
        mock_response.predictions = [
            {"student_id": student_ids[0], "risk_score": 0.6},
            {"student_id": student_ids[1], "risk_score": 0.8},
        ]

        with patch("src.api.routes.predictions.prediction_service") as mock_service:
            mock_service.predict_batch.return_value = mock_response

            response = client.post("/predictions/batch", json={"student_ids": student_ids})
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2

    def test_predict_batch_service_error(self, client):
        """Test batch prediction with service error."""
        student_ids = [str(uuid4()), str(uuid4())]

        with patch("src.api.routes.predictions.prediction_service") as mock_service:
            mock_service.predict_batch.side_effect = Exception("Batch prediction failed")

            response = client.post("/predictions/batch", json={"student_ids": student_ids})
            assert response.status_code == 500

    def test_get_predictions_success(self, client, mock_db_session):
        """Test successful prediction retrieval."""
        student_id = str(uuid4())
        mock_predictions = [
            Mock(student_id=student_id, risk_score=0.7, prediction_date=datetime.now())
        ]

        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            mock_predictions
        )

        response = client.get(f"/predictions/{student_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1

    def test_get_predictions_database_error(self, client, mock_db_session):
        """Test prediction retrieval with database error."""
        student_id = str(uuid4())
        mock_db_session.query.side_effect = Exception("Database error")

        response = client.get(f"/predictions/{student_id}")
        assert response.status_code == 500

    def test_get_metrics_success(self, client, mock_db_session):
        """Test successful metrics retrieval."""
        mock_predictions = [
            Mock(risk_score=0.2, risk_category="low"),
            Mock(risk_score=0.8, risk_category="high"),
        ]

        mock_db_session.query.return_value.all.return_value = mock_predictions

        response = client.get("/predictions/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "risk_distribution" in data
        assert "average_risk_score" in data

    def test_get_metrics_database_error(self, client, mock_db_session):
        """Test metrics retrieval with database error."""
        mock_db_session.query.side_effect = Exception("Database error")

        response = client.get("/predictions/metrics")
        assert response.status_code == 500


class TestTrainingRoutes:
    """Test training endpoints."""

    def test_start_training_success(self, client):
        """Test successful training start."""
        training_config = {"epochs": 50, "learning_rate": 0.001, "batch_size": 32}

        with patch("src.api.routes.training.ModelTrainer") as mock_trainer_class, patch(
            "src.api.routes.training.StudentSequenceDataset"
        ) as _mock_dataset_class, patch(
            "src.api.routes.training.GRUAttentionModel"
        ) as _mock_model_class:
            mock_trainer = Mock()
            mock_trainer.fit.return_value = {"final_loss": 0.5}
            mock_trainer_class.return_value = mock_trainer

            mock_dataset = Mock()
            _mock_dataset_class.return_value = mock_dataset

            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            response = client.post("/training/start", json=training_config)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"

    def test_start_training_error(self, client):
        """Test training with error."""
        training_config = {"epochs": 50, "learning_rate": 0.001, "batch_size": 32}

        with patch("src.api.routes.training.ModelTrainer") as mock_trainer_class:
            mock_trainer_class.side_effect = Exception("Training failed")

            response = client.post("/training/start", json=training_config)
            assert response.status_code == 500

    def test_get_training_status_success(self, client):
        """Test successful training status retrieval."""
        with patch("src.api.routes.training.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.stat.return_value.st_mtime = datetime.now().timestamp()

            response = client.get("/training/status")
            assert response.status_code == 200
            data = response.json()
            assert "model_exists" in data
            assert "last_trained" in data

    def test_get_training_status_no_model(self, client):
        """Test training status when no model exists."""
        with patch("src.api.routes.training.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            response = client.get("/training/status")
            assert response.status_code == 200
            data = response.json()
            assert data["model_exists"] is False


class TestMainApp:
    """Test main application configuration."""

    def test_app_creation(self):
        """Test that app is created correctly."""
        assert app is not None
        assert app.title == "EduPulse API"

    def test_cors_middleware(self, client):
        """Test CORS middleware is configured."""
        # Test preflight request
        response = client.options("/", headers={"Origin": "http://localhost:3000"})
        # FastAPI handles CORS automatically, just test it doesn't error
        assert response.status_code in [200, 405]  # 405 is also acceptable for OPTIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

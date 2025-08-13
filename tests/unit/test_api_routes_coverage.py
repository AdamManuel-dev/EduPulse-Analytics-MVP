"""
TARGETED API ROUTES COVERAGE
Focus on API routes which have the most missing lines.
"""

from datetime import datetime
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError


class TestHealthRoutes:
    """Target health routes missing lines."""

    @pytest.fixture
    def client(self):
        from src.api.main import app

        return TestClient(app)

    def test_health_check_complete(self, client):
        """Test all health check paths."""
        with patch("src.api.routes.health.get_db") as mock_get_db:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            # Test successful health check
            mock_session.execute.return_value = Mock()
            response = client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert "database" in data
            assert "timestamp" in data

            # Test database connection failure
            mock_get_db.return_value.__enter__.side_effect = Exception("Database connection failed")
            response = client.get("/health")
            assert response.status_code == 503

            data = response.json()
            assert data["status"] == "unhealthy"

    def test_readiness_check_complete(self, client):
        """Test readiness check paths."""
        with patch("src.api.routes.health.get_db") as mock_get_db, patch(
            "src.api.routes.health.prediction_service"
        ) as mock_pred_service:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value = Mock()

            # Test ready with model loaded
            mock_pred_service.model = Mock()  # Model exists
            response = client.get("/ready")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "ready"
            assert data["database_ready"] is True
            assert data["model_loaded"] is True

            # Test not ready - no model
            mock_pred_service.model = None
            response = client.get("/ready")
            assert response.status_code == 503

            data = response.json()
            assert data["status"] == "not ready"
            assert data["model_loaded"] is False

            # Test not ready - database failure
            mock_pred_service.model = Mock()
            mock_get_db.return_value.__enter__.side_effect = Exception("DB failed")
            response = client.get("/ready")
            assert response.status_code == 503


class TestStudentsRoutes:
    """Target students routes missing lines."""

    @pytest.fixture
    def client(self):
        from src.api.main import app

        return TestClient(app)

    def test_create_student_complete(self, client):
        """Test all create student paths."""
        with patch("src.api.routes.students.get_db") as mock_get_db:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            student_data = {
                "first_name": "John",
                "last_name": "Doe",
                "email": "john.doe@test.com",
                "date_of_birth": "2005-03-15",
                "grade_level": "10",
            }

            # Test successful creation
            mock_student = Mock()
            mock_student.id = str(uuid4())
            mock_student.first_name = "John"
            mock_student.last_name = "Doe"
            mock_student.email = "john.doe@test.com"

            with patch("src.api.routes.students.models.Student", return_value=mock_student):
                mock_session.add.return_value = None
                mock_session.commit.return_value = None
                mock_session.refresh.return_value = None

                response = client.post("/students", json=student_data)
                assert response.status_code == 201

                data = response.json()
                assert data["first_name"] == "John"
                assert data["last_name"] == "Doe"
                assert "id" in data

            # Test integrity error (duplicate email)
            mock_session.add.side_effect = IntegrityError("UNIQUE constraint failed", "", "")
            response = client.post("/students", json=student_data)
            assert response.status_code == 400

            data = response.json()
            assert "detail" in data
            assert "already exists" in data["detail"]

            # Test general database error
            mock_session.add.side_effect = Exception("Database error")
            response = client.post("/students", json=student_data)
            assert response.status_code == 500

    def test_get_student_complete(self, client):
        """Test all get student paths."""
        with patch("src.api.routes.students.get_db") as mock_get_db:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            student_id = str(uuid4())

            # Test successful get
            mock_student = Mock()
            mock_student.id = student_id
            mock_student.first_name = "Jane"
            mock_student.last_name = "Smith"
            mock_student.email = "jane.smith@test.com"

            mock_session.query.return_value.filter.return_value.first.return_value = mock_student

            response = client.get(f"/students/{student_id}")
            assert response.status_code == 200

            data = response.json()
            assert data["id"] == student_id
            assert data["first_name"] == "Jane"

            # Test student not found
            mock_session.query.return_value.filter.return_value.first.return_value = None
            response = client.get(f"/students/{student_id}")
            assert response.status_code == 404

            data = response.json()
            assert "not found" in data["detail"]

            # Test database error
            mock_session.query.side_effect = Exception("Database query failed")
            response = client.get(f"/students/{student_id}")
            assert response.status_code == 500

    def test_list_students_complete(self, client):
        """Test all list students paths."""
        with patch("src.api.routes.students.get_db") as mock_get_db:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            # Test successful list
            mock_students = []
            for i in range(5):
                student = Mock()
                student.id = str(uuid4())
                student.first_name = f"Student{i}"
                student.last_name = f"Lastname{i}"
                student.email = f"student{i}@test.com"
                mock_students.append(student)

            mock_session.query.return_value.offset.return_value.limit.return_value.all.return_value = (
                mock_students
            )
            mock_session.query.return_value.count.return_value = 5

            response = client.get("/students?skip=0&limit=10")
            assert response.status_code == 200

            data = response.json()
            assert "students" in data
            assert "total" in data
            assert data["total"] == 5
            assert len(data["students"]) == 5

            # Test with pagination
            response = client.get("/students?skip=2&limit=3")
            assert response.status_code == 200

            # Test database error
            mock_session.query.side_effect = Exception("Database list failed")
            response = client.get("/students")
            assert response.status_code == 500


class TestPredictionsRoutes:
    """Target predictions routes missing lines."""

    @pytest.fixture
    def client(self):
        from src.api.main import app

        return TestClient(app)

    def test_predict_single_complete(self, client):
        """Test all single prediction paths."""
        with patch("src.api.routes.predictions.prediction_service") as mock_service:
            student_id = str(uuid4())

            # Test successful prediction
            mock_response = Mock()
            mock_response.prediction.student_id = student_id
            mock_response.prediction.risk_score = 0.75
            mock_response.prediction.risk_category = "high"
            mock_response.prediction.confidence = 0.85
            mock_response.contributing_factors = [
                {"factor": "attendance", "weight": 0.6, "description": "Low attendance rate"},
                {"factor": "grades", "weight": 0.3, "description": "Declining grades"},
            ]
            mock_response.timestamp = datetime.now()

            mock_service.predict_risk.return_value = mock_response

            response = client.post(f"/predictions/predict/{student_id}")
            assert response.status_code == 200

            data = response.json()
            assert "prediction" in data
            assert "contributing_factors" in data
            assert "timestamp" in data
            assert data["prediction"]["risk_score"] == 0.75

            # Test service error
            mock_service.predict_risk.side_effect = Exception("Prediction service failed")
            response = client.post(f"/predictions/predict/{student_id}")
            assert response.status_code == 500

            data = response.json()
            assert "error" in data

    def test_predict_batch_complete(self, client):
        """Test all batch prediction paths."""
        with patch("src.api.routes.predictions.prediction_service") as mock_service:
            student_ids = [str(uuid4()) for _ in range(3)]

            # Test successful batch prediction
            mock_response = Mock()
            mock_response.predictions = []
            for i, sid in enumerate(student_ids):
                pred = {
                    "student_id": sid,
                    "risk_score": 0.5 + i * 0.1,
                    "risk_category": ["medium", "high", "high"][i],
                    "confidence": 0.8 + i * 0.05,
                }
                mock_response.predictions.append(pred)

            mock_response.processing_time_ms = 250.0
            mock_response.total_processed = 3
            mock_response.successful_predictions = 3
            mock_response.failed_predictions = 0

            mock_service.predict_batch.return_value = mock_response

            batch_request = {"student_ids": student_ids}
            response = client.post("/predictions/batch", json=batch_request)
            assert response.status_code == 200

            data = response.json()
            assert "predictions" in data
            assert "processing_time_ms" in data
            assert "summary" in data
            assert len(data["predictions"]) == 3

            # Test batch service error
            mock_service.predict_batch.side_effect = Exception("Batch prediction failed")
            response = client.post("/predictions/batch", json=batch_request)
            assert response.status_code == 500

    def test_get_predictions_complete(self, client):
        """Test get predictions paths."""
        with patch("src.api.routes.predictions.get_db") as mock_get_db:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            student_id = str(uuid4())

            # Test successful get predictions
            mock_predictions = []
            for i in range(3):
                pred = Mock()
                pred.student_id = student_id
                pred.risk_score = 0.6 + i * 0.1
                pred.risk_category = ["medium", "high", "high"][i]
                pred.confidence = 0.75 + i * 0.1
                pred.prediction_date = datetime.now()
                mock_predictions.append(pred)

            mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
                mock_predictions
            )

            response = client.get(f"/predictions/{student_id}")
            assert response.status_code == 200

            data = response.json()
            assert len(data) == 3

            # Test database error
            mock_session.query.side_effect = Exception("Database error")
            response = client.get(f"/predictions/{student_id}")
            assert response.status_code == 500

    def test_get_metrics_complete(self, client):
        """Test get metrics paths."""
        with patch("src.api.routes.predictions.get_db") as mock_get_db:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            # Test successful metrics
            mock_predictions = []
            risk_scores = [0.2, 0.4, 0.6, 0.8, 0.9, 0.3, 0.7, 0.1, 0.5, 0.95]
            categories = [
                "low",
                "medium",
                "high",
                "high",
                "high",
                "medium",
                "high",
                "low",
                "medium",
                "high",
            ]

            for i, (score, cat) in enumerate(zip(risk_scores, categories)):
                pred = Mock()
                pred.risk_score = score
                pred.risk_category = cat
                pred.confidence = 0.8
                mock_predictions.append(pred)

            mock_session.query.return_value.all.return_value = mock_predictions

            response = client.get("/predictions/metrics")
            assert response.status_code == 200

            data = response.json()
            assert "total_predictions" in data
            assert "average_risk_score" in data
            assert "risk_distribution" in data
            assert data["total_predictions"] == 10

            # Test database error
            mock_session.query.side_effect = Exception("Database metrics error")
            response = client.get("/predictions/metrics")
            assert response.status_code == 500


class TestTrainingRoutes:
    """Target training routes missing lines."""

    @pytest.fixture
    def client(self):
        from src.api.main import app

        return TestClient(app)

    def test_start_training_complete(self, client):
        """Test start training paths."""
        with patch("src.api.routes.training.ModelTrainer") as mock_trainer_class, patch(
            "src.api.routes.training.StudentSequenceDataset"
        ) as _mock_dataset_class, patch(
            "src.api.routes.training.GRUAttentionModel"
        ) as _mock_model_class, patch(
            "src.api.routes.training.get_db"
        ) as mock_get_db:
            mock_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            # Test successful training start
            mock_trainer = Mock()
            mock_trainer.fit.return_value = {
                "final_train_loss": 0.35,
                "final_val_loss": 0.42,
                "best_epoch": 18,
                "total_epochs": 25,
                "training_time": 3600.0,
                "early_stopped": False,
            }
            mock_trainer_class.return_value = mock_trainer

            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=200)
            _mock_dataset_class.return_value = mock_dataset

            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            training_config = {
                "epochs": 30,
                "learning_rate": 0.001,
                "batch_size": 32,
                "validation_split": 0.2,
                "early_stopping_patience": 5,
            }

            response = client.post("/training/start", json=training_config)
            assert response.status_code == 200

            data = response.json()
            assert "message" in data
            assert "training_results" in data
            assert data["training_results"]["final_train_loss"] == 0.35

            # Test training error
            mock_trainer_class.side_effect = Exception("Training failed to start")
            response = client.post("/training/start", json=training_config)
            assert response.status_code == 500

            data = response.json()
            assert "error" in data

    def test_get_training_status_complete(self, client):
        """Test training status paths."""
        with patch("src.api.routes.training.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance

            # Test with model existing
            mock_path_instance.exists.return_value = True
            mock_path_instance.stat.return_value.st_mtime = datetime.now().timestamp()

            response = client.get("/training/status")
            assert response.status_code == 200

            data = response.json()
            assert data["model_exists"] is True
            assert "last_trained" in data

            # Test without model
            mock_path_instance.exists.return_value = False
            response = client.get("/training/status")
            assert response.status_code == 200

            data = response.json()
            assert data["model_exists"] is False
            assert data["last_trained"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

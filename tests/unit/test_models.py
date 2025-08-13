"""
Unit tests for ML models and database models.
"""

import pytest
import torch
import numpy as np
from datetime import date, datetime
from uuid import uuid4

from src.models.gru_model import GRUAttentionModel, EarlyStopping
from src.models.schemas import (
    StudentCreate,
    PredictionRequest,
    PredictResponse,
    TrainingUpdateRequest,
    Prediction,
    RiskFactor,
    RiskCategory,
)
from src.db.models import Student, AttendanceRecord, Grade, DisciplineIncident


class TestGRUModel:
    """Test GRU attention model."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return GRUAttentionModel(
            input_size=42,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.1,
            bidirectional=False,
        )

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.input_size == 42
        assert model.hidden_size == 32
        assert model.num_layers == 1
        assert model.num_heads == 2
        assert not model.bidirectional

    def test_forward_pass(self, model):
        """Test forward pass through the model."""
        batch_size = 4
        seq_len = 10
        input_tensor = torch.randn(batch_size, seq_len, 42)

        model.eval()
        with torch.no_grad():
            risk_score, risk_category, attention_weights = model(input_tensor)

        # Check output shapes
        assert risk_score.shape == (batch_size, 1)
        assert risk_category.shape == (batch_size, 4)
        assert attention_weights is None  # Not requested

        # Check value ranges
        assert torch.all(risk_score >= 0) and torch.all(risk_score <= 1)

    def test_forward_with_attention(self, model):
        """Test forward pass with attention weights."""
        batch_size = 2
        seq_len = 5
        input_tensor = torch.randn(batch_size, seq_len, 42)

        model.eval()
        with torch.no_grad():
            risk_score, risk_category, attention_weights = model(
                input_tensor, return_attention=True
            )

        assert attention_weights is not None
        assert attention_weights.shape[0] == batch_size

    def test_predict_method(self, model):
        """Test predict method for single sample."""
        input_tensor = torch.randn(1, 10, 42)

        risk_value, risk_category, confidence = model.predict(input_tensor)

        assert isinstance(risk_value, float)
        assert 0 <= risk_value <= 1
        assert risk_category in ["low", "medium", "high", "critical"]
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_get_attention_weights(self, model):
        """Test attention weight extraction."""
        input_tensor = torch.randn(1, 10, 42)

        attention_weights = model.get_attention_weights(input_tensor)

        assert isinstance(attention_weights, np.ndarray)
        if attention_weights.size > 0:
            assert attention_weights.ndim >= 2

    def test_model_training_step(self, model):
        """Test a single training step."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create dummy data
        input_tensor = torch.randn(8, 10, 42)
        risk_targets = torch.rand(8, 1)
        category_targets = torch.randint(0, 4, (8,))

        # Forward pass
        risk_pred, category_pred, _ = model(input_tensor)

        # Calculate losses
        mse_loss = torch.nn.functional.mse_loss(risk_pred, risk_targets)
        ce_loss = torch.nn.functional.cross_entropy(category_pred, category_targets)
        total_loss = 0.5 * mse_loss + 0.5 * ce_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        assert total_loss.item() > 0
        assert not torch.isnan(total_loss)


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_init(self):
        """Test early stopping initialization."""
        early_stop = EarlyStopping(patience=5, min_delta=0.001)

        assert early_stop.patience == 5
        assert early_stop.min_delta == 0.001
        assert early_stop.counter == 0
        assert early_stop.best_loss is None
        assert not early_stop.early_stop

    def test_early_stopping_improvement(self):
        """Test early stopping with improving loss."""
        early_stop = EarlyStopping(patience=3, min_delta=0.001)

        losses = [0.5, 0.4, 0.3, 0.2]  # Improving
        for loss in losses:
            should_stop = early_stop(loss)
            assert not should_stop

        assert early_stop.best_loss == 0.2
        assert early_stop.counter == 0

    def test_early_stopping_trigger(self):
        """Test early stopping trigger."""
        early_stop = EarlyStopping(patience=2, min_delta=0.001)

        losses = [0.5, 0.4, 0.4, 0.4, 0.4]  # No improvement

        for i, loss in enumerate(losses):
            should_stop = early_stop(loss)
            if i < 3:
                assert not should_stop
            else:
                assert should_stop
                break


class TestPydanticSchemas:
    """Test Pydantic schemas."""

    def test_student_create_schema(self):
        """Test student creation schema."""
        student_data = {
            "district_id": "STU001",
            "first_name": "John",
            "last_name": "Doe",
            "grade_level": 10,
            "enrollment_date": str(date.today()),
            "date_of_birth": "2008-05-15",
            "gender": "M",
            "ethnicity": "White",
            "socioeconomic_status": "middle",
        }

        student = StudentCreate(**student_data)
        assert student.district_id == "STU001"
        assert student.grade_level == 10
        assert isinstance(student.enrollment_date, date)

    def test_prediction_request_schema(self):
        """Test prediction request schema."""
        test_uuid = uuid4()
        request_data = {
            "student_id": test_uuid,
            "include_factors": True,
        }

        request = PredictionRequest(**request_data)
        assert request.student_id == test_uuid
        assert request.include_factors is True

    def test_prediction_response_schema(self):
        """Test prediction response schema."""
        test_uuid = uuid4()
        prediction = Prediction(
            id=uuid4(),
            student_id=test_uuid,
            risk_score=0.75,
            risk_category=RiskCategory.HIGH,
            confidence=0.85,
            risk_factors=[],
            model_version="1.0.0",
            prediction_date=datetime.utcnow(),
            created_at=datetime.utcnow(),
        )

        response_data = {
            "prediction": prediction,
            "timestamp": datetime.utcnow(),
        }

        response = PredictResponse(**response_data)
        assert response.prediction.student_id == test_uuid
        assert response.prediction.risk_score == 0.75
        assert response.prediction.risk_category == RiskCategory.HIGH

    def test_training_config_schema(self):
        """Test training update request schema."""
        config_data = {
            "student_outcomes": [{"student_id": str(uuid4()), "actual_outcome": "dropout"}],
            "feedback_corrections": [{"prediction_id": str(uuid4()), "correction": "false_positive"}],
            "update_mode": "incremental",
        }

        config = TrainingUpdateRequest(**config_data)
        assert config.student_outcomes[0]["actual_outcome"] == "dropout"
        assert config.feedback_corrections[0]["correction"] == "false_positive"
        assert config.update_mode == "incremental"

    def test_risk_level_enum(self):
        """Test risk category enumeration."""
        assert RiskCategory.LOW.value == "low"
        assert RiskCategory.MEDIUM.value == "medium"
        assert RiskCategory.HIGH.value == "high"
        assert RiskCategory.CRITICAL.value == "critical"


class TestDatabaseModels:
    """Test SQLAlchemy database models."""

    def test_student_model(self, db_session):
        """Test student model creation and retrieval."""
        student = Student(
            district_id="TEST001",
            first_name="Test",
            last_name="Student",
            grade_level=9,
            enrollment_date=date.today(),
            date_of_birth=date(2009, 1, 1),
            gender="F",
            ethnicity="Asian",
            socioeconomic_status="low",
        )

        db_session.add(student)
        db_session.commit()

        retrieved = db_session.query(Student).filter_by(district_id="TEST001").first()
        assert retrieved is not None
        assert retrieved.first_name == "Test"
        assert retrieved.grade_level == 9

    def test_attendance_record_model(self, db_session, sample_student):
        """Test attendance record model."""
        attendance = AttendanceRecord(
            student_id=sample_student.id,
            date=date.today(),
            status="present",
            period=1,
            minutes_late=0,
        )

        db_session.add(attendance)
        db_session.commit()

        retrieved = (
            db_session.query(AttendanceRecord)
            .filter_by(student_id=sample_student.id)
            .first()
        )
        assert retrieved is not None
        assert retrieved.status == "present"

    def test_grade_model(self, db_session, sample_student):
        """Test grade model."""
        grade = Grade(
            student_id=sample_student.student_id,
            course_id="MATH101",
            assignment_id="HW001",
            assignment_type="homework",
            grade_value=95.0,
            max_grade_value=100.0,
            submission_date=date.today(),
            semester="Fall 2024",
            academic_year="2024-2025",
        )

        db_session.add(grade)
        db_session.commit()

        retrieved = db_session.query(Grade).filter_by(student_id=sample_student.student_id).first()
        assert retrieved is not None
        assert retrieved.grade_value == 95.0
        assert retrieved.course_id == "MATH101"

    def test_discipline_incident_model(self, db_session, sample_student):
        """Test discipline incident model."""
        incident = DisciplineIncident(
            student_id=sample_student.student_id,
            incident_date=date.today(),
            incident_type="tardiness",
            severity_level=1,
            description="Late to class",
            action_taken="Warning",
            reported_by="Teacher",
        )

        db_session.add(incident)
        db_session.commit()

        retrieved = (
            db_session.query(DisciplineIncident)
            .filter_by(student_id=sample_student.student_id)
            .first()
        )
        assert retrieved is not None
        assert retrieved.incident_type == "tardiness"
        assert retrieved.severity_level == 1

    def test_model_relationships(
        self,
        db_session,
        sample_student,
        sample_attendance_records,
        sample_grades,
        sample_discipline_incidents,
    ):
        """Test model relationships."""
        # Query student with related records
        student = db_session.query(Student).filter_by(student_id=sample_student.student_id).first()

        # Check relationships exist (would need to be defined in models)
        attendance_count = (
            db_session.query(AttendanceRecord).filter_by(student_id=student.student_id).count()
        )
        assert attendance_count == len(sample_attendance_records)

        grade_count = db_session.query(Grade).filter_by(student_id=student.student_id).count()
        assert grade_count == len(sample_grades)

        incident_count = (
            db_session.query(DisciplineIncident).filter_by(student_id=student.student_id).count()
        )
        assert incident_count == len(sample_discipline_incidents)

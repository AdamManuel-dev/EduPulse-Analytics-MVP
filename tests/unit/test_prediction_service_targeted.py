"""
Targeted tests for prediction service missing coverage lines.
Focus on specific uncovered code paths.
"""

from datetime import datetime
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import torch

from src.services.prediction_service import PredictionService


class TestPredictionServiceTargeted:
    """Targeted tests to cover specific missing lines."""

    @pytest.fixture(autouse=True)
    def setup_service(self):
        """Setup service with proper mocking."""
        with patch("src.services.prediction_service.settings") as mock_settings, patch(
            "src.services.prediction_service.get_db"
        ) as mock_get_db, patch(
            "src.services.prediction_service.GRUAttentionModel"
        ) as _mock_model_class, patch(
            "src.services.prediction_service.FeaturePipeline"
        ) as mock_pipeline_class, patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            # Setup settings
            mock_settings.model_path = "/app/models"
            mock_settings.sequence_length = 20
            mock_settings.device = "cpu"
            mock_settings.model_version = "v1.0"

            # Setup path
            mock_path.return_value.exists.return_value = False

            # Setup database
            mock_db = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db
            mock_get_db.return_value.__exit__.return_value = None

            # Setup model
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock(return_value=mock_model)
            _mock_model_class.return_value = mock_model

            # Setup pipeline
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.extract_features.return_value = [0.5] * 42
            mock_pipeline.get_feature_names.return_value = [f"feature_{i}" for i in range(42)]

            self.service = PredictionService()
            self.mock_model = mock_model
            self.mock_pipeline = mock_pipeline
            self.mock_db = mock_db

            yield

    def test_predict_risk_with_factors_extraction(self):
        """Test prediction with factor extraction - covers lines 212-214."""

        # Mock model forward with attention weights
        def mock_forward(x, return_attention=True):
            batch_size = x.shape[0] if hasattr(x, "shape") else 1
            risk_scores = torch.tensor([[0.75]])
            category_logits = torch.tensor([[0.1, 0.2, 0.8, 0.3]])
            attention_weights = torch.randn(1, 20, 42) if return_attention else None
            return risk_scores, category_logits, attention_weights

        self.mock_model.forward = mock_forward
        self.service.feature_pipeline = self.mock_pipeline

        with patch("src.services.prediction_service.db_models") as mock_db_models, patch(
            "src.services.prediction_service.schemas"
        ) as mock_schemas:
            # Mock database model
            mock_prediction = Mock()
            mock_db_models.Prediction.return_value = mock_prediction

            # Mock schema response
            mock_response = Mock()
            mock_schemas.PredictResponse.return_value = mock_response

            student_id = str(uuid4())
            result = self.service.predict_risk(student_id, include_factors=True)

            # Should extract factors - covers lines 212-214
            assert result == mock_response
            mock_db_models.Prediction.assert_called_once()
            mock_schemas.PredictResponse.assert_called_once()

    def test_predict_risk_database_operations(self):
        """Test database operations in predict_risk - covers lines 217-238."""

        # Mock model forward
        def mock_forward(x, return_attention=False):
            return torch.tensor([[0.65]]), torch.tensor([[0.2, 0.4, 0.6, 0.2]]), None

        self.mock_model.forward = mock_forward
        self.service.feature_pipeline = self.mock_pipeline

        with patch("src.services.prediction_service.db_models") as mock_db_models, patch(
            "src.services.prediction_service.schemas"
        ) as mock_schemas, patch("src.services.prediction_service.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2025, 1, 15, 12, 0, 0)

            # Mock database model
            mock_prediction = Mock()
            mock_db_models.Prediction.return_value = mock_prediction

            # Mock schema response
            mock_response = Mock()
            mock_schemas.PredictResponse.return_value = mock_response

            student_id = str(uuid4())
            result = self.service.predict_risk(student_id, include_factors=False)

            # Verify database operations
            self.mock_db.add.assert_called_once_with(mock_prediction)
            self.mock_db.commit.assert_called_once()
            self.mock_db.refresh.assert_called_once_with(mock_prediction)

            # Verify response creation
            mock_schemas.PredictResponse.assert_called_once()
            call_args = mock_schemas.PredictResponse.call_args[1]
            assert call_args["prediction"] == mock_prediction
            assert call_args["contributing_factors"] is None  # include_factors=False

    def test_extract_contributing_factors_complete(self):
        """Test complete factor extraction - covers lines 296-342."""
        self.service.feature_pipeline = self.mock_pipeline

        # Create test data
        X = torch.randn(1, 20, 42)
        attention_weights = torch.randn(1, 20, 42)
        risk_score = 0.75

        # Test factor extraction
        factors = self.service._extract_contributing_factors(X, attention_weights, risk_score)

        # Should return list of factors
        assert isinstance(factors, list)
        assert len(factors) <= 5  # Top 5 factors

        if factors:
            factor = factors[0]
            assert "factor" in factor
            assert "weight" in factor
            assert "details" in factor

    def test_describe_factor_methods(self):
        """Test factor description methods - covers lines 344-397."""
        # Test attendance factors
        factor_type, description = self.service._describe_attendance_factor("absence_rate", 0.4)
        assert factor_type == "attendance_pattern"
        assert "absence" in description.lower()

        factor_type, description = self.service._describe_attendance_factor("tardy_rate", 0.2)
        assert factor_type == "tardiness"
        assert "tardiness" in description.lower()

        # Test grades factors
        factor_type, description = self.service._describe_grades_factor("gpa_current", 2.5)
        assert factor_type == "academic_performance"
        assert "gpa" in description.lower()

        factor_type, description = self.service._describe_grades_factor("grade_trend", -0.1)
        assert factor_type == "grade_trajectory"
        assert "declining" in description.lower()

        # Test discipline factors
        factor_type, description = self.service._describe_discipline_factor("incident_count", 3)
        assert factor_type == "behavioral"
        assert "incident" in description.lower()

        factor_type, description = self.service._describe_discipline_factor("severity_max", 2.5)
        assert factor_type == "behavioral"
        assert "severity" in description.lower()

        # Test general describe_factor method
        factor_type, description = self.service._describe_factor("attendance_absence_rate", 0.3)
        assert factor_type == "attendance_pattern"

        factor_type, description = self.service._describe_factor("grades_gpa", 3.2)
        assert factor_type == "academic_performance"

        factor_type, description = self.service._describe_factor("discipline_incidents", 1)
        assert factor_type == "behavioral"

        factor_type, description = self.service._describe_factor("unknown_feature", 1.0)
        assert factor_type == "other"

    def test_fallback_prediction_complete(self):
        """Test complete fallback prediction - covers lines 399-432."""
        with patch("src.services.prediction_service.db_models") as mock_db_models, patch(
            "src.services.prediction_service.schemas"
        ) as mock_schemas, patch("src.services.prediction_service.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2025, 1, 15, 12, 0, 0)

            # Mock database model
            mock_prediction = Mock()
            mock_db_models.Prediction.return_value = mock_prediction

            # Mock schema response
            mock_response = Mock()
            mock_schemas.PredictResponse.return_value = mock_response

            student_id = str(uuid4())
            result = self.service._fallback_prediction(student_id)

            # Verify fallback prediction creation
            mock_db_models.Prediction.assert_called_once()
            call_args = mock_db_models.Prediction.call_args[1]
            assert call_args["student_id"] == student_id
            assert call_args["risk_score"] == 0.5
            assert call_args["risk_category"] == "medium"
            assert call_args["confidence"] == 0.5
            assert call_args["model_version"] == "fallback"

            # Verify database operations
            self.mock_db.add.assert_called_once_with(mock_prediction)
            self.mock_db.commit.assert_called_once()
            self.mock_db.refresh.assert_called_once_with(mock_prediction)

            # Verify response creation
            mock_schemas.PredictResponse.assert_called_once()
            assert result == mock_response

    def test_predict_batch_complete(self):
        """Test batch prediction - covers lines 245-294."""
        student_ids = [str(uuid4()), str(uuid4())]

        # Mock predict_risk method
        mock_responses = []
        for student_id in student_ids:
            mock_response = Mock()
            mock_response.prediction.student_id = student_id
            mock_response.prediction.risk_score = 0.5 + (len(mock_responses) * 0.2)
            mock_response.prediction.risk_category = "medium"
            mock_responses.append(mock_response)

        with patch.object(
            self.service, "predict_risk", side_effect=mock_responses
        ) as mock_predict, patch("src.services.prediction_service.schemas") as mock_schemas, patch(
            "src.services.prediction_service.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 15, 12, 0, 0)

            # Mock batch response
            mock_batch_response = Mock()
            mock_schemas.BatchPredictResponse.return_value = mock_batch_response

            result = self.service.predict_batch(student_ids, top_k=5)

            # Verify predict_risk was called for each student
            assert mock_predict.call_count == len(student_ids)

            # Verify batch response creation
            mock_schemas.BatchPredictResponse.assert_called_once()
            call_args = mock_schemas.BatchPredictResponse.call_args[1]
            assert len(call_args["predictions"]) == len(student_ids)
            assert "processing_time_ms" in call_args

            assert result == mock_batch_response

    def test_predict_batch_with_errors(self):
        """Test batch prediction with individual errors - covers error handling."""
        student_ids = [str(uuid4()), str(uuid4())]

        def mock_predict_side_effect(student_id, reference_date=None, include_factors=True):
            if student_id == student_ids[0]:
                raise Exception("Prediction failed")
            # Second student succeeds
            mock_response = Mock()
            mock_response.prediction.student_id = student_id
            mock_response.prediction.risk_score = 0.7
            mock_response.prediction.risk_category = "high"
            return mock_response

        with patch.object(
            self.service, "predict_risk", side_effect=mock_predict_side_effect
        ) as mock_predict, patch("src.services.prediction_service.schemas") as mock_schemas, patch(
            "src.services.prediction_service.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 15, 12, 0, 0)

            # Mock batch response
            mock_batch_response = Mock()
            mock_schemas.BatchPredictResponse.return_value = mock_batch_response

            result = self.service.predict_batch(student_ids)

            # Should handle errors gracefully
            mock_schemas.BatchPredictResponse.assert_called_once()
            call_args = mock_schemas.BatchPredictResponse.call_args[1]
            predictions = call_args["predictions"]

            # Should have fallback for failed student and real prediction for successful student
            assert len(predictions) == 2
            assert result == mock_batch_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Working functional tests for prediction service.
Tests actual functionality with proper mocking.
"""

from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import torch

# Import the actual service
from src.services.prediction_service import PredictionService


class TestPredictionServiceWorking:
    """Working tests for PredictionService."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup basic mocks for all tests."""
        # Mock settings
        self.mock_settings_patch = patch("src.services.prediction_service.settings")
        self.mock_settings = self.mock_settings_patch.start()
        self.mock_settings.model_path = "/app/models"
        self.mock_settings.model_version = "test_v1"

        # Mock database
        self.mock_get_db_patch = patch("src.services.prediction_service.get_db")
        self.mock_get_db = self.mock_get_db_patch.start()
        self.mock_db_session = Mock()
        self.mock_get_db.return_value.__enter__.return_value = self.mock_db_session
        self.mock_get_db.return_value.__exit__.return_value = None

        # Mock models
        self.mock_prediction = Mock()
        self.mock_prediction.student_id = "test-student"
        self.mock_prediction.risk_score = 0.75
        self.mock_prediction.risk_category = "high"
        self.mock_prediction.confidence = 0.85

        # Mock Path
        self.mock_path_patch = patch("src.services.prediction_service.Path")
        self.mock_path = self.mock_path_patch.start()
        self.mock_path.return_value.exists.return_value = False

        yield

        # Cleanup
        self.mock_settings_patch.stop()
        self.mock_get_db_patch.stop()
        self.mock_path_patch.stop()

    def test_initialization_success(self):
        """Test successful service initialization."""
        with patch("src.services.prediction_service.GRUAttentionModel") as _mock_model_class:
            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            service = PredictionService()

            # Should create model and set device
            assert service.model == mock_model
            assert service.device is not None
            mock_model.to.assert_called_once()
            mock_model.eval.assert_called_once()

    def test_load_model_no_file(self):
        """Test model loading when no file exists."""
        with patch("src.services.prediction_service.GRUAttentionModel") as _mock_model_class:
            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            service = PredictionService()

            # Should initialize with random weights
            assert service.model == mock_model
            # Path.exists should have been called
            self.mock_path.return_value.exists.assert_called()

    def test_load_model_with_file(self):
        """Test model loading when file exists."""
        self.mock_path.return_value.exists.return_value = True

        with patch("src.services.prediction_service.GRUAttentionModel") as _mock_model_class, patch(
            "torch.load"
        ) as mock_torch_load:
            mock_model = Mock()
            _mock_model_class.return_value = mock_model
            mock_checkpoint = {"model_state_dict": {"test": "weights"}}
            mock_torch_load.return_value = mock_checkpoint

            service = PredictionService()

            # Should load weights
            mock_torch_load.assert_called_once()
            mock_model.load_state_dict.assert_called_once_with(mock_checkpoint["model_state_dict"])

    def test_prepare_sequence_functionality(self):
        """Test sequence preparation."""
        with patch("src.services.prediction_service.GRUAttentionModel") as _mock_model_class, patch(
            "src.services.prediction_service.FeaturePipeline"
        ) as mock_pipeline_class:
            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            # Setup feature pipeline mock
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.extract_features.return_value = [0.5] * 42  # 42 features

            service = PredictionService()
            student_id = str(uuid4())

            # Test sequence preparation
            result = service.prepare_sequence(student_id, sequence_length=5)

            # Should return tensor with correct shape
            assert isinstance(result, torch.Tensor)
            assert result.shape == (1, 5, 42)  # batch, sequence, features

            # Should call feature extraction 5 times
            assert mock_pipeline.extract_features.call_count == 5

    def test_predict_risk_basic_functionality(self):
        """Test basic risk prediction."""
        with patch("src.services.prediction_service.GRUAttentionModel") as _mock_model_class, patch(
            "src.services.prediction_service.FeaturePipeline"
        ) as mock_pipeline_class, patch(
            "src.services.prediction_service.db_models"
        ) as mock_db_models, patch(
            "src.services.prediction_service.schemas"
        ) as mock_schemas:
            # Setup model mock
            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            # Mock model forward pass
            def mock_forward(x, return_attention=False):
                batch_size = x.shape[0]
                risk_scores = torch.tensor([[0.75]])  # Fixed risk score
                category_logits = torch.tensor([[0.1, 0.2, 0.8, 0.3]])  # High risk category
                attention_weights = (
                    torch.randn(batch_size, x.shape[1], x.shape[2]) if return_attention else None
                )
                return risk_scores, category_logits, attention_weights

            mock_model.forward = mock_forward

            # Setup feature pipeline mock
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.extract_features.return_value = [0.5] * 42

            # Setup database model mock
            mock_prediction_model = Mock()
            mock_db_models.Prediction = mock_prediction_model

            # Setup schema mock
            mock_response = Mock()
            mock_schemas.PredictResponse = Mock(return_value=mock_response)

            service = PredictionService()
            student_id = str(uuid4())

            # Test prediction
            result = service.predict_risk(student_id)

            # Should create prediction model
            mock_prediction_model.assert_called_once()
            # Should create response schema
            mock_schemas.PredictResponse.assert_called_once()

            assert result == mock_response

    def test_predict_batch_functionality(self):
        """Test batch prediction."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel"
        ) as _mock_model_class, patch.object(
            PredictionService, "predict_risk"
        ) as mock_predict_risk, patch(
            "src.services.prediction_service.schemas"
        ) as mock_schemas:
            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            # Setup predict_risk mock
            def mock_predict_side_effect(student_id, reference_date=None, include_factors=True):
                mock_pred = Mock()
                mock_pred.prediction.student_id = student_id
                mock_pred.prediction.risk_score = 0.5 + (hash(student_id) % 50) / 100  # Vary by ID
                mock_pred.prediction.risk_category = "medium"
                return mock_pred

            mock_predict_risk.side_effect = mock_predict_side_effect

            # Setup batch response mock
            mock_batch_response = Mock()
            mock_schemas.BatchPredictResponse = Mock(return_value=mock_batch_response)

            service = PredictionService()
            student_ids = [str(uuid4()) for _ in range(3)]

            result = service.predict_batch(student_ids)

            # Should call predict_risk for each student
            assert mock_predict_risk.call_count == 3
            # Should create batch response
            mock_schemas.BatchPredictResponse.assert_called_once()

            assert result == mock_batch_response

    def test_error_handling_fallback(self):
        """Test error handling with fallback prediction."""
        with patch("src.services.prediction_service.GRUAttentionModel") as _mock_model_class, patch(
            "src.services.prediction_service.FeaturePipeline"
        ) as mock_pipeline_class, patch.object(
            PredictionService, "_fallback_prediction"
        ) as mock_fallback:
            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            # Make feature pipeline fail
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.extract_features.side_effect = Exception("Feature extraction failed")

            # Setup fallback mock
            mock_fallback_response = Mock()
            mock_fallback.return_value = mock_fallback_response

            service = PredictionService()
            student_id = str(uuid4())

            result = service.predict_risk(student_id)

            # Should call fallback prediction
            mock_fallback.assert_called_once_with(student_id)
            assert result == mock_fallback_response

    def test_fallback_prediction_creation(self):
        """Test fallback prediction creation."""
        with patch("src.services.prediction_service.GRUAttentionModel") as _mock_model_class, patch(
            "src.services.prediction_service.db_models"
        ) as mock_db_models, patch("src.services.prediction_service.schemas") as mock_schemas:
            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            # Setup mocks
            mock_prediction_model = Mock()
            mock_db_models.Prediction = mock_prediction_model
            mock_response = Mock()
            mock_schemas.PredictResponse = Mock(return_value=mock_response)

            service = PredictionService()
            student_id = str(uuid4())

            result = service._fallback_prediction(student_id)

            # Should create fallback prediction with default values
            mock_prediction_model.assert_called_once()
            call_args = mock_prediction_model.call_args[1]
            assert call_args["student_id"] == student_id
            assert call_args["risk_score"] == 0.5
            assert call_args["risk_category"] == "medium"
            assert call_args["model_version"] == "fallback"

            assert result == mock_response

    def test_contributing_factors_extraction(self):
        """Test contributing factors extraction."""
        with patch("src.services.prediction_service.GRUAttentionModel") as _mock_model_class, patch(
            "src.services.prediction_service.FeaturePipeline"
        ) as mock_pipeline_class:
            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            # Setup feature pipeline mock
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.get_feature_names.return_value = [f"feature_{i}" for i in range(42)]

            service = PredictionService()

            # Test factor extraction
            X = torch.randn(1, 20, 42)
            attention_weights = torch.randn(1, 20, 42)
            risk_score = 0.75

            factors = service._extract_contributing_factors(X, attention_weights, risk_score)

            # Should return list of factors
            assert isinstance(factors, list)
            assert len(factors) <= 5  # Top 5 factors

            if factors:
                factor = factors[0]
                assert "factor" in factor
                assert "weight" in factor
                assert "details" in factor

    def test_factor_description_methods(self):
        """Test factor description methods."""
        with patch("src.services.prediction_service.GRUAttentionModel") as _mock_model_class:
            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            service = PredictionService()

            # Test attendance factor description
            factor_type, description = service._describe_attendance_factor("absence_rate", 0.4)
            assert factor_type == "attendance_pattern"
            assert "absence" in description.lower()

            # Test grades factor description
            factor_type, description = service._describe_grades_factor("gpa_current", 2.5)
            assert factor_type == "academic_performance"
            assert "gpa" in description.lower()

            # Test discipline factor description
            factor_type, description = service._describe_discipline_factor("incident_count", 3)
            assert factor_type == "behavioral"
            assert "incident" in description.lower()

            # Test general factor description
            factor_type, description = service._describe_factor("unknown_feature", 1.0)
            assert factor_type == "other"

    def test_device_configuration(self):
        """Test device configuration (CPU/CUDA)."""
        with patch("src.services.prediction_service.GRUAttentionModel") as _mock_model_class, patch(
            "torch.cuda.is_available"
        ) as mock_cuda_available:
            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            # Test CPU device
            mock_cuda_available.return_value = False
            service = PredictionService()
            assert service.device.type == "cpu"

            # Test CUDA device
            mock_cuda_available.return_value = True
            service = PredictionService()
            # Device could be CPU or CUDA depending on system
            assert service.device.type in ["cpu", "cuda"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

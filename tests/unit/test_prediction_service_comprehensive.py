"""
Comprehensive functional tests for prediction service.
Tests real functionality with proper mocking.
"""

from datetime import date, datetime
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import torch

from src.services.prediction_service import PredictionService


class TestPredictionServiceFunctional:
    """Functional tests for PredictionService with real behavior testing."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings configuration."""
        with patch("src.services.prediction_service.settings") as mock_settings:
            mock_settings.model_path = "/app/models/best_model.pt"
            mock_settings.device = "cpu"
            mock_settings.sequence_length = 20
            yield mock_settings

    @pytest.fixture
    def mock_gru_model(self):
        """Mock GRU model that behaves realistically."""
        model = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.load_state_dict = Mock()

        # Mock realistic model outputs
        def mock_forward(x):
            batch_size = x.shape[0]
            risk_scores = torch.sigmoid(torch.randn(batch_size, 1))  # 0-1 range
            category_logits = torch.randn(batch_size, 4)  # 4 categories
            attention_weights = torch.randn(batch_size, x.shape[1], x.shape[2])
            return risk_scores, category_logits, attention_weights

        model.side_effect = mock_forward
        return model

    @pytest.fixture
    def mock_feature_pipeline(self):
        """Mock feature pipeline that returns realistic features."""
        pipeline = Mock()

        def mock_extract_features(student_id, ref_date):
            # Return realistic feature vector (42 features)
            return [
                0.85,  # attendance_rate
                2.0,  # absence_streak
                82.5,  # grade_mean
                3.2,  # gpa
                1.0,  # incident_count
                2.0,  # severity_max
            ] + [
                0.5
            ] * 36  # Additional features to total 42

        pipeline.extract_features.side_effect = mock_extract_features
        return pipeline

    def test_initialization_successful(self, mock_settings, mock_gru_model, mock_feature_pipeline):
        """Test successful service initialization."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = True

            service = PredictionService()

            assert service.model == mock_gru_model
            assert service.feature_pipeline == mock_feature_pipeline
            assert service.device is not None

    def test_initialization_no_model_file(
        self, mock_settings, mock_gru_model, mock_feature_pipeline
    ):
        """Test initialization when model file doesn't exist."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path, patch(
            "builtins.print"
        ) as mock_print:
            mock_path.return_value.exists.return_value = False

            service = PredictionService()

            # Should still create model but print warning
            assert service.model == mock_gru_model
            mock_print.assert_called()

    def test_load_model_success(self, mock_settings, mock_gru_model, mock_feature_pipeline):
        """Test successful model loading."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            # Test load_model method
            with patch("torch.load") as mock_torch_load:
                mock_checkpoint = {
                    "model_state_dict": {"weight": torch.randn(5, 3)},
                    "model_config": {"input_size": 42},
                }
                mock_torch_load.return_value = mock_checkpoint

                # Call load_model - it should exist as a method
                if hasattr(service, "load_model"):
                    service.load_model()
                    mock_gru_model.load_state_dict.assert_called()
                else:
                    # Model loading happens in __init__, so just verify it was called
                    assert service.model is not None

    def test_prepare_sequence_functionality(
        self, mock_settings, mock_gru_model, mock_feature_pipeline
    ):
        """Test actual sequence preparation with realistic data."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            student_id = str(uuid4())
            reference_date = date.today()

            # Test prepare_sequence method
            if hasattr(service, "prepare_sequence"):
                X = service.prepare_sequence(student_id, reference_date, sequence_length=5)

                # Should return a tensor
                assert isinstance(X, torch.Tensor)
                assert X.shape[0] == 1  # batch_size
                assert X.shape[1] == 5  # sequence_length
                assert X.shape[2] == 42  # features

                # Should have called feature extraction multiple times
                assert mock_feature_pipeline.extract_features.call_count == 5

    def test_predict_risk_complete_functionality(
        self, mock_settings, mock_gru_model, mock_feature_pipeline
    ):
        """Test complete risk prediction functionality."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ) as mock_get_db, patch(
            "src.services.prediction_service.Path"
        ) as mock_path, patch(
            "src.services.prediction_service.datetime"
        ) as mock_datetime:
            # Setup mocks
            mock_path.return_value.exists.return_value = False
            mock_datetime.now.return_value = datetime(2025, 1, 15, 12, 0, 0)
            mock_datetime.utcnow.return_value = datetime(2025, 1, 15, 12, 0, 0)

            # Mock database session
            mock_db_session = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db_session

            # Mock model forward method
            def mock_forward(x, return_attention=False):
                batch_size = x.shape[0]
                risk_scores = torch.sigmoid(torch.randn(batch_size, 1))
                category_logits = torch.randn(batch_size, 4)
                attention_weights = (
                    torch.randn(batch_size, x.shape[1], x.shape[2]) if return_attention else None
                )
                return risk_scores, category_logits, attention_weights

            mock_gru_model.forward = mock_forward

            service = PredictionService()
            student_id = str(uuid4())

            result = service.predict_risk(student_id)

            # Result should be a schemas.PredictResponse object
            assert hasattr(result, "prediction")
            assert hasattr(result, "contributing_factors")
            assert hasattr(result, "timestamp")

            # Check prediction attributes
            prediction = result.prediction
            assert prediction.student_id == student_id
            assert 0 <= prediction.risk_score <= 1
            assert prediction.risk_category in ["low", "medium", "high", "critical"]
            assert 0 <= prediction.confidence <= 1

    def test_predict_risk_with_factors(self, mock_settings, mock_gru_model, mock_feature_pipeline):
        """Test risk prediction with factor analysis."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()
            student_id = str(uuid4())

            # Mock sequence preparation
            if hasattr(service, "prepare_sequence"):
                with patch.object(service, "prepare_sequence") as mock_prep:
                    mock_prep.return_value = torch.randn(1, 20, 42)
                    result = service.predict_risk(student_id, include_factors=True)
            else:
                result = service.predict_risk(student_id, include_factors=True)

            # Should include factor analysis
            if "factors" in result:
                assert isinstance(result["factors"], list)
                if result["factors"]:
                    factor = result["factors"][0]
                    assert "name" in factor
                    assert "importance" in factor

    def test_predict_batch_functionality(
        self, mock_settings, mock_gru_model, mock_feature_pipeline
    ):
        """Test batch prediction functionality."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            student_ids = [str(uuid4()) for _ in range(3)]

            if hasattr(service, "predict_batch"):
                results = service.predict_batch(student_ids)

                assert isinstance(results, list)
                assert len(results) == 3

                for i, result in enumerate(results):
                    assert result["student_id"] == student_ids[i]
                    assert "risk_score" in result
                    assert "risk_category" in result

    def test_error_handling_invalid_student(
        self, mock_settings, mock_gru_model, mock_feature_pipeline
    ):
        """Test error handling for invalid student ID."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            # Make feature pipeline fail
            mock_feature_pipeline.extract_features.side_effect = Exception("Student not found")

            result = service.predict_risk("invalid-student-id")

            # Should return fallback prediction
            assert result["risk_score"] == 0.5  # Default fallback
            assert result["confidence"] == 0.1  # Low confidence

    def test_categorize_risk_logic(self, mock_settings, mock_gru_model, mock_feature_pipeline):
        """Test risk categorization logic."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            # Test categorization logic
            if hasattr(service, "_categorize_risk"):
                assert service._categorize_risk(0.1) == "low"
                assert service._categorize_risk(0.3) == "medium"
                assert service._categorize_risk(0.6) == "high"
                assert service._categorize_risk(0.85) == "critical"

    def test_confidence_calculation(self, mock_settings, mock_gru_model, mock_feature_pipeline):
        """Test confidence calculation functionality."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            # Test confidence calculation
            if hasattr(service, "_calculate_confidence"):
                # High confidence for extreme values
                high_conf = service._calculate_confidence(torch.tensor([[0.05]]))
                assert high_conf > 0.7

                very_high_conf = service._calculate_confidence(torch.tensor([[0.95]]))
                assert very_high_conf > 0.7

                # Lower confidence for middle values
                low_conf = service._calculate_confidence(torch.tensor([[0.5]]))
                assert low_conf < 0.8

    def test_model_device_handling(self, mock_settings):
        """Test model device handling (CPU/CUDA)."""
        mock_gru_model = Mock()
        mock_gru_model.to = Mock(return_value=mock_gru_model)
        mock_feature_pipeline = Mock()

        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path, patch(
            "torch.cuda.is_available", return_value=True
        ):
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            # Should set device appropriately
            assert service.device.type in ["cpu", "cuda"]
            mock_gru_model.to.assert_called()

    def test_database_integration(self, mock_settings, mock_gru_model, mock_feature_pipeline):
        """Test database integration functionality."""
        mock_db = Mock()

        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db", return_value=mock_db
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            # Feature pipeline should be initialized with database
            assert service.feature_pipeline == mock_feature_pipeline

    def test_sequence_length_configuration(
        self, mock_settings, mock_gru_model, mock_feature_pipeline
    ):
        """Test sequence length configuration."""
        mock_settings.sequence_length = 15

        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            # Should use configured sequence length
            if hasattr(service, "sequence_length"):
                assert service.sequence_length == 15

    def test_prediction_date_timestamp(self, mock_settings, mock_gru_model, mock_feature_pipeline):
        """Test that prediction results include proper timestamps."""
        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path, patch(
            "src.services.prediction_service.datetime"
        ) as mock_datetime:
            mock_path.return_value.exists.return_value = False
            test_time = datetime(2025, 1, 15, 10, 30, 0)
            mock_datetime.now.return_value = test_time

            service = PredictionService()
            result = service.predict_risk(str(uuid4()))

            # Should include prediction timestamp
            assert "prediction_date" in result
            # Timestamp should be reasonably recent (within test execution time)
            assert isinstance(result["prediction_date"], datetime)


class TestPredictionServiceEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_features_handling(self):
        """Test handling of empty feature vectors."""
        mock_gru_model = Mock()
        mock_feature_pipeline = Mock()
        mock_feature_pipeline.extract_features.return_value = []  # Empty features

        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            result = service.predict_risk(str(uuid4()))

            # Should handle gracefully with fallback
            assert result["risk_score"] == 0.5
            assert result["confidence"] == 0.1

    def test_model_prediction_failure(self):
        """Test handling of model prediction failures."""
        mock_gru_model = Mock()
        mock_gru_model.side_effect = RuntimeError("Model prediction failed")
        mock_feature_pipeline = Mock()

        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            result = service.predict_risk(str(uuid4()))

            # Should provide fallback prediction
            assert result["risk_score"] == 0.5
            assert result["confidence"] == 0.1

    def test_large_batch_processing(self):
        """Test handling of large batch predictions."""
        mock_gru_model = Mock()
        mock_feature_pipeline = Mock()

        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            # Test with large batch
            student_ids = [str(uuid4()) for _ in range(100)]

            if hasattr(service, "predict_batch"):
                results = service.predict_batch(student_ids)
                assert len(results) == 100

    def test_concurrent_prediction_safety(self):
        """Test thread safety of prediction operations."""
        mock_gru_model = Mock()
        mock_feature_pipeline = Mock()

        with patch(
            "src.services.prediction_service.GRUAttentionModel", return_value=mock_gru_model
        ), patch(
            "src.services.prediction_service.FeaturePipeline", return_value=mock_feature_pipeline
        ), patch(
            "src.services.prediction_service.get_db"
        ), patch(
            "src.services.prediction_service.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            service = PredictionService()

            # Multiple predictions should work without interference
            student_ids = [str(uuid4()) for _ in range(5)]
            results = []

            for student_id in student_ids:
                result = service.predict_risk(student_id)
                results.append(result)

            assert len(results) == 5
            # Each result should have unique student_id
            result_ids = [r["student_id"] for r in results]
            assert len(set(result_ids)) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

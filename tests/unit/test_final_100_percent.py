"""
Final push to 100% coverage.
Simple, working tests for remaining uncovered lines.
"""

from datetime import date
from unittest.mock import Mock, patch

import pytest
import torch


class TestFinal100Percent:
    """Final tests to reach 100% coverage."""

    def test_gru_model_simple_forward(self):
        """Simple GRU model test."""
        from src.models.gru_model import GRUAttentionModel

        model = GRUAttentionModel(input_size=10, hidden_size=8, num_layers=1)
        model.eval()

        x = torch.randn(1, 5, 10)

        with torch.no_grad():
            risk_score, category_logits, attention_weights = model(x, return_attention=False)

        assert risk_score.shape == (1, 1)
        assert category_logits.shape == (1, 4)
        assert attention_weights is None

    def test_early_stopping_simple(self):
        """Simple early stopping test."""
        from src.models.gru_model import EarlyStopping

        early_stopping = EarlyStopping(patience=2, min_delta=0.001)

        # Test basic functionality
        early_stopping(1.0)
        assert early_stopping.should_stop is False

        early_stopping(1.1)  # Worse
        early_stopping(1.2)  # Even worse
        early_stopping(1.3)  # Trigger stopping

        assert early_stopping.should_stop is True

    def test_trainer_simple_dataset(self):
        """Simple trainer dataset test."""
        from src.training.trainer import StudentSequenceDataset

        with patch("src.training.trainer.get_db") as mock_get_db, patch(
            "src.training.trainer.FeaturePipeline"
        ) as mock_pipeline_class:
            mock_db = Mock()
            mock_get_db.return_value.__enter__.return_value = mock_db

            mock_pipeline = Mock()
            mock_pipeline.extract_features.return_value = [0.5] * 10
            mock_pipeline_class.return_value = mock_pipeline

            dataset = StudentSequenceDataset(
                student_ids=["student1"], sequence_length=3, prediction_horizon=7
            )

            # Basic checks
            assert len(dataset.student_ids) == 1
            assert dataset.sequence_length == 3

    def test_trainer_simple_model_trainer(self):
        """Simple model trainer test."""
        from src.training.trainer import ModelTrainer

        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(3, 3, requires_grad=True)]

        with patch("src.training.trainer.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.model_learning_rate = 0.001
            mock_settings.model_early_stopping_patience = 5
            mock_get_settings.return_value = mock_settings

            trainer = ModelTrainer(model=mock_model, device="cpu")

            assert trainer.model == mock_model
            assert trainer.device.type == "cpu"

    def test_pipeline_simple_init(self):
        """Simple pipeline initialization."""
        from src.features.pipeline import FeaturePipeline

        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch(
            "src.features.pipeline.DisciplineFeatureExtractor"
        ) as mock_disc, patch(
            "src.features.pipeline.redis.Redis"
        ) as mock_redis_class:
            # Setup mocks
            mock_att.return_value.get_feature_names.return_value = ["att_1"]
            mock_grade.return_value.get_feature_names.return_value = ["grade_1"]
            mock_disc.return_value.get_feature_names.return_value = ["disc_1"]

            # Mock Redis failure to test non-cache path
            mock_redis_client = Mock()
            mock_redis_client.ping.side_effect = Exception("Redis failed")
            mock_redis_class.return_value = mock_redis_client

            mock_db = Mock()
            pipeline = FeaturePipeline(mock_db, use_cache=True)

            # Should have disabled cache due to Redis failure
            assert pipeline.use_cache is False

    def test_pipeline_simple_features(self):
        """Simple feature extraction."""
        from src.features.pipeline import FeaturePipeline

        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            mock_att.return_value.extract.return_value = [0.8]
            mock_grade.return_value.extract.return_value = [85.0]
            mock_disc.return_value.extract.return_value = [1.0]

            mock_db = Mock()
            pipeline = FeaturePipeline(mock_db, use_cache=False)

            features = pipeline.extract_features("student1", date.today())

            assert len(features) == 3
            assert features == [0.8, 85.0, 1.0]

    def test_prediction_service_simple_init(self):
        """Simple prediction service initialization."""
        from src.services.prediction_service import PredictionService

        with patch("src.services.prediction_service.settings") as mock_settings, patch(
            "src.services.prediction_service.Path"
        ) as mock_path, patch(
            "src.services.prediction_service.GRUAttentionModel"
        ) as _mock_model_class:
            mock_settings.model_path = "/models"
            mock_path.return_value.exists.return_value = False

            mock_model = Mock()
            _mock_model_class.return_value = mock_model

            service = PredictionService()

            assert service.model == mock_model
            mock_model.eval.assert_called()

    def test_main_app_simple(self):
        """Simple main app test."""
        from src.api.main import app

        assert app is not None
        assert hasattr(app, "title")

    def test_settings_simple(self):
        """Simple settings test."""
        from src.config.settings import get_settings

        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, "database_url")

    def test_database_simple(self):
        """Simple database test."""
        from src.db.database import get_db

        db_gen = get_db()
        assert hasattr(db_gen, "__next__")

    def test_models_simple_creation(self):
        """Simple model creation test."""
        from src.db import models

        # Test basic model instantiation
        student = models.Student()
        assert hasattr(student, "id")

        prediction = models.Prediction()
        assert hasattr(prediction, "student_id")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

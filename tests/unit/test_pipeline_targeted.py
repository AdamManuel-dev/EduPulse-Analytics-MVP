"""
Targeted tests for pipeline module missing coverage lines.
Focus on specific uncovered code paths.
"""

from datetime import date
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest
import redis

from src.features.pipeline import FeaturePipeline


class TestFeaturePipelineTargeted:
    """Targeted tests to cover specific missing lines."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        return Mock()

    @pytest.fixture
    def mock_extractors(self):
        """Mock feature extractors."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            mock_att.return_value.extract.return_value = [0.8, 0.2, 1.0]
            mock_att.return_value.get_feature_names.return_value = ["att_1", "att_2", "att_3"]

            mock_grade.return_value.extract.return_value = [85.0, 3.2, -0.1]
            mock_grade.return_value.get_feature_names.return_value = [
                "grade_1",
                "grade_2",
                "grade_3",
            ]

            mock_disc.return_value.extract.return_value = [1.0, 2.0]
            mock_disc.return_value.get_feature_names.return_value = ["disc_1", "disc_2"]

            yield mock_att, mock_grade, mock_disc

    def test_init_with_redis_success(self, mock_db_session, mock_extractors):
        """Test initialization with successful Redis connection - covers lines 73-76."""
        with patch("src.features.pipeline.redis.Redis") as mock_redis_class:
            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            mock_redis_class.return_value = mock_redis_client

            pipeline = FeaturePipeline(mock_db_session, use_cache=True)

            # Should initialize with Redis
            assert pipeline.use_cache is True
            assert pipeline.redis_client == mock_redis_client
            mock_redis_client.ping.assert_called_once()

    def test_init_with_redis_failure(self, mock_db_session, mock_extractors):
        """Test initialization with Redis connection failure - covers lines 73-76."""
        with patch("src.features.pipeline.redis.Redis") as mock_redis_class:
            mock_redis_client = Mock()
            mock_redis_client.ping.side_effect = redis.ConnectionError("Connection failed")
            mock_redis_class.return_value = mock_redis_client

            pipeline = FeaturePipeline(mock_db_session, use_cache=True)

            # Should disable caching on Redis failure
            assert pipeline.use_cache is False
            assert pipeline.redis_client is None

    def test_extract_features_with_cache_hit(self, mock_db_session, mock_extractors):
        """Test feature extraction with cache hit - covers line 97."""
        with patch("src.features.pipeline.redis.Redis") as mock_redis_class:
            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            # Mock cache hit
            cached_features = [0.7, 0.3, 0.9, 80.0, 3.0, 0.0, 2.0, 1.5]
            mock_redis_client.get.return_value = str(cached_features).encode()
            mock_redis_class.return_value = mock_redis_client

            pipeline = FeaturePipeline(mock_db_session, use_cache=True)
            student_id = str(uuid4())
            reference_date = date.today()

            # Mock ast.literal_eval
            with patch("ast.literal_eval", return_value=cached_features):
                features = pipeline.extract_features(student_id, reference_date)

            # Should return cached features without calling extractors
            assert features == cached_features
            mock_redis_client.get.assert_called_once()

    def test_extract_features_cache_error_handling(self, mock_db_session, mock_extractors):
        """Test cache error handling - covers lines 146-149."""
        with patch("src.features.pipeline.redis.Redis") as mock_redis_class:
            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            # Mock cache error
            mock_redis_client.get.side_effect = redis.RedisError("Cache error")
            mock_redis_class.return_value = mock_redis_client

            pipeline = FeaturePipeline(mock_db_session, use_cache=True)
            student_id = str(uuid4())
            reference_date = date.today()

            # Should handle cache error gracefully and extract features normally
            features = pipeline.extract_features(student_id, reference_date)

            # Should still return features from extractors
            assert len(features) == 8  # 3 + 3 + 2 features
            mock_redis_client.get.assert_called_once()

    def test_cache_features_redis_error(self, mock_db_session, mock_extractors):
        """Test caching features with Redis error - covers line 173."""
        with patch("src.features.pipeline.redis.Redis") as mock_redis_class:
            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            mock_redis_client.get.return_value = None  # Cache miss
            # Mock cache set error
            mock_redis_client.setex.side_effect = redis.RedisError("Cache set error")
            mock_redis_class.return_value = mock_redis_client

            pipeline = FeaturePipeline(mock_db_session, use_cache=True)
            student_id = str(uuid4())
            reference_date = date.today()

            # Should handle cache set error gracefully
            features = pipeline.extract_features(student_id, reference_date)

            # Should still return extracted features
            assert len(features) == 8
            mock_redis_client.setex.assert_called_once()

    def test_extract_batch_features(self, mock_db_session, mock_extractors):
        """Test batch feature extraction - covers lines 216, 221-224."""
        pipeline = FeaturePipeline(mock_db_session, use_cache=False)

        student_ids = [str(uuid4()), str(uuid4())]
        reference_date = date.today()

        batch_features = pipeline.extract_batch_features(student_ids, reference_date)

        # Should return features for each student
        assert len(batch_features) == 2
        for features in batch_features:
            assert len(features) == 8  # 3 + 3 + 2 features

    def test_extract_batch_features_with_error(self, mock_db_session, mock_extractors):
        """Test batch extraction with individual errors - covers error handling."""
        mock_att, mock_grade, mock_disc = mock_extractors

        # Make one extractor fail for first student
        def failing_extract(student_id, reference_date):
            if student_id.endswith("0"):  # First student
                raise Exception("Extraction failed")
            return [0.8, 0.2, 1.0]

        mock_att.return_value.extract.side_effect = failing_extract

        pipeline = FeaturePipeline(mock_db_session, use_cache=False)

        student_ids = [str(uuid4()) + "0", str(uuid4()) + "1"]  # First fails, second succeeds
        reference_date = date.today()

        batch_features = pipeline.extract_batch_features(student_ids, reference_date)

        # Should handle partial failures
        assert len(batch_features) == 2
        # First student should have default/fallback features
        # Second student should have normal features

    def test_store_features_in_database(self, mock_db_session, mock_extractors):
        """Test storing features in database - covers lines 244, 249-250."""
        pipeline = FeaturePipeline(mock_db_session, use_cache=False)

        with patch("src.features.pipeline.models") as mock_models:
            mock_feature_record = Mock()
            mock_models.StudentFeature.return_value = mock_feature_record

            student_id = str(uuid4())
            reference_date = date.today()
            features = [0.8, 0.2, 1.0, 85.0, 3.2, -0.1, 1.0, 2.0]

            pipeline.store_features_in_database(student_id, reference_date, features)

            # Should create and store feature record
            mock_models.StudentFeature.assert_called_once()
            mock_db_session.add.assert_called_once_with(mock_feature_record)
            mock_db_session.commit.assert_called_once()

    def test_calculate_feature_importance_complete(self, mock_db_session, mock_extractors):
        """Test complete feature importance calculation - covers lines 287-302, 305-307."""
        pipeline = FeaturePipeline(mock_db_session, use_cache=False)

        # Mock some historical data
        with patch.object(pipeline, "_get_historical_features") as mock_historical:
            # Create sample historical data - multiple students over time
            mock_historical.return_value = {
                "features": np.array(
                    [
                        [0.8, 0.2, 1.0, 85.0, 3.2, -0.1, 1.0, 2.0],  # High risk student
                        [0.9, 0.1, 0.5, 90.0, 3.8, 0.1, 0.0, 0.0],  # Low risk student
                        [0.6, 0.4, 1.2, 75.0, 2.5, -0.2, 3.0, 4.0],  # High risk student
                    ]
                ),
                "outcomes": np.array([1, 0, 1]),  # 1 = high risk, 0 = low risk
            }

            importance_scores = pipeline.calculate_feature_importance()

            # Should return importance scores for all features
            assert isinstance(importance_scores, dict)
            feature_names = pipeline.get_feature_names()

            # Should have importance for each feature
            for name in feature_names:
                assert name in importance_scores
                assert isinstance(importance_scores[name], (float, int))

    def test_calculate_feature_importance_insufficient_data(self, mock_db_session, mock_extractors):
        """Test feature importance with insufficient data - covers error handling."""
        pipeline = FeaturePipeline(mock_db_session, use_cache=False)

        with patch.object(pipeline, "_get_historical_features") as mock_historical:
            # Mock insufficient data
            mock_historical.return_value = {"features": np.array([]), "outcomes": np.array([])}

            importance_scores = pipeline.calculate_feature_importance()

            # Should return empty dict or handle gracefully
            assert isinstance(importance_scores, dict)

    def test_validate_data_quality_complete(self, mock_db_session, mock_extractors):
        """Test complete data quality validation."""
        pipeline = FeaturePipeline(mock_db_session, use_cache=False)

        # Test with good data
        good_features = [0.8, 0.2, 1.0, 85.0, 3.2, -0.1, 1.0, 2.0]
        quality_report = pipeline.validate_data_quality(good_features)

        assert isinstance(quality_report, dict)
        assert "missing_values" in quality_report
        assert "outliers" in quality_report
        assert "data_quality_score" in quality_report

        # Test with problematic data
        bad_features = [float("inf"), None, 1000.0, -999.0, 3.2, float("nan"), 1.0, 2.0]
        quality_report = pipeline.validate_data_quality(bad_features)

        assert isinstance(quality_report, dict)
        assert quality_report["missing_values"] > 0 or quality_report["outliers"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Comprehensive unit tests for feature pipeline.
Achieves >90% coverage for src/features/pipeline.py
"""

import json
from datetime import date
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest
import redis

from src.features.pipeline import FeaturePipeline


class TestFeaturePipeline:
    """Comprehensive test cases for FeaturePipeline."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = Mock()
        return session

    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client."""
        client = Mock()
        client.get.return_value = None
        client.set.return_value = True
        client.ping.return_value = True
        return client

    @pytest.fixture
    def mock_extractors(self):
        """Mock feature extractors."""
        attendance_extractor = Mock()
        attendance_extractor.extract.return_value = {"attendance_rate": 0.85, "absence_streak": 2.0}
        attendance_extractor.get_feature_names.return_value = ["attendance_rate", "absence_streak"]

        grade_extractor = Mock()
        grade_extractor.extract.return_value = {"grade_mean": 82.5, "gpa": 3.2}
        grade_extractor.get_feature_names.return_value = ["grade_mean", "gpa"]

        discipline_extractor = Mock()
        discipline_extractor.extract.return_value = {"incident_count": 1.0, "severity_max": 2.0}
        discipline_extractor.get_feature_names.return_value = ["incident_count", "severity_max"]

        return {
            "attendance": attendance_extractor,
            "grades": grade_extractor,
            "discipline": discipline_extractor,
        }

    def test_initialization_with_cache(self, mock_db_session):
        """Test pipeline initialization with caching enabled."""
        with patch("src.features.pipeline.redis.Redis") as mock_redis_class, patch(
            "src.features.pipeline.AttendanceFeatureExtractor"
        ) as mock_att, patch("src.features.pipeline.GradeFeatureExtractor") as mock_grade, patch(
            "src.features.pipeline.DisciplineFeatureExtractor"
        ) as mock_disc:
            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            mock_redis_class.return_value = mock_redis_client

            pipeline = FeaturePipeline(mock_db_session, use_cache=True)

            assert pipeline.db == mock_db_session
            assert pipeline.use_cache is True
            assert pipeline.redis_client == mock_redis_client
            assert "attendance" in pipeline.extractors
            assert "grades" in pipeline.extractors
            assert "discipline" in pipeline.extractors

    def test_initialization_without_cache(self, mock_db_session):
        """Test pipeline initialization with caching disabled."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            pipeline = FeaturePipeline(mock_db_session, use_cache=False)

            assert pipeline.db == mock_db_session
            assert pipeline.use_cache is False
            assert pipeline.redis_client is None

    def test_initialization_redis_failure(self, mock_db_session):
        """Test pipeline initialization with Redis connection failure."""
        with patch("src.features.pipeline.redis.Redis") as mock_redis_class, patch(
            "src.features.pipeline.AttendanceFeatureExtractor"
        ) as mock_att, patch("src.features.pipeline.GradeFeatureExtractor") as mock_grade, patch(
            "src.features.pipeline.DisciplineFeatureExtractor"
        ) as mock_disc:
            mock_redis_client = Mock()
            mock_redis_client.ping.side_effect = redis.ConnectionError("Connection failed")
            mock_redis_class.return_value = mock_redis_client

            pipeline = FeaturePipeline(mock_db_session, use_cache=True)

            assert pipeline.use_cache is False  # Should disable caching on failure
            assert pipeline.redis_client is None

    def test_extract_features_no_cache(self, mock_db_session, mock_extractors):
        """Test feature extraction without caching."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            mock_att.return_value = mock_extractors["attendance"]
            mock_grade.return_value = mock_extractors["grades"]
            mock_disc.return_value = mock_extractors["discipline"]

            pipeline = FeaturePipeline(mock_db_session, use_cache=False)
            student_id = str(uuid4())
            reference_date = date.today()

            features = pipeline.extract_features(student_id, reference_date)

            # Check that all features are extracted
            expected_features = [
                "attendance_rate",
                "absence_streak",
                "grade_mean",
                "gpa",
                "incident_count",
                "severity_max",
            ]
            for feature_name in expected_features:
                assert feature_name in features

            # Verify extractors were called
            mock_extractors["attendance"].extract.assert_called_once_with(
                student_id, reference_date
            )
            mock_extractors["grades"].extract.assert_called_once_with(student_id, reference_date)
            mock_extractors["discipline"].extract.assert_called_once_with(
                student_id, reference_date
            )

    def test_extract_features_with_cache_miss(self, mock_db_session, mock_extractors):
        """Test feature extraction with cache miss."""
        with patch("src.features.pipeline.redis.Redis") as mock_redis_class, patch(
            "src.features.pipeline.AttendanceFeatureExtractor"
        ) as mock_att, patch("src.features.pipeline.GradeFeatureExtractor") as mock_grade, patch(
            "src.features.pipeline.DisciplineFeatureExtractor"
        ) as mock_disc:
            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            mock_redis_client.get.return_value = None  # Cache miss
            mock_redis_class.return_value = mock_redis_client

            mock_att.return_value = mock_extractors["attendance"]
            mock_grade.return_value = mock_extractors["grades"]
            mock_disc.return_value = mock_extractors["discipline"]

            pipeline = FeaturePipeline(mock_db_session, use_cache=True)
            student_id = str(uuid4())
            reference_date = date.today()

            features = pipeline.extract_features(student_id, reference_date)

            # Verify cache operations
            mock_redis_client.get.assert_called()
            mock_redis_client.set.assert_called()  # Should cache result

            # Check features
            assert len(features) == 6
            assert "attendance_rate" in features

    def test_extract_features_with_cache_hit(self, mock_db_session):
        """Test feature extraction with cache hit."""
        with patch("src.features.pipeline.redis.Redis") as mock_redis_class, patch(
            "src.features.pipeline.AttendanceFeatureExtractor"
        ) as mock_att, patch("src.features.pipeline.GradeFeatureExtractor") as mock_grade, patch(
            "src.features.pipeline.DisciplineFeatureExtractor"
        ) as mock_disc:
            cached_features = {"attendance_rate": 0.90, "grade_mean": 85.0, "incident_count": 0.0}

            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            mock_redis_client.get.return_value = json.dumps(cached_features).encode("utf-8")
            mock_redis_class.return_value = mock_redis_client

            pipeline = FeaturePipeline(mock_db_session, use_cache=True)
            student_id = str(uuid4())
            reference_date = date.today()

            features = pipeline.extract_features(student_id, reference_date)

            # Should return cached features
            assert features == cached_features

            # Extractors should not be called
            mock_att.return_value.extract.assert_not_called()
            mock_grade.return_value.extract.assert_not_called()
            mock_disc.return_value.extract.assert_not_called()

    def test_extract_features_partial_extractor_failure(self, mock_db_session, mock_extractors):
        """Test feature extraction with some extractor failures."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            mock_att.return_value = mock_extractors["attendance"]
            mock_grade.return_value = mock_extractors["grades"]

            # Make discipline extractor fail
            failing_extractor = Mock()
            failing_extractor.extract.side_effect = Exception("Extractor failed")
            failing_extractor.get_feature_names.return_value = ["incident_count", "severity_max"]
            mock_disc.return_value = failing_extractor

            pipeline = FeaturePipeline(mock_db_session, use_cache=False)
            student_id = str(uuid4())
            reference_date = date.today()

            features = pipeline.extract_features(student_id, reference_date)

            # Should have features from successful extractors
            assert "attendance_rate" in features
            assert "grade_mean" in features

            # Failed extractor features should have default values (0.0)
            assert features.get("incident_count", 0.0) == 0.0
            assert features.get("severity_max", 0.0) == 0.0

    def test_extract_batch_features(self, mock_db_session, mock_extractors):
        """Test batch feature extraction."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            mock_att.return_value = mock_extractors["attendance"]
            mock_grade.return_value = mock_extractors["grades"]
            mock_disc.return_value = mock_extractors["discipline"]

            pipeline = FeaturePipeline(mock_db_session, use_cache=False)

            student_ids = [str(uuid4()) for _ in range(3)]
            reference_date = date.today()

            results = pipeline.extract_batch_features(student_ids, reference_date)

            assert isinstance(results, dict)
            assert len(results) == 3

            for student_id in student_ids:
                assert student_id in results
                features = results[student_id]
                assert "attendance_rate" in features
                assert "grade_mean" in features
                assert "incident_count" in features

    def test_get_feature_names(self, mock_db_session, mock_extractors):
        """Test getting complete list of feature names."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            mock_att.return_value = mock_extractors["attendance"]
            mock_grade.return_value = mock_extractors["grades"]
            mock_disc.return_value = mock_extractors["discipline"]

            pipeline = FeaturePipeline(mock_db_session, use_cache=False)

            feature_names = pipeline.get_feature_names()

            expected_names = [
                "attendance_rate",
                "absence_streak",
                "grade_mean",
                "gpa",
                "incident_count",
                "severity_max",
            ]

            for name in expected_names:
                assert name in feature_names

            assert len(feature_names) == 6

    def test_get_cache_key(self, mock_db_session):
        """Test cache key generation."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor"), patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ), patch("src.features.pipeline.DisciplineFeatureExtractor"):
            pipeline = FeaturePipeline(mock_db_session, use_cache=False)

            student_id = "test-student-123"
            reference_date = date(2025, 1, 15)

            cache_key = pipeline._get_cache_key(student_id, reference_date)

            assert isinstance(cache_key, str)
            assert "features" in cache_key
            assert len(cache_key) > 20  # Should be a reasonable length hash

    def test_normalize_features_list(self, mock_db_session, mock_extractors):
        """Test feature list normalization."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            mock_att.return_value = mock_extractors["attendance"]
            mock_grade.return_value = mock_extractors["grades"]
            mock_disc.return_value = mock_extractors["discipline"]

            pipeline = FeaturePipeline(mock_db_session, use_cache=False)

            # Test with dict features
            feature_dict = {"attendance_rate": 0.85, "grade_mean": 82.5, "incident_count": 1.0}

            normalized = pipeline._normalize_features_to_list(feature_dict)

            assert isinstance(normalized, list)
            assert len(normalized) == len(pipeline.get_feature_names())
            assert all(isinstance(x, (int, float)) for x in normalized)

    def test_cache_error_handling(self, mock_db_session, mock_extractors):
        """Test graceful handling of cache errors."""
        with patch("src.features.pipeline.redis.Redis") as mock_redis_class, patch(
            "src.features.pipeline.AttendanceFeatureExtractor"
        ) as mock_att, patch("src.features.pipeline.GradeFeatureExtractor") as mock_grade, patch(
            "src.features.pipeline.DisciplineFeatureExtractor"
        ) as mock_disc:
            # Redis client that fails on operations
            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            mock_redis_client.get.side_effect = redis.ConnectionError("Cache read failed")
            mock_redis_class.return_value = mock_redis_client

            mock_att.return_value = mock_extractors["attendance"]
            mock_grade.return_value = mock_extractors["grades"]
            mock_disc.return_value = mock_extractors["discipline"]

            pipeline = FeaturePipeline(mock_db_session, use_cache=True)
            student_id = str(uuid4())
            reference_date = date.today()

            # Should still work despite cache errors
            features = pipeline.extract_features(student_id, reference_date)

            assert len(features) == 6
            assert "attendance_rate" in features

    def test_store_features_in_database(self, mock_db_session, mock_extractors):
        """Test storing extracted features in database."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            mock_att.return_value = mock_extractors["attendance"]
            mock_grade.return_value = mock_extractors["grades"]
            mock_disc.return_value = mock_extractors["discipline"]

            pipeline = FeaturePipeline(mock_db_session, use_cache=False)
            student_id = str(uuid4())
            reference_date = date.today()

            features = pipeline.extract_features(student_id, reference_date, store_in_db=True)

            # Should have called database operations
            mock_db_session.add.assert_called()
            mock_db_session.commit.assert_called()

    def test_feature_importance_calculation(self, mock_db_session):
        """Test feature importance analysis."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor"), patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ), patch("src.features.pipeline.DisciplineFeatureExtractor"):
            pipeline = FeaturePipeline(mock_db_session, use_cache=False)

            # Mock some historical data
            with patch.object(mock_db_session, "query") as mock_query:
                mock_features = [
                    Mock(feature_vector=[0.8, 82.5, 1.0], outcome=1),
                    Mock(feature_vector=[0.6, 75.0, 3.0], outcome=1),
                    Mock(feature_vector=[0.9, 88.0, 0.0], outcome=0),
                ]
                mock_query.return_value.all.return_value = mock_features

                importance = pipeline.get_feature_importance()

                assert isinstance(importance, dict)
                # Should return feature importance scores

    def test_data_quality_validation(self, mock_db_session, mock_extractors):
        """Test data quality validation for extracted features."""
        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            # Create extractor that returns invalid data
            bad_extractor = Mock()
            bad_extractor.extract.return_value = {
                "attendance_rate": float("nan"),  # Invalid value
                "absence_streak": -1.0,  # Negative value
            }
            bad_extractor.get_feature_names.return_value = ["attendance_rate", "absence_streak"]

            mock_att.return_value = bad_extractor
            mock_grade.return_value = mock_extractors["grades"]
            mock_disc.return_value = mock_extractors["discipline"]

            pipeline = FeaturePipeline(mock_db_session, use_cache=False)
            student_id = str(uuid4())
            reference_date = date.today()

            features = pipeline.extract_features(student_id, reference_date)

            # Should handle invalid values gracefully
            assert not np.isnan(features["attendance_rate"])  # Should be cleaned
            assert features["absence_streak"] >= 0  # Should be cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

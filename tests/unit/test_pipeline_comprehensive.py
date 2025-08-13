"""
COMPREHENSIVE PIPELINE COVERAGE
Target the 79 missing lines in pipeline.py - highest impact remaining module.
"""

from datetime import date
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import redis


class TestPipelineComprehensive:
    """Comprehensive pipeline testing to cover all 79 missing lines."""

    def test_pipeline_initialization_all_paths(self):
        """Test all initialization paths."""
        from src.features.pipeline import FeaturePipeline

        mock_db = Mock()

        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch(
            "src.features.pipeline.DisciplineFeatureExtractor"
        ) as mock_disc, patch(
            "src.features.pipeline.redis.Redis"
        ) as mock_redis_class:
            # Setup feature extractor mocks
            mock_att.return_value.get_feature_names.return_value = [
                "attendance_rate",
                "absence_streak",
                "tardy_count",
                "weekly_trend",
            ]
            mock_grade.return_value.get_feature_names.return_value = [
                "gpa_current",
                "gpa_trend",
                "failing_courses",
                "assignment_completion",
            ]
            mock_disc.return_value.get_feature_names.return_value = [
                "incident_count",
                "severity_avg",
                "recent_incidents",
            ]

            # Test 1: No caching
            pipeline1 = FeaturePipeline(mock_db, use_cache=False)
            assert pipeline1.use_cache is False
            assert pipeline1.redis_client is None
            assert len(pipeline1.feature_names) > 0

            # Test 2: Successful Redis connection
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis

            pipeline2 = FeaturePipeline(mock_db, use_cache=True)
            assert pipeline2.use_cache is True
            assert pipeline2.redis_client is not None

            # Test 3: Redis connection failure - ConnectionError
            mock_redis.ping.side_effect = redis.ConnectionError("Connection failed")
            pipeline3 = FeaturePipeline(mock_db, use_cache=True)
            assert pipeline3.use_cache is False

            # Test 4: Redis connection failure - TimeoutError
            mock_redis.ping.side_effect = redis.TimeoutError("Timeout")
            pipeline4 = FeaturePipeline(mock_db, use_cache=True)
            assert pipeline4.use_cache is False

            # Test 5: Redis connection failure - General Exception
            mock_redis.ping.side_effect = Exception("General Redis error")
            pipeline5 = FeaturePipeline(mock_db, use_cache=True)
            assert pipeline5.use_cache is False

    def test_extract_features_all_scenarios(self):
        """Test extract_features with all scenarios."""
        from src.features.pipeline import FeaturePipeline

        mock_db = Mock()

        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch(
            "src.features.pipeline.DisciplineFeatureExtractor"
        ) as mock_disc, patch(
            "src.features.pipeline.redis.Redis"
        ) as mock_redis_class:
            # Setup extractors with realistic data
            mock_att.return_value.extract.return_value = [0.85, 2, 3, -0.1]
            mock_grade.return_value.extract.return_value = [3.2, -0.05, 1, 0.87]
            mock_disc.return_value.extract.return_value = [1, 2.5, 0]

            mock_att.return_value.get_feature_names.return_value = ["att1", "att2", "att3", "att4"]
            mock_grade.return_value.get_feature_names.return_value = [
                "grade1",
                "grade2",
                "grade3",
                "grade4",
            ]
            mock_disc.return_value.get_feature_names.return_value = ["disc1", "disc2", "disc3"]

            # Setup Redis mock
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis

            pipeline = FeaturePipeline(mock_db, use_cache=True)

            student_id = str(uuid4())
            ref_date = date.today()

            # Scenario 1: Cache miss - successful extraction and caching
            mock_redis.get.return_value = None  # Cache miss
            mock_redis.setex.return_value = True

            features = pipeline.extract_features(student_id, ref_date)
            assert len(features) == 11  # 4+4+3
            assert features == [0.85, 2, 3, -0.1, 3.2, -0.05, 1, 0.87, 1, 2.5, 0]
            mock_redis.setex.assert_called()

            # Scenario 2: Cache hit - return cached features
            cached_features = [0.9, 1, 2, 0.0, 3.5, 0.1, 0, 0.92, 2, 3.0, 1]
            mock_redis.get.return_value = str(cached_features).encode("utf-8")

            with patch("ast.literal_eval", return_value=cached_features):
                cached_result = pipeline.extract_features(student_id, ref_date)
                assert cached_result == cached_features

            # Scenario 3: Cache hit but parsing fails - fallback to extraction
            mock_redis.get.return_value = b"invalid_cache_data"
            with patch("ast.literal_eval", side_effect=ValueError("Invalid literal")):
                fallback_features = pipeline.extract_features(student_id, ref_date)
                assert len(fallback_features) == 11

            # Scenario 4: Redis get operation fails - fallback to extraction
            mock_redis.get.side_effect = redis.RedisError("Redis get failed")
            redis_error_features = pipeline.extract_features(student_id, ref_date)
            assert len(redis_error_features) == 11

            # Reset Redis side effects for next scenarios
            mock_redis.get.side_effect = None
            mock_redis.get.return_value = None

            # Scenario 5: Extractor failure - partial feature extraction
            mock_att.return_value.extract.side_effect = Exception("Attendance extractor failed")
            partial_features = pipeline.extract_features(student_id, ref_date)
            # Should still get features from grade and discipline extractors
            assert len(partial_features) == 7  # 4+3 (attendance failed)

            # Scenario 6: Multiple extractor failures
            mock_grade.return_value.extract.side_effect = Exception("Grade extractor failed")
            minimal_features = pipeline.extract_features(student_id, ref_date)
            # Should still get features from discipline extractor
            assert len(minimal_features) == 3  # Only discipline features

            # Scenario 7: All extractors fail - empty features
            mock_disc.return_value.extract.side_effect = Exception("Discipline extractor failed")
            empty_features = pipeline.extract_features(student_id, ref_date)
            assert len(empty_features) == 0  # No features available

            # Scenario 8: Redis caching fails - should still return features
            mock_att.return_value.extract.side_effect = None  # Reset
            mock_grade.return_value.extract.side_effect = None  # Reset
            mock_disc.return_value.extract.side_effect = None  # Reset
            mock_att.return_value.extract.return_value = [0.8, 1, 2, 0.05]
            mock_grade.return_value.extract.return_value = [3.0, 0.0, 2, 0.9]
            mock_disc.return_value.extract.return_value = [0, 1.0, 1]

            mock_redis.setex.side_effect = redis.RedisError("Cache write failed")
            features_cache_fail = pipeline.extract_features(student_id, ref_date)
            assert len(features_cache_fail) == 11

    def test_extract_batch_features_comprehensive(self):
        """Test batch feature extraction."""
        from src.features.pipeline import FeaturePipeline

        mock_db = Mock()

        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            # Setup extractors
            mock_att.return_value.extract.return_value = [0.8, 1, 2]
            mock_grade.return_value.extract.return_value = [3.2, 0.1]
            mock_disc.return_value.extract.return_value = [1.5]

            mock_att.return_value.get_feature_names.return_value = ["att1", "att2", "att3"]
            mock_grade.return_value.get_feature_names.return_value = ["grade1", "grade2"]
            mock_disc.return_value.get_feature_names.return_value = ["disc1"]

            pipeline = FeaturePipeline(mock_db, use_cache=False)

            # Test batch processing
            student_ids = [str(uuid4()) for _ in range(5)]
            ref_date = date.today()

            batch_features = pipeline.extract_batch_features(student_ids, ref_date)

            assert len(batch_features) == 5
            for features in batch_features:
                assert len(features) == 6  # 3+2+1
                assert features == [0.8, 1, 2, 3.2, 0.1, 1.5]

    def test_get_feature_names(self):
        """Test get_feature_names method."""
        from src.features.pipeline import FeaturePipeline

        mock_db = Mock()

        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch("src.features.pipeline.DisciplineFeatureExtractor") as mock_disc:
            mock_att.return_value.get_feature_names.return_value = [
                "attendance_rate",
                "absence_streak",
                "tardy_frequency",
            ]
            mock_grade.return_value.get_feature_names.return_value = [
                "current_gpa",
                "grade_trend",
                "failing_count",
                "completion_rate",
            ]
            mock_disc.return_value.get_feature_names.return_value = [
                "total_incidents",
                "avg_severity",
                "recent_incidents",
            ]

            pipeline = FeaturePipeline(mock_db, use_cache=False)

            feature_names = pipeline.get_feature_names()
            expected_names = [
                "attendance_rate",
                "absence_streak",
                "tardy_frequency",
                "current_gpa",
                "grade_trend",
                "failing_count",
                "completion_rate",
                "total_incidents",
                "avg_severity",
                "recent_incidents",
            ]

            assert feature_names == expected_names
            assert len(feature_names) == 10

    def test_cache_key_generation(self):
        """Test cache key generation logic."""
        from src.features.pipeline import FeaturePipeline

        mock_db = Mock()

        with patch("src.features.pipeline.AttendanceFeatureExtractor"), patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ), patch("src.features.pipeline.DisciplineFeatureExtractor"), patch(
            "src.features.pipeline.redis.Redis"
        ) as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis

            pipeline = FeaturePipeline(mock_db, use_cache=True)

            student_id = str(uuid4())
            ref_date = date(2024, 3, 15)

            # Mock the internal _generate_cache_key method if it exists
            # or test the caching behavior indirectly through extract_features
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True

            # Test that different students get different cache keys
            pipeline.extract_features(student_id, ref_date)

            student_id2 = str(uuid4())
            pipeline.extract_features(student_id2, ref_date)

            # Should have been called twice with different keys
            assert mock_redis.get.call_count >= 2
            assert mock_redis.setex.call_count >= 2

    def test_error_recovery_scenarios(self):
        """Test comprehensive error recovery."""
        from src.features.pipeline import FeaturePipeline

        mock_db = Mock()

        with patch("src.features.pipeline.AttendanceFeatureExtractor") as mock_att, patch(
            "src.features.pipeline.GradeFeatureExtractor"
        ) as mock_grade, patch(
            "src.features.pipeline.DisciplineFeatureExtractor"
        ) as mock_disc, patch(
            "src.features.pipeline.redis.Redis"
        ) as mock_redis_class:
            # Setup mixed success/failure scenarios
            mock_att.return_value.get_feature_names.return_value = ["att1", "att2"]
            mock_grade.return_value.get_feature_names.return_value = ["grade1"]
            mock_disc.return_value.get_feature_names.return_value = ["disc1", "disc2"]

            # Attendance succeeds, grades fail, discipline succeeds
            mock_att.return_value.extract.return_value = [0.9, 3]
            mock_grade.return_value.extract.side_effect = Exception("Grades service down")
            mock_disc.return_value.extract.return_value = [2, 1.8]

            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True
            mock_redis_class.return_value = mock_redis

            pipeline = FeaturePipeline(mock_db, use_cache=True)

            student_id = str(uuid4())
            ref_date = date.today()

            # Should recover and return partial features
            features = pipeline.extract_features(student_id, ref_date)
            expected_features = [0.9, 3, 2, 1.8]  # attendance + discipline only
            assert features == expected_features

            # Should still attempt to cache the partial results
            mock_redis.setex.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

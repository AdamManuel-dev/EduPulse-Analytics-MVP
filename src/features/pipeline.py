"""
@fileoverview Feature extraction pipeline orchestrator with caching and batch processing
@lastmodified 2025-08-13T00:50:05-05:00

Features: Multi-extractor orchestration, Redis caching, batch processing, database storage
Main APIs: extract_features(), extract_batch_features(), get_feature_names(), get_feature_importance()
Constraints: Requires SQLAlchemy session, Redis URL, feature extractors, StudentFeature model
Patterns: Cache-aside pattern, error-tolerant extraction, consistent feature ordering
"""

from typing import Dict, List, Optional
from datetime import date
import numpy as np
from sqlalchemy.orm import Session
import redis
import json
import hashlib

from src.features.attendance import AttendanceFeatureExtractor
from src.features.grades import GradeFeatureExtractor
from src.features.discipline import DisciplineFeatureExtractor
from src.config.settings import get_settings
from src.db import models

settings = get_settings()


class FeaturePipeline:
    """
    Orchestrates feature extraction from multiple sources with caching and error handling.

    Coordinates multiple feature extractors (attendance, grades, discipline) to create
    a unified feature vector for machine learning models. Provides Redis caching for
    performance and database storage for persistence.

    Args:
        db_session: SQLAlchemy session for database operations
        use_cache: Whether to enable Redis caching for extracted features

    Attributes:
        db: Database session for data access
        extractors: Dictionary of named feature extractors
        redis_client: Redis client for caching (if enabled)
        use_cache: Boolean flag for cache usage
    """

    def __init__(self, db_session: Session, use_cache: bool = True):
        """
        Initialize the feature pipeline with database connection and optional caching.

        Sets up feature extractors and Redis connection if caching is enabled.
        Gracefully degrades to no caching if Redis connection fails.

        Args:
            db_session: SQLAlchemy session for database access
            use_cache: Enable Redis caching for performance optimization
        """
        self.db = db_session
        self.use_cache = use_cache and settings.feature_cache_enabled

        # Initialize extractors
        self.extractors = {
            "attendance": AttendanceFeatureExtractor(db_session),
            "grades": GradeFeatureExtractor(db_session),
            "discipline": DisciplineFeatureExtractor(db_session),
        }

        # Initialize Redis for caching if enabled
        if self.use_cache:
            try:
                self.redis_client = redis.from_url(str(settings.redis_url))
                self.cache_ttl = settings.feature_cache_ttl
            except Exception as e:
                print(f"Warning: Redis connection failed, caching disabled: {e}")
                self.use_cache = False
                self.redis_client = None
        else:
            self.redis_client = None

    def extract_features(self, student_id: str, reference_date: date) -> np.ndarray:
        """
        Extract all features for a student at a given date.

        Args:
            student_id: UUID of the student
            reference_date: Date to calculate features for

        Returns:
            Numpy array of feature values
        """
        # Try to get from cache first
        cache_key = self._get_cache_key(student_id, reference_date)

        if self.use_cache:
            cached_features = self._get_cached_features(cache_key)
            if cached_features is not None:
                return cached_features

        # Extract features from each source
        all_features = {}

        for name, extractor in self.extractors.items():
            try:
                features = extractor.extract(student_id, reference_date)
                # Prefix feature names with extractor name
                prefixed_features = {f"{name}_{k}": v for k, v in features.items()}
                all_features.update(prefixed_features)
            except Exception as e:
                print(f"Error extracting {name} features: {e}")
                # Add empty features on error
                empty_features = {f"{name}_{k}": 0.0 for k in extractor.get_feature_names()}
                all_features.update(empty_features)

        # Convert to numpy array with consistent ordering
        feature_names = self.get_feature_names()
        feature_vector = np.array([all_features.get(name, 0.0) for name in feature_names])

        # Cache the results
        if self.use_cache:
            self._cache_features(cache_key, feature_vector)

        # Also store in database
        self._store_features_in_db(student_id, reference_date, feature_vector)

        return feature_vector

    def extract_batch_features(
        self, student_ids: List[str], reference_date: date
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for multiple students.

        Args:
            student_ids: List of student UUIDs
            reference_date: Date to calculate features for

        Returns:
            Dictionary mapping student_id to feature vector
        """
        results = {}

        for student_id in student_ids:
            try:
                features = self.extract_features(student_id, reference_date)
                results[student_id] = features
            except Exception as e:
                print(f"Error extracting features for student {student_id}: {e}")
                # Return zero vector on error
                results[student_id] = np.zeros(len(self.get_feature_names()))

        return results

    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of all feature names.
        """
        names = []

        for extractor_name, extractor in self.extractors.items():
            for feature_name in extractor.get_feature_names():
                names.append(f"{extractor_name}_{feature_name}")

        return names

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (placeholder for now).
        """
        # This will be populated after model training
        feature_names = self.get_feature_names()
        # Return uniform importance for now
        importance = 1.0 / len(feature_names)
        return {name: importance for name in feature_names}

    def _get_cache_key(self, student_id: str, reference_date: date) -> str:
        """
        Generate a unique cache key for feature storage and retrieval.

        Creates a deterministic cache key based on student ID and reference date
        using MD5 hashing to ensure consistent key generation.

        Args:
            student_id: UUID string of the student
            reference_date: Date for feature calculation

        Returns:
            str: MD5 hash of the cache key components

        Examples:
            >>> key = pipeline._get_cache_key("123-456-789", date(2024, 6, 15))
            >>> print(len(key))
            32
        """
        key_data = f"features:{student_id}:{reference_date.isoformat()}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_features(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Retrieve cached feature vector from Redis if available.

        Attempts to fetch and deserialize a previously computed feature vector
        from Redis cache. Returns None if cache miss or error occurs.

        Args:
            cache_key: Unique cache key for the feature vector

        Returns:
            numpy.ndarray or None: Cached feature vector if found, None otherwise

        Examples:
            >>> features = pipeline._get_cached_features(cache_key)
            >>> if features is not None:
            ...     print(f"Cache hit: {len(features)} features")
        """
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                features_list = json.loads(cached_data)
                return np.array(features_list)
        except Exception as e:
            print(f"Cache retrieval error: {e}")

        return None

    def _cache_features(self, cache_key: str, features: np.ndarray) -> None:
        """
        Store computed feature vector in Redis cache with TTL.

        Serializes feature vector to JSON and stores in Redis with configurable
        time-to-live to balance performance and memory usage.

        Args:
            cache_key: Unique cache key for storage
            features: Feature vector to cache

        Examples:
            >>> pipeline._cache_features("abc123", np.array([0.8, 0.5, 2.1]))
            # Features stored in Redis with TTL
        """
        if not self.redis_client:
            return

        try:
            features_json = json.dumps(features.tolist())
            self.redis_client.setex(cache_key, self.cache_ttl, features_json)
        except Exception as e:
            print(f"Cache storage error: {e}")

    def _store_features_in_db(
        self, student_id: str, reference_date: date, feature_vector: np.ndarray
    ) -> None:
        """
        Persist computed features to database for long-term storage and analysis.

        Stores feature vectors in the StudentFeature table, updating existing
        records or creating new ones. Also extracts key aggregate metrics for
        easier querying and reporting.

        Args:
            student_id: UUID string of the student
            reference_date: Date the features were calculated for
            feature_vector: Complete feature vector as numpy array

        Examples:
            >>> pipeline._store_features_in_db("123-456", date(2024, 6, 15), features)
            # Features stored in StudentFeature table
        """
        try:
            # Check if features already exist for this date
            existing = (
                self.db.query(models.StudentFeature)
                .filter(
                    models.StudentFeature.student_id == student_id,
                    models.StudentFeature.feature_date == reference_date,
                )
                .first()
            )

            if existing:
                # Update existing record
                existing.feature_vector = feature_vector.tolist()
            else:
                # Create new record
                student_features = models.StudentFeature(
                    student_id=student_id,
                    feature_date=reference_date,
                    feature_vector=feature_vector.tolist(),
                    # Calculate some aggregate metrics
                    attendance_rate=feature_vector[
                        self.get_feature_names().index("attendance_attendance_rate")
                    ],
                    gpa_current=feature_vector[
                        self.get_feature_names().index("grades_gpa_current")
                    ],
                    discipline_incidents=int(
                        feature_vector[self.get_feature_names().index("discipline_incident_count")]
                    ),
                )
                self.db.add(student_features)

            self.db.commit()
        except Exception as e:
            print(f"Error storing features in database: {e}")
            self.db.rollback()

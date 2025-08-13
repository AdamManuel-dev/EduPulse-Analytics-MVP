"""
@fileoverview Abstract base class for feature extractors with common utilities
@lastmodified 2025-08-13T02:56:19-05:00

Features: BaseFeatureExtractor, rolling statistics, pattern analysis, date range calculation
Main APIs: extract(), get_feature_names(), calculate_rolling_stats(), calculate_pattern_features()
Constraints: Requires SQLAlchemy session, settings window/lag days, student data
Patterns: Abstract factory pattern, numpy statistics, linear trend calculation
"""

from abc import ABC, abstractmethod
from typing import Dict, List
from datetime import date, timedelta
import numpy as np
from sqlalchemy.orm import Session

from src.config.settings import get_settings

settings = get_settings()


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors with common utilities.

    Provides shared functionality for feature extraction including date range
    calculation, rolling statistics, and pattern analysis. All concrete feature
    extractors should inherit from this class.

    Args:
        db_session: SQLAlchemy database session for data access

    Attributes:
        db: Database session for queries
        window_days: Number of days to look back for feature calculation
        lag_days: Number of days to lag behind reference date to avoid data leakage
    """

    def __init__(self, db_session: Session):
        """
        Initialize the feature extractor with database connection and settings.

        Args:
            db_session: SQLAlchemy session for database access
        """
        self.db = db_session
        self.window_days = settings.feature_window_days
        self.lag_days = settings.feature_lag_days

    @abstractmethod
    def extract(self, student_id: str, reference_date: date) -> Dict[str, float]:
        """
        Extract features for a student at a specific date.

        Args:
            student_id: UUID of the student
            reference_date: Date to calculate features for

        Returns:
            Dictionary of feature names to values
        """

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this extractor produces.

        Returns:
            List of feature names
        """

    def get_date_range(self, reference_date: date) -> tuple[date, date]:
        """
        Calculate the date range for feature extraction with lag to prevent data leakage.

        Computes a date range that looks back from a lagged reference date to avoid
        using future information. The lag prevents data leakage in prediction scenarios.

        Args:
            reference_date: Reference date for calculation (typically prediction date)

        Returns:
            tuple[date, date]: Tuple of (start_date, end_date) where:
                - end_date = reference_date - lag_days
                - start_date = end_date - window_days

        Examples:
            >>> extractor = BaseFeatureExtractor(db_session)
            >>> start, end = extractor.get_date_range(date(2024, 6, 15))
            >>> print(f"Range: {start} to {end}")
            Range: 2024-05-01 to 2024-06-01
        """
        end_date = reference_date - timedelta(days=self.lag_days)
        start_date = end_date - timedelta(days=self.window_days)
        return start_date, end_date

    def calculate_rolling_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive rolling statistics for a time series of values.

        Computes descriptive statistics and linear trend analysis for a sequence
        of numeric values, providing robust handling of empty or single-value series.

        Args:
            values: List of numeric values in chronological order

        Returns:
            dict: Dictionary containing statistical measures:
                - mean: Average value across the series
                - std: Standard deviation (population)
                - min: Minimum value in the series
                - max: Maximum value in the series
                - trend: Linear trend coefficient (slope per time unit)

        Examples:
            >>> values = [85.0, 87.0, 83.0, 90.0, 88.0]
            >>> stats = extractor.calculate_rolling_stats(values)
            >>> print(f"Mean: {stats['mean']:.1f}, Trend: {stats['trend']:.2f}")
            Mean: 86.6, Trend: 0.70
        """
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "trend": 0.0}

        arr = np.array(values)

        # Calculate trend using linear regression
        if len(arr) > 1:
            x = np.arange(len(arr))
            trend = np.polyfit(x, arr, 1)[0]
        else:
            trend = 0.0

        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "trend": float(trend),
        }

    def calculate_pattern_features(self, dates: List[date]) -> Dict[str, float]:
        """
        Calculate temporal pattern features from a sequence of dates.

        Analyzes date patterns to identify behavioral indicators such as day-of-week
        preferences, consecutive streaks, and temporal gaps. Useful for detecting
        patterns in attendance, discipline incidents, or other time-based events.

        Args:
            dates: List of dates to analyze for temporal patterns

        Returns:
            dict: Dictionary containing pattern features:
                - monday_ratio: Proportion of events occurring on Mondays
                - friday_ratio: Proportion of events occurring on Fridays
                - consecutive_days: Maximum consecutive day streak
                - gaps_mean: Average gap in days between consecutive events

        Examples:
            >>> dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 5)]
            >>> patterns = extractor.calculate_pattern_features(dates)
            >>> print(f"Consecutive days: {patterns['consecutive_days']}")
            Consecutive days: 2
        """
        if not dates:
            return {
                "monday_ratio": 0.0,
                "friday_ratio": 0.0,
                "consecutive_days": 0,
                "gaps_mean": 0.0,
            }

        weekdays = [d.weekday() for d in dates]
        monday_count = sum(1 for w in weekdays if w == 0)
        friday_count = sum(1 for w in weekdays if w == 4)

        # Calculate consecutive days
        sorted_dates = sorted(dates)
        consecutive_counts = []
        current_streak = 1

        for i in range(1, len(sorted_dates)):
            diff = (sorted_dates[i] - sorted_dates[i - 1]).days
            if diff == 1:
                current_streak += 1
            else:
                consecutive_counts.append(current_streak)
                current_streak = 1
        consecutive_counts.append(current_streak)

        # Calculate gaps
        gaps = []
        for i in range(1, len(sorted_dates)):
            gap = (sorted_dates[i] - sorted_dates[i - 1]).days - 1
            if gap > 0:
                gaps.append(gap)

        return {
            "monday_ratio": monday_count / len(dates) if dates else 0.0,
            "friday_ratio": friday_count / len(dates) if dates else 0.0,
            "consecutive_days": max(consecutive_counts) if consecutive_counts else 0,
            "gaps_mean": np.mean(gaps) if gaps else 0.0,
        }

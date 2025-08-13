"""
Base classes for feature extraction in EduPulse Analytics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from src.db import models
from src.config.settings import get_settings

settings = get_settings()


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    """
    
    def __init__(self, db_session: Session):
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
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this extractor produces.
        
        Returns:
            List of feature names
        """
        pass
    
    def get_date_range(self, reference_date: date) -> tuple[date, date]:
        """
        Calculate the date range for feature extraction.
        
        Args:
            reference_date: Reference date for calculation
            
        Returns:
            Tuple of (start_date, end_date)
        """
        end_date = reference_date - timedelta(days=self.lag_days)
        start_date = end_date - timedelta(days=self.window_days)
        return start_date, end_date
    
    def calculate_rolling_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate rolling statistics for a series of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with mean, std, min, max, trend
        """
        if not values:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'trend': 0.0
            }
        
        arr = np.array(values)
        
        # Calculate trend using linear regression
        if len(arr) > 1:
            x = np.arange(len(arr))
            trend = np.polyfit(x, arr, 1)[0]
        else:
            trend = 0.0
        
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'trend': float(trend)
        }
    
    def calculate_pattern_features(self, dates: List[date]) -> Dict[str, float]:
        """
        Calculate pattern-based features from dates.
        
        Args:
            dates: List of dates to analyze
            
        Returns:
            Dictionary of pattern features
        """
        if not dates:
            return {
                'monday_ratio': 0.0,
                'friday_ratio': 0.0,
                'consecutive_days': 0,
                'gaps_mean': 0.0
            }
        
        weekdays = [d.weekday() for d in dates]
        monday_count = sum(1 for w in weekdays if w == 0)
        friday_count = sum(1 for w in weekdays if w == 4)
        
        # Calculate consecutive days
        sorted_dates = sorted(dates)
        consecutive_counts = []
        current_streak = 1
        
        for i in range(1, len(sorted_dates)):
            diff = (sorted_dates[i] - sorted_dates[i-1]).days
            if diff == 1:
                current_streak += 1
            else:
                consecutive_counts.append(current_streak)
                current_streak = 1
        consecutive_counts.append(current_streak)
        
        # Calculate gaps
        gaps = []
        for i in range(1, len(sorted_dates)):
            gap = (sorted_dates[i] - sorted_dates[i-1]).days - 1
            if gap > 0:
                gaps.append(gap)
        
        return {
            'monday_ratio': monday_count / len(dates) if dates else 0.0,
            'friday_ratio': friday_count / len(dates) if dates else 0.0,
            'consecutive_days': max(consecutive_counts) if consecutive_counts else 0,
            'gaps_mean': np.mean(gaps) if gaps else 0.0
        }
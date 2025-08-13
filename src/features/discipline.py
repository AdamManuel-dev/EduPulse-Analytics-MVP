"""
Discipline incident feature extraction for EduPulse Analytics.
"""

from typing import Dict, List
from datetime import date
from sqlalchemy import and_
import numpy as np

from src.features.base import BaseFeatureExtractor
from src.db import models


class DisciplineFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts discipline-related features for students.
    """
    
    def extract(self, student_id: str, reference_date: date) -> Dict[str, float]:
        """
        Extract discipline and behavioral features.
        
        Features include:
        - Incident frequency and severity
        - Escalation patterns
        - Time between incidents
        - Resolution effectiveness
        """
        start_date, end_date = self.get_date_range(reference_date)
        
        # Query discipline records
        incidents = self.db.query(models.DisciplineIncident).filter(
            and_(
                models.DisciplineIncident.student_id == student_id,
                models.DisciplineIncident.incident_date >= start_date,
                models.DisciplineIncident.incident_date <= end_date
            )
        ).order_by(models.DisciplineIncident.incident_date).all()
        
        if not incidents:
            return self._empty_features()
        
        # Calculate basic metrics
        total_incidents = len(incidents)
        severity_levels = [i.severity_level for i in incidents if i.severity_level is not None]
        
        # Severity statistics
        avg_severity = np.mean(severity_levels) if severity_levels else 0.0
        max_severity = np.max(severity_levels) if severity_levels else 0.0
        severity_std = np.std(severity_levels) if len(severity_levels) > 1 else 0.0
        
        # Calculate escalation trend
        severity_trend = self._calculate_severity_trend(incidents)
        
        # Time between incidents (recidivism measure)
        time_gaps = self._calculate_time_gaps(incidents)
        avg_time_between = np.mean(time_gaps) if time_gaps else 0.0
        min_time_between = np.min(time_gaps) if time_gaps else 0.0
        
        # Incident frequency per month
        days_in_window = (end_date - start_date).days
        monthly_rate = (total_incidents / days_in_window) * 30 if days_in_window > 0 else 0.0
        
        # Analyze incident types
        type_diversity = self._calculate_type_diversity(incidents)
        
        # Recent incidents (last 30% of window)
        recent_cutoff = int(len(incidents) * 0.7)
        recent_incidents = incidents[recent_cutoff:]
        recent_count = len(recent_incidents)
        recent_severity = np.mean([i.severity_level for i in recent_incidents 
                                  if i.severity_level is not None]) if recent_incidents else 0.0
        
        # Calculate acceleration (incidents becoming more frequent)
        acceleration = self._calculate_incident_acceleration(incidents)
        
        # High severity incident count
        high_severity_count = sum(1 for s in severity_levels if s >= 4)
        high_severity_rate = high_severity_count / total_incidents if total_incidents > 0 else 0.0
        
        features = {
            'incident_count': float(total_incidents),
            'incident_monthly_rate': monthly_rate,
            'severity_mean': avg_severity,
            'severity_max': max_severity,
            'severity_std': severity_std,
            'severity_trend': severity_trend,
            'time_between_mean': avg_time_between,
            'time_between_min': min_time_between,
            'incident_acceleration': acceleration,
            'type_diversity': type_diversity,
            'recent_incident_count': float(recent_count),
            'recent_severity_mean': recent_severity,
            'high_severity_rate': high_severity_rate
        }
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of discipline feature names.
        """
        return [
            'incident_count',
            'incident_monthly_rate',
            'severity_mean',
            'severity_max',
            'severity_std',
            'severity_trend',
            'time_between_mean',
            'time_between_min',
            'incident_acceleration',
            'type_diversity',
            'recent_incident_count',
            'recent_severity_mean',
            'high_severity_rate'
        ]
    
    def _empty_features(self) -> Dict[str, float]:
        """
        Return empty feature dictionary with zeros.
        """
        return {name: 0.0 for name in self.get_feature_names()}
    
    def _calculate_severity_trend(self, incidents: List[models.DisciplineIncident]) -> float:
        """
        Calculate trend in incident severity over time.
        """
        if len(incidents) < 2:
            return 0.0
        
        severities = [i.severity_level for i in incidents if i.severity_level is not None]
        
        if len(severities) < 2:
            return 0.0
        
        # Linear regression for trend
        x = np.arange(len(severities))
        trend = np.polyfit(x, severities, 1)[0]
        
        return float(trend)
    
    def _calculate_time_gaps(self, incidents: List[models.DisciplineIncident]) -> List[float]:
        """
        Calculate time gaps between consecutive incidents.
        """
        if len(incidents) < 2:
            return []
        
        gaps = []
        for i in range(1, len(incidents)):
            gap_days = (incidents[i].incident_date - incidents[i-1].incident_date).days
            gaps.append(float(gap_days))
        
        return gaps
    
    def _calculate_type_diversity(self, incidents: List[models.DisciplineIncident]) -> float:
        """
        Calculate diversity of incident types (0-1, higher = more diverse).
        """
        if not incidents:
            return 0.0
        
        types = [i.incident_type for i in incidents if i.incident_type]
        
        if not types:
            return 0.0
        
        unique_types = len(set(types))
        total_types = len(types)
        
        # Shannon entropy normalized
        if unique_types == 1:
            return 0.0
        
        return unique_types / total_types
    
    def _calculate_incident_acceleration(self, incidents: List[models.DisciplineIncident]) -> float:
        """
        Calculate if incidents are accelerating (becoming more frequent).
        Returns positive value if accelerating, negative if decelerating.
        """
        if len(incidents) < 3:
            return 0.0
        
        # Calculate time gaps
        gaps = self._calculate_time_gaps(incidents)
        
        if len(gaps) < 2:
            return 0.0
        
        # Calculate trend in gaps (negative trend = accelerating incidents)
        x = np.arange(len(gaps))
        gap_trend = np.polyfit(x, gaps, 1)[0]
        
        # Return negative to make positive value mean acceleration
        return float(-gap_trend)
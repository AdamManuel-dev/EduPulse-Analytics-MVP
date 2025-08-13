"""
Attendance feature extraction for EduPulse Analytics.
"""

from typing import Dict, List
from datetime import date
from sqlalchemy import and_

from src.features.base import BaseFeatureExtractor
from src.db import models


class AttendanceFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts attendance-related features for students.
    """
    
    def extract(self, student_id: str, reference_date: date) -> Dict[str, float]:
        """
        Extract attendance features for a student.
        
        Features include:
        - Attendance rate (overall and by status)
        - Absence patterns (consecutive, day of week)
        - Tardiness trends
        - Excused vs unexcused ratio
        """
        start_date, end_date = self.get_date_range(reference_date)
        
        # Query attendance records
        records = self.db.query(models.AttendanceRecord).filter(
            and_(
                models.AttendanceRecord.student_id == student_id,
                models.AttendanceRecord.date >= start_date,
                models.AttendanceRecord.date <= end_date
            )
        ).all()
        
        if not records:
            return self._empty_features()
        
        # Calculate basic attendance metrics
        total_days = len(records)
        present_count = sum(1 for r in records if r.status == 'present')
        absent_count = sum(1 for r in records if r.status == 'absent')
        tardy_count = sum(1 for r in records if r.status == 'tardy')
        excused_count = sum(1 for r in records if r.status == 'excused')
        
        # Calculate rates
        attendance_rate = present_count / total_days if total_days > 0 else 0.0
        absence_rate = absent_count / total_days if total_days > 0 else 0.0
        tardy_rate = tardy_count / total_days if total_days > 0 else 0.0
        excused_rate = excused_count / total_days if total_days > 0 else 0.0
        
        # Get dates for pattern analysis
        absent_dates = [r.date for r in records if r.status == 'absent']
        tardy_dates = [r.date for r in records if r.status == 'tardy']
        
        # Calculate pattern features
        absence_patterns = self.calculate_pattern_features(absent_dates)
        tardy_patterns = self.calculate_pattern_features(tardy_dates)
        
        # Calculate rolling statistics for weekly attendance
        weekly_rates = self._calculate_weekly_attendance_rates(records)
        rolling_stats = self.calculate_rolling_stats(weekly_rates)
        
        # Combine all features
        features = {
            'attendance_rate': attendance_rate,
            'absence_rate': absence_rate,
            'tardy_rate': tardy_rate,
            'excused_rate': excused_rate,
            'absence_monday_ratio': absence_patterns['monday_ratio'],
            'absence_friday_ratio': absence_patterns['friday_ratio'],
            'absence_consecutive_days': absence_patterns['consecutive_days'],
            'absence_gaps_mean': absence_patterns['gaps_mean'],
            'tardy_monday_ratio': tardy_patterns['monday_ratio'],
            'tardy_friday_ratio': tardy_patterns['friday_ratio'],
            'attendance_mean': rolling_stats['mean'],
            'attendance_std': rolling_stats['std'],
            'attendance_trend': rolling_stats['trend'],
            'total_days_tracked': total_days
        }
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of attendance feature names.
        """
        return [
            'attendance_rate',
            'absence_rate',
            'tardy_rate',
            'excused_rate',
            'absence_monday_ratio',
            'absence_friday_ratio',
            'absence_consecutive_days',
            'absence_gaps_mean',
            'tardy_monday_ratio',
            'tardy_friday_ratio',
            'attendance_mean',
            'attendance_std',
            'attendance_trend',
            'total_days_tracked'
        ]
    
    def _empty_features(self) -> Dict[str, float]:
        """
        Return empty feature dictionary with zeros.
        """
        return {name: 0.0 for name in self.get_feature_names()}
    
    def _calculate_weekly_attendance_rates(self, records: List[models.AttendanceRecord]) -> List[float]:
        """
        Calculate weekly attendance rates from records.
        """
        if not records:
            return []
        
        # Group by week and calculate rates
        weekly_data = {}
        for record in records:
            week_key = record.date.isocalendar()[1]  # Week number
            if week_key not in weekly_data:
                weekly_data[week_key] = {'present': 0, 'total': 0}
            
            weekly_data[week_key]['total'] += 1
            if record.status == 'present':
                weekly_data[week_key]['present'] += 1
        
        # Calculate rates for each week
        weekly_rates = []
        for week in sorted(weekly_data.keys()):
            data = weekly_data[week]
            rate = data['present'] / data['total'] if data['total'] > 0 else 0.0
            weekly_rates.append(rate)
        
        return weekly_rates
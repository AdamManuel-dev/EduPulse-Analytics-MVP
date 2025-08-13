"""
@fileoverview Academic grade feature extractor for performance metrics and trends
@lastmodified 2025-08-13T00:50:05-05:00

Features: GPA calculation, grade statistics, trends, subject analysis, assignment type analysis
Main APIs: extract(), get_feature_names(), _calculate_grade_trend(), _analyze_by_subject()
Constraints: Requires Grade model, numeric grade_value, submission_date, course_id
Patterns: 4.0 GPA scale conversion, linear trend fitting, core/elective classification
"""

from typing import Dict, List
from datetime import date
from sqlalchemy import and_
import numpy as np

from src.features.base import BaseFeatureExtractor
from src.db import models


class GradeFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts grade-related features for students.
    """

    def extract(self, student_id: str, reference_date: date) -> Dict[str, float]:
        """
        Extract academic performance features.

        Features include:
        - GPA and grade statistics
        - Performance trends
        - Subject-specific metrics
        - Assignment completion patterns
        """
        start_date, end_date = self.get_date_range(reference_date)

        # Query grade records
        grades = (
            self.db.query(models.Grade)
            .filter(
                and_(
                    models.Grade.student_id == student_id,
                    models.Grade.submission_date >= start_date,
                    models.Grade.submission_date <= end_date,
                )
            )
            .all()
        )

        if not grades:
            return self._empty_features()

        # Calculate basic grade metrics
        grade_values = [g.grade_value for g in grades if g.grade_value is not None]

        if not grade_values:
            return self._empty_features()

        # Overall statistics
        gpa_current = np.mean(grade_values) / 25.0  # Convert to 4.0 scale
        grade_mean = np.mean(grade_values)
        grade_std = np.std(grade_values) if len(grade_values) > 1 else 0.0
        grade_min = np.min(grade_values)
        grade_max = np.max(grade_values)

        # Calculate trends
        grade_trend = self._calculate_grade_trend(grades)

        # Subject-specific analysis
        subject_stats = self._analyze_by_subject(grades)

        # Assignment type analysis
        assignment_stats = self._analyze_by_assignment_type(grades)

        # Calculate volatility (consistency measure)
        grade_volatility = grade_std / grade_mean if grade_mean > 0 else 0.0

        # Failure risk indicators
        failing_count = sum(1 for g in grade_values if g < 60)
        failing_rate = failing_count / len(grade_values) if grade_values else 0.0

        # Recent performance (last 30% of window)
        recent_cutoff = int(len(grades) * 0.7)
        recent_grades = [g.grade_value for g in grades[recent_cutoff:] if g.grade_value is not None]
        recent_mean = np.mean(recent_grades) if recent_grades else grade_mean

        # Performance change
        performance_change = recent_mean - grade_mean

        features = {
            "gpa_current": gpa_current,
            "grade_mean": grade_mean,
            "grade_std": grade_std,
            "grade_min": grade_min,
            "grade_max": grade_max,
            "grade_trend": grade_trend,
            "grade_volatility": grade_volatility,
            "failing_rate": failing_rate,
            "recent_performance": recent_mean,
            "performance_change": performance_change,
            "core_subject_mean": subject_stats["core_mean"],
            "elective_subject_mean": subject_stats["elective_mean"],
            "test_score_mean": assignment_stats["test_mean"],
            "homework_score_mean": assignment_stats["homework_mean"],
            "total_grades_tracked": len(grade_values),
        }

        return features

    def get_feature_names(self) -> List[str]:
        """
        Get list of grade feature names.
        """
        return [
            "gpa_current",
            "grade_mean",
            "grade_std",
            "grade_min",
            "grade_max",
            "grade_trend",
            "grade_volatility",
            "failing_rate",
            "recent_performance",
            "performance_change",
            "core_subject_mean",
            "elective_subject_mean",
            "test_score_mean",
            "homework_score_mean",
            "total_grades_tracked",
        ]

    def _empty_features(self) -> Dict[str, float]:
        """
        Return empty feature dictionary with zeros.
        """
        return {name: 0.0 for name in self.get_feature_names()}

    def _calculate_grade_trend(self, grades: List[models.Grade]) -> float:
        """
        Calculate trend in grades over time.
        """
        if len(grades) < 2:
            return 0.0

        # Sort by date and get values
        sorted_grades = sorted(grades, key=lambda g: g.submission_date)
        values = [g.grade_value for g in sorted_grades if g.grade_value is not None]

        if len(values) < 2:
            return 0.0

        # Linear regression for trend
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]

        return float(trend)

    def _analyze_by_subject(self, grades: List[models.Grade]) -> Dict[str, float]:
        """
        Analyze grades by subject type (core vs elective).
        """
        # Define core subjects
        core_subjects = ["MATH", "ENG", "SCI", "HIST", "ENGLISH", "SCIENCE", "HISTORY"]

        core_grades = []
        elective_grades = []

        for grade in grades:
            if grade.grade_value is None:
                continue

            # Check if course is core subject
            is_core = any(subject in grade.course_id.upper() for subject in core_subjects)

            if is_core:
                core_grades.append(grade.grade_value)
            else:
                elective_grades.append(grade.grade_value)

        return {
            "core_mean": np.mean(core_grades) if core_grades else 0.0,
            "elective_mean": np.mean(elective_grades) if elective_grades else 0.0,
        }

    def _analyze_by_assignment_type(self, grades: List[models.Grade]) -> Dict[str, float]:
        """
        Analyze grades by assignment type.
        """
        test_grades = []
        homework_grades = []

        for grade in grades:
            if grade.grade_value is None or not grade.assignment_type:
                continue

            assignment_lower = grade.assignment_type.lower()

            if (
                "test" in assignment_lower
                or "exam" in assignment_lower
                or "quiz" in assignment_lower
            ):
                test_grades.append(grade.grade_value)
            elif (
                "homework" in assignment_lower
                or "hw" in assignment_lower
                or "assignment" in assignment_lower
            ):
                homework_grades.append(grade.grade_value)

        return {
            "test_mean": np.mean(test_grades) if test_grades else 0.0,
            "homework_mean": np.mean(homework_grades) if homework_grades else 0.0,
        }

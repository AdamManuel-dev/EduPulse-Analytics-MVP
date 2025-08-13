"""
Unit tests for feature extractors with real database integration.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, MagicMock
from uuid import uuid4
import numpy as np

from src.features.base import BaseFeatureExtractor
from src.features.attendance import AttendanceFeatureExtractor
from src.features.grades import GradeFeatureExtractor
from src.features.discipline import DisciplineFeatureExtractor
from src.db import models


class TestBaseFeatureExtractor:
    """Tests for base feature extractor."""

    def test_date_range_calculation(self):
        """Test date range calculation."""
        mock_db = Mock()

        class TestExtractor(BaseFeatureExtractor):
            def extract(self, student_id, reference_date):
                return {}

            def get_feature_names(self):
                return []

        extractor = TestExtractor(mock_db)
        extractor.window_days = 90
        extractor.lag_days = 7

        reference = date(2025, 8, 1)
        start, end = extractor.get_date_range(reference)

        assert end == date(2025, 7, 25)  # 7 days before reference
        assert start == date(2025, 4, 26)  # 90 days before end

    def test_rolling_stats_empty(self):
        """Test rolling stats with empty data."""
        mock_db = Mock()

        class TestExtractor(BaseFeatureExtractor):
            def extract(self, student_id, reference_date):
                return {}

            def get_feature_names(self):
                return []

        extractor = TestExtractor(mock_db)
        stats = extractor.calculate_rolling_stats([])

        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["trend"] == 0.0

    def test_rolling_stats_with_data(self):
        """Test rolling stats with actual data."""
        mock_db = Mock()

        class TestExtractor(BaseFeatureExtractor):
            def extract(self, student_id, reference_date):
                return {}

            def get_feature_names(self):
                return []

        extractor = TestExtractor(mock_db)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = extractor.calculate_rolling_stats(values)

        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["trend"] > 0  # Positive trend

    def test_rolling_stats_single_value(self):
        """Test rolling stats with single value - covers trend=0.0 case."""
        mock_db = Mock()

        class TestExtractor(BaseFeatureExtractor):
            def extract(self, student_id, reference_date):
                return {}

            def get_feature_names(self):
                return []

        extractor = TestExtractor(mock_db)
        values = [5.0]  # Single value
        stats = extractor.calculate_rolling_stats(values)

        assert stats["mean"] == 5.0
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0
        assert stats["trend"] == 0.0  # Should be 0 for single value


class TestAttendanceFeatureExtractor:
    """Tests for attendance feature extractor."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        return session

    def test_extract_no_records(self, mock_db_session):
        """Test extraction with no attendance records."""
        # Setup mock query to return empty list
        mock_db_session.query().filter().all.return_value = []

        extractor = AttendanceFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())

        # Should return zeros for all features
        assert len(features) == len(extractor.get_feature_names())
        assert all(v == 0.0 for v in features.values())

    def test_extract_with_records(self, mock_db_session):
        """Test extraction with attendance records."""
        # Create mock records
        mock_records = []
        for i in range(10):
            record = Mock(spec=models.AttendanceRecord)
            record.date = date.today() - timedelta(days=i)
            record.status = "present" if i % 2 == 0 else "absent"
            mock_records.append(record)

        mock_db_session.query().filter().all.return_value = mock_records

        extractor = AttendanceFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())

        # Check feature calculations
        assert "attendance_rate" in features
        assert features["attendance_rate"] == 0.5  # 5 present out of 10
        assert features["absence_rate"] == 0.5  # 5 absent out of 10
        assert features["total_days_tracked"] == 10

    def test_feature_names(self, mock_db_session):
        """Test feature names list."""
        extractor = AttendanceFeatureExtractor(mock_db_session)
        names = extractor.get_feature_names()

        assert "attendance_rate" in names
        assert "absence_rate" in names
        assert "tardy_rate" in names
        assert "excused_rate" in names
        assert len(names) == 14  # Total expected features

    def test_calculate_weekly_rates_empty(self, mock_db_session):
        """Test weekly rates calculation with empty records - covers line 154."""
        extractor = AttendanceFeatureExtractor(mock_db_session)
        
        # Test with empty list
        weekly_rates = extractor._calculate_weekly_attendance_rates([])
        assert weekly_rates == []  # Should return empty list


class TestGradeFeatureExtractor:
    """Tests for grade feature extractor."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        return session

    def test_extract_no_grades(self, mock_db_session):
        """Test extraction with no grades."""
        mock_db_session.query().filter().all.return_value = []

        extractor = GradeFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())

        assert len(features) == len(extractor.get_feature_names())
        assert all(v == 0.0 for v in features.values())

    def test_extract_with_grades(self, mock_db_session):
        """Test extraction with grade records."""
        # Create mock grades with declining trend (recent grades are lower)
        mock_grades = []
        for i in range(5):
            grade = Mock(spec=models.Grade)
            # Older dates first for proper trend calculation
            grade.submission_date = date.today() - timedelta(days=(4-i) * 7)  # 28, 21, 14, 7, 0 days ago
            grade.grade_value = 85.0 - (i * 5)  # 85, 80, 75, 70, 65 - declining over time
            grade.course_id = "MATH101"
            grade.assignment_type = "test"
            mock_grades.append(grade)

        mock_db_session.query().filter().all.return_value = mock_grades

        extractor = GradeFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())

        # Check calculations
        assert "gpa_current" in features
        assert "grade_mean" in features
        assert features["grade_mean"] == 75.0  # Average of 85, 80, 75, 70, 65
        assert features["grade_trend"] < 0  # Negative trend (declining)
        assert features["total_grades_tracked"] == 5

    def test_failing_rate(self, mock_db_session):
        """Test failing rate calculation."""
        # Create mix of passing and failing grades
        mock_grades = [
            Mock(spec=models.Grade, grade_value=80.0),
            Mock(spec=models.Grade, grade_value=55.0),  # Failing
            Mock(spec=models.Grade, grade_value=45.0),  # Failing
            Mock(spec=models.Grade, grade_value=70.0),
        ]
        for g in mock_grades:
            g.submission_date = date.today()
            g.course_id = "TEST"
            g.assignment_type = "test"

        mock_db_session.query().filter().all.return_value = mock_grades

        extractor = GradeFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())

        assert features["failing_rate"] == 0.5  # 2 out of 4 failing


class TestDisciplineFeatureExtractor:
    """Tests for discipline feature extractor."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        return session

    def test_extract_no_incidents(self, mock_db_session):
        """Test extraction with no discipline incidents."""
        mock_db_session.query().filter().order_by().all.return_value = []

        extractor = DisciplineFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())

        assert len(features) == len(extractor.get_feature_names())
        assert all(v == 0.0 for v in features.values())

    def test_extract_with_incidents(self, mock_db_session):
        """Test extraction with discipline incidents."""
        # Create mock incidents with escalating severity
        mock_incidents = []
        for i in range(3):
            incident = Mock(spec=models.DisciplineIncident)
            incident.incident_date = date.today() - timedelta(days=i * 10)
            incident.severity_level = i + 1  # Increasing severity
            incident.incident_type = f"type_{i}"
            mock_incidents.append(incident)

        mock_db_session.query().filter().order_by().all.return_value = mock_incidents

        extractor = DisciplineFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())

        # Check calculations
        assert features["incident_count"] == 3.0
        assert features["severity_mean"] == 2.0  # Average of 1, 2, 3
        assert features["severity_max"] == 3.0
        assert features["severity_trend"] > 0  # Positive trend (escalating)

    def test_incident_acceleration(self, mock_db_session):
        """Test incident acceleration calculation."""
        # Create incidents with decreasing time gaps (accelerating)
        dates = [
            date.today() - timedelta(days=20),
            date.today() - timedelta(days=10),
            date.today() - timedelta(days=3),
            date.today() - timedelta(days=1),
        ]

        mock_incidents = []
        for d in dates:
            incident = Mock(spec=models.DisciplineIncident)
            incident.incident_date = d
            incident.severity_level = 2
            incident.incident_type = "test"
            mock_incidents.append(incident)

        mock_db_session.query().filter().order_by().all.return_value = mock_incidents

        extractor = DisciplineFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())

        # Should show positive acceleration (incidents becoming more frequent)
        assert features["incident_acceleration"] > 0


class TestAttendanceFeatureExtractorIntegration:
    """Integration tests for attendance feature extractor using real database."""

    def test_extract_with_real_database(self, db_session, sample_student):
        """Test extraction with real database data."""
        student_id = sample_student.id
        
        # Create real attendance records
        attendance_records = []
        for i in range(20):
            record_date = date.today() - timedelta(days=i)
            status = "present" if i % 3 != 0 else ("absent" if i % 2 == 0 else "tardy")
            
            record = models.AttendanceRecord(
                student_id=student_id,
                date=record_date,
                status=status,
                period=1  # Add required period field
            )
            attendance_records.append(record)
            db_session.add(record)
        
        db_session.commit()

        # Test feature extraction
        extractor = AttendanceFeatureExtractor(db_session)
        features = extractor.extract(str(student_id), date.today())

        # Verify features are calculated correctly
        assert "attendance_rate" in features
        assert "absence_rate" in features  
        assert "tardy_rate" in features
        assert features["total_days_tracked"] > 0
        
        # Check that rates add up correctly
        attendance_rate = features["attendance_rate"]
        absence_rate = features["absence_rate"]
        tardy_rate = features["tardy_rate"]
        
        # Rates should be between 0 and 1
        assert 0 <= attendance_rate <= 1
        assert 0 <= absence_rate <= 1
        assert 0 <= tardy_rate <= 1

    def test_extract_real_stats_calculation(self, db_session, sample_student):
        """Test that rolling statistics are calculated correctly with real data."""
        student_id = sample_student.id
        
        # Create attendance with known pattern (accounting for 7-day lag)
        for i in range(10):
            # Create records starting from 2 weeks ago to account for the lag
            record_date = date.today() - timedelta(days=14 + i)  
            # Create a pattern: first 5 days present, last 5 days absent
            status = "present" if i < 5 else "absent"
            
            record = models.AttendanceRecord(
                student_id=student_id,
                date=record_date,
                status=status,
                period=1  # Add required period field
            )
            db_session.add(record)
        
        db_session.commit()

        extractor = AttendanceFeatureExtractor(db_session)
        features = extractor.extract(str(student_id), date.today())

        # Should show declining trend (present -> absent over time)
        assert "attendance_trend" in features
        assert features["attendance_rate"] == 0.5  # 5 out of 10 present


class TestGradeFeatureExtractorIntegration:
    """Integration tests for grade feature extractor using real database."""

    def test_extract_with_real_grades(self, db_session, sample_student):
        """Test grade feature extraction with real database data."""
        student_id = sample_student.id
        
        # Create real grade records with pattern (accounting for 7-day lag)
        grade_values = [75, 80, 85, 90, 95]  # Improving over time (oldest to newest)
        for i, grade_value in enumerate(grade_values):
            grade_date = date.today() - timedelta(days=42 - i * 7)  # 42, 35, 28, 21, 14 days ago
            
            grade = models.Grade(
                student_id=student_id,
                submission_date=grade_date,
                grade_value=grade_value,
                course_id="MATH101",
                assignment_type="test"
            )
            db_session.add(grade)
        
        db_session.commit()

        # Test feature extraction
        extractor = GradeFeatureExtractor(db_session)
        features = extractor.extract(str(student_id), date.today())

        # Verify calculations
        assert "grade_mean" in features
        assert "grade_trend" in features
        assert features["grade_mean"] == 85.0  # Mean of 75,80,85,90,95
        assert features["grade_trend"] > 0  # Positive trend (improving over time)
        assert features["total_grades_tracked"] == 5

    def test_gpa_calculation_real_data(self, db_session, sample_student):
        """Test GPA calculation with real grade data."""
        student_id = sample_student.id
        
        # Create grades across multiple courses
        courses = ["MATH101", "ENG101", "SCI101"]
        grade_values = [85, 92, 78]
        
        for course, grade_value in zip(courses, grade_values):
            grade = models.Grade(
                student_id=student_id,
                submission_date=date.today() - timedelta(days=14),  # Account for 7-day lag
                grade_value=grade_value,
                course_id=course,
                assignment_type="final"
            )
            db_session.add(grade)
        
        db_session.commit()

        extractor = GradeFeatureExtractor(db_session)
        features = extractor.extract(str(student_id), date.today())

        # Check GPA calculation
        expected_gpa = (85 + 92 + 78) / 3 / 25  # Convert to 4.0 scale
        assert "gpa_current" in features
        assert abs(features["gpa_current"] - expected_gpa) < 0.1


class TestDisciplineFeatureExtractorIntegration:
    """Integration tests for discipline feature extractor using real database."""

    def test_extract_with_real_incidents(self, db_session, sample_student):
        """Test discipline feature extraction with real database data."""
        student_id = sample_student.id
        
        # Create real discipline incidents
        incident_data = [
            (date.today() - timedelta(days=30), 1, "minor_disruption"),
            (date.today() - timedelta(days=20), 2, "defiance"),
            (date.today() - timedelta(days=10), 3, "fighting"),
        ]
        
        for incident_date, severity, incident_type in incident_data:
            incident = models.DisciplineIncident(
                student_id=student_id,
                incident_date=incident_date,
                incident_type=incident_type,
                severity_level=severity,
                description=f"Test incident: {incident_type}"
            )
            db_session.add(incident)
        
        db_session.commit()

        # Test feature extraction
        extractor = DisciplineFeatureExtractor(db_session)
        features = extractor.extract(str(student_id), date.today())

        # Verify calculations
        assert "incident_count" in features
        assert "severity_mean" in features
        assert "severity_trend" in features
        assert features["incident_count"] == 3.0
        assert features["severity_mean"] == 2.0  # Mean of 1,2,3
        assert features["severity_trend"] > 0  # Escalating severity

    def test_incident_frequency_calculation(self, db_session, sample_student):
        """Test incident frequency calculation with real data."""
        student_id = sample_student.id
        
        # Create incidents with specific timing to test frequency (accounting for 7-day lag)
        incident_dates = [
            date.today() - timedelta(days=70),  # Account for lag
            date.today() - timedelta(days=40),  # Account for lag
            date.today() - timedelta(days=25),  # Account for lag
            date.today() - timedelta(days=15),  # Account for lag
        ]
        
        for incident_date in incident_dates:
            incident = models.DisciplineIncident(
                student_id=student_id,
                incident_date=incident_date,
                incident_type="test_incident",
                severity_level=2,
                description="Test incident for frequency"
            )
            db_session.add(incident)
        
        db_session.commit()

        extractor = DisciplineFeatureExtractor(db_session)
        features = extractor.extract(str(student_id), date.today())

        # Should show increasing frequency (positive acceleration)
        assert "incident_acceleration" in features
        assert features["incident_count"] == 4.0
        # With decreasing gaps between incidents, acceleration should be positive
        assert features["incident_acceleration"] > 0


class TestFeatureIntegrationPipeline:
    """Integration tests for complete feature extraction pipeline."""

    def test_complete_feature_pipeline(self, db_session, sample_student):
        """Test complete feature extraction using all extractors."""
        student_id = sample_student.id
        reference_date = date.today()
        
        # Create comprehensive test data (accounting for 7-day lag)
        # Attendance data
        for i in range(30):
            record_date = reference_date - timedelta(days=i + 20)  # Account for lag
            status = "present" if i % 4 != 0 else "absent"
            
            record = models.AttendanceRecord(
                student_id=student_id,
                date=record_date,
                status=status,
                period=1  # Add required period field
            )
            db_session.add(record)
        
        # Grade data  
        for i in range(10):
            grade_date = reference_date - timedelta(days=i * 5 + 20)  # Account for lag
            
            grade = models.Grade(
                student_id=student_id,
                submission_date=grade_date,
                grade_value=85 - i * 2,  # Slightly declining grades
                course_id=f"COURSE_{i % 3}",
                assignment_type="test"
            )
            db_session.add(grade)
        
        # Discipline data
        for i in range(3):
            incident_date = reference_date - timedelta(days=i * 15 + 20)  # Account for lag
            
            incident = models.DisciplineIncident(
                student_id=student_id,
                incident_date=incident_date,
                incident_type="test_incident",
                severity_level=1 + i,
                description=f"Test incident {i}"
            )
            db_session.add(incident)
        
        db_session.commit()

        # Extract features from all extractors
        extractors = [
            AttendanceFeatureExtractor(db_session),
            GradeFeatureExtractor(db_session), 
            DisciplineFeatureExtractor(db_session)
        ]
        
        all_features = {}
        for extractor in extractors:
            features = extractor.extract(str(student_id), reference_date)
            all_features.update(features)
        
        # Verify we got features from all extractors
        expected_feature_types = [
            "attendance_rate", "grade_mean", "incident_count"
        ]
        
        for feature_type in expected_feature_types:
            assert feature_type in all_features, f"Missing {feature_type} feature"
        
        # Verify all features have valid values
        for feature_name, feature_value in all_features.items():
            # Check for numeric values (including numpy types)
            assert isinstance(feature_value, (int, float, np.integer, np.floating)), f"{feature_name} is not numeric: {feature_value} (type: {type(feature_value)})"
            assert not np.isnan(float(feature_value)), f"{feature_name} is NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

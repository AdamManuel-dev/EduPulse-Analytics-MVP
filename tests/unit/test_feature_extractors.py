"""
Unit tests for feature extractors.
"""

import pytest
import numpy as np
from datetime import date, timedelta
from unittest.mock import Mock, MagicMock
from uuid import uuid4

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
        assert start == date(2025, 4, 27)  # 90 days before end
    
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
        
        assert stats['mean'] == 0.0
        assert stats['std'] == 0.0
        assert stats['min'] == 0.0
        assert stats['max'] == 0.0
        assert stats['trend'] == 0.0
    
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
        
        assert stats['mean'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['trend'] > 0  # Positive trend


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
            record.status = 'present' if i % 2 == 0 else 'absent'
            mock_records.append(record)
        
        mock_db_session.query().filter().all.return_value = mock_records
        
        extractor = AttendanceFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())
        
        # Check feature calculations
        assert 'attendance_rate' in features
        assert features['attendance_rate'] == 0.5  # 5 present out of 10
        assert features['absence_rate'] == 0.5  # 5 absent out of 10
        assert features['total_days_tracked'] == 10
    
    def test_feature_names(self, mock_db_session):
        """Test feature names list."""
        extractor = AttendanceFeatureExtractor(mock_db_session)
        names = extractor.get_feature_names()
        
        assert 'attendance_rate' in names
        assert 'absence_rate' in names
        assert 'tardy_rate' in names
        assert 'excused_rate' in names
        assert len(names) == 14  # Total expected features


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
        # Create mock grades
        mock_grades = []
        for i in range(5):
            grade = Mock(spec=models.Grade)
            grade.submission_date = date.today() - timedelta(days=i*7)
            grade.grade_value = 85.0 - (i * 5)  # Declining grades
            grade.course_id = 'MATH101'
            grade.assignment_type = 'test'
            mock_grades.append(grade)
        
        mock_db_session.query().filter().all.return_value = mock_grades
        
        extractor = GradeFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())
        
        # Check calculations
        assert 'gpa_current' in features
        assert 'grade_mean' in features
        assert features['grade_mean'] == 75.0  # Average of 85, 80, 75, 70, 65
        assert features['grade_trend'] < 0  # Negative trend (declining)
        assert features['total_grades_tracked'] == 5
    
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
            g.course_id = 'TEST'
            g.assignment_type = 'test'
        
        mock_db_session.query().filter().all.return_value = mock_grades
        
        extractor = GradeFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())
        
        assert features['failing_rate'] == 0.5  # 2 out of 4 failing


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
            incident.incident_date = date.today() - timedelta(days=i*10)
            incident.severity_level = i + 1  # Increasing severity
            incident.incident_type = f'type_{i}'
            mock_incidents.append(incident)
        
        mock_db_session.query().filter().order_by().all.return_value = mock_incidents
        
        extractor = DisciplineFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())
        
        # Check calculations
        assert features['incident_count'] == 3.0
        assert features['severity_mean'] == 2.0  # Average of 1, 2, 3
        assert features['severity_max'] == 3.0
        assert features['severity_trend'] > 0  # Positive trend (escalating)
    
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
            incident.incident_type = 'test'
            mock_incidents.append(incident)
        
        mock_db_session.query().filter().order_by().all.return_value = mock_incidents
        
        extractor = DisciplineFeatureExtractor(mock_db_session)
        features = extractor.extract(str(uuid4()), date.today())
        
        # Should show positive acceleration (incidents becoming more frequent)
        assert features['incident_acceleration'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
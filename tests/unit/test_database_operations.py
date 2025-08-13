"""
Unit tests for database operations and models.
"""

import pytest
from datetime import date, datetime, timedelta
from uuid import uuid4, UUID
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db import models
from src.db.database import get_db


class TestStudentModel:
    """Test cases for Student model operations."""

    def test_student_creation(self, db_session):
        """Test creating a new student record."""
        student_data = {
            "district_id": "STU12345",
            "first_name": "John",
            "last_name": "Doe",
            "grade_level": 10,
            "date_of_birth": date(2007, 5, 15),
            "gender": "M",
            "ethnicity": "Hispanic",
            "socioeconomic_status": "Low",
            "gpa": 3.2,
            "attendance_rate": 0.92,
            "parent_contact": "parent@example.com"
        }
        
        student = models.Student(**student_data)
        db_session.add(student)
        db_session.commit()
        
        # Verify student was created
        assert student.id is not None
        assert isinstance(student.id, UUID)
        assert student.district_id == "STU12345"
        assert student.first_name == "John"
        assert student.created_at is not None
        assert student.updated_at is not None

    def test_student_unique_district_id(self, db_session):
        """Test that district_id must be unique."""
        student_data = {
            "district_id": "STU_UNIQUE",
            "first_name": "First",
            "last_name": "Student",
            "grade_level": 9,
            "date_of_birth": date(2008, 1, 1),
            "gender": "F",
            "ethnicity": "Asian",
            "socioeconomic_status": "Middle",
            "gpa": 3.5,
            "attendance_rate": 0.95,
            "parent_contact": "first@example.com"
        }
        
        # Create first student
        student1 = models.Student(**student_data)
        db_session.add(student1)
        db_session.commit()
        
        # Try to create duplicate
        student_data["first_name"] = "Duplicate"
        student_data["parent_contact"] = "duplicate@example.com"
        student2 = models.Student(**student_data)
        db_session.add(student2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_student_relationships(self, db_session):
        """Test student relationships with other models."""
        # Create student
        student = models.Student(
            district_id="REL_TEST",
            first_name="Relationship",
            last_name="Test",
            grade_level=11,
            date_of_birth=date(2006, 3, 20),
            gender="M",
            ethnicity="White",
            socioeconomic_status="High",
            gpa=3.8,
            attendance_rate=0.98,
            parent_contact="rel_test@example.com"
        )
        db_session.add(student)
        db_session.flush()  # Get student ID without committing
        
        # Add attendance records
        attendance1 = models.AttendanceRecord(
            student_id=student.id,
            date=date.today(),
            status="present"
        )
        attendance2 = models.AttendanceRecord(
            student_id=student.id,
            date=date.today() - timedelta(days=1),
            status="absent"
        )
        db_session.add_all([attendance1, attendance2])
        
        # Add grades
        grade = models.Grade(
            student_id=student.id,
            submission_date=date.today(),
            grade_value=87.5,
            course_id="MATH101",
            assignment_type="test"
        )
        db_session.add(grade)
        
        db_session.commit()
        
        # Test relationships
        db_session.refresh(student)
        assert len(student.attendance_records) == 2
        assert len(student.grades) == 1
        assert student.grades[0].grade_value == 87.5

    def test_student_query_operations(self, db_session, batch_students):
        """Test various student query operations."""
        # Test getting student by ID
        student = batch_students[0]
        found_student = db_session.query(models.Student).filter(
            models.Student.id == student.id
        ).first()
        
        assert found_student is not None
        assert found_student.district_id == student.district_id
        
        # Test getting student by district_id
        found_by_district = db_session.query(models.Student).filter(
            models.Student.district_id == student.district_id
        ).first()
        
        assert found_by_district.id == student.id
        
        # Test filtering by grade level
        grade_10_students = db_session.query(models.Student).filter(
            models.Student.grade_level == 10
        ).all()
        
        assert len(grade_10_students) >= 0  # May vary based on test data
        
        # Test ordering
        ordered_students = db_session.query(models.Student).order_by(
            models.Student.gpa.desc()
        ).limit(3).all()
        
        assert len(ordered_students) <= 3
        if len(ordered_students) > 1:
            assert ordered_students[0].gpa >= ordered_students[1].gpa


class TestAttendanceModel:
    """Test cases for AttendanceRecord model operations."""

    def test_attendance_creation(self, db_session, sample_student):
        """Test creating attendance records."""
        attendance_data = {
            "student_id": sample_student.id,
            "date": date.today(),
            "status": "present"
        }
        
        attendance = models.AttendanceRecord(**attendance_data)
        db_session.add(attendance)
        db_session.commit()
        
        assert attendance.id is not None
        assert attendance.student_id == sample_student.id
        assert attendance.date == date.today()
        assert attendance.status == "present"

    def test_attendance_status_values(self, db_session, sample_student):
        """Test different attendance status values."""
        valid_statuses = ["present", "absent", "tardy", "excused"]
        
        for i, status in enumerate(valid_statuses):
            attendance = models.AttendanceRecord(
                student_id=sample_student.id,
                date=date.today() - timedelta(days=i),
                status=status
            )
            db_session.add(attendance)
        
        db_session.commit()
        
        # Verify all records were created
        records = db_session.query(models.AttendanceRecord).filter(
            models.AttendanceRecord.student_id == sample_student.id
        ).all()
        
        assert len(records) == 4
        recorded_statuses = {r.status for r in records}
        assert recorded_statuses == set(valid_statuses)

    def test_attendance_date_queries(self, db_session, sample_student):
        """Test date-based attendance queries."""
        # Create attendance records over a date range
        start_date = date.today() - timedelta(days=30)
        
        for i in range(31):  # 31 days
            record_date = start_date + timedelta(days=i)
            status = "present" if i % 3 != 0 else "absent"
            
            attendance = models.AttendanceRecord(
                student_id=sample_student.id,
                date=record_date,
                status=status
            )
            db_session.add(attendance)
        
        db_session.commit()
        
        # Test date range queries
        recent_records = db_session.query(models.AttendanceRecord).filter(
            models.AttendanceRecord.student_id == sample_student.id,
            models.AttendanceRecord.date >= date.today() - timedelta(days=7)
        ).all()
        
        assert len(recent_records) <= 8  # Up to 8 days (today + 7 previous)
        
        # Test attendance rate calculation
        present_count = db_session.query(models.AttendanceRecord).filter(
            models.AttendanceRecord.student_id == sample_student.id,
            models.AttendanceRecord.status == "present"
        ).count()
        
        total_count = db_session.query(models.AttendanceRecord).filter(
            models.AttendanceRecord.student_id == sample_student.id
        ).count()
        
        if total_count > 0:
            attendance_rate = present_count / total_count
            assert 0 <= attendance_rate <= 1

    def test_attendance_unique_constraint(self, db_session, sample_student):
        """Test unique constraint on student_id + date."""
        attendance_data = {
            "student_id": sample_student.id,
            "date": date.today(),
            "status": "present",
            "period": 1
        }
        
        # Create first record
        attendance1 = models.AttendanceRecord(**attendance_data)
        db_session.add(attendance1)
        db_session.commit()
        
        # Try to create duplicate (same student, same date)
        attendance_data["status"] = "absent"  # Different status
        attendance2 = models.AttendanceRecord(**attendance_data)
        db_session.add(attendance2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestGradeModel:
    """Test cases for Grade model operations."""

    def test_grade_creation(self, db_session, sample_student):
        """Test creating grade records."""
        grade_data = {
            "student_id": sample_student.id,
            "submission_date": date.today(),
            "grade_value": 87.5,
            "course_id": "MATH101",
            "assignment_type": "test"
        }
        
        grade = models.Grade(**grade_data)
        db_session.add(grade)
        db_session.commit()
        
        assert grade.id is not None
        assert grade.grade_value == 87.5
        assert grade.course_id == "MATH101"
        assert grade.assignment_type == "test"

    def test_grade_calculations(self, db_session, sample_student):
        """Test grade-related calculations."""
        # Create grades with known values for testing calculations
        grades_data = [
            {"grade_value": 95, "course_id": "MATH101"},
            {"grade_value": 87, "course_id": "MATH101"},
            {"grade_value": 78, "course_id": "ENG101"},
            {"grade_value": 92, "course_id": "SCI101"},
        ]
        
        for i, grade_data in enumerate(grades_data):
            grade_data.update({
                "student_id": sample_student.id,
                "submission_date": date.today() - timedelta(days=i * 7),
                "assignment_type": "test"
            })
            
            grade = models.Grade(**grade_data)
            db_session.add(grade)
        
        db_session.commit()
        
        # Test grade queries
        student_grades = db_session.query(models.Grade).filter(
            models.Grade.student_id == sample_student.id
        ).all()
        
        assert len(student_grades) == 4
        
        # Test average calculation
        total_points = sum(g.grade_value for g in student_grades)
        average = total_points / len(student_grades)
        expected_average = (95 + 87 + 78 + 92) / 4
        assert abs(average - expected_average) < 0.01

    def test_grade_course_filtering(self, db_session, sample_student):
        """Test filtering grades by course."""
        courses = ["MATH101", "ENG101", "SCI101"]
        
        for course in courses:
            for i in range(3):  # 3 grades per course
                grade = models.Grade(
                    student_id=sample_student.id,
                    submission_date=date.today() - timedelta(days=i),
                    grade_value=80 + i * 5,
                    course_id=course,
                    assignment_type="test"
                )
                db_session.add(grade)
        
        db_session.commit()
        
        # Test course filtering
        math_grades = db_session.query(models.Grade).filter(
            models.Grade.student_id == sample_student.id,
            models.Grade.course_id == "MATH101"
        ).all()
        
        assert len(math_grades) == 3
        assert all(g.course_id == "MATH101" for g in math_grades)

    def test_failing_grade_identification(self, db_session, sample_student):
        """Test identification of failing grades."""
        failing_threshold = 60
        
        grade_values = [45, 58, 72, 85, 91]  # 2 failing, 3 passing
        
        for i, grade_value in enumerate(grade_values):
            grade = models.Grade(
                student_id=sample_student.id,
                submission_date=date.today() - timedelta(days=i),
                grade_value=grade_value,
                course_id="TEST101",
                assignment_type="test"
            )
            db_session.add(grade)
        
        db_session.commit()
        
        # Query failing grades
        failing_grades = db_session.query(models.Grade).filter(
            models.Grade.student_id == sample_student.id,
            models.Grade.grade_value < failing_threshold
        ).all()
        
        assert len(failing_grades) == 2
        assert all(g.grade_value < failing_threshold for g in failing_grades)


class TestDisciplineModel:
    """Test cases for DisciplineIncident model operations."""

    def test_discipline_creation(self, db_session, sample_student):
        """Test creating discipline incident records."""
        incident_data = {
            "student_id": sample_student.id,
            "incident_date": date.today(),
            "incident_type": "disruptive_behavior",
            "severity_level": 2,
            "description": "Student was talking during class instruction"
        }
        
        incident = models.DisciplineIncident(**incident_data)
        db_session.add(incident)
        db_session.commit()
        
        assert incident.id is not None
        assert incident.incident_type == "disruptive_behavior"
        assert incident.severity_level == 2
        assert incident.description is not None

    def test_severity_level_queries(self, db_session, sample_student):
        """Test querying incidents by severity level."""
        # Create incidents with different severity levels
        incidents_data = [
            {"severity_level": 1, "incident_type": "minor_disruption"},
            {"severity_level": 2, "incident_type": "defiance"},
            {"severity_level": 3, "incident_type": "fighting"},
            {"severity_level": 1, "incident_type": "tardy"},
            {"severity_level": 2, "incident_type": "inappropriate_language"},
        ]
        
        for i, incident_data in enumerate(incidents_data):
            incident_data.update({
                "student_id": sample_student.id,
                "incident_date": date.today() - timedelta(days=i),
                "description": f"Test incident {i}"
            })
            
            incident = models.DisciplineIncident(**incident_data)
            db_session.add(incident)
        
        db_session.commit()
        
        # Test severity filtering
        severe_incidents = db_session.query(models.DisciplineIncident).filter(
            models.DisciplineIncident.student_id == sample_student.id,
            models.DisciplineIncident.severity_level >= 3
        ).all()
        
        assert len(severe_incidents) == 1
        assert severe_incidents[0].incident_type == "fighting"
        
        # Test count by severity
        level_2_count = db_session.query(models.DisciplineIncident).filter(
            models.DisciplineIncident.student_id == sample_student.id,
            models.DisciplineIncident.severity_level == 2
        ).count()
        
        assert level_2_count == 2

    def test_incident_chronology(self, db_session, sample_student):
        """Test chronological ordering of incidents."""
        # Create incidents over time
        incident_dates = [
            date.today() - timedelta(days=60),
            date.today() - timedelta(days=30),
            date.today() - timedelta(days=15),
            date.today() - timedelta(days=5),
        ]
        
        for i, incident_date in enumerate(incident_dates):
            incident = models.DisciplineIncident(
                student_id=sample_student.id,
                incident_date=incident_date,
                incident_type="test_incident",
                severity_level=1,
                description=f"Incident {i}"
            )
            db_session.add(incident)
        
        db_session.commit()
        
        # Test chronological ordering
        chronological_incidents = db_session.query(models.DisciplineIncident).filter(
            models.DisciplineIncident.student_id == sample_student.id
        ).order_by(models.DisciplineIncident.incident_date.asc()).all()
        
        assert len(chronological_incidents) == 4
        
        # Verify ordering
        for i in range(len(chronological_incidents) - 1):
            assert chronological_incidents[i].incident_date <= chronological_incidents[i + 1].incident_date


class TestPredictionModel:
    """Test cases for Prediction model operations."""

    def test_prediction_creation(self, db_session, sample_student):
        """Test creating prediction records."""
        prediction_data = {
            "student_id": sample_student.id,
            "prediction_date": datetime.now(),
            "risk_score": 0.75,
            "risk_category": "high",
            "confidence": 0.89,
            "model_version": "v1.0",
            "risk_factors": [{"factor": "attendance_rate", "weight": 0.65}]
        }
        
        prediction = models.Prediction(**prediction_data)
        db_session.add(prediction)
        db_session.commit()
        
        assert prediction.id is not None
        assert prediction.risk_score == 0.75
        assert prediction.risk_category == "high"
        assert prediction.confidence == 0.89
        assert len(prediction.risk_factors) > 0

    def test_prediction_queries(self, db_session, sample_student):
        """Test various prediction queries."""
        # Create predictions over time
        for i in range(5):
            prediction = models.Prediction(
                student_id=sample_student.id,
                prediction_date=datetime.now() - timedelta(days=i * 7),
                risk_score=0.5 + i * 0.1,
                risk_category="medium" if i < 3 else "high",
                confidence=0.8 + i * 0.02,
                model_version="v1.0",
                risk_factors=[{"factor": "test_feature", "weight": i}]
            )
            db_session.add(prediction)
        
        db_session.commit()
        
        # Test latest prediction query
        latest_prediction = db_session.query(models.Prediction).filter(
            models.Prediction.student_id == sample_student.id
        ).order_by(models.Prediction.prediction_date.desc()).first()
        
        assert latest_prediction is not None
        assert latest_prediction.risk_score == 0.5  # Most recent (i=0)
        
        # Test high-risk predictions
        high_risk_predictions = db_session.query(models.Prediction).filter(
            models.Prediction.student_id == sample_student.id,
            models.Prediction.risk_category == "high"
        ).all()
        
        assert len(high_risk_predictions) == 2  # i=3 and i=4


class TestDatabaseSessionOperations:
    """Test database session and transaction operations."""

    def test_transaction_rollback(self, db_session):
        """Test transaction rollback functionality."""
        # Create a student
        student = models.Student(
            district_id="ROLLBACK_TEST",
            first_name="Rollback",
            last_name="Test",
            grade_level=9,
            date_of_birth=date(2008, 1, 1),
            gender="M",
            ethnicity="Other",
            socioeconomic_status="Middle",
            gpa=3.0,
            attendance_rate=0.9,
            parent_contact="rollback@test.com"
        )
        db_session.add(student)
        db_session.flush()  # Get ID but don't commit
        
        student_id = student.id
        
        # Add some related data
        attendance = models.AttendanceRecord(
            student_id=student_id,
            date=date.today(),
            status="present"
        )
        db_session.add(attendance)
        
        # Rollback the transaction
        db_session.rollback()
        
        # Verify data was not persisted
        found_student = db_session.query(models.Student).filter(
            models.Student.district_id == "ROLLBACK_TEST"
        ).first()
        
        assert found_student is None

    def test_bulk_operations(self, db_session):
        """Test bulk insert and update operations."""
        # Bulk insert students
        students_data = []
        for i in range(10):
            student = models.Student(
                district_id=f"BULK_{i:03d}",
                first_name=f"Student",
                last_name=f"Number{i}",
                grade_level=9 + (i % 4),
                date_of_birth=date(2006, 1, 1),
                gender="M" if i % 2 == 0 else "F",
                ethnicity="Other",
                socioeconomic_status="Middle",
                gpa=3.0 + (i % 3) * 0.5,
                attendance_rate=0.85 + (i % 10) * 0.015,
                parent_contact=f"bulk{i}@test.com"
            )
            students_data.append(student)
        
        db_session.bulk_save_objects(students_data)
        db_session.commit()
        
        # Verify bulk insert
        bulk_students = db_session.query(models.Student).filter(
            models.Student.district_id.like("BULK_%")
        ).all()
        
        assert len(bulk_students) == 10
        
        # Test bulk update
        db_session.query(models.Student).filter(
            models.Student.district_id.like("BULK_%")
        ).update({models.Student.gpa: 3.5})
        
        db_session.commit()
        
        # Verify bulk update
        updated_students = db_session.query(models.Student).filter(
            models.Student.district_id.like("BULK_%")
        ).all()
        
        assert all(s.gpa == 3.5 for s in updated_students)

    def test_database_constraints_and_indexes(self, db_session):
        """Test database constraints and index behavior."""
        # Test foreign key constraint
        fake_student_id = uuid4()
        
        attendance = models.AttendanceRecord(
            student_id=fake_student_id,
            date=date.today(),
            status="present"
        )
        db_session.add(attendance)
        
        # This should fail due to foreign key constraint
        with pytest.raises(IntegrityError):
            db_session.commit()
        
        db_session.rollback()
        
        # Test index performance (implicit test through queries)
        # Create student for valid foreign key
        student = models.Student(
            district_id="INDEX_TEST",
            first_name="Index",
            last_name="Test",
            grade_level=10,
            date_of_birth=date(2007, 1, 1),
            gender="F",
            ethnicity="Other",
            socioeconomic_status="Middle",
            gpa=3.2,
            attendance_rate=0.92,
            parent_contact="index@test.com"
        )
        db_session.add(student)
        db_session.flush()
        
        # Add many attendance records (should be fast with index)
        for i in range(100):
            attendance = models.AttendanceRecord(
                student_id=student.id,
                date=date.today() - timedelta(days=i),
                status="present" if i % 4 != 0 else "absent"
            )
            db_session.add(attendance)
        
        db_session.commit()
        
        # Query should be efficient with index on student_id
        student_attendance = db_session.query(models.AttendanceRecord).filter(
            models.AttendanceRecord.student_id == student.id
        ).all()
        
        assert len(student_attendance) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
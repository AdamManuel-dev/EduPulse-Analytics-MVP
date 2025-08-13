"""
SQLAlchemy ORM models for EduPulse Analytics.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Date, DateTime, Boolean,
    ForeignKey, JSON, ARRAY, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from .database import Base


class Student(Base):
    __tablename__ = 'students'
    __table_args__ = {'schema': 'edupulse'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    district_id = Column(String(50), unique=True, nullable=False)
    grade_level = Column(Integer, CheckConstraint('grade_level >= 0 AND grade_level <= 12'))
    enrollment_date = Column(Date, nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    features = relationship("StudentFeature", back_populates="student", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="student", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="student", cascade="all, delete-orphan")
    grades = relationship("Grade", back_populates="student", cascade="all, delete-orphan")
    discipline_incidents = relationship("DisciplineIncident", back_populates="student", cascade="all, delete-orphan")


class StudentFeature(Base):
    __tablename__ = 'student_features'
    __table_args__ = (
        UniqueConstraint('student_id', 'feature_date'),
        {'schema': 'edupulse'}
    )

    student_id = Column(UUID(as_uuid=True), ForeignKey('edupulse.students.id'), primary_key=True)
    feature_date = Column(Date, primary_key=True, nullable=False)
    attendance_rate = Column(Float, CheckConstraint('attendance_rate >= 0 AND attendance_rate <= 1'))
    gpa_current = Column(Float, CheckConstraint('gpa_current >= 0 AND gpa_current <= 5'))
    discipline_incidents = Column(Integer, CheckConstraint('discipline_incidents >= 0'), default=0)
    feature_vector = Column(ARRAY(Float))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    student = relationship("Student", back_populates="features")


class Prediction(Base):
    __tablename__ = 'predictions'
    __table_args__ = {'schema': 'edupulse'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey('edupulse.students.id'))
    prediction_date = Column(DateTime(timezone=True), server_default=func.now())
    risk_score = Column(Float, CheckConstraint('risk_score >= 0 AND risk_score <= 1'))
    risk_category = Column(String(20), CheckConstraint("risk_category IN ('low', 'medium', 'high', 'critical')"))
    confidence = Column(Float, CheckConstraint('confidence >= 0 AND confidence <= 1'))
    risk_factors = Column(JSON, default=[])
    model_version = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    student = relationship("Student", back_populates="predictions")
    feedback = relationship("TrainingFeedback", back_populates="prediction", cascade="all, delete-orphan")


class TrainingFeedback(Base):
    __tablename__ = 'training_feedback'
    __table_args__ = {'schema': 'edupulse'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey('edupulse.predictions.id'))
    outcome_date = Column(Date)
    outcome_type = Column(String(50))
    feedback_type = Column(String(50), CheckConstraint(
        "feedback_type IN ('true_positive', 'false_positive', 'true_negative', 'false_negative')"
    ))
    educator_notes = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    prediction = relationship("Prediction", back_populates="feedback")


class AttendanceRecord(Base):
    __tablename__ = 'attendance_records'
    __table_args__ = (
        UniqueConstraint('student_id', 'date', 'period'),
        {'schema': 'edupulse'}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey('edupulse.students.id'))
    date = Column(Date, nullable=False)
    status = Column(String(20), CheckConstraint("status IN ('present', 'absent', 'tardy', 'excused')"))
    period = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    student = relationship("Student", back_populates="attendance_records")


class Grade(Base):
    __tablename__ = 'grades'
    __table_args__ = {'schema': 'edupulse'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey('edupulse.students.id'))
    course_id = Column(String(50), nullable=False)
    course_name = Column(String(200))
    grade_value = Column(Float, CheckConstraint('grade_value >= 0 AND grade_value <= 100'))
    grade_letter = Column(String(2))
    submission_date = Column(Date, nullable=False)
    assignment_type = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    student = relationship("Student", back_populates="grades")


class DisciplineIncident(Base):
    __tablename__ = 'discipline_incidents'
    __table_args__ = {'schema': 'edupulse'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey('edupulse.students.id'))
    incident_date = Column(Date, nullable=False)
    severity_level = Column(Integer, CheckConstraint('severity_level >= 1 AND severity_level <= 5'))
    incident_type = Column(String(100))
    description = Column(String)
    resolution = Column(String(200))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    student = relationship("Student", back_populates="discipline_incidents")


class ModelMetadata(Base):
    __tablename__ = 'model_metadata'
    __table_args__ = {'schema': 'edupulse'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version = Column(String(50), unique=True, nullable=False)
    model_type = Column(String(50))
    training_date = Column(DateTime(timezone=True), server_default=func.now())
    performance_metrics = Column(JSON, default={})
    hyperparameters = Column(JSON, default={})
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
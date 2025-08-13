"""
@fileoverview Pydantic data models for API serialization and validation
@lastmodified 2025-08-13T00:50:05-05:00

Features: Student/feature/prediction schemas, enums, request/response models, validation
Main APIs: Student, Prediction, PredictRequest, BatchPredictRequest, TrainingFeedback
Constraints: Pydantic v2, UUID fields, field validators, from_attributes config
Patterns: Base/Create/Update pattern, enum validation, field constraints, nested models
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from uuid import UUID
from enum import Enum


class RiskCategory(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttendanceStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    TARDY = "tardy"
    EXCUSED = "excused"


class FeedbackType(str, Enum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_NEGATIVE = "false_negative"


# Base schemas
class StudentBase(BaseModel):
    district_id: str = Field(..., max_length=50)
    first_name: str = Field(..., max_length=100)
    last_name: str = Field(..., max_length=100)
    grade_level: int = Field(..., ge=0, le=12)
    date_of_birth: date
    gender: Optional[str] = Field(None, max_length=10)
    ethnicity: Optional[str] = Field(None, max_length=100)
    socioeconomic_status: Optional[str] = Field(None, max_length=50)
    gpa: Optional[float] = Field(None, ge=0, le=4.0)
    attendance_rate: Optional[float] = Field(None, ge=0, le=1.0)
    parent_contact: Optional[str] = Field(None, max_length=200)
    student_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class StudentCreate(StudentBase):
    enrollment_date: Optional[date] = None


class StudentUpdate(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    grade_level: Optional[int] = Field(None, ge=0, le=12)
    date_of_birth: Optional[date] = None
    gender: Optional[str] = Field(None, max_length=10)
    ethnicity: Optional[str] = Field(None, max_length=100)
    socioeconomic_status: Optional[str] = Field(None, max_length=50)
    gpa: Optional[float] = Field(None, ge=0, le=4.0)
    attendance_rate: Optional[float] = Field(None, ge=0, le=1.0)
    parent_contact: Optional[str] = Field(None, max_length=200)
    student_metadata: Optional[Dict[str, Any]] = None


class Student(StudentBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    enrollment_date: Optional[date] = None
    created_at: datetime
    updated_at: datetime


# Feature schemas
class StudentFeatureBase(BaseModel):
    attendance_rate: float = Field(..., ge=0, le=1)
    gpa_current: float = Field(..., ge=0, le=5)
    discipline_incidents: int = Field(default=0, ge=0)
    feature_vector: Optional[List[float]] = None


class StudentFeatureCreate(StudentFeatureBase):
    student_id: UUID
    feature_date: date


class StudentFeature(StudentFeatureBase):
    model_config = ConfigDict(from_attributes=True)

    student_id: UUID
    feature_date: date
    created_at: datetime


# Prediction schemas
class RiskFactor(BaseModel):
    factor: str
    weight: float
    details: str


class PredictionBase(BaseModel):
    risk_score: float = Field(..., ge=0, le=1)
    risk_category: RiskCategory
    confidence: float = Field(..., ge=0, le=1)
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    model_version: str


class PredictionCreate(PredictionBase):
    student_id: UUID


class Prediction(PredictionBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    student_id: UUID
    prediction_date: datetime
    created_at: datetime


# Training feedback schemas
class TrainingFeedbackBase(BaseModel):
    outcome_date: date
    outcome_type: str
    feedback_type: FeedbackType
    educator_notes: Optional[str] = None


class TrainingFeedbackCreate(TrainingFeedbackBase):
    prediction_id: UUID


class TrainingFeedback(TrainingFeedbackBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    prediction_id: UUID
    created_at: datetime


# Attendance schemas
class AttendanceRecordBase(BaseModel):
    date: date
    status: AttendanceStatus
    period: Optional[int] = None


class AttendanceRecordCreate(AttendanceRecordBase):
    student_id: UUID


class AttendanceRecord(AttendanceRecordBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    student_id: UUID
    created_at: datetime


# Grade schemas
class GradeBase(BaseModel):
    course_id: str = Field(..., max_length=50)
    course_name: Optional[str] = Field(None, max_length=200)
    grade_value: float = Field(..., ge=0, le=100)
    grade_letter: Optional[str] = Field(None, max_length=2)
    submission_date: date
    assignment_type: Optional[str] = Field(None, max_length=50)


class GradeCreate(GradeBase):
    student_id: UUID


class Grade(GradeBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    student_id: UUID
    created_at: datetime


# Discipline incident schemas
class DisciplineIncidentBase(BaseModel):
    incident_date: date
    severity_level: int = Field(..., ge=1, le=5)
    incident_type: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    resolution: Optional[str] = Field(None, max_length=200)


class DisciplineIncidentCreate(DisciplineIncidentBase):
    student_id: UUID


class DisciplineIncident(DisciplineIncidentBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    student_id: UUID
    created_at: datetime


# Model metadata schemas
class ModelMetadataBase(BaseModel):
    model_version: str = Field(..., max_length=50)
    model_type: Optional[str] = Field(None, max_length=50)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = False


class ModelMetadataCreate(ModelMetadataBase):
    pass


class ModelMetadata(ModelMetadataBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    training_date: datetime
    created_at: datetime


# API Request/Response schemas
class PredictRequest(BaseModel):
    student_id: UUID
    date_range: Optional[Dict[str, date]] = None
    include_factors: bool = True


class BatchPredictRequest(BaseModel):
    student_ids: List[UUID]
    top_k: int = Field(default=10, gt=0)


class PredictResponse(BaseModel):
    prediction: Prediction
    contributing_factors: Optional[List[RiskFactor]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    processing_time_ms: float


class TrainingUpdateRequest(BaseModel):
    student_outcomes: List[Dict[str, Any]]
    feedback_corrections: List[Dict[str, Any]]
    update_mode: str = "incremental"


class TrainingUpdateResponse(BaseModel):
    update_id: UUID
    status: str
    estimated_completion: datetime


class MetricsResponse(BaseModel):
    performance_metrics: Dict[str, float]
    data_coverage: Dict[str, int]


# Legacy alias schemas for backward compatibility
PredictionRequest = PredictRequest
PredictionResponse = PredictResponse
TrainingConfig = TrainingUpdateRequest


class RiskLevel(str, Enum):
    """Legacy enum for risk level (use RiskCategory instead)."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

"""
Pytest configuration and shared fixtures for all tests.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Generator, Dict
from uuid import uuid4

import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app  # noqa: E402
from src.db.database import Base, get_db  # noqa: E402
from src.db.models import Student, AttendanceRecord, Grade, DisciplineIncident  # noqa: E402
from src.config.settings import Settings, get_settings  # noqa: E402
from src.models.gru_model import GRUAttentionModel  # noqa: E402


# Test database URL
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Override settings for testing."""
    return Settings(
        database_url="postgresql://user:pass@localhost/testdb",  # Valid postgres URL for validation
        redis_url="redis://localhost:6379/15",  # Use test redis db
        environment="testing",  # Valid environment value
        secret_key="test-secret-key-for-testing-only",
        jwt_secret_key="test-jwt-secret-key-for-testing-only",
        celery_broker_url="redis://localhost:6379/15",
        celery_result_backend="redis://localhost:6379/15",
        log_level="DEBUG",
    )


@pytest.fixture(scope="session")
def engine():
    """Create test database engine."""
    from sqlalchemy import event
    
    engine = create_engine(
        TEST_DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
    
    # Enable foreign key constraints for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        if 'sqlite' in engine.url.drivername:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(engine) -> Generator[Session, None, None]:
    """Create a test database session."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    yield session

    session.rollback()
    session.close()


@pytest.fixture(scope="function")
def client(db_session, test_settings, engine) -> TestClient:
    """Create test client with overridden dependencies."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from src.api.routes import health, predictions, students, training
    
    # Create a test-specific FastAPI app without lifespan
    test_app = FastAPI(
        title="EduPulse Analytics API (Test)",
        description="Test version of EduPulse API",
        version="1.0.0-test"
    )
    
    # Configure CORS
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    test_app.include_router(health.router, tags=["health"])
    test_app.include_router(students.router, prefix="/api/v1/students", tags=["students"])
    test_app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
    test_app.include_router(training.router, prefix="/api/v1/train", tags=["training"])
    
    @test_app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "EduPulse Analytics API (Test)", "version": "1.0.0-test", "docs": "/docs"}
    
    # Ensure tables are created
    from src.db.database import Base
    Base.metadata.create_all(bind=engine)
    
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    def override_get_settings():
        return test_settings

    # Override dependencies
    test_app.dependency_overrides[get_db] = override_get_db
    test_app.dependency_overrides[get_settings] = override_get_settings

    with TestClient(test_app) as test_client:
        yield test_client

    test_app.dependency_overrides.clear()


# Remove async_client since we're using the standard client for e2e tests


@pytest.fixture
def sample_student(db_session) -> Student:
    """Create a sample student."""
    import uuid
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for uniqueness
    student = Student(
        district_id=f"STU{unique_id}",
        first_name="Test",
        last_name="Student",
        grade_level=10,
        enrollment_date=date.today() - timedelta(days=365),
        date_of_birth=date(2008, 5, 15),
        gender="M",
        ethnicity="Hispanic",
        socioeconomic_status="middle",
    )
    db_session.add(student)
    db_session.commit()
    db_session.refresh(student)
    return student


@pytest.fixture
def sample_attendance_records(db_session, sample_student) -> list[AttendanceRecord]:
    """Create sample attendance records."""
    records = []
    for i in range(30):
        date_val = date.today() - timedelta(days=i)
        status = np.random.choice(
            ["present", "absent", "tardy", "excused"], p=[0.7, 0.15, 0.1, 0.05]
        )

        record = AttendanceRecord(
            student_id=sample_student.id,
            date=date_val,
            status=status,
            period=1,
        )
        records.append(record)
        db_session.add(record)

    db_session.commit()
    return records


@pytest.fixture
def sample_grades(db_session, sample_student) -> list[Grade]:
    """Create sample grade records."""
    grades = []
    courses = ["MATH101", "ENG101", "SCI101", "HIST101"]
    assignment_types = ["homework", "quiz", "test", "project"]

    for course in courses:
        for i in range(10):
            grade = Grade(
                student_id=sample_student.id,
                course_id=course,
                assignment_type=np.random.choice(assignment_types),
                grade_value=np.random.uniform(60, 100),
                submission_date=date.today() - timedelta(days=i * 7),
            )
            grades.append(grade)
            db_session.add(grade)

    db_session.commit()
    return grades


@pytest.fixture
def sample_discipline_incidents(db_session, sample_student) -> list[DisciplineIncident]:
    """Create sample discipline incidents."""
    incidents = []
    incident_types = ["tardiness", "disruption", "insubordination", "fighting"]

    for i in range(5):
        incident = DisciplineIncident(
            student_id=sample_student.id,
            incident_date=date.today() - timedelta(days=i * 30),
            incident_type=np.random.choice(incident_types),
            severity_level=np.random.randint(1, 4),
            description=f"Test incident {i}",
        )
        incidents.append(incident)
        db_session.add(incident)

    db_session.commit()
    return incidents


@pytest.fixture
def mock_ml_model():
    """Create a mock ML model for testing."""
    model = GRUAttentionModel(
        input_size=42, hidden_size=64, num_layers=1, num_heads=2, dropout=0.1, bidirectional=False
    )
    model.eval()
    return model


@pytest.fixture
def sample_feature_vector():
    """Create a sample feature vector for testing."""
    return np.random.randn(1, 10, 42).astype(np.float32)  # batch_size=1, seq_len=10, features=42


@pytest.fixture
def auth_headers(client) -> Dict[str, str]:
    """Create authentication headers for testing."""
    # In a real app, you'd generate a JWT token here
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def batch_students(db_session) -> list[Student]:
    """Create multiple students for batch testing."""
    students = []
    for i in range(10):
        student = Student(
            district_id=f"STU{i:05d}",
            first_name=f"Student{i}",
            last_name="Test",
            grade_level=9 + (i % 4),
            enrollment_date=date.today() - timedelta(days=365),
            date_of_birth=date(2008 + (i % 3), (i % 12) + 1, (i % 28) + 1),
            gender="M" if i % 2 == 0 else "F",
            ethnicity=np.random.choice(["White", "Black", "Hispanic", "Asian"]),
            socioeconomic_status=np.random.choice(["low", "middle", "high"]),
        )
        students.append(student)
        db_session.add(student)

    db_session.commit()
    return students


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset any singleton instances between tests."""
    # Add any singleton resets here if needed
    yield


@pytest.fixture
def mock_redis(monkeypatch):
    """Mock Redis for testing."""

    class MockRedis:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value, ex=None):
            self.data[key] = value
            return True

        def delete(self, key):
            if key in self.data:
                del self.data[key]
                return 1
            return 0

        def exists(self, key):
            return key in self.data

    mock_redis_instance = MockRedis()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_redis_instance)
    return mock_redis_instance

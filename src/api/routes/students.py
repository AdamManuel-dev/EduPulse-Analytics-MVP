"""
@fileoverview Student CRUD API endpoints for managing student records
@lastmodified 2025-08-13T02:56:19-05:00

Features: Create, read, update, delete student records with validation and pagination
Main APIs: create_student(), get_student(), list_students(), update_student(), delete_student()
Constraints: Requires UUID student ID, unique district_id, SQLAlchemy session
Patterns: FastAPI dependency injection, Pydantic validation, HTTP status codes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from src.db.database import get_db
from src.db import models
from src.models import schemas

router = APIRouter()


@router.post("/", response_model=schemas.Student)
async def create_student(student: schemas.StudentCreate, db: Session = Depends(get_db)):
    """
    Create a new student record in the system.

    Validates that no student with the same district_id already exists,
    then creates and persists the new student record to the database.

    Args:
        student: Student creation data including required fields like district_id,
                first_name, last_name, and optional demographic information
        db: Database session for validation and persistence

    Returns:
        Student: The created student record with generated UUID and timestamps

    Raises:
        HTTPException: 400 if student with district_id already exists

    Examples:
        >>> student_data = StudentCreate(district_id="STU123", first_name="John", last_name="Doe")
        >>> new_student = await create_student(student_data, db)
        >>> print(new_student.id)
        550e8400-e29b-41d4-a716-446655440000
    """
    # Check if student with district_id already exists
    existing = (
        db.query(models.Student).filter(models.Student.district_id == student.district_id).first()
    )

    if existing:
        raise HTTPException(status_code=400, detail="Student with this district ID already exists")

    db_student = models.Student(**student.dict())
    db.add(db_student)
    db.commit()
    db.refresh(db_student)

    return db_student


@router.get("/{student_id}", response_model=schemas.Student)
async def get_student(student_id: UUID, db: Session = Depends(get_db)):
    """
    Retrieve a single student record by their unique identifier.

    Looks up a student in the database using their UUID and returns
    the complete student record if found.

    Args:
        student_id: UUID of the student to retrieve
        db: Database session for student lookup

    Returns:
        Student: Complete student record with all available fields

    Raises:
        HTTPException: 404 if student with the given ID is not found

    Examples:
        >>> student_uuid = UUID("550e8400-e29b-41d4-a716-446655440000")
        >>> student = await get_student(student_uuid, db)
        >>> print(student.district_id)
        STU123
    """
    student = db.query(models.Student).filter(models.Student.id == student_id).first()

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    return student


@router.get("/", response_model=List[schemas.Student])
async def list_students(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve a paginated list of all students in the system.

    Returns student records with pagination support to handle large datasets
    efficiently. Results are ordered by creation date.

    Args:
        skip: Number of records to skip for pagination (default: 0)
        limit: Maximum number of records to return (default: 100, max: 1000)
        db: Database session for student queries

    Returns:
        List[Student]: List of student records within the specified range

    Examples:
        >>> # Get first 10 students
        >>> students = await list_students(skip=0, limit=10, db=db)
        >>> print(len(students))
        10
        >>> # Get next page
        >>> next_students = await list_students(skip=10, limit=10, db=db)
    """
    students = db.query(models.Student).offset(skip).limit(limit).all()
    return students


@router.patch("/{student_id}", response_model=schemas.Student)
async def update_student(
    student_id: UUID, student_update: schemas.StudentUpdate, db: Session = Depends(get_db)
):
    """
    Update specific fields of an existing student record.

    Performs a partial update using PATCH semantics, updating only the
    fields provided in the request while leaving other fields unchanged.

    Args:
        student_id: UUID of the student to update
        student_update: Partial student data with only fields to be updated
        db: Database session for student lookup and persistence

    Returns:
        Student: The updated student record with all current field values

    Raises:
        HTTPException: 404 if student with the given ID is not found

    Examples:
        >>> update_data = StudentUpdate(grade_level=11, gpa=3.8)
        >>> updated_student = await update_student(student_uuid, update_data, db)
        >>> print(updated_student.grade_level)
        11
    """
    student = db.query(models.Student).filter(models.Student.id == student_id).first()

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    update_data = student_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(student, field, value)

    db.commit()
    db.refresh(student)

    return student


@router.delete("/{student_id}")
async def delete_student(student_id: UUID, db: Session = Depends(get_db)):
    """
    Permanently delete a student record and all associated data.
    
    Removes the student record from the database along with all related
    data including predictions, attendance records, grades, and discipline
    incidents through cascade deletion.
    
    Args:
        student_id: UUID of the student to delete
        db: Database session for student lookup and deletion
        
    Returns:
        dict: Confirmation message with the deleted student's district_id
        
    Raises:
        HTTPException: 404 if student with the given ID is not found
        
    Examples:
        >>> result = await delete_student(student_uuid, db)
        >>> print(result["message"])
        Student STU123 deleted successfully
        
    Warning:
        This operation cannot be undone. All associated student data
        including predictions, grades, and attendance records will be
        permanently removed.
    """
    student = db.query(models.Student).filter(models.Student.id == student_id).first()
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    district_id = student.district_id
    db.delete(student)
    db.commit()
    
    return {"message": f"Student {district_id} deleted successfully"}

"""
Student management endpoints.
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
async def create_student(
    student: schemas.StudentCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new student record.
    """
    # Check if student with district_id already exists
    existing = db.query(models.Student).filter(
        models.Student.district_id == student.district_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Student with this district ID already exists")
    
    db_student = models.Student(**student.dict())
    db.add(db_student)
    db.commit()
    db.refresh(db_student)
    
    return db_student


@router.get("/{student_id}", response_model=schemas.Student)
async def get_student(
    student_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a student by ID.
    """
    student = db.query(models.Student).filter(models.Student.id == student_id).first()
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    return student


@router.get("/", response_model=List[schemas.Student])
async def list_students(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all students with pagination.
    """
    students = db.query(models.Student).offset(skip).limit(limit).all()
    return students


@router.patch("/{student_id}", response_model=schemas.Student)
async def update_student(
    student_id: UUID,
    student_update: schemas.StudentUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a student's information.
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
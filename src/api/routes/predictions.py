"""
Prediction endpoints for risk assessment.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID, uuid4
from datetime import datetime

from src.db.database import get_db
from src.db import models
from src.models import schemas
from src.config.settings import get_settings
from src.services.prediction_service import prediction_service

settings = get_settings()

router = APIRouter()


@router.post("/predict", response_model=schemas.PredictResponse)
async def predict_single(
    request: schemas.PredictRequest,
    db: Session = Depends(get_db)
):
    """
    Generate risk prediction for a single student.
    """
    # Check if student exists
    student = db.query(models.Student).filter(
        models.Student.id == request.student_id
    ).first()
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Use real ML prediction service
    try:
        response = prediction_service.predict_risk(
            student_id=str(request.student_id),
            reference_date=request.date_range.get('end') if request.date_range else None,
            include_factors=request.include_factors
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=schemas.BatchPredictResponse)
async def predict_batch(
    request: schemas.BatchPredictRequest,
    db: Session = Depends(get_db)
):
    """
    Generate predictions for multiple students.
    """
    if len(request.student_ids) > settings.max_prediction_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {settings.max_prediction_batch_size}"
        )
    
    # Verify all students exist
    students = db.query(models.Student).filter(
        models.Student.id.in_(request.student_ids)
    ).all()
    
    if len(students) != len(request.student_ids):
        found_ids = {str(s.id) for s in students}
        missing_ids = [str(sid) for sid in request.student_ids if str(sid) not in found_ids]
        raise HTTPException(
            status_code=404,
            detail=f"Students not found: {missing_ids}"
        )
    
    # Use real ML prediction service for batch
    try:
        response = prediction_service.predict_batch(
            student_ids=[str(sid) for sid in request.student_ids],
            top_k=request.top_k
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/metrics", response_model=schemas.MetricsResponse)
async def get_metrics(
    start_date: str = None,
    end_date: str = None,
    db: Session = Depends(get_db)
):
    """
    Get model performance metrics.
    """
    # TODO: Implement actual metrics calculation
    # For now, return mock metrics
    
    total_students = db.query(models.Student).count()
    total_predictions = db.query(models.Prediction).count()
    
    return schemas.MetricsResponse(
        performance_metrics={
            "precision_at_10": 0.87,
            "recall_at_10": 0.82,
            "average_lead_time_days": 68,
            "false_positive_rate": 0.15
        },
        data_coverage={
            "total_students": total_students,
            "predictions_made": total_predictions,
            "outcomes_tracked": 0  # TODO: Implement
        }
    )
"""
Prediction service for real-time ML inference.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, date, timedelta
from uuid import UUID
import os
from pathlib import Path

from src.models.gru_model import GRUAttentionModel
from src.features.pipeline import FeaturePipeline
from src.config.settings import get_settings
from src.db.database import get_db
from src.db import models as db_models
from src.models import schemas

settings = get_settings()


class PredictionService:
    """
    Service for making student risk predictions using trained ML models.
    """
    
    def __init__(self):
        self.model = None
        self.feature_pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(settings.model_path) / 'best_model.pt'
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self) -> None:
        """
        Load the trained model from disk.
        """
        try:
            # Initialize model architecture
            self.model = GRUAttentionModel(
                input_size=42,
                hidden_size=128,
                num_layers=2,
                num_heads=4,
                dropout=0.3,
                bidirectional=True
            )
            
            # Load weights if model file exists
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {self.model_path}")
            else:
                print(f"No saved model found at {self.model_path}, using random weights")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Initialize with random weights as fallback
            self.model = GRUAttentionModel().to(self.device)
            self.model.eval()
    
    def prepare_sequence(
        self, 
        student_id: str,
        reference_date: Optional[date] = None,
        sequence_length: int = 20
    ) -> torch.Tensor:
        """
        Prepare input sequence for a student.
        
        Args:
            student_id: Student UUID
            reference_date: End date for sequence (defaults to today)
            sequence_length: Number of weeks in sequence
            
        Returns:
            Input tensor for model
        """
        if reference_date is None:
            reference_date = datetime.now().date()
        
        # Initialize feature pipeline
        with get_db() as db:
            if self.feature_pipeline is None:
                self.feature_pipeline = FeaturePipeline(db)
            
            # Extract features for sequence of dates
            sequence_features = []
            
            for i in range(sequence_length):
                # Calculate date for this point in sequence
                seq_date = reference_date - timedelta(weeks=i)
                
                # Extract features
                features = self.feature_pipeline.extract_features(
                    student_id, 
                    seq_date
                )
                sequence_features.append(features)
            
            # Reverse to have chronological order
            sequence_features.reverse()
            
            # Stack into tensor
            X = torch.FloatTensor(np.stack(sequence_features))
            
            # Add batch dimension
            X = X.unsqueeze(0)
            
            return X
    
    def predict_risk(
        self,
        student_id: str,
        reference_date: Optional[date] = None,
        include_factors: bool = True
    ) -> schemas.PredictResponse:
        """
        Generate risk prediction for a student.
        
        Args:
            student_id: Student UUID
            reference_date: Date for prediction
            include_factors: Whether to include contributing factors
            
        Returns:
            Prediction response
        """
        try:
            # Prepare input sequence
            X = self.prepare_sequence(student_id, reference_date)
            X = X.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                risk_score, category_logits, attention_weights = self.model.forward(
                    X, 
                    return_attention=include_factors
                )
                
                # Extract values
                risk_value = float(risk_score[0, 0].cpu().item())
                
                # Get category prediction
                category_probs = torch.softmax(category_logits[0], dim=-1)
                category_idx = torch.argmax(category_probs).item()
                confidence = float(category_probs[category_idx].cpu().item())
                
                # Map to category name
                categories = ['low', 'medium', 'high', 'critical']
                risk_category = categories[category_idx]
            
            # Extract contributing factors if requested
            contributing_factors = []
            if include_factors and attention_weights is not None:
                factors = self._extract_contributing_factors(
                    X, 
                    attention_weights,
                    risk_value
                )
                contributing_factors = factors
            
            # Create prediction record
            with get_db() as db:
                prediction = db_models.Prediction(
                    student_id=student_id,
                    risk_score=risk_value,
                    risk_category=risk_category,
                    confidence=confidence,
                    risk_factors=contributing_factors,
                    model_version=settings.model_version,
                    prediction_date=datetime.utcnow()
                )
                db.add(prediction)
                db.commit()
                db.refresh(prediction)
                
                # Create response
                response = schemas.PredictResponse(
                    prediction=prediction,
                    contributing_factors=contributing_factors if include_factors else None,
                    timestamp=datetime.utcnow()
                )
            
            return response
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return fallback prediction on error
            return self._fallback_prediction(student_id)
    
    def predict_batch(
        self,
        student_ids: List[str],
        reference_date: Optional[date] = None,
        top_k: int = 10
    ) -> schemas.BatchPredictResponse:
        """
        Generate predictions for multiple students.
        
        Args:
            student_ids: List of student UUIDs
            reference_date: Date for predictions
            top_k: Number of top risk students to return
            
        Returns:
            Batch prediction response
        """
        start_time = datetime.now()
        predictions = []
        
        # Generate predictions for each student
        for student_id in student_ids:
            try:
                response = self.predict_risk(
                    student_id, 
                    reference_date,
                    include_factors=False
                )
                
                predictions.append({
                    'student_id': str(student_id),
                    'risk_score': response.prediction.risk_score,
                    'risk_category': response.prediction.risk_category
                })
            except Exception as e:
                print(f"Error predicting for student {student_id}: {e}")
                # Add fallback prediction
                predictions.append({
                    'student_id': str(student_id),
                    'risk_score': 0.5,
                    'risk_category': 'medium'
                })
        
        # Sort by risk score and take top k
        predictions.sort(key=lambda x: x['risk_score'], reverse=True)
        top_predictions = predictions[:top_k]
        
        # Add rank
        for i, pred in enumerate(top_predictions):
            pred['rank'] = i + 1
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return schemas.BatchPredictResponse(
            predictions=top_predictions,
            processing_time_ms=processing_time
        )
    
    def _extract_contributing_factors(
        self,
        X: torch.Tensor,
        attention_weights: torch.Tensor,
        risk_score: float
    ) -> List[Dict]:
        """
        Extract contributing factors from attention weights.
        
        Args:
            X: Input features
            attention_weights: Attention weights from model
            risk_score: Predicted risk score
            
        Returns:
            List of contributing factors
        """
        factors = []
        
        # Get feature names
        feature_names = self.feature_pipeline.get_feature_names()
        
        # Calculate feature importance from attention
        # This is simplified - in production would use more sophisticated methods
        attention_mean = attention_weights.mean(dim=1).squeeze().cpu().numpy()
        
        # Get last time step features
        last_features = X[0, -1, :].cpu().numpy()
        
        # Combine attention and feature values for importance
        importance_scores = []
        for i, (name, value) in enumerate(zip(feature_names, last_features)):
            # Simple heuristic: high attention + abnormal value = important
            attention_score = attention_mean[-1] if i < len(attention_mean) else 0.0
            importance = attention_score * abs(value)
            importance_scores.append((name, importance, value))
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 5 factors
        for name, importance, value in importance_scores[:5]:
            # Determine factor description based on feature name and value
            factor_type, description = self._describe_factor(name, value)
            
            factors.append({
                'factor': factor_type,
                'weight': float(importance),
                'details': description
            })
        
        return factors
    
    def _describe_factor(self, feature_name: str, value: float) -> Tuple[str, str]:
        """
        Generate human-readable description for a feature.
        
        Args:
            feature_name: Name of the feature
            value: Feature value
            
        Returns:
            Tuple of (factor_type, description)
        """
        # Parse feature name
        parts = feature_name.split('_')
        category = parts[0] if parts else 'unknown'
        
        # Generate descriptions based on category and value
        if category == 'attendance':
            if 'absence' in feature_name:
                if value > 0.3:
                    return ('attendance_pattern', f'High absence rate: {value:.1%}')
                else:
                    return ('attendance_pattern', f'Absence rate: {value:.1%}')
            elif 'tardy' in feature_name:
                return ('tardiness', f'Tardiness rate: {value:.1%}')
        
        elif category == 'grades':
            if 'gpa' in feature_name:
                return ('academic_performance', f'GPA: {value:.2f}')
            elif 'trend' in feature_name:
                if value < 0:
                    return ('grade_trajectory', 'Declining academic performance')
                else:
                    return ('grade_trajectory', 'Stable academic performance')
        
        elif category == 'discipline':
            if 'incident' in feature_name:
                return ('behavioral', f'Discipline incidents: {int(value)}')
            elif 'severity' in feature_name:
                return ('behavioral', f'Incident severity: {value:.1f}')
        
        # Default
        return ('other', f'{feature_name}: {value:.2f}')
    
    def _fallback_prediction(self, student_id: str) -> schemas.PredictResponse:
        """
        Generate fallback prediction when ML fails.
        
        Args:
            student_id: Student UUID
            
        Returns:
            Fallback prediction response
        """
        # Create simple rule-based prediction
        with get_db() as db:
            prediction = db_models.Prediction(
                student_id=student_id,
                risk_score=0.5,
                risk_category='medium',
                confidence=0.5,
                risk_factors=[{
                    'factor': 'system',
                    'weight': 1.0,
                    'details': 'Using fallback prediction due to system error'
                }],
                model_version='fallback',
                prediction_date=datetime.utcnow()
            )
            db.add(prediction)
            db.commit()
            db.refresh(prediction)
            
            return schemas.PredictResponse(
                prediction=prediction,
                contributing_factors=None,
                timestamp=datetime.utcnow()
            )


# Global prediction service instance
prediction_service = PredictionService()
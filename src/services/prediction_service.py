"""
@fileoverview ML prediction service for real-time student risk assessment
@lastmodified 2025-08-13T02:56:19-05:00

Features: Model loading, sequence preparation, risk prediction, batch processing, factor analysis
Main APIs: predict_risk(), predict_batch(), prepare_sequence(), load_model()
Constraints: Requires GRU model, feature pipeline, PyTorch, database session
Patterns: Singleton service, attention-based interpretability, fallback predictions, caching
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config.settings import get_settings
from src.db import models as db_models
from src.db.database import get_db
from src.features.pipeline import FeaturePipeline
from src.models import schemas
from src.models.gru_model import GRUAttentionModel

settings = get_settings()


class PredictionService:
    """
    Service for making student dropout risk predictions using trained ML models.

    Provides a high-level interface for real-time risk assessment using GRU-based
    neural networks with attention mechanisms. Handles model loading, feature
    preparation, batch processing, and result interpretation.

    Attributes:
        model: GRU attention model for predictions
        feature_pipeline: Pipeline for feature extraction
        device: PyTorch device (CPU/CUDA)
        model_path: Path to saved model weights
    """

    def __init__(self):
        """
        Initialize the prediction service with model loading and device setup.

        Automatically loads the best available model weights or initializes
        with random weights if no saved model is found.
        """
        self.model = None
        self.feature_pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(settings.model_path) / "best_model.pt"

        # Load model on initialization
        self.load_model()

    def load_model(self) -> None:
        """
        Load the trained GRU model from disk with graceful fallback handling.

        Initializes the model architecture and attempts to load saved weights.
        If no saved model exists, initializes with random weights. Configures
        the model for evaluation mode and moves to appropriate device.

        Raises:
            Exception: Logged but not raised - gracefully falls back to random weights

        Examples:
            >>> service = PredictionService()
            >>> service.load_model()  # Called automatically in __init__
            Model loaded from /models/best_model.pt
        """
        try:
            # Initialize model architecture
            self.model = GRUAttentionModel(
                input_size=42,
                hidden_size=128,
                num_layers=2,
                num_heads=4,
                dropout=0.3,
                bidirectional=True,
            )

            # Load weights if model file exists
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
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
        self, student_id: str, reference_date: Optional[date] = None, sequence_length: int = 20
    ) -> torch.Tensor:
        """
        Prepare sequential feature input for GRU model prediction.

        Extracts features for multiple time points to create a temporal sequence
        that captures student behavior patterns over time. Features are extracted
        weekly going backwards from the reference date.

        Args:
            student_id: UUID string of the student
            reference_date: End date for sequence (defaults to current date)
            sequence_length: Number of weekly time steps in sequence (default: 20)

        Returns:
            torch.Tensor: Feature tensor of shape (1, sequence_length, num_features)
                         ready for model input

        Examples:
            >>> tensor = service.prepare_sequence("123-456", date(2024, 6, 15), 10)
            >>> print(tensor.shape)
            torch.Size([1, 10, 42])
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
                features = self.feature_pipeline.extract_features(student_id, seq_date)
                sequence_features.append(features)

            # Reverse to have chronological order
            sequence_features.reverse()

            # Stack into tensor
            X = torch.FloatTensor(np.stack(sequence_features))

            # Add batch dimension
            X = X.unsqueeze(0)

            return X

    def predict_risk(
        self, student_id: str, reference_date: Optional[date] = None, include_factors: bool = True
    ) -> schemas.PredictResponse:
        """
        Generate comprehensive dropout risk prediction for a single student.

        Performs end-to-end risk assessment including feature sequence preparation,
        model inference, attention-based factor analysis, and result persistence.
        Returns risk score, category classification, and optional contributing factors.

        Args:
            student_id: UUID string of the student to assess
            reference_date: Date to predict risk for (defaults to current date)
            include_factors: Whether to extract and return contributing risk factors
                            using attention weights (default: True)

        Returns:
            PredictResponse: Complete prediction response containing:
                - prediction: Stored prediction record with risk score/category
                - contributing_factors: List of interpretable risk factors (if requested)
                - timestamp: When the prediction was generated

        Raises:
            Exception: Falls back to rule-based prediction if ML model fails

        Examples:
            >>> response = service.predict_risk("123-456-789", include_factors=True)
            >>> print(f"Risk: {response.prediction.risk_score:.2f}")
            Risk: 0.73
            >>> print(f"Category: {response.prediction.risk_category}")
            Category: high
        """
        try:
            # Prepare input sequence
            X = self.prepare_sequence(student_id, reference_date)
            X = X.to(self.device)

            # Get prediction
            with torch.no_grad():
                risk_score, category_logits, attention_weights = self.model.forward(
                    X, return_attention=include_factors
                )

                # Extract values
                risk_value = float(risk_score[0, 0].cpu().item())

                # Get category prediction
                category_probs = torch.softmax(category_logits[0], dim=-1)
                category_idx = torch.argmax(category_probs).item()
                confidence = float(category_probs[category_idx].cpu().item())

                # Map to category name
                categories = ["low", "medium", "high", "critical"]
                risk_category = categories[category_idx]

            # Extract contributing factors if requested
            contributing_factors = []
            if include_factors and attention_weights is not None:
                factors = self._extract_contributing_factors(X, attention_weights, risk_value)
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
                    prediction_date=datetime.utcnow(),
                )
                db.add(prediction)
                db.commit()
                db.refresh(prediction)

                # Create response
                response = schemas.PredictResponse(
                    prediction=prediction,
                    contributing_factors=contributing_factors if include_factors else None,
                    timestamp=datetime.utcnow(),
                )

            return response

        except Exception as e:
            print(f"Prediction error: {e}")
            # Return fallback prediction on error
            return self._fallback_prediction(student_id)

    def predict_batch(
        self, student_ids: List[str], reference_date: Optional[date] = None, top_k: int = 10
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
                response = self.predict_risk(student_id, reference_date, include_factors=False)

                predictions.append(
                    {
                        "student_id": str(student_id),
                        "risk_score": response.prediction.risk_score,
                        "risk_category": response.prediction.risk_category,
                    }
                )
            except Exception as e:
                print(f"Error predicting for student {student_id}: {e}")
                # Add fallback prediction
                predictions.append(
                    {"student_id": str(student_id), "risk_score": 0.5, "risk_category": "medium"}
                )

        # Sort by risk score and take top k
        predictions.sort(key=lambda x: x["risk_score"], reverse=True)
        top_predictions = predictions[:top_k]

        # Add rank
        for i, pred in enumerate(top_predictions):
            pred["rank"] = i + 1

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return schemas.BatchPredictResponse(
            predictions=top_predictions, processing_time_ms=processing_time
        )

    def _extract_contributing_factors(
        self, X: torch.Tensor, attention_weights: torch.Tensor, risk_score: float
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

            factors.append(
                {"factor": factor_type, "weight": float(importance), "details": description}
            )

        return factors

    def _describe_attendance_factor(self, feature_name: str, value: float) -> Tuple[str, str]:
        """Describe attendance-related factors"""
        if "absence" in feature_name:
            prefix = "High " if value > 0.3 else ""
            return ("attendance_pattern", f"{prefix}absence rate: {value:.1%}")
        elif "tardy" in feature_name:
            return ("tardiness", f"Tardiness rate: {value:.1%}")
        return ("attendance_pattern", f"Attendance: {value:.2f}")

    def _describe_grades_factor(self, feature_name: str, value: float) -> Tuple[str, str]:
        """Describe grades-related factors"""
        if "gpa" in feature_name:
            return ("academic_performance", f"GPA: {value:.2f}")
        elif "trend" in feature_name:
            trajectory = "Declining" if value < 0 else "Stable"
            return ("grade_trajectory", f"{trajectory} academic performance")
        return ("academic_performance", f"Grades: {value:.2f}")

    def _describe_discipline_factor(self, feature_name: str, value: float) -> Tuple[str, str]:
        """Describe discipline-related factors"""
        if "incident" in feature_name:
            return ("behavioral", f"Discipline incidents: {int(value)}")
        elif "severity" in feature_name:
            return ("behavioral", f"Incident severity: {value:.1f}")
        return ("behavioral", f"Discipline: {value:.2f}")

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
        parts = feature_name.split("_")
        category = parts[0] if parts else "unknown"

        # Delegate to specific handlers
        handlers = {
            "attendance": self._describe_attendance_factor,
            "grades": self._describe_grades_factor,
            "discipline": self._describe_discipline_factor,
        }

        handler = handlers.get(category)
        if handler:
            return handler(feature_name, value)

        # Default
        return ("other", f"{feature_name}: {value:.2f}")

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
                risk_category="medium",
                confidence=0.5,
                risk_factors=[
                    {
                        "factor": "system",
                        "weight": 1.0,
                        "details": "Using fallback prediction due to system error",
                    }
                ],
                model_version="fallback",
                prediction_date=datetime.utcnow(),
            )
            db.add(prediction)
            db.commit()
            db.refresh(prediction)

            return schemas.PredictResponse(
                prediction=prediction, contributing_factors=None, timestamp=datetime.utcnow()
            )


# Global prediction service instance
prediction_service = PredictionService()

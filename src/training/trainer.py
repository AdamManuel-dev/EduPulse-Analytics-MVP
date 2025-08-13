"""
@fileoverview Model training pipeline with PyTorch dataset and trainer classes
@lastmodified 2025-08-13T02:56:19-05:00

Features: StudentSequenceDataset, ModelTrainer, epoch training/validation, early stopping
Main APIs: StudentSequenceDataset(), ModelTrainer(), fit(), save_model(), load_model()
Constraints: Requires PyTorch, GRU model, feature pipeline, student sequence data
Patterns: PyTorch Dataset/DataLoader, combined loss (risk + category), gradient clipping
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.config.settings import get_settings
from src.db.database import get_db
from src.features.pipeline import FeaturePipeline
from src.models.gru_model import EarlyStopping, GRUAttentionModel

settings = get_settings()


class StudentSequenceDataset(Dataset):
    """
    PyTorch dataset for temporal student feature sequences with sliding window sampling.

    Creates training samples by extracting sequential feature vectors for students
    over configurable time windows. Supports multiple samples per student using
    sliding window approach to increase training data diversity.

    Attributes:
        student_ids: List of student UUID strings to include in dataset
        sequence_length: Number of weekly time steps in each sequence
        prediction_horizon: Days into future for prediction target
        feature_pipeline: Pipeline for extracting student features
        samples: Prepared list of training samples with targets
    """

    def __init__(
        self,
        student_ids: List[str],
        sequence_length: int = 20,
        prediction_horizon: int = 30,
        feature_pipeline: Optional[FeaturePipeline] = None,
    ):
        """
        Initialize the dataset with student IDs and temporal configuration.

        Sets up the dataset for training with configurable sequence length and
        prediction horizon. Automatically prepares all training samples during
        initialization for efficient batch loading.

        Args:
            student_ids: List of student UUID strings to include in training
            sequence_length: Number of weekly time steps for input sequences (default: 20)
            prediction_horizon: Days ahead to predict dropout risk (default: 30)
            feature_pipeline: Pre-configured feature extraction pipeline, or None
                             to create a new one with database connection

        Examples:
            >>> student_ids = ["123-456", "789-012"]
            >>> dataset = StudentSequenceDataset(student_ids, sequence_length=15)
            >>> print(len(dataset))
            20  # 10 samples per student
        """
        self.student_ids = student_ids
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Initialize feature pipeline if not provided
        if feature_pipeline is None:
            with get_db() as db:
                self.feature_pipeline = FeaturePipeline(db)
        else:
            self.feature_pipeline = feature_pipeline

        # Prepare data samples
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict]:
        """
        Prepare training samples using sliding window approach across time.

        Creates multiple training samples per student by sliding a temporal window
        across different time periods. Each sample contains a sequence of feature
        vectors and corresponding target risk labels.

        Returns:
            list: List of sample dictionaries containing:
                - student_id: Student UUID
                - sequence_dates: List of dates for feature extraction
                - target_date: Date for prediction target
                - target_risk: Risk score target (0-1)
                - target_category: Risk category target (0-3)

        Note:
            In production, target values would be computed from actual outcomes.
            Current implementation uses mock targets for demonstration.
        """
        samples = []

        # For each student, create multiple samples with sliding window
        for student_id in self.student_ids:
            # This is simplified - in production, would query actual date ranges
            base_date = datetime.now().date()

            for i in range(10):  # Create 10 samples per student
                # Calculate dates for sequence
                sequence_dates = []
                for j in range(self.sequence_length):
                    date = base_date - timedelta(weeks=(i + j))
                    sequence_dates.append(date)

                # Target date (prediction_horizon days after last sequence date)
                target_date = sequence_dates[-1] + timedelta(days=self.prediction_horizon)

                samples.append(
                    {
                        "student_id": student_id,
                        "sequence_dates": sequence_dates,
                        "target_date": target_date,
                        # In production, would fetch actual outcome
                        "target_risk": np.random.random(),  # Mock target
                        "target_category": np.random.randint(0, 4),  # Mock category
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single training sample by index with feature extraction.

        Extracts features for all dates in the sample's sequence and returns
        formatted tensors ready for model training. Features are extracted
        on-demand to maintain memory efficiency.

        Args:
            idx: Sample index in the dataset

        Returns:
            tuple: Three-element tuple containing:
                - torch.Tensor: Feature sequence of shape (sequence_length, num_features)
                - torch.Tensor: Risk score target of shape (1,)
                - torch.Tensor: Category target of shape (1,) with values 0-3

        Examples:
            >>> X, y_risk, y_cat = dataset[0]
            >>> print(f"Features: {X.shape}, Risk: {y_risk.item():.2f}")
            Features: torch.Size([20, 42]), Risk: 0.73
        """
        sample = self.samples[idx]

        # Extract features for each date in sequence
        sequence_features = []
        for date in sample["sequence_dates"]:
            features = self.feature_pipeline.extract_features(sample["student_id"], date)
            sequence_features.append(features)

        # Stack into sequence tensor
        X = torch.FloatTensor(np.stack(sequence_features))

        # Targets
        y_risk = torch.FloatTensor([sample["target_risk"]])
        y_category = torch.LongTensor([sample["target_category"]])

        return X, y_risk, y_category


class ModelTrainer:
    """
    Comprehensive trainer for GRU attention model with multi-task learning.

    Handles end-to-end training of dropout prediction models with combined loss
    functions for risk score regression and category classification. Includes
    early stopping, learning rate scheduling, and comprehensive metrics tracking.

    Attributes:
        model: GRU attention model being trained
        device: PyTorch device for computation
        risk_criterion: Loss function for risk score regression
        category_criterion: Loss function for category classification
        optimizer: Adam optimizer for parameter updates
        scheduler: Learning rate scheduler for training stability
        early_stopping: Early stopping handler to prevent overfitting
        history: Training metrics history for analysis
    """

    def __init__(self, model: GRUAttentionModel, device: str = "cpu"):
        """
        Initialize the trainer with model and optimization configuration.

        Sets up loss functions, optimizer, learning rate scheduler, and early
        stopping for robust training. Configures multi-task learning for both
        continuous risk scores and discrete risk categories.

        Args:
            model: Pre-initialized GRU attention model to train
            device: Computation device ('cpu' or 'cuda' if available)

        Examples:
            >>> model = GRUAttentionModel(input_size=42, hidden_size=128)
            >>> trainer = ModelTrainer(model, device='cuda')
            >>> print(trainer.device)
            cuda
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        # Loss functions
        self.risk_criterion = nn.BCELoss()  # Binary cross-entropy for risk score
        self.category_criterion = nn.CrossEntropyLoss()  # Cross-entropy for categories

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), lr=settings.model_learning_rate, weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=settings.model_early_stopping_patience)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_risk_loss": [],
            "val_risk_loss": [],
            "train_category_loss": [],
            "val_category_loss": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Execute one complete training epoch with gradient updates.

        Performs forward pass, loss calculation, backpropagation, and parameter
        updates for all batches in the training set. Includes gradient clipping
        for training stability.

        Args:
            dataloader: PyTorch DataLoader with training batches

        Returns:
            dict: Training metrics containing:
                - loss: Combined average loss across all batches
                - risk_loss: Risk regression loss component
                - category_loss: Category classification loss component

        Examples:
            >>> metrics = trainer.train_epoch(train_loader)
            >>> print(f"Training loss: {metrics['loss']:.4f}")
            Training loss: 0.7234
        """
        self.model.train()

        total_loss = 0.0
        total_risk_loss = 0.0
        total_category_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            X, y_risk, y_category = batch
            X = X.to(self.device)
            y_risk = y_risk.to(self.device)
            y_category = y_category.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            risk_pred, category_logits, _ = self.model(X)

            # Calculate losses
            risk_loss = self.risk_criterion(risk_pred, y_risk)
            category_loss = self.category_criterion(category_logits.squeeze(), y_category.squeeze())

            # Combined loss
            loss = risk_loss + category_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update weights
            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_risk_loss += risk_loss.item()
            total_category_loss += category_loss.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "risk_loss": total_risk_loss / num_batches,
            "category_loss": total_category_loss / num_batches,
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance on validation set without parameter updates.

        Performs forward pass through validation data in evaluation mode
        (no gradient computation) to assess model generalization performance.

        Args:
            dataloader: PyTorch DataLoader with validation batches

        Returns:
            dict: Validation metrics containing:
                - loss: Combined average validation loss
                - risk_loss: Risk regression validation loss
                - category_loss: Category classification validation loss

        Examples:
            >>> val_metrics = trainer.validate(val_loader)
            >>> print(f"Validation loss: {val_metrics['loss']:.4f}")
            Validation loss: 0.8156
        """
        self.model.eval()

        total_loss = 0.0
        total_risk_loss = 0.0
        total_category_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                X, y_risk, y_category = batch
                X = X.to(self.device)
                y_risk = y_risk.to(self.device)
                y_category = y_category.to(self.device)

                # Forward pass
                risk_pred, category_logits, _ = self.model(X)

                # Calculate losses
                risk_loss = self.risk_criterion(risk_pred, y_risk)
                category_loss = self.category_criterion(
                    category_logits.squeeze(), y_category.squeeze()
                )

                # Combined loss
                loss = risk_loss + category_loss

                # Track losses
                total_loss += loss.item()
                total_risk_loss += risk_loss.item()
                total_category_loss += category_loss.item()
                num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "risk_loss": total_risk_loss / num_batches,
            "category_loss": total_category_loss / num_batches,
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100) -> None:
        """
        Execute complete model training with early stopping and learning rate scheduling.

        Runs the full training loop including epoch training, validation,
        metrics tracking, and early stopping. Automatically saves training
        history for analysis and applies learning rate decay on plateaus.

        Args:
            train_loader: DataLoader containing training batches
            val_loader: DataLoader containing validation batches
            epochs: Maximum number of training epochs (may stop early)

        Examples:
            >>> trainer.fit(train_loader, val_loader, epochs=50)
            Starting training for 50 epochs...
            Epoch 10/50
              Train Loss: 0.7234
              Val Loss: 0.8156
            Early stopping triggered at epoch 23
        """
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_metrics["loss"])

            # Track history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_risk_loss"].append(train_metrics["risk_loss"])
            self.history["val_risk_loss"].append(val_metrics["risk_loss"])
            self.history["train_category_loss"].append(train_metrics["category_loss"])
            self.history["val_category_loss"].append(val_metrics["category_loss"])

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")

            # Early stopping
            if self.early_stopping(val_metrics["loss"]):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    def save_model(self, path: str) -> None:
        """
        Save complete model state including weights, optimizer, and training history.

        Persists all necessary components for model restoration including
        model weights, optimizer state, training history, and model configuration.
        Creates parent directories if they don't exist.

        Args:
            path: File path to save the model checkpoint

        Examples:
            >>> trainer.save_model('/models/best_model.pt')
            Model saved to /models/best_model.pt
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "model_config": {
                    "input_size": self.model.input_size,
                    "hidden_size": self.model.hidden_size,
                    "num_layers": self.model.num_layers,
                    "num_heads": self.model.num_heads,
                },
            },
            path,
        )

        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load complete model state from saved checkpoint.

        Restores model weights, optimizer state, and training history from
        a previously saved checkpoint. Ensures model is ready for continued
        training or inference.

        Args:
            path: File path to the saved model checkpoint

        Examples:
            >>> trainer.load_model('/models/best_model.pt')
            Model loaded from /models/best_model.pt
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {})

        print(f"Model loaded from {path}")

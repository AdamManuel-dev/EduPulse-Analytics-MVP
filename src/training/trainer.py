"""
Training pipeline for the GRU attention model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

from src.models.gru_model import GRUAttentionModel, EarlyStopping
from src.features.pipeline import FeaturePipeline
from src.config.settings import get_settings
from src.db.database import get_db

settings = get_settings()


class StudentSequenceDataset(Dataset):
    """
    PyTorch dataset for student temporal sequences.
    """
    
    def __init__(
        self, 
        student_ids: List[str],
        sequence_length: int = 20,
        prediction_horizon: int = 30,
        feature_pipeline: Optional[FeaturePipeline] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            student_ids: List of student IDs to include
            sequence_length: Number of weeks to use as input
            prediction_horizon: Days ahead to predict
            feature_pipeline: Feature extraction pipeline
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
        Prepare training samples from student data.
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
                
                samples.append({
                    'student_id': student_id,
                    'sequence_dates': sequence_dates,
                    'target_date': target_date,
                    # In production, would fetch actual outcome
                    'target_risk': np.random.random(),  # Mock target
                    'target_category': np.random.randint(0, 4)  # Mock category
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (features, risk_target, category_target)
        """
        sample = self.samples[idx]
        
        # Extract features for each date in sequence
        sequence_features = []
        for date in sample['sequence_dates']:
            features = self.feature_pipeline.extract_features(
                sample['student_id'], 
                date
            )
            sequence_features.append(features)
        
        # Stack into sequence tensor
        X = torch.FloatTensor(np.stack(sequence_features))
        
        # Targets
        y_risk = torch.FloatTensor([sample['target_risk']])
        y_category = torch.LongTensor([sample['target_category']])
        
        return X, y_risk, y_category


class ModelTrainer:
    """
    Trainer class for the GRU attention model.
    """
    
    def __init__(
        self,
        model: GRUAttentionModel,
        device: str = 'cpu'
    ):
        """
        Initialize the trainer.
        
        Args:
            model: GRU attention model to train
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Loss functions
        self.risk_criterion = nn.BCELoss()  # Binary cross-entropy for risk score
        self.category_criterion = nn.CrossEntropyLoss()  # Cross-entropy for categories
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=settings.model_learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=settings.model_early_stopping_patience
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_risk_loss': [],
            'val_risk_loss': [],
            'train_category_loss': [],
            'val_category_loss': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of training metrics
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
            category_loss = self.category_criterion(
                category_logits.squeeze(),
                y_category.squeeze()
            )
            
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
            'loss': total_loss / num_batches,
            'risk_loss': total_risk_loss / num_batches,
            'category_loss': total_category_loss / num_batches
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
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
                    category_logits.squeeze(),
                    y_category.squeeze()
                )
                
                # Combined loss
                loss = risk_loss + category_loss
                
                # Track losses
                total_loss += loss.item()
                total_risk_loss += risk_loss.item()
                total_category_loss += category_loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'risk_loss': total_risk_loss / num_batches,
            'category_loss': total_category_loss / num_batches
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100
    ) -> None:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
        """
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_risk_loss'].append(train_metrics['risk_loss'])
            self.history['val_risk_loss'].append(val_metrics['risk_loss'])
            self.history['train_category_loss'].append(train_metrics['category_loss'])
            self.history['val_category_loss'].append(val_metrics['category_loss'])
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads
            }
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {})
        
        print(f"Model loaded from {path}")
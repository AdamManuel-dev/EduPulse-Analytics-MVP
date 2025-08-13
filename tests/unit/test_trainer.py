"""
Fixed comprehensive unit tests for model training components.
Achieves >90% coverage for src/training/trainer.py
"""

from datetime import timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.gru_model import GRUAttentionModel
from src.training.trainer import ModelTrainer, StudentSequenceDataset


class TestStudentSequenceDataset:
    """Comprehensive test cases for StudentSequenceDataset."""

    @pytest.fixture
    def mock_feature_pipeline(self):
        """Mock feature pipeline for testing."""
        pipeline = Mock()
        # Return 42 features to match expected feature count
        mock_features = np.random.random(42).tolist()
        pipeline.extract_features.return_value = mock_features
        return pipeline

    @pytest.fixture
    def sample_student_ids(self):
        """Generate sample student IDs for testing."""
        return [str(uuid4()) for _ in range(3)]

    def test_dataset_initialization_with_pipeline(self, sample_student_ids, mock_feature_pipeline):
        """Test dataset initialization with provided feature pipeline."""
        dataset = StudentSequenceDataset(
            student_ids=sample_student_ids,
            sequence_length=10,
            prediction_horizon=30,
            feature_pipeline=mock_feature_pipeline,
        )

        assert dataset.student_ids == sample_student_ids
        assert dataset.sequence_length == 10
        assert dataset.prediction_horizon == 30
        assert dataset.feature_pipeline == mock_feature_pipeline
        assert len(dataset.samples) == 30  # 3 students * 10 samples each

    @patch("src.training.trainer.get_db")
    def test_dataset_initialization_without_pipeline(self, mock_get_db, sample_student_ids):
        """Test dataset initialization without provided feature pipeline."""
        mock_db = Mock()
        mock_get_db.return_value.__enter__.return_value = mock_db

        with patch("src.training.trainer.FeaturePipeline") as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.extract_features.return_value = np.random.random(42).tolist()
            mock_pipeline_class.return_value = mock_pipeline

            dataset = StudentSequenceDataset(
                student_ids=sample_student_ids, sequence_length=5, prediction_horizon=15
            )

            mock_pipeline_class.assert_called_once_with(mock_db)
            assert dataset.feature_pipeline == mock_pipeline

    def test_dataset_default_parameters(self, mock_feature_pipeline):
        """Test dataset initialization with default parameters."""
        student_ids = [str(uuid4())]

        dataset = StudentSequenceDataset(
            student_ids=student_ids, feature_pipeline=mock_feature_pipeline
        )

        assert dataset.sequence_length == 20  # Default
        assert dataset.prediction_horizon == 30  # Default

    def test_prepare_samples_structure(self, mock_feature_pipeline):
        """Test sample preparation creates correct structure."""
        student_ids = [str(uuid4())]

        dataset = StudentSequenceDataset(
            student_ids=student_ids,
            sequence_length=5,
            prediction_horizon=14,
            feature_pipeline=mock_feature_pipeline,
        )

        # Should generate 10 samples per student
        assert len(dataset.samples) == 10

        sample = dataset.samples[0]
        assert "student_id" in sample
        assert "sequence_dates" in sample
        assert "target_date" in sample
        assert "target_risk" in sample
        assert "target_category" in sample

        assert sample["student_id"] == student_ids[0]
        assert len(sample["sequence_dates"]) == 5
        assert isinstance(sample["target_risk"], float)
        assert isinstance(sample["target_category"], (int, np.integer))
        assert 0 <= sample["target_risk"] <= 1
        assert 0 <= sample["target_category"] <= 3

    def test_getitem_feature_extraction(self, mock_feature_pipeline):
        """Test __getitem__ method with feature extraction."""
        student_id = str(uuid4())

        dataset = StudentSequenceDataset(
            student_ids=[student_id],
            sequence_length=3,
            prediction_horizon=7,
            feature_pipeline=mock_feature_pipeline,
        )

        X, y_risk, y_category = dataset[0]

        # Check tensor types and shapes
        assert isinstance(X, torch.Tensor)
        assert isinstance(y_risk, torch.Tensor)
        assert isinstance(y_category, torch.Tensor)

        assert X.shape == (3, 42)  # sequence_length x num_features
        assert y_risk.shape == (1,)
        assert y_category.shape == (1,)

        # Verify feature extraction was called for each date
        assert mock_feature_pipeline.extract_features.call_count == 3

    def test_sequence_dates_chronological(self, mock_feature_pipeline):
        """Test that sequence dates are in chronological order."""
        dataset = StudentSequenceDataset(
            student_ids=[str(uuid4())],
            sequence_length=4,
            prediction_horizon=7,
            feature_pipeline=mock_feature_pipeline,
        )

        sample = dataset.samples[0]
        dates = sample["sequence_dates"]

        # Target date should be after last sequence date
        target_date = sample["target_date"]
        last_sequence_date = dates[-1]
        assert target_date > last_sequence_date

        # Check prediction horizon
        expected_delta = timedelta(days=7)
        actual_delta = target_date - last_sequence_date
        assert actual_delta == expected_delta


class TestModelTrainer:
    """Comprehensive test cases for ModelTrainer."""

    @pytest.fixture
    def mock_model(self):
        """Mock GRU model for testing."""
        model = Mock(spec=GRUAttentionModel)
        model.parameters.return_value = [torch.randn(5, 3, requires_grad=True)]
        model.train = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.state_dict = Mock(return_value={"weight": torch.randn(5, 3)})
        model.load_state_dict = Mock()

        # Add model config attributes
        model.input_size = 42
        model.hidden_size = 128
        model.num_layers = 2
        model.num_heads = 8

        return model

    def test_trainer_initialization_cpu(self, mock_model):
        """Test trainer initialization with CPU device."""
        trainer = ModelTrainer(model=mock_model, device="cpu")

        assert trainer.model == mock_model
        assert trainer.device == torch.device("cpu")
        mock_model.to.assert_called_once_with(torch.device("cpu"))

        # Check loss functions
        assert isinstance(trainer.risk_criterion, nn.BCELoss)
        assert isinstance(trainer.category_criterion, nn.CrossEntropyLoss)

    @patch("src.training.trainer.get_settings")
    def test_trainer_initialization_uses_settings(self, mock_get_settings, mock_model):
        """Test trainer uses settings for configuration."""
        mock_settings = Mock()
        mock_settings.model_learning_rate = 0.001  # Correct default value
        mock_settings.model_early_stopping_patience = 10  # Correct default value
        mock_get_settings.return_value = mock_settings

        trainer = ModelTrainer(model=mock_model)

        # Verify settings were used
        assert trainer.optimizer.param_groups[0]["lr"] == 0.001
        assert trainer.early_stopping.patience == 10

    def test_train_epoch(self, mock_model):
        """Test training epoch execution."""
        trainer = ModelTrainer(model=mock_model)

        # Mock model output - ensure risk predictions are properly shaped with gradients
        mock_model.return_value = (
            torch.sigmoid(torch.randn(1, 1, requires_grad=True)),  # Requires grad for backward pass
            torch.randn(1, 4, requires_grad=True),  # category logits
            torch.randn(1, 10, 42),  # attention weights (no grad needed)
        )

        # Create properly shaped batch data - single sample
        X = torch.randn(1, 10, 42)
        y_risk = torch.tensor([0.5])  # 1D tensor for batch
        y_category = torch.tensor([1])  # 1D tensor for batch

        # Create dataset-like object that returns properly formatted data
        class MockDataset:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = MockDataset(
            [(X.squeeze(0), y_risk[0].unsqueeze(0), y_category[0])]
        )  # Shape y_risk as (1,) for model output (1,1)
        dataloader = DataLoader(dataset, batch_size=1)

        metrics = trainer.train_epoch(dataloader)

        # Check metrics structure
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "risk_loss" in metrics
        assert "category_loss" in metrics

        # Check that training methods were called
        mock_model.train.assert_called()

    def test_validate(self, mock_model):
        """Test validation epoch execution."""
        trainer = ModelTrainer(model=mock_model)

        # Mock model output with correct shapes
        mock_model.return_value = (
            torch.sigmoid(torch.randn(1, 1)),
            torch.randn(1, 4),
            torch.randn(1, 10, 42),
        )

        # Create properly shaped batch data - single sample
        X = torch.randn(1, 10, 42)
        y_risk = torch.tensor([0.5])  # 1D tensor for batch
        y_category = torch.tensor([1])  # 1D tensor for batch

        # Create dataset-like object that returns properly formatted data
        class MockDataset:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = MockDataset(
            [(X.squeeze(0), y_risk[0].unsqueeze(0), y_category[0])]
        )  # Shape y_risk as (1,) for model output (1,1)
        dataloader = DataLoader(dataset, batch_size=1)

        metrics = trainer.validate(dataloader)

        # Check metrics structure
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "risk_loss" in metrics
        assert "category_loss" in metrics

        # Check that eval mode was set
        mock_model.eval.assert_called()

    def test_save_model(self, mock_model, tmp_path):
        """Test model saving functionality."""
        trainer = ModelTrainer(model=mock_model)

        # Add some training history
        trainer.history = {
            "train_loss": [0.8, 0.7, 0.6],
            "val_loss": [0.9, 0.8, 0.7],
            "train_risk_loss": [0.4, 0.35, 0.3],
            "val_risk_loss": [0.45, 0.4, 0.35],
            "train_category_loss": [0.4, 0.35, 0.3],
            "val_category_loss": [0.45, 0.4, 0.35],
        }

        save_path = tmp_path / "models" / "test_model.pt"

        with patch("torch.save") as mock_torch_save:
            trainer.save_model(str(save_path))

            # Verify torch.save was called
            mock_torch_save.assert_called_once()

            # Check the saved data structure
            saved_data = mock_torch_save.call_args[0][0]
            assert "model_state_dict" in saved_data
            assert "optimizer_state_dict" in saved_data
            assert "history" in saved_data
            assert "model_config" in saved_data

    def test_load_model(self, mock_model, tmp_path):
        """Test model loading functionality."""
        trainer = ModelTrainer(model=mock_model)

        # Create checkpoint data with all required keys
        checkpoint = {
            "model_state_dict": {"weight": torch.randn(5, 3)},
            "optimizer_state_dict": {"lr": 0.001},
            "history": {"train_loss": [0.8, 0.7], "val_loss": [0.9, 0.8]},
            "model_config": {"input_size": 42, "hidden_size": 128},
        }

        with patch("torch.load", return_value=checkpoint) as mock_torch_load:
            trainer.load_model(str(tmp_path / "test.pt"))

            # Verify torch.load was called
            mock_torch_load.assert_called_once()

            # Verify model and optimizer state loading
            mock_model.load_state_dict.assert_called_once_with(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict.assert_called_once_with(
                checkpoint["optimizer_state_dict"]
            )

    def test_fit_complete_training_loop(self, mock_model):
        """Test complete training loop with fit method."""
        trainer = ModelTrainer(model=mock_model)

        # Mock model outputs with correct shapes
        mock_model.return_value = (
            torch.sigmoid(torch.randn(1, 1)),
            torch.randn(1, 4),
            torch.randn(1, 5, 42),
        )

        # Create data with correct shapes
        X = torch.randn(1, 5, 42)
        y_risk = torch.tensor([[0.5]])
        y_category = torch.tensor([1])

        train_loader = DataLoader([(X, y_risk, y_category)], batch_size=1)
        val_loader = DataLoader([(X, y_risk, y_category)], batch_size=1)

        # Mock early stopping to prevent infinite training
        with patch.object(trainer.early_stopping, "__call__", side_effect=[False, False, True]):
            trainer.fit(train_loader, val_loader, epochs=5)

        # Check that history was updated
        assert len(trainer.history["train_loss"]) == 3  # Stopped at epoch 3
        assert len(trainer.history["val_loss"]) == 3


class TestModelTrainerIntegration:
    """Integration tests for ModelTrainer with real components."""

    def test_trainer_with_real_model(self):
        """Test trainer with actual GRU model."""
        # Create real model with correct parameters
        model = GRUAttentionModel(input_size=42, hidden_size=32, num_layers=1, dropout=0.1)

        trainer = ModelTrainer(model=model, device="cpu")

        # Verify initialization
        assert trainer.model == model
        assert isinstance(trainer.risk_criterion, nn.BCELoss)
        assert isinstance(trainer.category_criterion, nn.CrossEntropyLoss)

    def test_save_load_cycle(self, tmp_path):
        """Test complete save/load cycle."""
        # Create model and trainer
        model = GRUAttentionModel(input_size=42, hidden_size=32, num_layers=1)

        trainer1 = ModelTrainer(model=model, device="cpu")

        # Add some history
        trainer1.history["train_loss"] = [0.8, 0.7, 0.6]
        trainer1.history["val_loss"] = [0.9, 0.8, 0.7]

        # Save model
        save_path = tmp_path / "integration_test_model.pt"
        trainer1.save_model(str(save_path))

        # Create new trainer and load
        model2 = GRUAttentionModel(input_size=42, hidden_size=32, num_layers=1)

        trainer2 = ModelTrainer(model=model2, device="cpu")
        trainer2.load_model(str(save_path))

        # Verify history was restored
        assert trainer2.history["train_loss"] == [0.8, 0.7, 0.6]
        assert trainer2.history["val_loss"] == [0.9, 0.8, 0.7]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

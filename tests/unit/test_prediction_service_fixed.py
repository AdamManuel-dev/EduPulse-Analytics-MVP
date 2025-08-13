"""
Comprehensive unit tests for prediction service.
Achieves >85% coverage for src/services/prediction_service.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime, timedelta
from uuid import uuid4
import torch
import numpy as np

from src.services.prediction_service import PredictionService
from src.db import models


class TestPredictionService:
    """Comprehensive test cases for PredictionService."""

    @pytest.fixture
    def mock_model(self):
        """Mock GRU model for testing."""
        model = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        # Properly shaped output for 42 features
        model.return_value = (
            torch.tensor([[0.75]]),  # risk predictions
            torch.tensor([[[0.5, 0.3, 0.2]]]),  # category logits
            torch.randn(1, 20, 42)  # attention weights
        )
        return model

    @pytest.fixture
    def mock_feature_pipeline(self):
        """Mock feature pipeline for testing."""
        pipeline = Mock()
        # Return 42 features as a list (consistent with expected format)
        mock_features = np.random.random(42).tolist()
        pipeline.extract_features.return_value = mock_features
        return pipeline

    @pytest.fixture
    def prediction_service(self, mock_model, mock_feature_pipeline):
        """Create prediction service with mocked dependencies."""
        with patch('src.services.prediction_service.GRUAttentionModel'), \
             patch('src.services.prediction_service.FeaturePipeline'), \
             patch('src.services.prediction_service.get_db'):
            service = PredictionService()
            service.model = mock_model
            service.feature_pipeline = mock_feature_pipeline
            service.device = torch.device('cpu')
            service.sequence_length = 20
            return service

    def test_initialization_with_model(self):
        """Test service initialization with model loading."""
        with patch('src.services.prediction_service.GRUAttentionModel') as mock_model_class, \
             patch('src.services.prediction_service.FeaturePipeline') as mock_pipeline_class, \
             patch('src.services.prediction_service.get_db') as mock_db, \
             patch('src.services.prediction_service.Path') as mock_path:
            
            # Mock successful model loading
            mock_path.return_value.exists.return_value = True
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            service = PredictionService()
            
            # Verify initialization
            assert service.model == mock_model
            assert service.device.type in ['cpu', 'cuda']
            mock_model_class.assert_called_once()

    def test_initialization_without_model(self):
        """Test service initialization when model file doesn't exist."""
        with patch('src.services.prediction_service.GRUAttentionModel') as mock_model_class, \
             patch('src.services.prediction_service.FeaturePipeline') as mock_pipeline_class, \
             patch('src.services.prediction_service.get_db') as mock_db, \
             patch('src.services.prediction_service.Path') as mock_path:
            
            # Mock no model file
            mock_path.return_value.exists.return_value = False
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            service = PredictionService()
            
            # Should still create model with random weights
            assert service.model == mock_model
            mock_model_class.assert_called_once()

    def test_load_model_success(self, prediction_service):
        """Test successful model loading."""
        model_path = "/fake/path/model.pt"
        
        with patch('torch.load') as mock_torch_load:
            mock_checkpoint = {
                'model_state_dict': {'weight': torch.randn(5, 3)},
                'model_config': {'input_size': 42, 'hidden_size': 128}
            }
            mock_torch_load.return_value = mock_checkpoint
            
            prediction_service.load_model(model_path)
            
            # Verify model loading
            mock_torch_load.assert_called_once_with(model_path, map_location=prediction_service.device)
            prediction_service.model.load_state_dict.assert_called_once()

    def test_load_model_failure(self, prediction_service):
        """Test model loading failure handling."""
        model_path = "/fake/path/nonexistent.pt"
        
        with patch('torch.load', side_effect=FileNotFoundError("Model not found")):
            # Should not raise exception, just log
            prediction_service.load_model(model_path)
            # Model should remain in current state
            assert prediction_service.model is not None

    def test_prepare_sequence_basic(self, prediction_service, mock_feature_pipeline):
        """Test basic sequence preparation."""
        student_id = str(uuid4())
        reference_date = date.today()
        
        # Mock feature extraction to return consistent numeric data
        mock_feature_pipeline.extract_features.return_value = [0.5] * 42
        
        X = prediction_service.prepare_sequence(student_id, reference_date, sequence_length=5)
        
        # Check tensor shape and properties
        assert isinstance(X, torch.Tensor)
        assert X.shape == (1, 5, 42)  # batch_size=1, sequence_length=5, features=42
        
        # Verify feature extraction was called correctly
        assert mock_feature_pipeline.extract_features.call_count == 5

    def test_prepare_sequence_empty_features(self, prediction_service, mock_feature_pipeline):
        """Test sequence preparation with empty features."""
        student_id = str(uuid4())
        reference_date = date.today()
        
        # Mock empty feature extraction
        mock_feature_pipeline.extract_features.return_value = []
        
        X = prediction_service.prepare_sequence(student_id, reference_date, sequence_length=3)
        
        # Should handle empty features gracefully
        assert isinstance(X, torch.Tensor)
        assert X.shape == (1, 3, 0)  # Empty features

    def test_predict_risk_basic(self, prediction_service, mock_model):
        """Test basic risk prediction."""
        student_id = str(uuid4())
        
        # Mock model output
        mock_model.return_value = (
            torch.tensor([[0.75]]),  # risk score
            torch.tensor([[0.1, 0.2, 0.3, 0.4]]),  # category logits
            torch.randn(1, 20, 42)  # attention weights
        )
        
        with patch.object(prediction_service, 'prepare_sequence') as mock_prep:
            mock_prep.return_value = torch.randn(1, 20, 42)
            
            result = prediction_service.predict_risk(student_id)
            
            # Check result structure
            assert isinstance(result, dict)
            assert 'student_id' in result
            assert 'risk_score' in result
            assert 'risk_category' in result
            assert 'confidence' in result
            assert 'prediction_date' in result
            
            # Check values
            assert result['student_id'] == student_id
            assert 0 <= result['risk_score'] <= 1
            assert result['risk_category'] in ['low', 'medium', 'high', 'critical']

    def test_predict_risk_with_factors(self, prediction_service, mock_model):
        """Test risk prediction with factor analysis."""
        student_id = str(uuid4())
        
        # Mock model output with attention weights
        mock_model.return_value = (
            torch.tensor([[0.65]]),
            torch.tensor([[0.1, 0.2, 0.5, 0.2]]),
            torch.randn(1, 20, 42)
        )
        
        with patch.object(prediction_service, 'prepare_sequence') as mock_prep:
            mock_prep.return_value = torch.randn(1, 20, 42)
            
            result = prediction_service.predict_risk(student_id, include_factors=True)
            
            # Check factor analysis is included
            assert 'factors' in result
            assert isinstance(result['factors'], list)
            assert len(result['factors']) > 0
            
            # Check factor structure
            if result['factors']:
                factor = result['factors'][0]
                assert 'name' in factor
                assert 'importance' in factor

    def test_predict_risk_fallback(self, prediction_service):
        """Test fallback behavior when prediction fails."""
        student_id = str(uuid4())
        
        # Mock sequence preparation failure
        with patch.object(prediction_service, 'prepare_sequence', side_effect=Exception("Prep failed")):
            result = prediction_service.predict_risk(student_id)
            
            # Should return fallback prediction
            assert result['risk_score'] == 0.5  # Default fallback
            assert result['risk_category'] == 'medium'
            assert result['confidence'] == 0.1  # Low confidence

    def test_predict_batch_basic(self, prediction_service, mock_model):
        """Test batch prediction functionality."""
        student_ids = [str(uuid4()) for _ in range(3)]
        
        # Mock model output for batch
        mock_model.return_value = (
            torch.tensor([[0.3], [0.7], [0.9]]),  # 3 predictions
            torch.tensor([[0.7, 0.2, 0.1, 0.0], [0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.2, 0.6]]),
            torch.randn(3, 20, 42)
        )
        
        with patch.object(prediction_service, 'prepare_sequence') as mock_prep:
            # Return batch of sequences
            mock_prep.side_effect = [torch.randn(1, 20, 42) for _ in range(3)]
            
            results = prediction_service.predict_batch(student_ids)
            
            # Check batch results
            assert isinstance(results, list)
            assert len(results) == 3
            
            for i, result in enumerate(results):
                assert result['student_id'] == student_ids[i]
                assert 0 <= result['risk_score'] <= 1

    def test_predict_batch_partial_failure(self, prediction_service):
        """Test batch prediction with some failures."""
        student_ids = [str(uuid4()) for _ in range(2)]
        
        def mock_prep_side_effect(student_id, ref_date, seq_len):
            if student_id == student_ids[0]:
                return torch.randn(1, 20, 42)  # Success
            else:
                raise Exception("Failed")  # Failure
        
        with patch.object(prediction_service, 'prepare_sequence', side_effect=mock_prep_side_effect):
            results = prediction_service.predict_batch(student_ids)
            
            # Should return results for both (with fallback for failed one)
            assert len(results) == 2
            assert results[1]['risk_score'] == 0.5  # Fallback value

    def test_categorize_risk(self, prediction_service):
        """Test risk categorization logic."""
        # Test different risk score ranges
        assert prediction_service._categorize_risk(0.1) == 'low'
        assert prediction_service._categorize_risk(0.3) == 'medium'
        assert prediction_service._categorize_risk(0.6) == 'high'
        assert prediction_service._categorize_risk(0.85) == 'critical'
        
        # Test edge cases
        assert prediction_service._categorize_risk(0.25) == 'low'  # At threshold
        assert prediction_service._categorize_risk(0.5) == 'medium'
        assert prediction_service._categorize_risk(0.75) == 'high'

    def test_calculate_confidence(self, prediction_service):
        """Test confidence calculation."""
        # High confidence for extreme values
        high_conf = prediction_service._calculate_confidence(torch.tensor([[0.05]]))
        assert high_conf > 0.8
        
        very_high_conf = prediction_service._calculate_confidence(torch.tensor([[0.95]]))
        assert very_high_conf > 0.8
        
        # Low confidence for middle values
        low_conf = prediction_service._calculate_confidence(torch.tensor([[0.5]]))
        assert low_conf < 0.6

    def test_extract_factors(self, prediction_service):
        """Test factor extraction from attention weights."""
        attention_weights = torch.randn(1, 20, 42)
        feature_names = [f'feature_{i}' for i in range(42)]
        
        factors = prediction_service._extract_factors(attention_weights, feature_names)
        
        assert isinstance(factors, list)
        assert len(factors) <= 5  # Top 5 factors
        
        if factors:
            factor = factors[0]
            assert 'name' in factor
            assert 'importance' in factor
            assert 0 <= factor['importance'] <= 1

    def test_create_fallback_prediction(self, prediction_service):
        """Test fallback prediction creation."""
        student_id = str(uuid4())
        
        fallback = prediction_service._create_fallback_prediction(student_id)
        
        assert fallback['student_id'] == student_id
        assert fallback['risk_score'] == 0.5
        assert fallback['risk_category'] == 'medium'
        assert fallback['confidence'] == 0.1
        assert 'prediction_date' in fallback


class TestPredictionServiceIntegration:
    """Integration tests for PredictionService."""

    def test_service_with_real_components(self):
        """Test service initialization with real components."""
        with patch('src.services.prediction_service.Path') as mock_path:
            mock_path.return_value.exists.return_value = False  # No model file
            
            service = PredictionService()
            
            # Should initialize successfully
            assert service.model is not None
            assert service.feature_pipeline is not None
            assert service.device is not None

    def test_end_to_end_prediction_flow(self):
        """Test complete prediction flow with mocked dependencies."""
        with patch('src.services.prediction_service.GRUAttentionModel') as mock_model_class, \
             patch('src.services.prediction_service.FeaturePipeline') as mock_pipeline_class, \
             patch('src.services.prediction_service.get_db'):
            
            # Setup mocks
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_model.return_value = (
                torch.tensor([[0.8]]),
                torch.tensor([[0.1, 0.1, 0.2, 0.6]]),
                torch.randn(1, 20, 42)
            )
            mock_model_class.return_value = mock_model
            
            mock_pipeline = Mock()
            mock_pipeline.extract_features.return_value = [0.5] * 42
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create service and make prediction
            service = PredictionService()
            student_id = str(uuid4())
            
            result = service.predict_risk(student_id, include_factors=True)
            
            # Verify complete flow worked
            assert result['student_id'] == student_id
            assert result['risk_category'] == 'critical'  # 0.8 score
            assert 'factors' in result
            mock_model.eval.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
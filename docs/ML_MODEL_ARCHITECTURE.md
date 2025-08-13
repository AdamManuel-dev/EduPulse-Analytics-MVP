# ML Model Architecture Documentation

> **Last Updated**: 2025-08-13 02:56:19 CDT  
> **Model Version**: v1.2.0  
> **Framework**: PyTorch 2.0+

## Overview

EduPulse uses a sophisticated multi-modal neural network architecture based on Gated Recurrent Units (GRUs) with multi-head attention mechanisms to predict student dropout risk. The model processes temporal sequences of student behavioral and academic data to generate risk assessments with interpretable explanations.

## Architecture Philosophy

- **Multi-Modal Processing**: Separate neural pathways for different data types (attendance, grades, discipline)
- **Temporal Modeling**: Captures patterns and trends over time using recurrent networks
- **Attention Mechanisms**: Identifies which time periods and features are most important
- **Interpretability**: Provides actionable insights through attention weights and feature analysis
- **Robustness**: Handles missing data and varying sequence lengths gracefully

---

## Model Architecture

### High-Level Architecture

```
Input Data (3 Modalities)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│  Attendance     │    Grades       │   Discipline    │
│  Features       │   Features      │   Features      │
│  (14 dims)      │   (15 dims)     │   (13 dims)     │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                   ↓                   ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Attendance GRU  │  Grades GRU     │ Discipline GRU  │
│ (Bi-directional)│ (Bi-directional)│ (Bi-directional)│
└─────────────────┴─────────────────┴─────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│           Feature Concatenation & Normalization       │
└───────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│             Multi-Head Self-Attention                 │
│            (4 heads, 128-dim hidden)                  │
└───────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│           Residual Connection & LayerNorm             │
└───────────────────────────────────────────────────────┘
    ↓
┌─────────────────┬─────────────────────────────────────┐
│   Risk Score    │        Risk Category                │
│   Output        │        Output                       │
│   (Sigmoid)     │        (Softmax)                    │
│   [0.0, 1.0]    │        [low, med, high, critical]   │
└─────────────────┴─────────────────────────────────────┘
```

### Detailed Architecture

```python
class GRUAttentionModel(nn.Module):
    """
    Multi-modal GRU model with self-attention for dropout risk prediction.
    
    Input: (batch_size, sequence_length, 42)
    Output: (risk_score, risk_category_logits)
    """
    
    def __init__(self, 
                 input_size=42,        # 14 + 15 + 13 features
                 hidden_size=128,      # Hidden state dimension
                 num_layers=2,         # GRU layers per modality
                 num_heads=4,          # Attention heads
                 dropout=0.3,          # Dropout rate
                 bidirectional=True):  # Bidirectional GRUs
```

---

## Feature Engineering

### Input Feature Dimensions

| Modality | Features | Dimensions | Description |
|----------|----------|------------|-------------|
| **Attendance** | 14 | `[0:14]` | Daily attendance patterns, rates, streaks |
| **Academic** | 15 | `[14:29]` | Grades, GPA trends, assignment completion |
| **Behavioral** | 13 | `[29:42]` | Discipline incidents, severity, frequency |

### Attendance Features (14 dimensions)

```python
attendance_features = {
    # Core metrics
    'attendance_rate_7d': float,      # Weekly attendance rate
    'attendance_rate_30d': float,     # Monthly attendance rate
    'attendance_rate_semester': float, # Semester attendance rate
    
    # Pattern analysis
    'consecutive_absences': int,      # Current absence streak
    'max_absence_streak': int,        # Longest absence streak
    'absence_frequency': float,       # Absences per week
    'tardy_frequency': float,         # Tardiness per week
    
    # Temporal trends
    'attendance_trend_slope': float,  # Linear trend coefficient
    'attendance_variance': float,     # Consistency measure
    'monday_absence_rate': float,     # Day-specific patterns
    'friday_absence_rate': float,
    
    # Advanced patterns
    'absence_clustering': float,      # Temporal clustering of absences
    'seasonal_adjustment': float,     # Adjustment for school calendar
    'attendance_momentum': float      # Recent change momentum
}
```

### Academic Features (15 dimensions)

```python
academic_features = {
    # Current performance
    'current_gpa': float,             # Most recent GPA
    'semester_gpa': float,            # Current semester GPA
    'cumulative_gpa': float,          # Overall GPA
    
    # Grade trends
    'gpa_trend_slope': float,         # GPA change rate
    'gpa_volatility': float,          # Grade consistency
    'failing_courses_count': int,     # Number of failing grades
    'failing_rate': float,            # Percentage of failing assignments
    
    # Assignment completion
    'assignment_completion_rate': float, # Homework completion rate
    'late_submission_rate': float,    # Late assignment rate
    'missing_assignment_count': int,  # Count of missing work
    
    # Subject-specific
    'math_performance': float,        # Math-specific GPA
    'english_performance': float,     # English-specific GPA
    'core_subject_avg': float,        # Core subjects average
    
    # Advanced metrics
    'grade_recovery_rate': float,     # Recovery from low grades
    'academic_momentum': float        # Recent academic trajectory
}
```

### Behavioral Features (13 dimensions)

```python
behavioral_features = {
    # Incident counts
    'discipline_incidents_30d': int,   # Recent incidents
    'discipline_incidents_semester': int, # Semester incidents
    'total_discipline_incidents': int, # All-time incidents
    
    # Severity analysis
    'avg_severity_level': float,      # Average incident severity
    'max_severity_level': int,        # Worst incident severity
    'severe_incidents_count': int,    # Count of severe incidents (level 4-5)
    
    # Behavioral patterns
    'incident_frequency': float,      # Incidents per month
    'incident_escalation': float,     # Severity trend
    'days_since_last_incident': int,  # Time since last incident
    
    # Type classification
    'disruptive_incidents': int,      # Classroom disruption
    'aggressive_incidents': int,      # Fighting, aggression
    'defiance_incidents': int,        # Authority defiance
    
    # Behavioral momentum
    'behavioral_improvement': float   # Recent behavior trajectory
}
```

### Feature Preprocessing Pipeline

```python
class FeaturePipeline:
    """Comprehensive feature extraction and preprocessing pipeline."""
    
    def extract_student_features(self, student_id: str, reference_date: date) -> Dict:
        """Extract all features for a student at a specific date."""
        
        # Set temporal window (20 weeks = ~5 months)
        end_date = reference_date - timedelta(days=7)  # 1-week lag
        start_date = end_date - timedelta(days=140)    # 20 weeks back
        
        features = {}
        
        # Extract modality-specific features
        features.update(self._extract_attendance_features(student_id, start_date, end_date))
        features.update(self._extract_academic_features(student_id, start_date, end_date))  
        features.update(self._extract_behavioral_features(student_id, start_date, end_date))
        
        # Normalize and validate
        features = self._normalize_features(features)
        features = self._validate_features(features)
        
        return features
    
    def _normalize_features(self, features: Dict) -> Dict:
        """Apply z-score normalization to continuous features."""
        
        # Define feature scaling parameters (learned from training data)
        scaling_params = {
            'attendance_rate_30d': {'mean': 0.87, 'std': 0.12},
            'current_gpa': {'mean': 2.8, 'std': 0.9},
            'discipline_incidents_30d': {'mean': 0.3, 'std': 0.8},
            # ... additional parameters for all features
        }
        
        normalized = {}
        for feature, value in features.items():
            if feature in scaling_params:
                params = scaling_params[feature]
                normalized[feature] = (value - params['mean']) / params['std']
            else:
                normalized[feature] = value  # Keep as-is for categorical features
                
        return normalized
```

---

## Neural Network Components

### Modality-Specific GRU Networks

Each data modality is processed by a dedicated bidirectional GRU network:

```python
# Attendance processing pathway
self.attendance_gru = nn.GRU(
    input_size=14,                    # Attendance feature dimensions
    hidden_size=hidden_size // 3,     # Balanced hidden state allocation
    num_layers=num_layers,            # Deep network (typically 2 layers)
    batch_first=True,
    bidirectional=bidirectional,      # Forward and backward temporal processing
    dropout=dropout if num_layers > 1 else 0
)

# Academic performance pathway  
self.grades_gru = nn.GRU(
    input_size=15,                    # Academic feature dimensions
    hidden_size=hidden_size // 3,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=bidirectional,
    dropout=dropout if num_layers > 1 else 0
)

# Behavioral pathway
self.discipline_gru = nn.GRU(
    input_size=13,                    # Behavioral feature dimensions  
    hidden_size=hidden_size // 3,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=bidirectional,
    dropout=dropout if num_layers > 1 else 0
)
```

**Why Separate GRUs?**
- **Specialized Processing**: Each modality has unique temporal patterns
- **Feature Independence**: Prevents cross-contamination between modalities
- **Interpretability**: Enables modality-specific attention analysis
- **Scalability**: Easy to add new modalities or modify existing ones

### Multi-Head Self-Attention

After GRU processing, features are combined and processed through multi-head attention:

```python
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for temporal pattern recognition."""
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query, Key, Value projection layers
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.shape
        
        # Project to Q, K, V and reshape for multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, hidden_size)
        
        return self.out_proj(attention_output), attention_weights
```

### Output Heads

The model has dual output heads for different prediction tasks:

```python
# Risk score regression head (continuous)
self.risk_score_head = nn.Sequential(
    nn.Linear(combined_hidden_size, 64),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(64, 1),
    nn.Sigmoid()  # Output between 0 and 1
)

# Risk category classification head (discrete)
self.risk_category_head = nn.Sequential(
    nn.Linear(combined_hidden_size, 64),
    nn.ReLU(), 
    nn.Dropout(dropout),
    nn.Linear(64, 4),  # 4 risk categories
    # No activation - raw logits for CrossEntropyLoss
)
```

---

## Training Process

### Loss Function

The model uses a combined loss function that balances regression and classification:

```python
def combined_loss(risk_pred, risk_true, category_pred, category_true, alpha=0.7):
    """
    Combined loss function for multi-task learning.
    
    Args:
        risk_pred: Predicted risk scores (0-1)
        risk_true: True risk scores (0-1)
        category_pred: Predicted category logits (4 classes)
        category_true: True category labels (0-3)
        alpha: Weight between regression and classification loss
    """
    
    # Risk score regression loss (MSE)
    regression_loss = F.mse_loss(risk_pred, risk_true)
    
    # Risk category classification loss (Cross-entropy)
    classification_loss = F.cross_entropy(category_pred, category_true)
    
    # Combined weighted loss
    total_loss = alpha * regression_loss + (1 - alpha) * classification_loss
    
    return total_loss, regression_loss, classification_loss
```

### Training Configuration

```python
training_config = {
    # Model hyperparameters
    'input_size': 42,
    'hidden_size': 128,
    'num_layers': 2,
    'num_heads': 4,
    'dropout': 0.3,
    'bidirectional': True,
    
    # Training hyperparameters
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15,
    'weight_decay': 0.01,
    
    # Sequence parameters
    'sequence_length': 20,      # 20 weeks of data
    'prediction_horizon': 30,   # Predict 30 days ahead
    'feature_lag': 7,          # 1-week lag to avoid data leakage
    
    # Loss weighting
    'loss_alpha': 0.7,         # Regression vs classification balance
    'gradient_clip_norm': 1.0, # Gradient clipping for stability
}
```

### Data Augmentation and Regularization

```python
class DropoutRegularization:
    """Advanced dropout strategies for temporal data."""
    
    def temporal_dropout(self, sequences, drop_prob=0.1):
        """Randomly drop entire time steps during training."""
        mask = torch.rand(sequences.size(0), sequences.size(1)) > drop_prob
        return sequences * mask.unsqueeze(-1)
    
    def feature_dropout(self, sequences, drop_prob=0.1):
        """Randomly drop entire features during training."""
        mask = torch.rand(sequences.size(0), sequences.size(2)) > drop_prob
        return sequences * mask.unsqueeze(1)
    
    def sequence_augmentation(self, sequences):
        """Apply sequence-level augmentation techniques."""
        
        # Random sequence length (between 15-20 weeks)
        seq_len = random.randint(15, 20)
        if sequences.size(1) > seq_len:
            start_idx = random.randint(0, sequences.size(1) - seq_len)
            sequences = sequences[:, start_idx:start_idx + seq_len, :]
        
        # Gaussian noise injection (small amount)
        noise = torch.randn_like(sequences) * 0.01
        sequences = sequences + noise
        
        return sequences
```

---

## Model Interpretability

### Attention Visualization

The model provides attention weights that show which time periods and features are most important:

```python
def get_attention_weights(self, sequences, student_id=None):
    """Extract attention weights for interpretability analysis."""
    
    with torch.no_grad():
        # Forward pass through modality-specific GRUs
        attendance_out, _ = self.attendance_gru(sequences[:, :, :14])
        grades_out, _ = self.grades_gru(sequences[:, :, 14:29]) 
        discipline_out, _ = self.discipline_gru(sequences[:, :, 29:42])
        
        # Combine modality outputs
        combined_features = torch.cat([
            attendance_out, grades_out, discipline_out
        ], dim=-1)
        
        # Apply attention and extract weights
        attention_out, attention_weights = self.attention(combined_features)
        
        return {
            'temporal_attention': attention_weights.cpu().numpy(),
            'feature_importance': self._compute_feature_importance(combined_features),
            'modality_contributions': self._compute_modality_contributions(
                attendance_out, grades_out, discipline_out
            )
        }
```

### Risk Factor Explanation

```python
class RiskExplainer:
    """Generate human-readable explanations for predictions."""
    
    def explain_prediction(self, student_id, features, attention_weights, risk_score):
        """Generate comprehensive risk factor explanations."""
        
        explanations = []
        
        # Analyze attention-weighted features
        feature_importance = self._compute_weighted_importance(features, attention_weights)
        
        # Top contributing factors
        top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for factor_name, importance in top_factors:
            explanation = self._generate_factor_explanation(
                factor_name, features[factor_name], importance
            )
            explanations.append(explanation)
        
        return explanations
    
    def _generate_factor_explanation(self, factor_name, value, importance):
        """Generate human-readable explanation for a specific factor."""
        
        explanations_map = {
            'attendance_rate_30d': {
                'description': f"Student has attended {value:.1%} of classes in the last 30 days",
                'threshold': 0.85,
                'recommendation': "Contact family to discuss attendance barriers" if value < 0.85 
                               else "Maintain current attendance patterns"
            },
            'current_gpa': {
                'description': f"Current GPA of {value:.2f}",
                'threshold': 2.5,
                'recommendation': "Academic support and tutoring intervention needed" if value < 2.5
                               else "Continue current academic support"
            },
            'discipline_incidents_30d': {
                'description': f"{int(value)} discipline incidents in the last 30 days",
                'threshold': 1,
                'recommendation': "Behavioral intervention and support plan needed" if value >= 1
                               else "Continue positive behavior reinforcement"
            }
        }
        
        factor_info = explanations_map.get(factor_name, {
            'description': f"Factor {factor_name}: {value}",
            'recommendation': "Monitor closely"
        })
        
        return {
            'factor': factor_name,
            'weight': importance,
            'value': value,
            'description': factor_info['description'],
            'impact': 'negative' if value < factor_info.get('threshold', 0) else 'positive',
            'recommendation': factor_info['recommendation']
        }
```

---

## Model Evaluation

### Performance Metrics

```python
class ModelEvaluator:
    """Comprehensive model evaluation metrics."""
    
    def evaluate_model(self, model, test_loader):
        """Calculate comprehensive evaluation metrics."""
        
        all_predictions = []
        all_targets = []
        all_categories = []
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                sequences, risk_targets, category_targets = batch
                
                risk_pred, category_pred = model(sequences)
                
                all_predictions.extend(risk_pred.cpu().numpy())
                all_targets.extend(risk_targets.cpu().numpy())
                all_categories.extend(category_targets.cpu().numpy())
        
        metrics = {}
        
        # Regression metrics
        metrics.update(self._regression_metrics(all_predictions, all_targets))
        
        # Classification metrics  
        category_pred = self._convert_risk_to_category(all_predictions)
        metrics.update(self._classification_metrics(category_pred, all_categories))
        
        # Business metrics
        metrics.update(self._business_metrics(all_predictions, all_targets))
        
        return metrics
    
    def _business_metrics(self, predictions, targets):
        """Calculate business-relevant metrics."""
        
        # Precision and Recall at different thresholds
        precisions_at_k = {}
        recalls_at_k = {}
        
        for k in [5, 10, 20, 50]:
            if len(predictions) >= k:
                top_k_indices = np.argsort(predictions)[-k:]
                top_k_targets = np.array(targets)[top_k_indices]
                
                # High-risk threshold (>0.5)
                high_risk_in_top_k = np.sum(top_k_targets > 0.5)
                total_high_risk = np.sum(np.array(targets) > 0.5)
                
                precisions_at_k[f'precision_at_{k}'] = high_risk_in_top_k / k
                recalls_at_k[f'recall_at_{k}'] = high_risk_in_top_k / max(total_high_risk, 1)
        
        return {**precisions_at_k, **recalls_at_k}
```

### Cross-Validation and Temporal Validation

```python
class TemporalValidator:
    """Time-aware validation to prevent data leakage."""
    
    def temporal_split(self, student_data, test_months=3):
        """Create temporal train/test split."""
        
        cutoff_date = datetime.now() - timedelta(days=test_months * 30)
        
        train_data = []
        test_data = []
        
        for student_id, sequences in student_data.items():
            for sequence in sequences:
                if sequence['reference_date'] < cutoff_date:
                    train_data.append(sequence)
                else:
                    test_data.append(sequence)
        
        return train_data, test_data
    
    def walk_forward_validation(self, student_data, window_months=6, step_months=1):
        """Walk-forward validation for time series."""
        
        results = []
        start_date = min(seq['reference_date'] for seq in student_data)
        end_date = max(seq['reference_date'] for seq in student_data)
        
        current_date = start_date + timedelta(days=window_months * 30)
        
        while current_date < end_date:
            # Training window
            train_end = current_date
            train_start = train_end - timedelta(days=window_months * 30)
            
            # Test window  
            test_start = current_date
            test_end = test_start + timedelta(days=step_months * 30)
            
            train_data = [seq for seq in student_data 
                         if train_start <= seq['reference_date'] < train_end]
            test_data = [seq for seq in student_data
                        if test_start <= seq['reference_date'] < test_end]
            
            if len(train_data) > 100 and len(test_data) > 20:
                fold_result = self._evaluate_fold(train_data, test_data)
                results.append(fold_result)
            
            current_date += timedelta(days=step_months * 30)
        
        return results
```

---

## Deployment and Inference

### Model Serving

```python
class PredictionService:
    """Production model serving with caching and error handling."""
    
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.feature_pipeline = FeaturePipeline()
        self.cache = {}  # Simple prediction cache
        
    def predict_risk(self, student_id, reference_date=None, include_factors=True):
        """Generate risk prediction with optional explanations."""
        
        # Check cache first
        cache_key = f"{student_id}_{reference_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Extract features
            features = self.feature_pipeline.extract_student_features(
                student_id, reference_date or date.today()
            )
            
            # Prepare sequence
            sequence = self._prepare_sequence(features)
            
            # Model inference
            with torch.no_grad():
                risk_score, category_logits = self.model(sequence)
                risk_score = risk_score.item()
                risk_category = self._logits_to_category(category_logits)
                confidence = self._calculate_confidence(risk_score, category_logits)
            
            # Generate explanations if requested
            explanations = None
            if include_factors:
                attention_weights = self.model.get_attention_weights(sequence)
                explanations = self._generate_explanations(
                    features, attention_weights, risk_score
                )
            
            result = {
                'student_id': student_id,
                'risk_score': risk_score,
                'risk_category': risk_category,
                'confidence': confidence,
                'explanations': explanations,
                'model_version': self.model.version,
                'prediction_date': datetime.now().isoformat()
            }
            
            # Cache result for 1 hour
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for student {student_id}: {e}")
            return self._fallback_prediction(student_id)
```

### Batch Processing

```python
def predict_batch(self, student_ids, batch_size=32):
    """Efficient batch prediction processing."""
    
    results = []
    
    for i in range(0, len(student_ids), batch_size):
        batch_ids = student_ids[i:i + batch_size]
        
        # Extract features for batch
        batch_sequences = []
        valid_ids = []
        
        for student_id in batch_ids:
            try:
                features = self.feature_pipeline.extract_student_features(student_id)
                sequence = self._prepare_sequence(features)
                batch_sequences.append(sequence)
                valid_ids.append(student_id)
            except Exception as e:
                logger.warning(f"Failed to extract features for {student_id}: {e}")
                continue
        
        if not batch_sequences:
            continue
            
        # Batch inference
        batch_tensor = torch.stack(batch_sequences)
        
        with torch.no_grad():
            risk_scores, category_logits = self.model(batch_tensor)
            
            for j, student_id in enumerate(valid_ids):
                result = {
                    'student_id': student_id,
                    'risk_score': risk_scores[j].item(),
                    'risk_category': self._logits_to_category(category_logits[j]),
                    'confidence': self._calculate_confidence(
                        risk_scores[j], category_logits[j]
                    )
                }
                results.append(result)
    
    return results
```

---

## Model Monitoring and Maintenance

### Performance Monitoring

```python
class ModelMonitor:
    """Monitor model performance in production."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        
    def log_prediction(self, prediction_result, processing_time):
        """Log prediction for monitoring."""
        
        self.metrics_collector.increment('predictions_total')
        self.metrics_collector.histogram('prediction_latency', processing_time)
        self.metrics_collector.histogram('risk_score_distribution', 
                                        prediction_result['risk_score'])
        
        # Log risk category distribution
        category = prediction_result['risk_category']
        self.metrics_collector.increment(f'predictions_by_category.{category}')
        
        # Log confidence distribution
        confidence = prediction_result['confidence']
        if confidence < 0.7:
            self.metrics_collector.increment('low_confidence_predictions')
    
    def detect_drift(self, recent_predictions, historical_baseline):
        """Detect distribution drift in predictions."""
        
        from scipy import stats
        
        # Extract risk scores
        recent_scores = [p['risk_score'] for p in recent_predictions]
        baseline_scores = historical_baseline['risk_scores']
        
        # Statistical tests for drift
        ks_stat, ks_p_value = stats.ks_2samp(recent_scores, baseline_scores)
        
        drift_detected = ks_p_value < 0.05  # Significance threshold
        
        if drift_detected:
            self.metrics_collector.increment('drift_alerts')
            logger.warning(f"Model drift detected: KS statistic={ks_stat:.4f}, p={ks_p_value:.4f}")
        
        return {
            'drift_detected': drift_detected,
            'ks_statistic': ks_stat,
            'p_value': ks_p_value,
            'recent_mean': np.mean(recent_scores),
            'baseline_mean': np.mean(baseline_scores)
        }
```

### Model Updates and Versioning

```python
class ModelUpdater:
    """Handle model updates and version management."""
    
    def __init__(self):
        self.model_registry = ModelRegistry()
        
    def trigger_retraining(self, feedback_data, config_updates=None):
        """Trigger model retraining with new feedback."""
        
        training_config = self._get_base_config()
        if config_updates:
            training_config.update(config_updates)
        
        # Prepare training data with new feedback
        training_data = self._prepare_training_data(feedback_data)
        
        # Start training job
        training_job = ModelTrainer(training_config)
        new_model = training_job.fit(training_data)
        
        # Validate new model
        validation_metrics = self._validate_model(new_model)
        
        if validation_metrics['accuracy'] > self.current_model_accuracy * 0.95:
            # Deploy new model if performance is acceptable
            new_version = self._deploy_model(new_model, validation_metrics)
            logger.info(f"Deployed new model version: {new_version}")
            return new_version
        else:
            logger.warning("New model performance insufficient, keeping current model")
            return None
```

---

## Future Enhancements

### Planned Improvements

1. **Transformer Architecture**: Evaluate self-attention without recurrence
2. **Multi-Task Learning**: Add graduation prediction and intervention recommendation
3. **Federated Learning**: Enable privacy-preserving multi-school training
4. **Causal Inference**: Incorporate causal reasoning for intervention planning
5. **Real-Time Updates**: Streaming model updates with new data
6. **Ensemble Methods**: Combine multiple models for improved robustness

### Research Directions

- **Graph Neural Networks**: Model social and academic relationships
- **Meta-Learning**: Quick adaptation to new schools/districts
- **Uncertainty Quantification**: Better confidence estimation
- **Fairness-Aware ML**: Reduce bias across demographic groups
- **Counterfactual Explanations**: "What-if" analysis for interventions

---

This comprehensive documentation provides the foundation for understanding, maintaining, and extending EduPulse's machine learning architecture. The model's design prioritizes both predictive accuracy and interpretability to serve the educational mission of supporting at-risk students.
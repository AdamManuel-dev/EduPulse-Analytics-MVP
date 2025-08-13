# Machine Learning Model Module

## Overview

The ML model module implements a sophisticated GRU-based neural network with multi-head attention for temporal student risk prediction. The model processes multi-modal time-series data to predict both risk scores and categories.

## Model Architecture

### High-Level Architecture

```
Input (42 features)
       │
    Split by Modality
       │
┌──────┼──────┐
▼      ▼      ▼
GRU    GRU    GRU
(Att)  (Grd)  (Dis)
│      │      │
└──────┼──────┘
       │
  Concatenate
       │
Multi-Head Attention
       │
  Feature Fusion
       │
    ┌──┴──┐
    │     │
Risk Score  Category
  (0-1)    (4 classes)
```

### Detailed Components

#### 1. Input Layer
- **Dimensions**: (batch_size, sequence_length, 42)
- **Features**: 14 attendance + 15 grades + 13 discipline

#### 2. Modality-Specific GRUs
Three parallel GRU networks process different aspects:

**Attendance GRU**:
- Input: 14 features
- Hidden size: 42 (128/3)
- Layers: 2
- Bidirectional: True

**Grades GRU**:
- Input: 15 features
- Hidden size: 42
- Layers: 2
- Bidirectional: True

**Discipline GRU**:
- Input: 13 features
- Hidden size: 42
- Layers: 2
- Bidirectional: True

#### 3. Multi-Head Attention
- **Heads**: 4
- **Embed dimension**: 256 (combined hidden)
- **Dropout**: 0.3
- **Purpose**: Capture temporal dependencies

#### 4. Feature Fusion
Sequential layers for combining modalities:
```
Linear(256 → 128) → ReLU → Dropout(0.3) →
Linear(128 → 64) → ReLU → Dropout(0.3)
```

#### 5. Output Heads

**Risk Score Head**:
```
Linear(64 → 32) → ReLU → Dropout(0.3) →
Linear(32 → 1) → Sigmoid
```

**Category Head**:
```
Linear(64 → 32) → ReLU → Dropout(0.3) →
Linear(32 → 4) → Softmax
```

## Risk Categories

The model predicts four risk levels:

| Category | Risk Score Range | Description | Intervention |
|----------|-----------------|-------------|--------------|
| Low | 0.0 - 0.25 | On track | Standard monitoring |
| Medium | 0.25 - 0.50 | Early warning | Proactive support |
| High | 0.50 - 0.75 | At risk | Immediate intervention |
| Critical | 0.75 - 1.0 | Crisis | Emergency response |

## Training Process

### Dataset Preparation

```python
class StudentSequenceDataset(Dataset):
    def __init__(self, sequences, labels, seq_length=10):
        self.sequences = sequences
        self.labels = labels
        self.seq_length = seq_length
    
    def __getitem__(self, idx):
        # Apply sliding window
        seq = self.sequences[idx:idx+self.seq_length]
        label = self.labels[idx+self.seq_length-1]
        return torch.FloatTensor(seq), torch.FloatTensor(label)
```

### Loss Functions

**Combined Loss**:
```python
def combined_loss(risk_pred, cat_pred, risk_true, cat_true):
    mse_loss = F.mse_loss(risk_pred, risk_true)
    ce_loss = F.cross_entropy(cat_pred, cat_true)
    return 0.5 * mse_loss + 0.5 * ce_loss
```

### Training Configuration

```python
config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping_patience': 10,
    'gradient_clip': 1.0,
    'weight_decay': 0.0001,
    'scheduler': 'ReduceLROnPlateau',
    'optimizer': 'Adam'
}
```

### Training Loop

```python
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        sequences, labels = batch
        
        # Forward pass
        risk_scores, categories, _ = model(sequences)
        loss = criterion(risk_scores, categories, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

## Model Evaluation

### Metrics

**Risk Score Metrics**:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score

**Category Metrics**:
- Accuracy
- Precision per class
- Recall per class
- F1 Score
- Confusion Matrix
- AUC-ROC

### Evaluation Code

```python
def evaluate_model(model, test_loader):
    model.eval()
    metrics = {
        'mae': [],
        'accuracy': [],
        'precision': [],
        'recall': []
    }
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            risk_scores, categories, _ = model(sequences)
            
            # Calculate metrics
            mae = mean_absolute_error(labels[:, 0], risk_scores)
            acc = accuracy_score(labels[:, 1], categories.argmax(1))
            
            metrics['mae'].append(mae)
            metrics['accuracy'].append(acc)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

## Interpretability

### Attention Weights

Extract attention weights for understanding model focus:

```python
def get_attention_analysis(model, sequence):
    model.eval()
    with torch.no_grad():
        _, _, attention_weights = model(sequence, return_attention=True)
    
    # Visualize attention
    plt.imshow(attention_weights[0].cpu().numpy())
    plt.colorbar()
    plt.xlabel('Time Steps')
    plt.ylabel('Features')
    plt.title('Attention Heatmap')
```

### Feature Importance

```python
def calculate_feature_importance(model, data_loader):
    importances = np.zeros(42)
    
    for sequences, _ in data_loader:
        sequences.requires_grad = True
        risk_scores, _, _ = model(sequences)
        
        # Calculate gradients
        risk_scores.sum().backward()
        
        # Aggregate importance
        importances += sequences.grad.abs().mean(dim=(0, 1)).numpy()
    
    return importances / len(data_loader)
```

## Model Deployment

### Serialization

```python
def save_model(model, path='models/gru_model.pt'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
        'feature_stats': model.feature_stats,
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }, path)
```

### Loading

```python
def load_model(path='models/gru_model.pt'):
    checkpoint = torch.load(path)
    
    model = GRUAttentionModel(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model
```

### Optimization for Inference

**Quantization**:
```python
# Dynamic quantization for CPU inference
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.GRU}, dtype=torch.qint8
)
```

**TorchScript**:
```python
# Compile for production
scripted_model = torch.jit.script(model)
scripted_model.save('models/gru_model_scripted.pt')
```

## Performance Benchmarks

### Inference Speed

| Batch Size | CPU (ms) | GPU (ms) | Quantized (ms) |
|------------|----------|----------|----------------|
| 1 | 15 | 3 | 8 |
| 32 | 120 | 10 | 45 |
| 128 | 450 | 25 | 150 |

### Model Size

| Version | Size (MB) | Accuracy |
|---------|-----------|----------|
| Full Precision | 12.5 | 89.2% |
| Quantized | 3.2 | 88.7% |
| Pruned | 8.1 | 88.9% |

## Hyperparameter Tuning

### Grid Search Configuration

```python
param_grid = {
    'hidden_size': [64, 128, 256],
    'num_layers': [1, 2, 3],
    'num_heads': [2, 4, 8],
    'dropout': [0.2, 0.3, 0.4],
    'learning_rate': [0.0001, 0.001, 0.01]
}
```

### Optuna Integration

```python
def objective(trial):
    config = {
        'hidden_size': trial.suggest_int('hidden_size', 64, 256),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2)
    }
    
    model = GRUAttentionModel(**config)
    accuracy = train_and_evaluate(model)
    
    return accuracy
```

## Continuous Learning

### Online Learning Pipeline

```python
def update_model_online(model, new_data, learning_rate=0.0001):
    """Update model with new data without full retraining."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for batch in new_data:
        sequences, labels = batch
        
        # Forward pass
        predictions = model(sequences)
        loss = calculate_loss(predictions, labels)
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model
```

### A/B Testing Framework

```python
class ModelABTest:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
    
    def predict(self, student_id, features):
        if hash(student_id) % 100 < self.split_ratio * 100:
            return self.model_a.predict(features), 'A'
        else:
            return self.model_b.predict(features), 'B'
```

## Monitoring

### Model Drift Detection

```python
def detect_drift(current_predictions, baseline_predictions, threshold=0.05):
    """Detect if model predictions have drifted."""
    ks_statistic, p_value = ks_2samp(current_predictions, baseline_predictions)
    
    if p_value < threshold:
        alert("Model drift detected", severity="high")
        return True
    return False
```

### Performance Tracking

```python
metrics_to_track = {
    'daily_accuracy': [],
    'daily_mae': [],
    'inference_latency': [],
    'prediction_volume': [],
    'false_positive_rate': [],
    'false_negative_rate': []
}
```

## Troubleshooting

### Common Issues

**Issue**: Model overfitting
- **Solution**: Increase dropout, add L2 regularization, reduce model size

**Issue**: Poor convergence
- **Solution**: Adjust learning rate, use gradient clipping, normalize inputs

**Issue**: Class imbalance
- **Solution**: Use weighted loss, oversample minority class, adjust thresholds

## Future Improvements

1. **Architecture Enhancements**:
   - Transformer-based models
   - Graph neural networks for peer relationships
   - Ensemble methods

2. **Feature Engineering**:
   - Automated feature learning
   - Cross-modal attention
   - Temporal convolutions

3. **Deployment**:
   - Edge deployment for privacy
   - Federated learning
   - Real-time streaming predictions

---

*For implementation details, see [source code](../../src/models/).*
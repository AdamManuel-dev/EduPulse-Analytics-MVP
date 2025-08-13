"""
GRU-based neural network model for student risk prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class GRUAttentionModel(nn.Module):
    """
    GRU model with self-attention mechanism for temporal student data.
    """
    
    def __init__(
        self,
        input_size: int = 42,  # Number of features
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize the GRU attention model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of GRU hidden state
            num_layers: Number of GRU layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super(GRUAttentionModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # GRU layers for each modality
        self.attendance_gru = nn.GRU(
            input_size=14,  # Attendance features
            hidden_size=hidden_size // 3,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.grades_gru = nn.GRU(
            input_size=15,  # Grade features
            hidden_size=hidden_size // 3,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.discipline_gru = nn.GRU(
            input_size=13,  # Discipline features
            hidden_size=hidden_size // 3,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust hidden size for concatenation
        self.combined_hidden_size = (hidden_size // 3) * 3 * self.num_directions
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.combined_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.combined_hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.combined_hidden_size)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Feature fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.combined_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers for risk prediction
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Risk score between 0 and 1
        )
        
        # Category classification head
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 4)  # 4 risk categories: low, medium, high, critical
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (risk_score, risk_category_logits, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Split features by modality
        attendance_features = x[:, :, :14]
        grades_features = x[:, :, 14:29]
        discipline_features = x[:, :, 29:42]
        
        # Process each modality through its GRU
        attendance_out, _ = self.attendance_gru(attendance_features)
        grades_out, _ = self.grades_gru(grades_features)
        discipline_out, _ = self.discipline_gru(discipline_features)
        
        # Concatenate modality outputs
        combined = torch.cat([attendance_out, grades_out, discipline_out], dim=-1)
        
        # Apply layer normalization
        combined = self.layer_norm1(combined)
        
        # Apply self-attention
        if return_attention:
            attended, attention_weights = self.attention(
                combined, combined, combined, 
                need_weights=True, average_attn_weights=True
            )
        else:
            attended, _ = self.attention(
                combined, combined, combined,
                need_weights=False
            )
            attention_weights = None
        
        # Residual connection and normalization
        combined = self.layer_norm2(combined + attended)
        combined = self.dropout(combined)
        
        # Global pooling (take the last time step for now)
        # Could also use mean pooling or attention-weighted pooling
        pooled = combined[:, -1, :]
        
        # Feature fusion
        fused = self.fusion_layer(pooled)
        
        # Generate outputs
        risk_score = self.risk_head(fused)
        risk_category_logits = self.category_head(fused)
        
        return risk_score, risk_category_logits, attention_weights
    
    def predict(self, x: torch.Tensor) -> Tuple[float, str, float]:
        """
        Make a prediction for a single sample.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (risk_score, risk_category, confidence)
        """
        self.eval()
        
        with torch.no_grad():
            risk_score, category_logits, _ = self.forward(x)
            
            # Get risk score
            risk_value = risk_score.item()
            
            # Get category prediction
            category_probs = F.softmax(category_logits, dim=-1)
            category_idx = torch.argmax(category_probs, dim=-1).item()
            confidence = category_probs[0, category_idx].item()
            
            # Map to category names
            categories = ['low', 'medium', 'high', 'critical']
            risk_category = categories[category_idx]
        
        return risk_value, risk_category, confidence
    
    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract attention weights for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights as numpy array
        """
        self.eval()
        
        with torch.no_grad():
            _, _, attention_weights = self.forward(x, return_attention=True)
            
            if attention_weights is not None:
                return attention_weights.cpu().numpy()
            else:
                return np.array([])


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop
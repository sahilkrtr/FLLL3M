"""
Lightweight Transformer Model for FLLL³M

This module implements the lightweight transformer encoder used as the local
client model in the FLLL³M framework. The model predicts the next embedding
given a sequence of previous embeddings using masked attention.

Architecture:
- 6 transformer layers
- 4 attention heads
- 256 hidden size
- Masked attention for sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, Union

from config.model_config import TransformerConfig

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        
        assert self.head_dim * config.num_heads == config.hidden_size, \
            "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Output projection
        output = self.output(context)
        
        return output

class TransformerLayer(nn.Module):
    """Single transformer layer with attention and feed-forward network."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        
        # Multi-head attention
        self.attention = MultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        self.ff_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attention_output = self.attention(x, mask)
        x = self.attention_norm(x + attention_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + ff_output)
        
        return x

class LightweightTransformer(nn.Module):
    """
    Lightweight Transformer Encoder for FLLL³M.
    
    This model predicts the next embedding given a sequence of previous embeddings
    using masked attention. It serves as the local client model in federated learning.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Input projection (if needed)
        self.input_projection = nn.Linear(
            config.combined_embedding_dim, 
            config.hidden_size
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(
            config.hidden_size, 
            config.combined_embedding_dim
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask
        
    def forward(self, 
                embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            mask: Optional attention mask
            
        Returns:
            Output embeddings [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, embedding_dim = embeddings.shape
        
        # Input projection
        x = self.input_projection(embeddings)
        x = self.dropout(x)
        
        # Create causal mask if not provided
        if mask is None:
            mask = self.create_causal_mask(seq_len, x.device)
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
            
        # Output projection
        output = self.output_projection(x)
        
        return output
        
    def predict_next_embedding(self, 
                              embeddings: torch.Tensor, 
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict the next embedding given a sequence.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            mask: Optional attention mask
            
        Returns:
            Predicted next embedding [batch_size, embedding_dim]
        """
        # Get transformer output
        transformer_output = self.forward(embeddings, mask)
        
        # Return the last position prediction
        return transformer_output[:, -1, :]
        
    def compute_loss(self, 
                    embeddings: torch.Tensor, 
                    targets: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute MSE loss between predictions and targets.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            targets: Target embeddings [batch_size, seq_len, embedding_dim]
            mask: Optional attention mask
            
        Returns:
            MSE loss
        """
        predictions = self.forward(embeddings, mask)
        loss = F.mse_loss(predictions, targets, reduction='mean')
        return loss
        
    def get_outer_product(self, 
                         embeddings: torch.Tensor, 
                         next_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute outer product for federated aggregation.
        
        Args:
            embeddings: Current embeddings [batch_size, seq_len, embedding_dim]
            next_embeddings: Next embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            Outer products [batch_size, seq_len, embedding_dim, embedding_dim]
        """
        batch_size, seq_len, embedding_dim = embeddings.shape
        
        # Reshape for outer product computation
        e_t = embeddings.view(batch_size, seq_len, embedding_dim, 1)
        e_t_plus_1 = next_embeddings.view(batch_size, seq_len, 1, embedding_dim)
        
        # Compute outer product: O_t = e_t ⊗ e_{t+1}
        outer_products = torch.matmul(e_t, e_t_plus_1)
        
        return outer_products
        
    def add_noise_for_privacy(self, 
                             outer_products: torch.Tensor, 
                             noise_std: float = 0.1) -> torch.Tensor:
        """
        Add Gaussian noise for differential privacy.
        
        Args:
            outer_products: Outer products [batch_size, seq_len, dim, dim]
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Noisy outer products
        """
        noise = torch.randn_like(outer_products) * noise_std
        return outer_products + noise
        
    def save_model(self, filepath: str) -> None:
        """Save the model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__
        }, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """Load the model."""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")
        
    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        } 
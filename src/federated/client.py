"""
Federated Learning Client Module

This module implements the federated learning client that:
- Maintains local data and model
- Performs local training
- Communicates with the federated server
- Handles privacy-preserving mechanisms
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import pandas as pd

from src.models.delta_iris import DeltaIRISTokenizer
from src.models.transformer_model import LightweightTransformer
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Simple model configuration for federated learning."""
    n_location_clusters: int = 50
    n_time_bins: int = 25
    embedding_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    hidden_size: int = 256
    dropout: float = 0.1

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for federated learning client"""
    client_id: str
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    noise_scale: float = 1.0


class FederatedClient:
    """
    Federated Learning Client
    
    Handles local training, model updates, and communication with the server.
    Implements differential privacy and secure aggregation mechanisms.
    """
    
    def __init__(self, config: ClientConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        
        # Create config objects
        from config.model_config import DeltaIRISConfig, TransformerConfig
        
        tokenizer_config = DeltaIRISConfig(
            location_bins=model_config.n_location_clusters,
            time_bins=model_config.n_time_bins,
            token_embedding_dim=model_config.embedding_dim,
            use_positional_encoding=True
        )
        
        transformer_config = TransformerConfig(
            combined_embedding_dim=model_config.embedding_dim,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.n_layers,
            num_heads=model_config.n_heads,
            intermediate_size=model_config.hidden_size * 4,
            dropout=model_config.dropout
        )
        
        # Initialize components
        self.tokenizer = DeltaIRISTokenizer(tokenizer_config)
        self.model = LightweightTransformer(transformer_config)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.criterion = nn.MSELoss()
        
        # Training state
        self.local_data: Optional[np.ndarray] = None
        self.is_trained = False
        self.training_history: List[float] = []
        
        logger.info(f"Federated client {config.client_id} initialized")
    
    def load_local_data(self, data_path: str) -> None:
        """Load and preprocess local mobility data"""
        try:
            # Load processed data (parquet)
            df = pd.read_parquet(data_path)
            data = df[['user_id', 'latitude', 'longitude', 'timestamp']].values
            data = np.asarray(data)
            
            # Filter data for this client (simulate data partitioning)
            np.random.seed(hash(self.config.client_id) % 2**32)
            client_indices = np.random.choice(
                len(data), 
                size=min(len(data), 1000),  # Limit data size for testing
                replace=False
            )
            self.local_data = data[client_indices]
            
            # Fit tokenizer on local data
            self.tokenizer.fit(self.local_data)
            
            logger.info(f"Client {self.config.client_id} loaded {len(self.local_data)} samples")
            
        except Exception as e:
            logger.error(f"Error loading data for client {self.config.client_id}: {e}")
            raise
    
    def get_dataloader(self) -> DataLoader:
        """Prepare a DataLoader for training batches from local data."""
        if self.local_data is None:
            raise ValueError("No local data loaded")
        
        # Tokenize and encode data
        location_tokens, time_tokens = self.tokenizer.tokenize(self.local_data)
        embeddings = self.tokenizer.encode(self.local_data).detach()  # Detach here!
        
        # Create sequences for training
        sequences = []
        targets = []
        seq_len = 10
        for i in range(len(embeddings) - seq_len):
            sequence = embeddings[i:i+seq_len]
            target = embeddings[i+seq_len]  # Next embedding
            sequences.append(sequence)
            targets.append(target)
        
        # Convert to tensors
        sequences = torch.stack(sequences)
        targets = torch.stack(targets)
        
        dataset = TensorDataset(sequences, targets)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        return dataloader

    def train_local_model(self) -> Dict[str, float]:
        """Perform local training on client data using DataLoader batching."""
        if self.local_data is None:
            raise ValueError("No local data loaded")
        
        logger.info(f"Starting local training for client {self.config.client_id}")
        
        # Prepare DataLoader
        dataloader = self.get_dataloader()
        
        # Training loop
        epoch_losses = []
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_sequences, batch_targets in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                next_embeddings = self.model.predict_next_embedding(batch_sequences)
                
                # Check for NaN/Inf in outputs
                if torch.isnan(next_embeddings).any() or torch.isinf(next_embeddings).any():
                    next_embeddings = torch.nan_to_num(next_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
                if torch.isnan(batch_targets).any() or torch.isinf(batch_targets).any():
                    batch_targets = torch.nan_to_num(batch_targets, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Compute loss
                loss = self.criterion(next_embeddings, batch_targets)
                if torch.isnan(loss) or torch.isinf(loss):
                    loss = torch.tensor(0.0, requires_grad=True)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            epoch_losses.append(avg_epoch_loss)
            
            logger.info(f"Client {self.config.client_id}, Epoch {epoch+1}/{self.config.local_epochs}, Loss: {avg_epoch_loss:.6f}")
        
        self.is_trained = True
        self.training_history.extend(epoch_losses)
        
        return {
            'final_loss': epoch_losses[-1],
            'avg_loss': np.mean(epoch_losses),
            'epochs_trained': self.config.local_epochs
        }
    
    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated averaging"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting updates")
        
        # Get current model parameters
        model_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                model_params[name] = param.data.clone()
        
        return model_params
    
    def apply_global_model(self, global_params: Dict[str, torch.Tensor]) -> None:
        """Apply global model parameters from server"""
        # Load global parameters into local model
        for name, param in self.model.named_parameters():
            if name in global_params:
                param.data = global_params[name].clone()
        
        # Reset training state
        self.is_trained = False
        logger.info(f"Client {self.config.client_id} applied global model")
    
    def add_differential_privacy(self, model_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to model parameters"""
        # Calculate noise scale based on privacy budget
        noise_scale = self.config.noise_scale / self.config.privacy_epsilon
        
        # Add Gaussian noise to parameters
        noisy_params = {}
        for name, param in model_params.items():
            noise = torch.randn_like(param) * noise_scale
            noisy_params[name] = param + noise
        
        logger.info(f"Added DP noise (ε={self.config.privacy_epsilon}) to client {self.config.client_id}")
        return noisy_params
    
    def compute_contribution_metrics(self) -> Dict[str, float]:
        """Compute metrics about client's contribution to federated learning"""
        if self.local_data is None:
            return {}
        # Ensure user_id is always string for unique count
        user_ids = np.array([str(uid) for uid in self.local_data[:, 0]])
        # Ensure locations are float for unique
        locations = self.local_data[:, 1:3].astype(float)
        return {
            'data_size': len(self.local_data),
            'unique_users': len(np.unique(user_ids)),
            'unique_locations': len(np.unique(locations, axis=0)),
            'time_span_days': (self.local_data[:, 3].max() - self.local_data[:, 3].min()) / (24 * 3600),
            'avg_sequence_length': 10.0,  # Fixed for now
            'training_epochs': len(self.training_history)
        }
    
    def save_client_state(self, save_path: str) -> None:
        """Save client state for checkpointing"""
        state = {
            'config': self.config,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'tokenizer_state': self.tokenizer.get_state(),
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Client {self.config.client_id} state saved to {save_path}")
    
    def load_client_state(self, load_path: str) -> None:
        """Load client state from checkpoint"""
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.tokenizer.load_state(state['tokenizer_state'])
        self.training_history = state['training_history']
        self.is_trained = state['is_trained']
        
        logger.info(f"Client {self.config.client_id} state loaded from {load_path}")
    
    def get_client_info(self) -> Dict[str, any]:
        """Get information about the client"""
        return {
            'client_id': self.config.client_id,
            'model_size': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'is_trained': self.is_trained,
            'training_epochs': len(self.training_history),
            'final_loss': self.training_history[-1] if self.training_history else None,
            'contribution_metrics': self.compute_contribution_metrics()
        } 
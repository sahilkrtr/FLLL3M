"""
Federated Learning Server Module

This module implements the federated learning server that:
- Orchestrates federated learning rounds
- Implements FedAvg algorithm
- Manages client communication
- Handles model aggregation and distribution
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import json

from src.models.delta_iris import DeltaIRISTokenizer
from src.models.transformer_model import LightweightTransformer
from .client import FederatedClient, ClientConfig, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for federated learning server"""
    num_rounds: int = 100
    num_clients_per_round: int = 10
    min_clients_per_round: int = 5
    aggregation_method: str = "fedavg"  # fedavg, fedprox, fednova
    client_selection_strategy: str = "random"  # random, weighted, adaptive
    evaluation_frequency: int = 5
    save_frequency: int = 10
    early_stopping_patience: int = 20
    convergence_threshold: float = 1e-4
    secure_aggregation: bool = False
    differential_privacy: bool = True


class FederatedServer:
    """
    Federated Learning Server
    
    Orchestrates federated learning rounds, implements FedAvg,
    and manages client communication and model aggregation.
    """
    
    def __init__(self, config: ServerConfig, model_config: ModelConfig):
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
        
        # Initialize global model
        self.global_model = LightweightTransformer(transformer_config)
        
        # Global tokenizer (will be updated from clients)
        self.global_tokenizer = DeltaIRISTokenizer(tokenizer_config)
        
        # Training state
        self.round_history: List[Dict[str, Any]] = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.current_round = 0
        
        # Client management
        self.clients: Dict[str, FederatedClient] = {}
        self.client_weights: Dict[str, float] = {}
        self.client_metrics: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Federated server initialized")
    
    def register_client(self, client: FederatedClient) -> None:
        """Register a client with the server"""
        client_id = client.config.client_id
        self.clients[client_id] = client
        
        # Calculate client weight based on data size
        if client.local_data is not None:
            self.client_weights[client_id] = len(client.local_data)
        else:
            self.client_weights[client_id] = 1.0
        
        logger.info(f"Client {client_id} registered with weight {self.client_weights[client_id]}")
    
    def select_clients_for_round(self) -> List[str]:
        """Select clients to participate in the current round"""
        available_clients = list(self.clients.keys())
        
        if len(available_clients) < self.config.min_clients_per_round:
            raise ValueError(f"Not enough clients available. Need at least {self.config.min_clients_per_round}")
        
        if self.config.client_selection_strategy == "random":
            # Random selection
            selected = np.random.choice(
                available_clients,
                size=min(self.config.num_clients_per_round, len(available_clients)),
                replace=False
            )
        elif self.config.client_selection_strategy == "weighted":
            # Weighted selection based on data size
            weights = [self.client_weights[client_id] for client_id in available_clients]
            weights = np.array(weights) / sum(weights)
            selected = np.random.choice(
                available_clients,
                size=min(self.config.num_clients_per_round, len(available_clients)),
                replace=False,
                p=weights
            )
        else:
            # Default to random
            selected = np.random.choice(
                available_clients,
                size=min(self.config.num_clients_per_round, len(available_clients)),
                replace=False
            )
        
        return selected.tolist()
    
    def distribute_global_model(self, selected_clients: List[str]) -> None:
        """Distribute global model to selected clients"""
        # Get global model parameters
        global_params = {}
        for name, param in self.global_model.named_parameters():
            if param.requires_grad:
                global_params[name] = param.data.clone()
        
        # Distribute to clients
        for client_id in selected_clients:
            client = self.clients[client_id]
            client.apply_global_model(global_params)
        
        logger.info(f"Distributed global model to {len(selected_clients)} clients")
    
    def collect_client_updates(self, selected_clients: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collect model updates from selected clients"""
        client_updates = {}
        
        for client_id in selected_clients:
            try:
                client = self.clients[client_id]
                
                # Train client model
                training_metrics = client.train_local_model()
                
                # Get model update
                model_update = client.get_model_update()
                
                # Add differential privacy if enabled
                if self.config.differential_privacy:
                    model_update = client.add_differential_privacy(model_update)
                
                client_updates[client_id] = model_update
                
                # Store client metrics
                self.client_metrics[client_id] = {
                    'training_metrics': training_metrics,
                    'contribution_metrics': client.compute_contribution_metrics(),
                    'round': self.current_round
                }
                
                logger.info(f"Collected update from client {client_id}, loss: {training_metrics['final_loss']:.6f}")
                
            except Exception as e:
                logger.error(f"Error collecting update from client {client_id}: {e}")
                continue
        
        return client_updates
    
    def aggregate_model_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Aggregate client model updates using FedAvg"""
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return
        
        # Calculate aggregation weights
        total_weight = sum(self.client_weights[client_id] for client_id in client_updates.keys())
        
        # Initialize aggregated parameters
        aggregated_params = {}
        param_names = list(client_updates[list(client_updates.keys())[0]].keys())
        
        for param_name in param_names:
            aggregated_params[param_name] = torch.zeros_like(
                client_updates[list(client_updates.keys())[0]][param_name]
            )
        
        # Weighted averaging
        for client_id, update in client_updates.items():
            weight = self.client_weights[client_id] / total_weight
            
            for param_name, param_update in update.items():
                aggregated_params[param_name] += weight * param_update
        
        # Update global model
        for name, param in self.global_model.named_parameters():
            if name in aggregated_params:
                param.data = aggregated_params[name]
        
        logger.info(f"Aggregated updates from {len(client_updates)} clients")
    
    def evaluate_global_model(self, test_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate the global model performance"""
        # Fit the global tokenizer if not already fitted
        if not self.global_tokenizer.is_fitted:
            all_client_data = []
            for client in self.clients.values():
                if client.local_data is not None:
                    all_client_data.append(client.local_data)
            if all_client_data:
                combined_data = np.vstack(all_client_data)
                self.global_tokenizer.fit(combined_data)
        
        if test_data is None:
            # Use a sample from clients for evaluation
            test_data = self._get_evaluation_data()
        
        # Tokenize and encode test data
        location_tokens, time_tokens = self.global_tokenizer.tokenize(test_data)
        embeddings = self.global_tokenizer.encode(test_data)
        
        # Create test sequences
        test_sequences = []
        test_targets = []
        
        for i in range(len(embeddings) - 10):
            sequence = embeddings[i:i+10]
            target = embeddings[i+10]
            test_sequences.append(sequence)
            test_targets.append(target)
        
        if not test_sequences:
            return {'loss': float('inf')}
        
        # Convert to tensors
        test_sequences = torch.stack(test_sequences)
        test_targets = torch.stack(test_targets)
        
        # Evaluate
        self.global_model.eval()
        with torch.no_grad():
            next_embeddings = self.global_model.predict_next_embedding(test_sequences)
            loss = nn.MSELoss()(next_embeddings, test_targets)
        
        self.global_model.train()
        
        return {'loss': loss.item()}
    
    def _get_evaluation_data(self) -> np.ndarray:
        """Get evaluation data from clients"""
        # Combine data from all clients for evaluation
        all_data = []
        for client in self.clients.values():
            if client.local_data is not None:
                all_data.append(client.local_data)
        
        if not all_data:
            return np.array([])
        
        # Take a sample for evaluation
        combined_data = np.vstack(all_data)
        sample_size = min(1000, len(combined_data))
        indices = np.random.choice(len(combined_data), sample_size, replace=False)
        
        return combined_data[indices]
    
    def run_federated_learning(self, test_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run the complete federated learning process"""
        logger.info(f"Starting federated learning for {self.config.num_rounds} rounds")
        
        start_time = time.time()
        
        for round_num in range(self.config.num_rounds):
            self.current_round = round_num
            
            logger.info(f"\n=== Round {round_num + 1}/{self.config.num_rounds} ===")
            
            # Select clients for this round
            selected_clients = self.select_clients_for_round()
            logger.info(f"Selected clients: {selected_clients}")
            
            # Distribute global model
            self.distribute_global_model(selected_clients)
            
            # Collect client updates
            client_updates = self.collect_client_updates(selected_clients)
            
            # Aggregate updates
            self.aggregate_model_updates(client_updates)
            
            # Evaluate global model
            if (round_num + 1) % self.config.evaluation_frequency == 0:
                eval_metrics = self.evaluate_global_model(test_data)
                logger.info(f"Round {round_num + 1} evaluation: {eval_metrics}")
                
                # Check for improvement
                current_loss = eval_metrics['loss']
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at round {round_num + 1}")
                    break
            
            # Save checkpoint
            if (round_num + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(f"checkpoints/round_{round_num + 1}")
            
            # Store round history
            round_info = {
                'round': round_num + 1,
                'selected_clients': selected_clients,
                'num_updates': len(client_updates),
                'evaluation_metrics': eval_metrics if (round_num + 1) % self.config.evaluation_frequency == 0 else None
            }
            self.round_history.append(round_info)
        
        total_time = time.time() - start_time
        
        # Final evaluation
        final_eval = self.evaluate_global_model(test_data)
        
        results = {
            'total_rounds': self.current_round + 1,
            'total_time': total_time,
            'final_evaluation': final_eval,
            'best_loss': self.best_loss,
            'round_history': self.round_history,
            'client_metrics': self.client_metrics
        }
        
        logger.info(f"Federated learning completed in {total_time:.2f} seconds")
        logger.info(f"Final evaluation: {final_eval}")
        
        return results
    
    def save_checkpoint(self, save_path: str) -> None:
        """Save server checkpoint"""
        checkpoint = {
            'server_config': self.config,
            'model_config': self.model_config,
            'global_model_state': self.global_model.state_dict(),
            'global_tokenizer_state': self.global_tokenizer.get_state(),
            'round_history': self.round_history,
            'client_weights': self.client_weights,
            'client_metrics': self.client_metrics,
            'current_round': self.current_round,
            'best_loss': self.best_loss
        }
        # Ensure directory exists
        dir_path = os.path.dirname(save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(f"{save_path}.pkl", 'wb') as f:
            pickle.dump(checkpoint, f)
        # Save round history as JSON for easy inspection
        with open(f"{save_path}_history.json", 'w') as f:
            json.dump(self.round_history, f, indent=2, default=str)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, load_path: str) -> None:
        """Load server checkpoint"""
        with open(f"{load_path}.pkl", 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.config = checkpoint['server_config']
        self.global_model.load_state_dict(checkpoint['global_model_state'])
        self.global_tokenizer.load_state(checkpoint['global_tokenizer_state'])
        self.round_history = checkpoint['round_history']
        self.client_weights = checkpoint['client_weights']
        self.client_metrics = checkpoint['client_metrics']
        self.current_round = checkpoint['current_round']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Checkpoint loaded from {load_path}")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            'total_clients': len(self.clients),
            'current_round': self.current_round,
            'best_loss': self.best_loss,
            'total_rounds_completed': len(self.round_history),
            'client_weights': self.client_weights,
            'model_size': sum(p.numel() for p in self.global_model.parameters()),
            'trainable_params': sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
        } 
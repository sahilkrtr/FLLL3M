"""
Step 4: Training and Evaluation Pipeline
Main training pipeline for FLLL³M federated learning experiments
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from ..federated.server import FederatedServer, ServerConfig
from ..federated.client import FederatedClient, ClientConfig, ModelConfig
from ..models.delta_iris import DeltaIRISTokenizer
from ..models.transformer_model import LightweightTransformer
from ..utils.config import TrainingConfig
from ..utils.metrics import MobilityMetrics
from ..utils.data_loader import MobilityDataLoader


class TrainingPipeline:
    """Main training pipeline for FLLL³M federated learning experiments"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.tokenizer = None
        self.global_model = None
        self.server = None
        self.clients = []
        self.metrics = MobilityMetrics()
        
        # Training state
        self.current_round = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'test_metrics': []
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_components(self):
        """Initialize all training components"""
        self.logger.info("Initializing training components...")
        
        # Create model config
        model_config = ModelConfig(
            n_location_clusters=50,
            n_time_bins=25,
            embedding_dim=self.config.d_model,
            n_layers=self.config.num_layers,
            n_heads=self.config.nhead,
            hidden_size=self.config.d_model,
            dropout=self.config.dropout
        )
        
        # Initialize tokenizer
        from config.model_config import DeltaIRISConfig
        tokenizer_config = DeltaIRISConfig(
            location_bins=model_config.n_location_clusters,
            time_bins=model_config.n_time_bins,
            token_embedding_dim=model_config.embedding_dim,
            use_positional_encoding=True
        )
        self.tokenizer = DeltaIRISTokenizer(tokenizer_config)
        
        # Initialize global model
        from config.model_config import TransformerConfig
        transformer_config = TransformerConfig(
            combined_embedding_dim=model_config.embedding_dim,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.n_layers,
            num_heads=model_config.n_heads,
            intermediate_size=model_config.hidden_size * 4,
            dropout=model_config.dropout
        )
        self.global_model = LightweightTransformer(transformer_config).to(self.device)
        
        # Initialize federated server
        server_config = ServerConfig(
            num_rounds=self.config.num_rounds,
            num_clients_per_round=self.config.num_clients,
            aggregation_method=self.config.aggregation_method
        )
        self.server = FederatedServer(server_config, model_config)
        
        # Initialize federated clients
        self.initialize_clients(model_config)
        
        self.logger.info("All components initialized successfully")
        
    def initialize_clients(self, model_config: ModelConfig):
        """Initialize federated learning clients"""
        self.logger.info("Initializing federated clients...")
        
        # Load processed data
        data_dir = Path(self.config.data_dir)
        client_data_files = list(data_dir.glob("client_*.parquet"))
        
        if not client_data_files:
            raise ValueError(f"No client data files found in {data_dir}")
        
        self.clients = []
        for i, data_file in enumerate(client_data_files[:self.config.num_clients]):
            client_id = f"client_{i}"
            
            # Load client data
            client_data = pd.read_parquet(data_file)
            
            # Create client config
            client_config = ClientConfig(
                client_id=client_id,
                local_epochs=self.config.local_epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                max_grad_norm=self.config.gradient_clip_norm,
                privacy_epsilon=self.config.epsilon,
                privacy_delta=self.config.delta,
                noise_scale=self.config.noise_std
            )
            
            # Create client
            client = FederatedClient(client_config, model_config)
            
            # Load data into client
            client.load_local_data(str(data_file))
            
            # Register client with server
            self.server.register_client(client)
            
            self.clients.append(client)
            
        self.logger.info(f"Initialized {len(self.clients)} federated clients")
        
    def fit_tokenizer(self):
        """Fit the tokenizer on all training data"""
        self.logger.info("Fitting tokenizer on training data...")
        
        # For now, we'll fit the tokenizer on the first client's data
        # In a real implementation, you might want to aggregate data from all clients
        if self.clients and self.clients[0].local_data is not None:
            self.tokenizer.fit(self.clients[0].local_data)
            self.logger.info(f"Tokenizer fitted on {len(self.clients[0].local_data)} samples")
        else:
            self.logger.warning("No client data available for tokenizer fitting")
        
    def train_round(self) -> Dict[str, float]:
        """Execute one round of federated training"""
        self.logger.info(f"Starting training round {self.current_round + 1}")
        
        # Use the server's federated learning process for one round
        # Select clients for this round
        selected_clients = self.server.select_clients_for_round()
        
        # Distribute global model
        self.server.distribute_global_model(selected_clients)
        
        # Collect client updates
        client_updates = self.server.collect_client_updates(selected_clients)
        
        # Aggregate updates
        self.server.aggregate_model_updates(client_updates)
        
        # Calculate average metrics from client updates
        client_metrics = []
        for client_id in selected_clients:
            if client_id in self.server.client_metrics:
                metrics = self.server.client_metrics[client_id]['training_metrics']
                client_metrics.append(metrics)
        
        avg_metrics = self.calculate_average_metrics(client_metrics)
        
        # Ensure we always return a loss value
        if 'loss' not in avg_metrics and client_metrics:
            # Extract final_loss from client metrics if available
            losses = [m.get('final_loss', 0.0) for m in client_metrics]
            avg_metrics['loss'] = np.mean(losses)
        elif 'loss' not in avg_metrics:
            avg_metrics['loss'] = 0.0
        
        self.current_round += 1
        return avg_metrics
        
    def calculate_average_metrics(self, client_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average metrics across all clients"""
        avg_metrics = {}
        
        if not client_metrics:
            return avg_metrics
            
        # Get all metric keys
        metric_keys = client_metrics[0].keys()
        
        for key in metric_keys:
            values = [metrics.get(key, 0.0) for metrics in client_metrics]
            avg_metrics[key] = np.mean(values)
            
        return avg_metrics
        
    def evaluate(self, split: str = 'test') -> Dict[str, float]:
        """Evaluate the global model"""
        self.logger.info(f"Evaluating model on {split} split...")
        
        # Use the server's evaluation method
        try:
            eval_metrics = self.server.evaluate_global_model()
            return eval_metrics
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return {'loss': float('inf')}
        
    def evaluate_client_data(self, client: FederatedClient, eval_data: pd.DataFrame) -> Tuple[List, List, Dict[str, float]]:
        """Evaluate model on client data"""
        self.global_model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in client.data_loader.get_batches(eval_data):
                # Forward pass
                outputs = self.global_model(batch['input_ids'].to(self.device))
                
                # Get predictions and targets
                batch_predictions = outputs.cpu().numpy()
                batch_targets = batch['targets'].cpu().numpy()
                
                predictions.extend(batch_predictions)
                targets.extend(batch_targets)
                
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(predictions, targets)
        
        return predictions, targets, metrics
        
    def save_checkpoint(self, metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'round': self.current_round,
            'model_state_dict': self.global_model.state_dict(),
            'tokenizer_state': self.tokenizer.get_state(),
            'config': self.config.to_dict(),
            'metrics': metrics,
            'training_history': self.training_history,
            'best_loss': self.best_loss
        }
        
        checkpoint_path = checkpoint_dir / f'checkpoint_round_{self.current_round}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        current_loss = metrics.get('loss', float('inf'))
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            best_model_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"New best model saved with loss: {self.best_loss:.6f}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_round = checkpoint['round']
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer.load_state(checkpoint['tokenizer_state'])
        self.training_history = checkpoint['training_history']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"Loaded checkpoint from round {self.current_round}")
        
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Test loss
        if self.training_history['test_loss']:
            axes[0, 1].plot(self.training_history['test_loss'], label='Test Loss', color='red')
            axes[0, 1].set_title('Test Loss')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Metrics plots
        if self.training_history['train_metrics']:
            train_metrics_df = pd.DataFrame(self.training_history['train_metrics'])
            for metric in ['mse', 'mae', 'rmse']:
                if metric in train_metrics_df.columns:
                    axes[1, 0].plot(train_metrics_df[metric], label=f'Train {metric.upper()}')
            
            axes[1, 0].set_title('Training Metrics')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Metric Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        if self.training_history['val_metrics']:
            val_metrics_df = pd.DataFrame(self.training_history['val_metrics'])
            for metric in ['mse', 'mae', 'rmse']:
                if metric in val_metrics_df.columns:
                    axes[1, 1].plot(val_metrics_df[metric], label=f'Val {metric.upper()}')
            
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Metric Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_path = results_dir / f'training_history_round_{self.current_round}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history plot saved to {plot_path}")
        
    def save_results(self, final_metrics: Dict[str, float]):
        """Save final training results"""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        results = {
            'final_round': self.current_round,
            'best_loss': self.best_loss,
            'final_metrics': final_metrics,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        
        results_path = results_dir / f'final_results_round_{self.current_round}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Final results saved to {results_path}")
        
    def train(self):
        """Main training loop"""
        self.logger.info("Starting FLLL³M training pipeline...")
        
        # Initialize components
        self.initialize_components()
        
        # Fit tokenizer
        self.fit_tokenizer()
        
        # Training loop
        for round_num in range(self.config.num_rounds):
            start_time = time.time()
            
            # Train one round
            train_metrics = self.train_round()
            
            # Evaluate
            if round_num % self.config.eval_interval == 0:
                val_metrics = self.evaluate('val')
                test_metrics = self.evaluate('test')
                
                # Update training history
                self.training_history['train_loss'].append(train_metrics.get('loss', 0.0))
                self.training_history['val_loss'].append(val_metrics.get('loss', 0.0))
                self.training_history['test_loss'].append(test_metrics.get('loss', 0.0))
                self.training_history['train_metrics'].append(train_metrics)
                self.training_history['val_metrics'].append(val_metrics)
                self.training_history['test_metrics'].append(test_metrics)
                
                # Log metrics
                self.logger.info(f"Round {self.current_round}: "
                               f"Train Loss: {train_metrics.get('loss', 0.0):.6f}, "
                               f"Val Loss: {val_metrics.get('loss', 0.0):.6f}, "
                               f"Test Loss: {test_metrics.get('loss', 0.0):.6f}")
                
                # Save checkpoint
                self.save_checkpoint(val_metrics)
                
                # Plot training history
                if round_num % self.config.plot_interval == 0:
                    self.plot_training_history()
            else:
                # Update only training metrics
                self.training_history['train_loss'].append(train_metrics.get('loss', 0.0))
                self.training_history['train_metrics'].append(train_metrics)
                
                self.logger.info(f"Round {self.current_round}: "
                               f"Train Loss: {train_metrics.get('loss', 0.0):.6f}")
            
            # Check early stopping
            if self.config.early_stopping_patience > 0:
                if len(self.training_history['val_loss']) >= self.config.early_stopping_patience:
                    recent_losses = self.training_history['val_loss'][-self.config.early_stopping_patience:]
                    if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                        self.logger.info("Early stopping triggered")
                        break
            
            round_time = time.time() - start_time
            self.logger.info(f"Round {self.current_round} completed in {round_time:.2f}s")
        
        # Final evaluation
        final_metrics = self.evaluate('test')
        self.logger.info(f"Final test metrics: {final_metrics}")
        
        # Save final results
        self.save_results(final_metrics)
        
        # Final plot
        self.plot_training_history()
        
        self.logger.info("Training completed successfully!")
        return final_metrics
        self.logger.info("Training completed successfully!")
        return final_metrics 
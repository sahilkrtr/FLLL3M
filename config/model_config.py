"""
Model configuration for FLLL³M.

This module contains hyperparameters and configuration settings for all models
in the FLLL³M framework.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class TransformerConfig:
    """Configuration for the lightweight transformer model."""
    
    # Architecture
    num_layers: int = 6
    num_heads: int = 4
    hidden_size: int = 256
    intermediate_size: int = 1024
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 64
    max_sequence_length: int = 512
    
    # Embeddings
    location_embedding_dim: int = 128
    time_embedding_dim: int = 128
    combined_embedding_dim: int = 128

@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    
    # Model selection
    llm_name: str = "gpt2-small"  # gpt2-small, gpt2-medium, gpt2-large
    
    # Projection
    projection_input_dim: int = 128 * 128  # Outer product flattened
    projection_hidden_dim: int = 512
    projection_output_dim: int = 256
    
    # Injection
    injection_layer: int = 6  # Which layer to inject federated knowledge
    
    # Fine-tuning
    freeze_llm: bool = True
    fine_tune_projection: bool = True
    fine_tune_output_head: bool = True

@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    
    # Privacy
    noise_std: float = 0.1  # Gaussian noise for differential privacy
    
    # Aggregation
    aggregation_method: str = "outer_product"  # outer_product, simple_average
    
    # Training
    num_fl_rounds: int = 100
    local_epochs: int = 1
    client_fraction: float = 0.1  # Fraction of clients to sample per round
    
    # Communication
    communication_rounds: int = 10

@dataclass
class DeltaIRISConfig:
    """Configuration for Δ-IRIS tokenizer."""
    
    # Tokenization
    location_bins: int = 1000
    time_bins: int = 1000
    context_window: int = 5
    
    # Embedding
    token_embedding_dim: int = 128
    use_positional_encoding: bool = True

@dataclass
class TrainingConfig:
    """General training configuration."""
    
    # Optimization
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Training
    batch_size: int = 64
    num_epochs: int = 100
    gradient_clip_norm: float = 1.0
    
    # Evaluation
    eval_steps: int = 1000
    save_steps: int = 5000
    logging_steps: int = 100
    
    # Hardware
    device: str = "cuda"  # cuda, cpu
    num_workers: int = 4
    
    # Checkpointing
    save_dir: str = "results/models"
    load_best_model: bool = True

@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    
    # Filtering
    min_checkins_per_user: int = 10
    min_visits_per_venue: int = 10
    
    # Splitting
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Preprocessing
    gps_noise_reduction: bool = True
    median_filter_window: int = 5
    normalize_timestamps: bool = True
    
    # Supported datasets
    supported_datasets: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.supported_datasets is None:
            self.supported_datasets = ["gowalla", "brightkite", "foursquare", "weeplace"]

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    
    # Experiment
    experiment_name: str = "flll3m_experiment"
    seed: int = 42
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "flll3m"
    log_dir: str = "results/logs"
    
    # Results
    results_dir: str = "results"
    save_plots: bool = True
    save_metrics: bool = True

# Default configurations
DEFAULT_TRANSFORMER_CONFIG = TransformerConfig()
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_FEDERATED_CONFIG = FederatedConfig()
DEFAULT_DELTA_IRIS_CONFIG = DeltaIRISConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()

# Configuration presets
CONFIG_PRESETS = {
    "small": {
        "transformer": TransformerConfig(num_layers=4, hidden_size=128),
        "llm": LLMConfig(llm_name="gpt2-small"),
        "federated": FederatedConfig(num_fl_rounds=50),
    },
    "medium": {
        "transformer": TransformerConfig(),
        "llm": LLMConfig(llm_name="gpt2-medium"),
        "federated": FederatedConfig(),
    },
    "large": {
        "transformer": TransformerConfig(num_layers=8, hidden_size=512),
        "llm": LLMConfig(llm_name="gpt2-large"),
        "federated": FederatedConfig(num_fl_rounds=200),
    }
} 
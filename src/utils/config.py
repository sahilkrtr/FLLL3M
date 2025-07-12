"""
Configuration utilities for FLLL³M training and evaluation
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class TrainingConfig:
    """Configuration for FLLL³M training and evaluation"""
    
    # Data configuration
    data_dir: str = "data/processed"
    num_clients: int = 5
    
    # Model architecture
    vocab_size: int = 1000
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    max_sequence_length: int = 512
    dropout: float = 0.1
    
    # Training parameters
    num_rounds: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Evaluation
    eval_interval: int = 5
    plot_interval: int = 10
    early_stopping_patience: int = 10
    
    # Federated learning
    client_fraction: float = 1.0
    local_epochs: int = 1
    aggregation_method: str = "fedavg"
    
    # Privacy
    noise_std: float = 0.0
    differential_privacy: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    log_dir: str = "logs"
    experiment_name: str = "flll3m_experiment"
    
    # Hardware
    device: str = "auto"  # auto, cuda, cpu
    num_workers: int = 4
    
    # Logging
    log_level: str = "INFO"
    save_checkpoints: bool = True
    save_plots: bool = True
    save_metrics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'TrainingConfig':
        """Update config with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


@dataclass
class ExperimentConfig:
    """Configuration for experiment management"""
    
    # Experiment settings
    experiment_name: str = "flll3m_experiment"
    seed: int = 42
    max_experiments: Optional[int] = None
    
    # Hyperparameter search
    hyperparameter_grid: Optional[Dict[str, list]] = None
    
    # Results
    results_dir: str = "experiments"
    save_intermediate_results: bool = True
    
    # Logging
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "flll3m"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()

# Configuration presets
TRAINING_PRESETS = {
    "small": TrainingConfig(
        num_clients=3,
        num_rounds=20,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        batch_size=16,
        max_sequence_length=256
    ),
    "medium": TrainingConfig(
        num_clients=5,
        num_rounds=50,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        batch_size=32,
        max_sequence_length=512
    ),
    "large": TrainingConfig(
        num_clients=10,
        num_rounds=100,
        d_model=512,
        nhead=16,
        num_layers=8,
        dim_feedforward=2048,
        batch_size=64,
        max_sequence_length=1024
    )
} 
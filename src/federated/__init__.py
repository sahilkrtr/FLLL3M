"""
Federated Learning Package

This package implements federated learning components for FLLL³M:
- FederatedClient: Handles local training and model updates
- FederatedServer: Orchestrates federated learning rounds
- FedAvg algorithm implementation
- Privacy-preserving mechanisms
"""

from .client import FederatedClient, ClientConfig, ModelConfig
from .server import FederatedServer, ServerConfig

__all__ = [
    'FederatedClient',
    'ClientConfig',
    'ModelConfig',
    'FederatedServer',
    'ServerConfig'
] 
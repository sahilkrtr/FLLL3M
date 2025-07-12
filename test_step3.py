#!/usr/bin/env python3
"""
FLLL³M - Step 3: Federated Learning Setup Tests

This script tests the federated learning components:
- FederatedClient: Local training and model updates
- FederatedServer: FedAvg aggregation and round management
- End-to-end federated learning pipeline
"""

import sys
import os
import logging
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.federated import FederatedClient, ClientConfig, FederatedServer, ServerConfig, ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_federated_learning():
    """Test the complete federated learning setup"""
    print("FLLL³M - Step 3: Federated Learning Setup Tests")
    print("=" * 60)
    
    # Configuration
    model_config = ModelConfig(
        n_location_clusters=50,
        n_time_bins=25,
        embedding_dim=128,
        n_layers=2,
        n_heads=4,
        hidden_size=256,
        dropout=0.1
    )
    
    # Server configuration (small scale for testing)
    server_config = ServerConfig(
        num_rounds=3,  # Small number for testing
        num_clients_per_round=3,
        min_clients_per_round=2,
        evaluation_frequency=1,
        save_frequency=2,
        early_stopping_patience=5,
        differential_privacy=True
    )
    
    print("=" * 50)
    print("Testing Federated Learning Setup")
    print("=" * 50)
    
    # Create server
    print("Creating federated server...")
    server = FederatedServer(server_config, model_config)
    
    # Create clients
    print("Creating federated clients...")
    clients = []
    num_clients = 5
    
    for i in range(num_clients):
        client_config = ClientConfig(
            client_id=f"client_{i}",
            local_epochs=2,  # Small number for testing
            batch_size=16,
            learning_rate=1e-4,
            privacy_epsilon=1.0
        )
        
        client = FederatedClient(client_config, model_config)
        clients.append(client)
    
    # Load data for clients
    print("Loading data for clients...")
    data_path = "data/processed/processed_data.npy"
    
    if not os.path.exists(data_path):
        print(f" Processed data not found at {data_path}")
        print("Please run preprocessing first (Step 1)")
        return False
    
    # Load and distribute data
    full_data = np.load(data_path, allow_pickle=True)
    
    for i, client in enumerate(clients):
        # Each client gets a different subset of data
        np.random.seed(i)
        client_indices = np.random.choice(
            len(full_data),
            size=min(500, len(full_data)),  # Small sample for testing
            replace=False
        )
        client_data = full_data[client_indices]
        
        # Save client data temporarily
        client_data_path = f"temp_client_{i}_data.npy"
        np.save(client_data_path, client_data)
        
        # Load data into client
        client.load_local_data(client_data_path)
        
        # Register client with server
        server.register_client(client)
        
        print(f"  Client {i}: {len(client_data)} samples")
    
    print(f" Created {num_clients} clients with data")
    
    # Test client training
    print("\n" + "=" * 50)
    print("Testing Client Training")
    print("=" * 50)
    
    for i, client in enumerate(clients[:2]):  # Test first 2 clients
        print(f"Testing client {i} training...")
        
        try:
            # Train client model
            training_metrics = client.train_local_model()
            
            print(f"   Client {i} training completed")
            print(f"  Final loss: {training_metrics['final_loss']:.6f}")
            print(f"  Average loss: {training_metrics['avg_loss']:.6f}")
            
            # Test model update
            model_update = client.get_model_update()
            print(f"  Model update size: {len(model_update)} parameters")
            
        except Exception as e:
            print(f"   Client {i} training failed: {e}")
            return False
    
    # Test server operations
    print("\n" + "=" * 50)
    print("Testing Server Operations")
    print("=" * 50)
    
    # Test client selection
    selected_clients = server.select_clients_for_round()
    print(f"Selected clients for round: {selected_clients}")
    
    # Test model distribution
    server.distribute_global_model(selected_clients)
    print(" Global model distributed to clients")
    
    # Test client updates collection
    client_updates = server.collect_client_updates(selected_clients)
    print(f" Collected updates from {len(client_updates)} clients")
    
    # Test model aggregation
    server.aggregate_model_updates(client_updates)
    print(" Model updates aggregated")
    
    # Test evaluation
    eval_metrics = server.evaluate_global_model()
    print(f" Global model evaluation: {eval_metrics}")
    
    # Test server stats
    server_stats = server.get_server_stats()
    print(f" Server stats: {server_stats}")
    
    # Test end-to-end federated learning (small scale)
    print("\n" + "=" * 50)
    print("Testing End-to-End Federated Learning")
    print("=" * 50)
    
    try:
        # Run a few rounds of federated learning
        results = server.run_federated_learning()
        
        print(" Federated learning completed successfully!")
        print(f"Total rounds: {results['total_rounds']}")
        print(f"Total time: {results['total_time']:.2f} seconds")
        print(f"Final evaluation: {results['final_evaluation']}")
        print(f"Best loss: {results['best_loss']}")
        
    except Exception as e:
        print(f" Federated learning failed: {e}")
        return False
    
    # Test checkpointing
    print("\n" + "=" * 50)
    print("Testing Checkpointing")
    print("=" * 50)
    
    try:
        # Save checkpoint
        server.save_checkpoint("test_checkpoint")
        print(" Checkpoint saved")
        
        # Load checkpoint
        server.load_checkpoint("test_checkpoint")
        print(" Checkpoint loaded")
        
    except Exception as e:
        print(f" Checkpointing failed: {e}")
        return False
    
    # Cleanup
    print("\nCleaning up temporary files...")
    for i in range(num_clients):
        temp_file = f"temp_client_{i}_data.npy"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    if os.path.exists("test_checkpoint.pkl"):
        os.remove("test_checkpoint.pkl")
    if os.path.exists("test_checkpoint_history.json"):
        os.remove("test_checkpoint_history.json")
    
    print(" Cleanup completed")
    
    print("\n" + "=" * 60)
    print("All Step 3 tests passed successfully!")
    print(" Federated client working correctly")
    print(" Federated server working correctly")
    print(" FedAvg aggregation working correctly")
    print(" End-to-end federated learning working correctly")
    print(" Checkpointing working correctly")
    print("\nReady to proceed to Step 4: Training and Evaluation")
    
    return True

if __name__ == "__main__":
    success = test_federated_learning()
    sys.exit(0 if success else 1) 
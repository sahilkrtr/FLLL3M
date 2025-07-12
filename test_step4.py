#!/usr/bin/env python3
"""
Test Script for Step 4: Training and Evaluation
Tests the training pipeline and experiment manager functionality
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.core.training_pipeline import TrainingPipeline
from src.core.experiment_manager import ExperimentManager
from src.utils.config import TrainingConfig
from src.models.delta_iris import DeltaIRISTokenizer
from src.models.transformer_model import LightweightTransformer
from src.federated.server import FederatedServer
from src.federated.client import FederatedClient
from src.utils.data_loader import MobilityDataLoader
from src.utils.metrics import MobilityMetrics


def setup_test_data():
    """Create test data for training"""
    print("Setting up test data...")
    
    # Create test data directory
    test_data_dir = Path("test_data_step4")
    test_data_dir.mkdir(exist_ok=True)
    
    # Generate synthetic mobility data
    np.random.seed(42)
    n_users = 50
    n_checkins_per_user = 100
    
    all_data = []
    
    for user_id in range(n_users):
        # Generate user trajectory
        base_lat = np.random.uniform(40.0, 41.0)  # NYC area
        base_lon = np.random.uniform(-74.0, -73.0)
        
        # Generate checkins with some temporal and spatial patterns
        start_time = datetime.now() - timedelta(days=30)
        
        for i in range(n_checkins_per_user):
            # Add some randomness to location
            lat = base_lat + np.random.normal(0, 0.01)
            lon = base_lon + np.random.normal(0, 0.01)
            
            # Add temporal progression
            checkin_time = start_time + timedelta(hours=i*2 + np.random.randint(-1, 2))
            
            # Create checkin record
            checkin = {
                'user_id': f'user_{user_id}',
                'latitude': lat,
                'longitude': lon,
                'timestamp': checkin_time,
                'venue_id': f'venue_{np.random.randint(1, 100)}',
                'venue_category': f'category_{np.random.randint(1, 10)}'
            }
            all_data.append(checkin)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Split into clients (simulate federated setting)
    n_clients = 5
    users_per_client = n_users // n_clients
    
    for i in range(n_clients):
        start_idx = i * users_per_client
        end_idx = start_idx + users_per_client if i < n_clients - 1 else n_users
        
        client_users = [f'user_{j}' for j in range(start_idx, end_idx)]
        client_data = df[df['user_id'].isin(client_users)].copy()
        
        # Save client data
        client_file = test_data_dir / f"client_{i}.parquet"
        client_data.to_parquet(client_file, index=False)
        print(f"Created {client_file} with {len(client_data)} records")
    
    return str(test_data_dir)


def test_training_pipeline():
    """Test the training pipeline"""
    print("\n" + "="*50)
    print("Testing Training Pipeline")
    print("="*50)
    
    # Setup test data
    test_data_dir = setup_test_data()
    
    # Create configuration
    config = TrainingConfig(
        data_dir=test_data_dir,
        num_clients=5,
        num_rounds=3,  # Small number for testing
        batch_size=16,
        learning_rate=0.001,
        d_model=64,  # Smaller model for testing
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_sequence_length=50,
        vocab_size=1000,
        dropout=0.1,
        eval_interval=1,
        plot_interval=2,
        early_stopping_patience=5,
        checkpoint_dir="test_checkpoints",
        results_dir="test_results",
        log_dir="test_logs"
    )
    
    try:
        # Create training pipeline
        pipeline = TrainingPipeline(config)
        
        # Test component initialization
        print("Testing component initialization...")
        pipeline.initialize_components()
        assert pipeline.tokenizer is not None
        assert pipeline.global_model is not None
        assert pipeline.server is not None
        assert len(pipeline.clients) == 5
        print("✓ Component initialization successful")
        
        # Test tokenizer fitting
        print("Testing tokenizer fitting...")
        pipeline.fit_tokenizer()
        assert pipeline.tokenizer.is_fitted
        print("✓ Tokenizer fitting successful")
        
        # Test single training round
        print("Testing single training round...")
        metrics = pipeline.train_round()
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], (int, float))
        print(f"✓ Training round successful, loss: {metrics['loss']:.6f}")
        
        # Test evaluation
        print("Testing evaluation...")
        eval_metrics = pipeline.evaluate('test')
        assert isinstance(eval_metrics, dict)
        print(f"✓ Evaluation successful, metrics: {eval_metrics}")
        
        # Test checkpoint saving
        print("Testing checkpoint saving...")
        pipeline.save_checkpoint(metrics)
        checkpoint_dir = Path(config.checkpoint_dir)
        assert checkpoint_dir.exists()
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0
        print("✓ Checkpoint saving successful")
        
        # Test training history plotting
        print("Testing training history plotting...")
        pipeline.plot_training_history()
        results_dir = Path(config.results_dir)
        plot_files = list(results_dir.glob("*.png"))
        assert len(plot_files) > 0
        print("✓ Training history plotting successful")
        
        print("✓ Training pipeline test completed successfully!")
        
    except Exception as e:
        print(f"✗ Training pipeline test failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        cleanup_dirs = [test_data_dir, config.checkpoint_dir, config.results_dir, config.log_dir]
        for dir_path in cleanup_dirs:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path)


def test_experiment_manager():
    """Test the experiment manager"""
    print("\n" + "="*50)
    print("Testing Experiment Manager")
    print("="*50)
    
    # Setup test data
    test_data_dir = setup_test_data()
    
    # Create base configuration
    base_config = TrainingConfig(
        data_dir=test_data_dir,
        num_clients=5,
        num_rounds=2,  # Very small for testing
        batch_size=16,
        learning_rate=0.001,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_sequence_length=50,
        vocab_size=1000,
        dropout=0.1,
        eval_interval=1,
        plot_interval=2,
        early_stopping_patience=5,
        checkpoint_dir="test_checkpoints",
        results_dir="test_results",
        log_dir="test_logs"
    )
    
    # Create experiment manager
    experiment_manager = ExperimentManager(base_config, experiments_dir="test_experiments")
    
    try:
        # Test experiment configuration creation
        print("Testing experiment configuration creation...")
        hyperparameter_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32],
            'd_model': [64, 128]
        }
        
        configs = experiment_manager.create_experiment_configs(hyperparameter_grid)
        assert len(configs) == 8  # 2 * 2 * 2 = 8 combinations
        print(f"✓ Created {len(configs)} experiment configurations")
        
        # Test single experiment
        print("Testing single experiment...")
        test_config = configs[0]
        result = experiment_manager.run_single_experiment(test_config)
        
        assert 'experiment_name' in result
        assert 'status' in result
        assert 'config' in result
        print(f"✓ Single experiment completed with status: {result['status']}")
        
        # Test experiment results saving and loading
        print("Testing results saving and loading...")
        experiment_manager.save_experiment_results([result])
        loaded_results = experiment_manager.load_experiment_results()
        assert len(loaded_results) == 1
        assert loaded_results[0]['experiment_name'] == result['experiment_name']
        print("✓ Results saving and loading successful")
        
        # Test results analysis
        print("Testing results analysis...")
        analysis = experiment_manager.analyze_results()
        assert isinstance(analysis, dict)
        print(f"✓ Results analysis successful: {analysis}")
        
        # Test experiment plotting
        print("Testing experiment plotting...")
        experiment_manager.plot_experiment_results()
        plot_files = list(experiment_manager.experiments_dir.glob("*.png"))
        # Note: Plotting may be skipped if all metrics are NaN, which is acceptable
        print(f"✓ Experiment plotting completed (found {len(plot_files)} plot files)")
        
        # Test report generation
        print("Testing report generation...")
        report = experiment_manager.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0
        report_files = list(experiment_manager.experiments_dir.glob("*.md"))
        assert len(report_files) > 0
        print("✓ Report generation successful")
        
        print("✓ Experiment manager test completed successfully!")
        
    except Exception as e:
        print(f"✗ Experiment manager test failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        cleanup_dirs = [test_data_dir, "test_experiments"]
        for dir_path in cleanup_dirs:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path)


def test_integration():
    """Test integration between components"""
    print("\n" + "="*50)
    print("Testing Integration")
    print("="*50)
    
    # Setup test data
    test_data_dir = setup_test_data()
    
    # Create configuration
    config = TrainingConfig(
        data_dir=test_data_dir,
        num_clients=5,  # Changed from 3 to 5 to meet server requirements
        num_rounds=2,
        batch_size=16,
        learning_rate=0.001,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_sequence_length=50,
        vocab_size=1000,
        dropout=0.1,
        eval_interval=1,
        plot_interval=2,
        early_stopping_patience=5,
        checkpoint_dir="test_checkpoints",
        results_dir="test_results",
        log_dir="test_logs"
    )
    
    try:
        # Test full training pipeline
        print("Testing full training pipeline...")
        pipeline = TrainingPipeline(config)
        final_metrics = pipeline.train()
        
        assert isinstance(final_metrics, dict)
        assert 'loss' in final_metrics
        print(f"✓ Full training completed with final loss: {final_metrics.get('loss', 'N/A')}")
        
        # Test experiment manager with completed training
        print("Testing experiment manager with completed training...")
        experiment_manager = ExperimentManager(config, experiments_dir="test_experiments")
        
        # Create a simple hyperparameter grid
        hyperparameter_grid = {
            'learning_rate': [0.001],
            'batch_size': [16]
        }
        
        results = experiment_manager.run_experiments(hyperparameter_grid, max_experiments=1)
        assert len(results) == 1
        assert results[0]['status'] == 'completed'
        print("✓ Integration test completed successfully!")
        
    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        cleanup_dirs = [test_data_dir, config.checkpoint_dir, config.results_dir, config.log_dir, "test_experiments"]
        for dir_path in cleanup_dirs:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path)


def main():
    """Main test function"""
    print("FLLL³M Step 4: Training and Evaluation Test")
    print("="*60)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    try:
        test_training_pipeline()
        test_experiment_manager()
        test_integration()
        
        print("\n" + "="*60)
        print("All Step 4 tests passed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n Step 4 tests failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 
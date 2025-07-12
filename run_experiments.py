"""
FLLL³M Experiment Runner
Run experiments on real mobility datasets with configurable parameters.
"""

import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.training_pipeline import TrainingPipeline
from src.core.experiment_manager import ExperimentManager
from src.core.llm_integration import FLLL3MLLMIntegration, LLMConfig
from src.models.delta_iris import DeltaIRISTokenizer
from src.models.transformer_model import LightweightTransformer
from src.utils.config import TrainingConfig
from config.model_config import TransformerConfig, DeltaIRISConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str) -> str:
    """Load dataset and return path to processed data."""
    print(f"Loading {dataset_name} dataset...")
    
    # Define dataset paths
    dataset_paths = {
        'gowalla': 'Gowalla/loc-gowalla_totalCheckins.txt/Gowalla_totalCheckins.txt',
        'brightkite': 'Brightkite/loc-brightkite_totalCheckins.txt/Brightkite_totalCheckins.txt',
        'foursquare': 'Foursquare/dataset_tsmc2014/dataset_TSMC2014_NYC.txt',
        'weeplace': 'Weeplace/weeplace_checkins.csv'
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(dataset_paths.keys())}")
    
    data_path = dataset_paths[dataset_name]
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    # Process the dataset
    processed_path = f"processed_data/{dataset_name}_processed.csv"
    
    if not os.path.exists(processed_path):
        print(f"   Processing {dataset_name} dataset...")
        process_dataset(data_path, processed_path, dataset_name)
    
    print(f"Dataset loaded: {processed_path}")
    return processed_path


def process_dataset(raw_path: str, output_path: str, dataset_name: str):
    """Process raw dataset into standard format."""
    os.makedirs("processed_data", exist_ok=True)
    
    if dataset_name == 'gowalla':
        # Gowalla format: user_id, check-in_time, latitude, longitude, location_id
        df = pd.read_csv(raw_path, sep='\t', header=None, 
                        names=['user_id', 'timestamp', 'lat', 'lon', 'location_id'])
        
    elif dataset_name == 'brightkite':
        # Brightkite format: user_id, check-in_time, latitude, longitude, location_id
        df = pd.read_csv(raw_path, sep='\t', header=None,
                        names=['user_id', 'timestamp', 'lat', 'lon', 'location_id'])
        
    elif dataset_name == 'foursquare':
        # Foursquare format: user_id, venue_id, venue_category_id, venue_category_name, 
        # latitude, longitude, timezone_offset, utc_time
        df = pd.read_csv(raw_path, sep='\t', header=None,
                        names=['user_id', 'venue_id', 'venue_category_id', 'venue_category_name',
                               'lat', 'lon', 'timezone_offset', 'timestamp'])
        
    elif dataset_name == 'weeplace':
        # Weeplace format: user_id, timestamp, lat, lon, location_id
        df = pd.read_csv(raw_path)
        if 'location_id' not in df.columns:
            df['location_id'] = df.index  # Create location_id if not present
    
    # Standardize column names
    df = df[['user_id', 'timestamp', 'lat', 'lon']].copy()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter data (remove outliers, etc.)
    df = df.dropna()
    
    # Filter by geographic bounds (remove extreme outliers)
    lat_bounds = (df['lat'].quantile(0.001), df['lat'].quantile(0.999))
    lon_bounds = (df['lon'].quantile(0.001), df['lon'].quantile(0.999))
    
    df = df[
        (df['lat'].between(*lat_bounds)) & 
        (df['lon'].between(*lon_bounds))
    ]
    
    # Sort by user_id and timestamp
    df = df.sort_values(['user_id', 'timestamp'])
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"   Processed {len(df)} check-ins from {df['user_id'].nunique()} users")


def run_experiment(dataset_name: str, config: dict):
    """Run a complete FLLL³M experiment."""
    print(f"\nStarting FLLL³M experiment on {dataset_name}")
    print("="*60)
    
    # Load dataset
    data_path = load_dataset(dataset_name)
    
    # Initialize components
    print("Initializing components...")
    
    # Create training config
    training_config = TrainingConfig(
        batch_size=config.get('batch_size', 64),
        learning_rate=config.get('learning_rate', 1e-4),
        num_rounds=config.get('num_epochs', 10),
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir=f"experiment_results/{dataset_name}",
        log_dir=f"experiment_results/{dataset_name}/logs"
    )
    
    # Training pipeline
    pipeline = TrainingPipeline(training_config)
    
    # Initialize components
    pipeline.initialize_components()
    
    # Experiment manager
    experiment_name = f"flll3m_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_manager = ExperimentManager(
        base_config=training_config,
        experiments_dir=f"experiment_results/{dataset_name}"
    )
    
    # Train model
    print("Starting training...")
    training_history = pipeline.train()
    
    # Evaluate model
    print("Evaluating model...")
    evaluation_results = pipeline.evaluate()
    
    # Save experiment
    print("Saving experiment results...")
    experiment_manager.save_experiment_results([{
        'experiment_name': experiment_name,
        'config': config,
        'results': evaluation_results,
        'training_history': training_history,
        'status': 'completed',
        'timestamp': datetime.now().isoformat()
    }])
    
    # Test LLM integration
    print("Testing LLM integration...")
    llm_config = LLMConfig(
        model_name=config.get('llm_model', 'microsoft/DialoGPT-small'),
        use_mobility_context=True,
        fusion_method=config.get('fusion_method', 'attention')
    )
    
    # Save model checkpoint for LLM integration
    checkpoint_path = f"experiment_results/{dataset_name}/model_checkpoint.pth"
    if pipeline.global_model is not None:
        torch.save({
            'model_state_dict': pipeline.global_model.state_dict(),
            'config': {
                'vocab_size': 1000,
                'd_model': training_config.d_model,
                'nhead': training_config.nhead,
                'num_layers': training_config.num_layers
            }
        }, checkpoint_path)
    else:
        print("Warning: Global model not initialized, skipping checkpoint save")
    
    # Initialize LLM integration
    llm_integration = FLLL3MLLMIntegration(
        config=llm_config,
        mobility_model_path=checkpoint_path
    )
    
    # Test LLM functionality
    test_user_data = {
        'mobility_sequences': [
            "HOME_WORK_0.8_9:00",
            "WORK_RESTAURANT_0.6_12:00",
            "RESTAURANT_WORK_0.7_13:00"
        ]
    }
    
    llm_response = llm_integration.generate_mobility_aware_response(
        "What can you tell me about this user's mobility patterns?",
        test_user_data
    )
    
    print(f"   LLM Response: {llm_response}")
    
    # Print results
    print("\nExperiment Results:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Training epochs: {len(training_history)}")
    print(f"   Final loss: {evaluation_results.get('loss', 'N/A'):.4f}")
    print(f"   Final accuracy: {evaluation_results.get('accuracy', 'N/A'):.4f}")
    print(f"   Experiment saved: {experiment_name}")
    
    return {
        'dataset': dataset_name,
        'config': config,
        'results': evaluation_results,
        'experiment_name': experiment_name
    }


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description='Run FLLL³M experiments')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['gowalla', 'brightkite', 'foursquare', 'weeplace'],
                       help='Dataset to use for experiment')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_federated_rounds', type=int, default=20,
                       help='Number of federated learning rounds')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden size of transformer')
    parser.add_argument('--llm_model', type=str, default='microsoft/DialoGPT-small',
                       help='LLM model to use for integration')
    parser.add_argument('--fusion_method', type=str, default='attention',
                       choices=['attention', 'concat', 'weighted'],
                       help='Fusion method for LLM integration')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_federated_rounds': args.num_federated_rounds,
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'llm_model': args.llm_model,
        'fusion_method': args.fusion_method,
        'location_bins': 1000,
        'time_bins': 1000,
        'token_embedding_dim': 128,
        'num_heads': 4,
        'intermediate_size': 1024,
        'dropout': 0.1
    }
    
    # Run experiment
    try:
        results = run_experiment(args.dataset, config)
        print(f"\nExperiment completed successfully!")
        print(f"Results saved in: experiment_results/{args.dataset}/")
        return True
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
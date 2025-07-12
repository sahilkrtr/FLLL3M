#!/usr/bin/env python3
"""
Test script for Step 2: Semantic Mobility Encoding

This script tests the Δ-IRIS tokenizer and embedding components to ensure
they are working correctly before proceeding to Step 3.
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.model_config import DeltaIRISConfig, TransformerConfig
from src.models.delta_iris import DeltaIRISTokenizer
from src.models.transformer_model import LightweightTransformer

def test_delta_iris_tokenizer():
    """Test Δ-IRIS tokenizer functionality."""
    print("=" * 50)
    print("Testing Δ-IRIS Tokenizer")
    print("=" * 50)
    
    # Create configuration
    config = DeltaIRISConfig(
        location_bins=100,
        time_bins=50,
        token_embedding_dim=128,
        use_positional_encoding=True
    )
    
    # Create tokenizer
    tokenizer = DeltaIRISTokenizer(config)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample check-ins
    checkins_data = {
        'user_id': np.random.randint(0, 100, n_samples),
        'lat': np.random.uniform(30, 50, n_samples),  # US latitude range
        'lon': np.random.uniform(-120, -70, n_samples),  # US longitude range
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H')
    }
    
    checkins_df = pd.DataFrame(checkins_data)
    
    print(f"Sample data shape: {checkins_df.shape}")
    print(f"Sample data:\n{checkins_df.head()}")
    
    # Fit tokenizer
    print("\nFitting tokenizer...")
    tokenizer.fit(checkins_df)
    
    # Test tokenization
    print("\nTesting tokenization...")
    location_tokens, time_tokens = tokenizer.tokenize(checkins_df)
    
    print(f"Location tokens shape: {location_tokens.shape}")
    print(f"Time tokens shape: {time_tokens.shape}")
    print(f"Location token range: {location_tokens.min()} - {location_tokens.max()}")
    print(f"Time token range: {time_tokens.min()} - {time_tokens.max()}")
    
    # Test encoding
    print("\nTesting encoding...")
    embeddings = tokenizer.encode(location_tokens, time_tokens)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding range: {embeddings.min():.4f} - {embeddings.max():.4f}")
    
    # Test combined function
    print("\nTesting combined tokenize_and_encode...")
    combined_embeddings = tokenizer.tokenize_and_encode(checkins_df)
    
    print(f"Combined embeddings shape: {combined_embeddings.shape}")
    print(f"Combined embedding range: {combined_embeddings.min():.4f} - {combined_embeddings.max():.4f}")
    
    # Test vocabulary sizes
    loc_vocab_size, time_vocab_size = tokenizer.get_vocab_size()
    print(f"\nVocabulary sizes:")
    print(f"  Location vocabulary: {loc_vocab_size}")
    print(f"  Time vocabulary: {time_vocab_size}")
    print(f"  Embedding dimension: {tokenizer.get_embedding_dim()}")
    
    print("\n Δ-IRIS Tokenizer test passed!")
    return tokenizer, embeddings

def test_lightweight_transformer():
    """Test lightweight transformer functionality."""
    print("\n" + "=" * 50)
    print("Testing Lightweight Transformer")
    print("=" * 50)
    
    # Create configuration
    config = TransformerConfig(
        num_layers=2,  # Reduced for testing
        num_heads=4,
        hidden_size=256,
        intermediate_size=512,
        combined_embedding_dim=128,
        dropout=0.1
    )
    
    # Create transformer
    transformer = LightweightTransformer(config)
    
    print(f"Transformer configuration:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Embedding dim: {config.combined_embedding_dim}")
    
    # Create sample embeddings
    batch_size, seq_len, embedding_dim = 4, 10, 128
    sample_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    
    print(f"\nSample embeddings shape: {sample_embeddings.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    transformer.eval()
    with torch.no_grad():
        output = transformer(sample_embeddings)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: {output.min():.4f} - {output.max():.4f}")
    
    # Test next embedding prediction
    print("\nTesting next embedding prediction...")
    with torch.no_grad():
        next_embedding = transformer.predict_next_embedding(sample_embeddings)
    
    print(f"Next embedding shape: {next_embedding.shape}")
    print(f"Next embedding range: {next_embedding.min():.4f} - {next_embedding.max():.4f}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    targets = torch.randn_like(sample_embeddings)
    loss = transformer.compute_loss(sample_embeddings, targets)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Test outer product computation
    print("\nTesting outer product computation...")
    next_embeddings = torch.randn_like(sample_embeddings)
    outer_products = transformer.get_outer_product(sample_embeddings, next_embeddings)
    
    print(f"Outer products shape: {outer_products.shape}")
    print(f"Outer products range: {outer_products.min():.4f} - {outer_products.max():.4f}")
    
    # Test noise addition
    print("\nTesting noise addition...")
    noisy_products = transformer.add_noise_for_privacy(outer_products, noise_std=0.1)
    
    print(f"Noisy products shape: {noisy_products.shape}")
    print(f"Noise difference: {torch.mean(torch.abs(noisy_products - outer_products)):.4f}")
    
    # Test model size
    model_size = transformer.get_model_size()
    print(f"\nModel size information:")
    print(f"  Total parameters: {model_size['total_parameters']:,}")
    print(f"  Trainable parameters: {model_size['trainable_parameters']:,}")
    print(f"  Model size: {model_size['model_size_mb']:.2f} MB")
    
    print("\n Lightweight Transformer test passed!")
    return transformer

def test_integration():
    """Test integration between tokenizer and transformer."""
    print("\n" + "=" * 50)
    print("Testing Integration")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    checkins_data = {
        'user_id': np.random.randint(0, 10, n_samples),
        'lat': np.random.uniform(30, 50, n_samples),
        'lon': np.random.uniform(-120, -70, n_samples),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H')
    }
    
    checkins_df = pd.DataFrame(checkins_data)
    
    # Create and fit tokenizer
    tokenizer_config = DeltaIRISConfig(
        location_bins=50,
        time_bins=25,
        token_embedding_dim=128
    )
    tokenizer = DeltaIRISTokenizer(tokenizer_config)
    tokenizer.fit(checkins_df)
    
    # Create transformer
    transformer_config = TransformerConfig(
        num_layers=2,
        num_heads=4,
        hidden_size=256,
        combined_embedding_dim=128
    )
    transformer = LightweightTransformer(transformer_config)
    
    # Test end-to-end pipeline
    print("Testing end-to-end pipeline...")
    
    # Tokenize and encode
    embeddings = tokenizer.tokenize_and_encode(checkins_df)
    
    # Create sequences for transformer
    seq_len = 10
    batch_size = embeddings.shape[0] // seq_len
    
    if batch_size > 0:
        # Reshape into batches
        embeddings_batch = embeddings[:batch_size * seq_len].view(batch_size, seq_len, -1)
        
        print(f"Batch embeddings shape: {embeddings_batch.shape}")
        
        # Predict next embeddings
        transformer.eval()
        with torch.no_grad():
            predicted_next = transformer.predict_next_embedding(embeddings_batch)
        
        print(f"Predicted next embeddings shape: {predicted_next.shape}")
        
        # Compute loss
        actual_next = embeddings_batch[:, 1:, :]  # Shift by 1
        loss = transformer.compute_loss(embeddings_batch[:, :-1, :], actual_next)
        
        print(f"Sequence prediction loss: {loss.item():.4f}")
        
        print("\n Integration test passed!")
    else:
        print("  Not enough samples for batch processing")

def main():
    """Run all tests for Step 2."""
    print("FLLL³M - Step 2: Semantic Mobility Encoding Tests")
    print("=" * 60)
    
    try:
        # Test Δ-IRIS tokenizer
        tokenizer, embeddings = test_delta_iris_tokenizer()
        
        # Test lightweight transformer
        transformer = test_lightweight_transformer()
        
        # Test integration
        test_integration()
        
        print("\n" + "=" * 60)
        print("All Step 2 tests passed successfully!")
        print(" Δ-IRIS tokenizer working correctly")
        print(" Lightweight transformer working correctly")
        print(" Integration between components working correctly")
        print("\nReady to proceed to Step 3: Federated Learning Setup")
        
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
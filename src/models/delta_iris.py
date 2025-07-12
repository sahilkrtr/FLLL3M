"""
Δ-IRIS Tokenizer for Semantic Mobility Encoding

This module implements the Δ-IRIS (Delta-Information Retrieval and Indexing System)
tokenizer for converting location-time tuples into semantic tokens as described
in the FLLL³M paper.

The tokenizer includes:
- Location discretization and binning
- Time discretization and binning  
- Context-aware token generation
- Embedding generation for locations and times
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

from config.model_config import DeltaIRISConfig

class DeltaIRISTokenizer:
    """
    Δ-IRIS Tokenizer for semantic mobility encoding.
    
    Converts (location, timestamp) tuples into semantic tokens using:
    1. Location discretization via clustering
    2. Time discretization via temporal binning
    3. Context-aware token generation
    4. Embedding generation
    """
    
    def __init__(self, config: DeltaIRISConfig):
        """
        Initialize Δ-IRIS tokenizer.
        
        Args:
            config: Configuration for the tokenizer
        """
        self.config = config
        
        # Location discretization
        self.location_kmeans = None
        self.location_scaler = StandardScaler()
        self.location_bins = config.location_bins
        
        # Time discretization
        self.time_bins = config.time_bins
        self.time_scaler = StandardScaler()
        
        # Embeddings
        self.location_embedding = None
        self.time_embedding = None
        
        # Vocabulary
        self.location_vocab = {}
        self.time_vocab = {}
        self.combined_vocab = {}
        
        # Positional encoding
        self.use_positional_encoding = config.use_positional_encoding
        self.positional_encoding = None
        
        # Training state
        self.is_fitted = False
        
    def fit(self, data: np.ndarray) -> None:
        """
        Fit the tokenizer on mobility data.
        
        Args:
            data: numpy array with columns [user_id, lat, lon, timestamp]
        """
        print("Fitting Δ-IRIS tokenizer...")
        import pandas as pd
        # Ensure data is a numpy array of shape (n, 4) and dtype=object
        data = np.array(data, dtype=object)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != 4:
            raise ValueError("Input data must have 4 columns: user_id, lat, lon, timestamp")
        df = pd.DataFrame(data, columns=['user_id', 'lat', 'lon', 'timestamp'])
        # Convert timestamp to numeric if needed
        if not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['timestamp'] = df['timestamp'].astype('int64') // 10**9  # Unix timestamp
        
        # Extract location and time features
        locations = df[['lat', 'lon']].astype(float).values
        timestamps = df['timestamp'].astype(float).values
        
        # Fit location discretization
        self._fit_location_discretization(locations)
        
        # Fit time discretization
        self._fit_time_discretization(timestamps)
        
        # Create embeddings
        self._create_embeddings()
        
        # Create positional encoding if needed
        if self.use_positional_encoding:
            self._create_positional_encoding()
        
        self.is_fitted = True
        print("Δ-IRIS tokenizer fitted successfully!")
        
    def _fit_location_discretization(self, locations: np.ndarray) -> None:
        """Fit location discretization using K-means clustering."""
        print("Fitting location discretization...")
        
        # Normalize locations
        locations_normalized = self.location_scaler.fit_transform(locations)
        
        # Apply K-means clustering
        self.location_kmeans = KMeans(
            n_clusters=self.location_bins,
            random_state=42,
            n_init='auto'
        )
        self.location_kmeans.fit(locations_normalized)
        
        # Create location vocabulary
        for i in range(self.location_bins):
            self.location_vocab[f"LOC_{i}"] = i
            
        print(f"Location discretization: {self.location_bins} clusters")
        
    def _fit_time_discretization(self, timestamps) -> None:
        """Fit time discretization using temporal binning."""
        print("Fitting time discretization...")
        
        # Convert timestamps to numerical features
        time_features = self._extract_time_features(timestamps)
        
        # Normalize time features
        time_features_normalized = self.time_scaler.fit_transform(time_features)
        
        # Create time bins using quantiles
        time_bin_edges = np.percentile(
            time_features_normalized[:, 0],  # Use hour of day as primary feature
            np.linspace(0, 100, self.time_bins + 1)
        )
        
        # Create time vocabulary
        for i in range(self.time_bins):
            self.time_vocab[f"TIME_{i}"] = i
            
        print(f"Time discretization: {self.time_bins} bins")
        
    def _extract_time_features(self, timestamps) -> np.ndarray:
        """Extract temporal features from timestamps."""
        # Convert to pandas datetime if needed
        if isinstance(timestamps[0], str):
            timestamps = pd.to_datetime(timestamps, utc=True).tz_localize(None)
        elif isinstance(timestamps, np.ndarray):
            # If it's already a numpy array, convert to pandas datetime
            timestamps = pd.to_datetime(timestamps, utc=True).tz_localize(None)
        
        # Extract various temporal features using pandas datetime methods
        hours = timestamps.hour.to_numpy()
        days_of_week = timestamps.weekday.to_numpy()
        days_of_month = timestamps.day.to_numpy()
        months = timestamps.month.to_numpy()
        
        # Normalize features to [0, 1]
        hours_normalized = hours / 24.0
        days_of_week_normalized = days_of_week / 7.0
        days_of_month_normalized = days_of_month / 31.0
        months_normalized = months / 12.0
        
        return np.column_stack([
            hours_normalized,
            days_of_week_normalized,
            days_of_month_normalized,
            months_normalized
        ])
        
    def _create_embeddings(self) -> None:
        """Create embedding layers for locations and times."""
        print("Creating embeddings...")
        
        # Location embeddings
        self.location_embedding = nn.Embedding(
            num_embeddings=self.location_bins,
            embedding_dim=self.config.token_embedding_dim
        )
        
        # Time embeddings
        self.time_embedding = nn.Embedding(
            num_embeddings=self.time_bins,
            embedding_dim=self.config.token_embedding_dim
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.location_embedding.weight)
        nn.init.xavier_uniform_(self.time_embedding.weight)
        
        print(f"Embeddings created: {self.config.token_embedding_dim} dimensions")
        
    def _create_positional_encoding(self) -> None:
        """Create positional encoding for sequence modeling."""
        max_seq_len = 512  # Maximum sequence length
        
        pe = torch.zeros(max_seq_len, self.config.token_embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.config.token_embedding_dim, 2).float() *
                           -(np.log(10000.0) / self.config.token_embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.positional_encoding = pe
        
    def tokenize(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize mobility data.
        
        Args:
            data: numpy array with columns [user_id, lat, lon, timestamp]
            
        Returns:
            Tuple of (location_tokens, time_tokens)
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before tokenization")
            
        # Extract features
        locations = data[:, 1:3].astype(np.float64)  # lat, lon
        timestamps = data[:, 3]  # timestamp
        
        # Generate tokens
        location_tokens = self._tokenize_locations(locations)
        time_tokens = self._tokenize_times(timestamps)
        
        return location_tokens, time_tokens
        
    def _tokenize_locations(self, locations: np.ndarray) -> torch.Tensor:
        """Tokenize locations using fitted K-means clusters."""
        # Normalize locations
        locations_normalized = self.location_scaler.transform(locations)
        
        # Predict clusters
        cluster_labels = self.location_kmeans.predict(locations_normalized)
        
        return torch.tensor(cluster_labels, dtype=torch.long)
        
    def _tokenize_times(self, timestamps) -> torch.Tensor:
        """Tokenize timestamps using temporal binning."""
        # Extract time features
        time_features = self._extract_time_features(timestamps)
        
        # Normalize time features
        time_features_normalized = self.time_scaler.transform(time_features)
        
        # Assign to bins based on hour of day
        hours_normalized = time_features_normalized[:, 0]
        time_bins = np.digitize(hours_normalized, 
                               np.percentile(hours_normalized, 
                                           np.linspace(0, 100, self.time_bins + 1)[1:-1]))
        
        # Ensure bins are within valid range
        time_bins = np.clip(time_bins, 0, self.time_bins - 1)
        
        return torch.tensor(time_bins, dtype=torch.long)
        
    def encode(self, data: np.ndarray) -> torch.Tensor:
        """
        Encode mobility data into embeddings.
        
        Args:
            data: numpy array with columns [user_id, lat, lon, timestamp]
            
        Returns:
            Combined embeddings: e_t = φ_loc(x_t) + φ_time(τ_t)
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding")
            
        # Tokenize first
        location_tokens, time_tokens = self.tokenize(data)
        
        # Get embeddings
        location_embeddings = self.location_embedding(location_tokens)
        time_embeddings = self.time_embedding(time_tokens)
        
        # Combine embeddings
        combined_embeddings = location_embeddings + time_embeddings
        
        # Add positional encoding if enabled
        if self.use_positional_encoding and self.positional_encoding is not None:
            seq_len = combined_embeddings.size(0)
            if seq_len <= self.positional_encoding.size(0):
                pos_enc = self.positional_encoding[:seq_len].to(combined_embeddings.device)
                combined_embeddings = combined_embeddings + pos_enc
                
        return combined_embeddings
        
    def tokenize_and_encode(self, data: np.ndarray) -> torch.Tensor:
        """
        Tokenize and encode mobility data in one step.
        
        Args:
            data: numpy array with columns [user_id, lat, lon, timestamp]
            
        Returns:
            Combined embeddings
        """
        return self.encode(data)
        
    def save(self, filepath: str) -> None:
        """Save the fitted tokenizer."""
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before saving")
            
        save_dict = {
            'config': self.config.__dict__,
            'location_kmeans': self.location_kmeans,
            'location_scaler': self.location_scaler,
            'time_scaler': self.time_scaler,
            'location_vocab': self.location_vocab,
            'time_vocab': self.time_vocab,
            'location_embedding_state': self.location_embedding.state_dict(),
            'time_embedding_state': self.time_embedding.state_dict(),
            'use_positional_encoding': self.use_positional_encoding,
            'is_fitted': self.is_fitted
        }
        
        if self.positional_encoding is not None:
            save_dict['positional_encoding'] = self.positional_encoding
            
        torch.save(save_dict, filepath)
        print(f"Tokenizer saved to {filepath}")
        
    def load(self, filepath: str) -> None:
        """Load a fitted tokenizer."""
        save_dict = torch.load(filepath, map_location='cpu')
        
        # Restore configuration
        self.config = DeltaIRISConfig(**save_dict['config'])
        
        # Restore fitted components
        self.location_kmeans = save_dict['location_kmeans']
        self.location_scaler = save_dict['location_scaler']
        self.time_scaler = save_dict['time_scaler']
        self.location_vocab = save_dict['location_vocab']
        self.time_vocab = save_dict['time_vocab']
        self.use_positional_encoding = save_dict['use_positional_encoding']
        self.is_fitted = save_dict['is_fitted']
        
        # Restore embeddings
        self._create_embeddings()
        self.location_embedding.load_state_dict(save_dict['location_embedding_state'])
        self.time_embedding.load_state_dict(save_dict['time_embedding_state'])
        
        # Restore positional encoding
        if self.use_positional_encoding and 'positional_encoding' in save_dict:
            self.positional_encoding = save_dict['positional_encoding']
            
        print(f"Tokenizer loaded from {filepath}")
        
    def get_vocab_size(self) -> Tuple[int, int]:
        """Get vocabulary sizes for locations and times."""
        return len(self.location_vocab), len(self.time_vocab)
        
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.config.token_embedding_dim
    
    def get_state(self) -> Dict:
        """Get the current state of the tokenizer for checkpointing."""
        return {
            'config': self.config,
            'location_kmeans': self.location_kmeans,
            'location_scaler': self.location_scaler,
            'time_scaler': self.time_scaler,
            'location_embedding': self.location_embedding.state_dict() if self.location_embedding else None,
            'time_embedding': self.time_embedding.state_dict() if self.time_embedding else None,
            'location_vocab': self.location_vocab,
            'time_vocab': self.time_vocab,
            'combined_vocab': self.combined_vocab,
            'positional_encoding': self.positional_encoding,
            'is_fitted': self.is_fitted
        }
    
    def load_state(self, state: Dict) -> None:
        """Load the tokenizer state from checkpoint."""
        self.config = state['config']
        self.location_kmeans = state['location_kmeans']
        self.location_scaler = state['location_scaler']
        self.time_scaler = state['time_scaler']
        
        if state['location_embedding'] and self.location_embedding:
            self.location_embedding.load_state_dict(state['location_embedding'])
        if state['time_embedding'] and self.time_embedding:
            self.time_embedding.load_state_dict(state['time_embedding'])
        
        self.location_vocab = state['location_vocab']
        self.time_vocab = state['time_vocab']
        self.combined_vocab = state['combined_vocab']
        self.positional_encoding = state['positional_encoding']
        self.is_fitted = state['is_fitted'] 
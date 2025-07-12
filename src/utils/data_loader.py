"""
Data Loader for FLLL³M Mobility Data
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset
import logging

from ..models.delta_iris import DeltaIRISTokenizer

logger = logging.getLogger(__name__)


class MobilityDataLoader:
    """Data loader for mobility data with tokenization and batching"""
    
    def __init__(self, data: pd.DataFrame, tokenizer: DeltaIRISTokenizer, 
                 batch_size: int = 32, max_length: int = 512, shuffle: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        
        # Split data into train/val/test
        self._split_data()
        
        # Prepare sequences
        self.train_sequences = None
        self.val_sequences = None
        self.test_sequences = None
        
    def _split_data(self):
        """Split data into train/val/test sets"""
        n_samples = len(self.data)
        train_size = int(0.6 * n_samples)
        val_size = int(0.2 * n_samples)
        
        # Shuffle data
        if self.shuffle:
            self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        self.train_data = self.data[:train_size]
        self.val_data = self.data[train_size:train_size + val_size]
        self.test_data = self.data[train_size + val_size:]
        
        logger.info(f"Data split: Train={len(self.train_data)}, Val={len(self.val_data)}, Test={len(self.test_data)}")
    
    def get_all_sequences(self) -> List[np.ndarray]:
        """Get all sequences for tokenizer fitting"""
        sequences = []
        
        # Convert DataFrame to numpy array format expected by tokenizer
        for _, row in self.data.iterrows():
            sequence = np.array([
                row['user_id'],
                row['latitude'],
                row['longitude'],
                row['timestamp']
            ])
            sequences.append(sequence)
        
        return sequences
    
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences and targets for training"""
        if len(data) == 0:
            return torch.tensor([]), torch.tensor([])
        
        # Convert to numpy array format
        data_array = data[['user_id', 'latitude', 'longitude', 'timestamp']].values
        
        # Tokenize and encode
        try:
            embeddings = self.tokenizer.encode(data_array)
        except Exception as e:
            logger.error(f"Error encoding data: {e}")
            # Return empty tensors if encoding fails
            return torch.tensor([]), torch.tensor([])
        
        # Create sequences for training
        sequences = []
        targets = []
        seq_len = min(10, self.max_length - 1)  # Leave room for target
        
        for i in range(len(embeddings) - seq_len):
            sequence = embeddings[i:i+seq_len]
            target = embeddings[i+seq_len]  # Next embedding
            
            # Pad sequence if needed
            if sequence.shape[0] < seq_len:
                padding = torch.zeros(seq_len - sequence.shape[0], sequence.shape[1])
                sequence = torch.cat([sequence, padding], dim=0)
            
            sequences.append(sequence)
            targets.append(target)
        
        if not sequences:
            return torch.tensor([]), torch.tensor([])
        
        return torch.stack(sequences), torch.stack(targets)
    
    def get_batches(self, data: pd.DataFrame) -> List[Dict[str, torch.Tensor]]:
        """Get batches of data"""
        sequences, targets = self._prepare_sequences(data)
        
        if len(sequences) == 0:
            return []
        
        # Create batches
        batches = []
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i+self.batch_size]
            batch_targets = targets[i:i+self.batch_size]
            
            batches.append({
                'input_ids': batch_sequences,
                'targets': batch_targets
            })
        
        return batches
    
    def get_evaluation_data(self, split: str = 'test') -> Optional[pd.DataFrame]:
        """Get evaluation data for specified split"""
        if split == 'train':
            return self.train_data
        elif split == 'val':
            return self.val_data
        elif split == 'test':
            return self.test_data
        else:
            logger.warning(f"Unknown split: {split}")
            return None
    
    def get_dataloader(self, split: str = 'train') -> Optional[DataLoader]:
        """Get PyTorch DataLoader for specified split"""
        data = self.get_evaluation_data(split)
        if data is None or len(data) == 0:
            return None
        
        sequences, targets = self._prepare_sequences(data)
        if len(sequences) == 0:
            return None
        
        dataset = TensorDataset(sequences, targets)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=(self.shuffle and split == 'train')
        )
        
        return dataloader 
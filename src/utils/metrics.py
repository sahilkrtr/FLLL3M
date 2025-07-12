"""
Mobility Metrics for FLLL³M Evaluation
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MobilityMetrics:
    """Calculate evaluation metrics for mobility prediction"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_metrics(self, predictions: List, targets: List) -> Dict[str, float]:
        """Calculate various mobility prediction metrics"""
        if not predictions or not targets:
            return {'loss': float('inf'), 'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}
        
        # Convert to numpy arrays
        pred_array = np.array(predictions)
        target_array = np.array(targets)
        
        # Handle NaN and Inf values
        pred_array = np.nan_to_num(pred_array, nan=0.0, posinf=0.0, neginf=0.0)
        target_array = np.nan_to_num(target_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate metrics
        mse = mean_squared_error(target_array, pred_array)
        mae = mean_absolute_error(target_array, pred_array)
        rmse = np.sqrt(mse)
        
        # Calculate custom loss (can be modified based on requirements)
        loss = mse  # Using MSE as default loss
        
        metrics = {
            'loss': float(loss),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse)
        }
        
        # Store metrics history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all calculated metrics"""
        if not self.metrics_history:
            return {}
        
        # Convert to DataFrame-like structure
        all_metrics = {}
        for metric_name in self.metrics_history[0].keys():
            values = [m[metric_name] for m in self.metrics_history]
            all_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        return all_metrics
    
    def reset(self):
        """Reset metrics history"""
        self.metrics_history = [] 
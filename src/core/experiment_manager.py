"""
Step 4: Experiment Manager
Manages multiple training experiments with different configurations
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import itertools

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

from .training_pipeline import TrainingPipeline
from ..utils.config import TrainingConfig


class ExperimentManager:
    """Manages multiple training experiments with different configurations"""
    
    def __init__(self, base_config: TrainingConfig, experiments_dir: str = "experiments"):
        self.base_config = base_config
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Experiment tracking
        self.experiments = []
        self.results = []
        
    def setup_logging(self):
        """Setup logging for experiment manager"""
        log_dir = self.experiments_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_experiment_configs(self, hyperparameter_grid: Dict[str, List]) -> List[TrainingConfig]:
        """Create experiment configurations from hyperparameter grid"""
        configs = []
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(hyperparameter_grid))
        
        for i, params in enumerate(param_combinations):
            # Create new config based on base config
            config = TrainingConfig(**self.base_config.to_dict())
            
            # Update with experiment parameters
            for key, value in params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    self.logger.warning(f"Unknown parameter {key} in hyperparameter grid")
            
            # Set experiment-specific paths
            experiment_name = f"exp_{i:03d}_{'_'.join([f'{k}_{v}' for k, v in params.items()])}"
            config.experiment_name = experiment_name
            config.checkpoint_dir = str(self.experiments_dir / experiment_name / "checkpoints")
            config.results_dir = str(self.experiments_dir / experiment_name / "results")
            config.log_dir = str(self.experiments_dir / experiment_name / "logs")
            
            configs.append(config)
            
        return configs
        
    def run_single_experiment(self, config: TrainingConfig) -> Dict[str, Any]:
        """Run a single training experiment"""
        experiment_start = time.time()
        
        self.logger.info(f"Starting experiment: {config.experiment_name}")
        self.logger.info(f"Config: {config.to_dict()}")
        
        try:
            # Create training pipeline
            pipeline = TrainingPipeline(config)
            
            # Run training
            final_metrics = pipeline.train()
            
            # Record experiment results
            experiment_time = time.time() - experiment_start
            
            results = {
                'experiment_name': config.experiment_name,
                'config': config.to_dict(),
                'final_metrics': final_metrics,
                'training_time': experiment_time,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Experiment {config.experiment_name} completed successfully in {experiment_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Experiment {config.experiment_name} failed: {str(e)}")
            
            results = {
                'experiment_name': config.experiment_name,
                'config': config.to_dict(),
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
        return results
        
    def run_experiments(self, hyperparameter_grid: Dict[str, List], 
                       max_experiments: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run multiple experiments with different configurations"""
        self.logger.info("Starting experiment suite...")
        
        # Create experiment configurations
        configs = self.create_experiment_configs(hyperparameter_grid)
        
        if max_experiments:
            configs = configs[:max_experiments]
            
        self.logger.info(f"Running {len(configs)} experiments")
        
        # Run experiments
        results = []
        for i, config in enumerate(configs):
            self.logger.info(f"Progress: {i+1}/{len(configs)}")
            
            result = self.run_single_experiment(config)
            results.append(result)
            
            # Save intermediate results
            self.save_experiment_results(results)
            
        self.results = results
        self.logger.info("All experiments completed!")
        
        return results
        
    def save_experiment_results(self, results: List[Dict[str, Any]]):
        """Save experiment results to file"""
        results_file = self.experiments_dir / "experiment_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
    def load_experiment_results(self) -> List[Dict[str, Any]]:
        """Load experiment results from file"""
        results_file = self.experiments_dir / "experiment_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        else:
            return []
            
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results"""
        if not self.results:
            self.results = self.load_experiment_results()
            
        if not self.results:
            return {}
            
        # Filter completed experiments
        completed_results = [r for r in self.results if r['status'] == 'completed']
        
        if not completed_results:
            return {'error': 'No completed experiments found'}
            
        analysis = {
            'total_experiments': len(self.results),
            'completed_experiments': len(completed_results),
            'failed_experiments': len(self.results) - len(completed_results),
            'best_experiment': None,
            'parameter_analysis': {},
            'metric_summary': {}
        }
        
        # Find best experiment
        if completed_results:
            best_exp = min(completed_results, 
                          key=lambda x: x['final_metrics'].get('loss', float('inf')))
            analysis['best_experiment'] = best_exp
            
        # Analyze metrics
        metrics_data = []
        for result in completed_results:
            metrics = result['final_metrics']
            config = result['config']
            
            row = {
                'experiment_name': result['experiment_name'],
                'training_time': result['training_time'],
                **metrics,
                **{k: v for k, v in config.items() if k not in ['experiment_name', 'checkpoint_dir', 'results_dir', 'log_dir']}
            }
            metrics_data.append(row)
            
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            analysis['metric_summary'] = {
                'mean_loss': df['loss'].mean(),
                'std_loss': df['loss'].std(),
                'min_loss': df['loss'].min(),
                'max_loss': df['loss'].max(),
                'mean_training_time': df['training_time'].mean()
            }
            
            # Parameter analysis
            numeric_params = ['learning_rate', 'batch_size', 'd_model', 'num_layers']
            for param in numeric_params:
                if param in df.columns:
                    analysis['parameter_analysis'][param] = {
                        'correlation_with_loss': df[param].corr(df['loss']),
                        'mean_value': df[param].mean(),
                        'std_value': df[param].std()
                    }
                    
        return analysis
        
    def plot_experiment_results(self):
        """Plot experiment results"""
        if not self.results:
            self.results = self.load_experiment_results()
            
        completed_results = [r for r in self.results if r['status'] == 'completed']
        
        if not completed_results:
            self.logger.warning("No completed experiments to plot")
            return
            
        # Create results DataFrame
        data = []
        for result in completed_results:
            row = {
                'experiment_name': result['experiment_name'],
                'loss': result['final_metrics'].get('loss', float('inf')),
                'mse': result['final_metrics'].get('mse', 0.0),
                'mae': result['final_metrics'].get('mae', 0.0),
                'training_time': result['training_time']
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Filter out NaN values for plotting
        df_plot = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df_plot) == 0:
            self.logger.warning("No valid data for plotting after filtering NaN values")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss distribution
        if 'loss' in df_plot.columns and len(df_plot['loss']) > 0:
            axes[0, 0].hist(df_plot['loss'], bins=min(20, len(df_plot)), alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Distribution of Final Loss')
            axes[0, 0].set_xlabel('Loss')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Training time vs Loss
        if 'training_time' in df_plot.columns and 'loss' in df_plot.columns:
            axes[0, 1].scatter(df_plot['training_time'], df_plot['loss'], alpha=0.6)
            axes[0, 1].set_title('Training Time vs Loss')
            axes[0, 1].set_xlabel('Training Time (s)')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # MSE vs MAE
        if 'mse' in df_plot.columns and 'mae' in df_plot.columns:
            axes[1, 0].scatter(df_plot['mse'], df_plot['mae'], alpha=0.6)
            axes[1, 0].set_title('MSE vs MAE')
            axes[1, 0].set_xlabel('MSE')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Top experiments
        if 'loss' in df_plot.columns and len(df_plot) > 0:
            top_experiments = df_plot.nsmallest(min(10, len(df_plot)), 'loss')
            axes[1, 1].barh(range(len(top_experiments)), top_experiments['loss'])
            axes[1, 1].set_yticks(range(len(top_experiments)))
            axes[1, 1].set_yticklabels(top_experiments['experiment_name'], fontsize=8)
            axes[1, 1].set_title('Top Experiments by Loss')
            axes[1, 1].set_xlabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.experiments_dir / "experiment_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Experiment analysis plot saved to {plot_path}")
        
    def generate_report(self) -> str:
        """Generate a comprehensive experiment report"""
        analysis = self.analyze_results()
        
        report = f"""
# FLLL³M Experiment Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Total Experiments: {analysis.get('total_experiments', 0)}
- Completed: {analysis.get('completed_experiments', 0)}
- Failed: {analysis.get('failed_experiments', 0)}

## Best Experiment
"""
        
        if analysis.get('best_experiment'):
            best = analysis['best_experiment']
            report += f"""
- Name: {best['experiment_name']}
- Final Loss: {best['final_metrics'].get('loss', 'N/A'):.6f}
- Training Time: {best['training_time']:.2f}s
- Config: {json.dumps(best['config'], indent=2)}
"""
        
        if analysis.get('metric_summary'):
            metrics = analysis['metric_summary']
            report += f"""
## Metric Summary
- Mean Loss: {metrics.get('mean_loss', 'N/A'):.6f}
- Std Loss: {metrics.get('std_loss', 'N/A'):.6f}
- Min Loss: {metrics.get('min_loss', 'N/A'):.6f}
- Max Loss: {metrics.get('max_loss', 'N/A'):.6f}
- Mean Training Time: {metrics.get('mean_training_time', 'N/A'):.2f}s
"""
        
        if analysis.get('parameter_analysis'):
            report += "\n## Parameter Analysis\n"
            for param, stats in analysis['parameter_analysis'].items():
                report += f"""
### {param}
- Correlation with Loss: {stats.get('correlation_with_loss', 'N/A'):.4f}
- Mean Value: {stats.get('mean_value', 'N/A'):.4f}
- Std Value: {stats.get('std_value', 'N/A'):.4f}
"""
        
        # Save report
        report_path = self.experiments_dir / "experiment_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
            
        self.logger.info(f"Experiment report saved to {report_path}")
        
        return report 
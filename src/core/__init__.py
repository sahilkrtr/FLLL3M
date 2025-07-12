"""
Step 4: Core Training and Evaluation Module
Main entry point for FLLL³M training and evaluation
"""

from .training_pipeline import TrainingPipeline
from .experiment_manager import ExperimentManager

__all__ = [
    'TrainingPipeline',
    'ExperimentManager'
] 
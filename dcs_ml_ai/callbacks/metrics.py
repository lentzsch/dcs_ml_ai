"""
Custom metrics for aircraft simulation training.

Metric Interpretations:
- thrust_usage: Value between 0 and 1
    * 0: No thrust used
    * 1: Maximum thrust used
    * Ideal: Lower values indicate more efficient thrust management

- fuel_efficiency: Value between 0 and 1
    * 0: Poor fuel efficiency (excessive consumption)
    * 1: Optimal fuel efficiency
    * Ideal: Higher values indicate better fuel economy

- landing_quality: Value between 0 and 1
    * 0: Poor landing (crash or hard landing)
    * 1: Perfect landing (smooth touchdown at target)
    * Ideal: Higher values indicate better landing performance

Note: Currently using placeholder values. Real metrics will be implemented
when environment attributes become available.
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

class CustomMetricCallback(BaseCallback):
    """
    Custom callback for tracking domain-specific metrics for aircraft simulation.
    
    Currently tracks:
    - Thrust usage metrics
    - Fuel efficiency metrics
    - Landing behavior metrics
    
    Future metrics to implement:
    - Engine temperature
    - G-force measurements
    - Control surface deflection
    - Navigation accuracy
    - Mission completion rate
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Initialize metric storage
        self.metrics: Dict[str, List[float]] = {
            'thrust_usage': [],
            'fuel_efficiency': [],
            'landing_quality': []
        }
        # Track total steps for potential future per-step logging
        self.total_steps = 0
        
    def _on_step(self) -> bool:
        """
        Called at each step during training.
        
        TODO: Implement actual metric collection:
        - Get thrust values from env.get_attr('thrust')
        - Calculate fuel efficiency from state/action data
        - Analyze landing behavior from episode data
        - Add per-step metrics (e.g., every 100 steps)
        """
        self.total_steps += 1
        
        # Check if any environment in the vector has completed an episode
        if any(self.locals.get("dones", [])):
            # Example of how metrics would be collected
            # thrust = self.training_env.get_attr('thrust')[0]
            # fuel = self.training_env.get_attr('fuel')[0]
            # landing = self.training_env.get_attr('landing_status')[0]
            
            # For now, just log dummy data
            self.metrics['thrust_usage'].append(np.random.random())
            self.metrics['fuel_efficiency'].append(np.random.random())
            self.metrics['landing_quality'].append(np.random.random())
            
            # Log individual episode metrics to tensorboard
            for metric_name, values in self.metrics.items():
                if values:  # Only log if we have values
                    self.logger.record(f"custom/{metric_name}/episode", values[-1])
        
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout.
        Aggregates metrics and logs summary statistics to TensorBoard.
        """
        # Calculate and log aggregate statistics
        for metric_name, values in self.metrics.items():
            if values:  # Only process if we have values
                # Calculate statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                # Log to tensorboard
                self.logger.record(f"custom/{metric_name}/mean", mean_val)
                self.logger.record(f"custom/{metric_name}/std", std_val)
                self.logger.record(f"custom/{metric_name}/min", min_val)
                self.logger.record(f"custom/{metric_name}/max", max_val)
        
        # Reset metrics for next rollout
        for key in self.metrics:
            self.metrics[key] = []

    def save_metrics_to_csv(self, save_dir: Optional[str] = None) -> None:
        """
        Export tracked metrics to CSV files for offline analysis.
        
        Args:
            save_dir: Directory to save CSV files. If None, uses 'metrics' in current directory.
        """
        if save_dir is None:
            save_dir = "metrics"
        
        # Create save directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual metrics
        for metric_name, values in self.metrics.items():
            if values:  # Only save if we have values
                df = pd.DataFrame({
                    'step': range(len(values)),
                    metric_name: values
                })
                filename = save_path / f"{metric_name}_{timestamp}.csv"
                df.to_csv(filename, index=False)
        
        # Save summary statistics
        summary_data = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary_data[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data).T
            filename = save_path / f"metrics_summary_{timestamp}.csv"
            df_summary.to_csv(filename) 
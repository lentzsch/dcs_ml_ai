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
from datetime import datetime
from dcs_ml_ai.utils import unwrap_env

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
    def __init__(self, verbose: int = 0, run_id: Optional[str] = None):
        super().__init__(verbose)
        # Initialize metric storage
        self.metrics: Dict[str, List[float]] = {
            'thrust_usage': [],
            'fuel_efficiency': [],
            'landing_quality': []
        }
        # Track total steps for potential future per-step logging
        self.total_steps = 0
        
        # Set run ID for tracking training sessions
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _on_step(self) -> bool:
        """
        Called at each step during training.
        Collects real metrics from the environment when an episode ends.
        """
        self.total_steps += 1
        
        # Check if any environment in the vector has completed an episode
        if any(self.locals.get("dones", [])):
            # Get the core environment by unwrapping all layers
            core_env = unwrap_env(self.training_env)
            
            # Safely get metrics using getattr with None as default
            thrust = getattr(core_env, "thrust", None)
            fuel = getattr(core_env, "fuel", None)
            landing_status = getattr(core_env, "landing_status", None)
            
            # Process thrust data
            if thrust is not None:
                self.metrics['thrust_usage'].append(float(thrust))
                self.logger.record("custom/thrust_usage/episode", float(thrust))
            elif self.verbose > 0:
                print("Warning: Could not get thrust data from environment")
            
            # Process fuel data
            if fuel is not None:
                self.metrics['fuel_efficiency'].append(float(fuel))
                self.logger.record("custom/fuel_efficiency/episode", float(fuel))
            elif self.verbose > 0:
                print("Warning: Could not get fuel data from environment")
            
            # Process landing data
            if landing_status is not None:
                self.metrics['landing_quality'].append(float(landing_status))
                self.logger.record("custom/landing_quality/episode", float(landing_status))
            elif self.verbose > 0:
                print("Warning: Could not get landing status data from environment")
        
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout.
        Aggregates metrics and logs summary statistics to TensorBoard.
        """
        # Calculate and log aggregate statistics for each metric
        for metric_name, values in self.metrics.items():
            if values:  # Only process if we have values
                # Convert to numpy array for efficient calculation
                values_array = np.array(values)
                
                # Calculate statistics
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                min_val = np.min(values_array)
                max_val = np.max(values_array)
                
                # Log to tensorboard (using record since these are already aggregated)
                self.logger.record(f"custom/{metric_name}/mean", mean_val)
                self.logger.record(f"custom/{metric_name}/std", std_val)
                self.logger.record(f"custom/{metric_name}/min", min_val)
                self.logger.record(f"custom/{metric_name}/max", max_val)
                
                if self.verbose > 1:
                    print(f"{metric_name} stats - Mean: {mean_val:.3f}, Std: {std_val:.3f}, "
                          f"Min: {min_val:.3f}, Max: {max_val:.3f}")
        
        # Ensure all metrics are flushed to TensorBoard
        self.logger.dump(self.num_timesteps)
        
        # Reset metrics for next rollout
        for key in self.metrics:
            self.metrics[key] = []

    def _on_training_end(self) -> None:
        """
        Called at the end of training.
        Saves final metrics to CSV files.
        """
        if self.verbose > 0:
            print(f"\nSaving final metrics for run {self.run_id}...")
        self.save_metrics_to_csv()

    def save_metrics_to_csv(self, save_dir: Optional[str] = None) -> None:
        """
        Export tracked metrics to CSV files for offline analysis.
        
        Args:
            save_dir: Directory to save CSV files. If None, uses 'metrics' in current directory.
        """
        if save_dir is None:
            save_dir = "metrics"
        
        # Create save directory with run ID
        save_path = Path(save_dir) / self.run_id
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose > 0:
            print(f"Saving metrics to {save_path}")
        
        # Save individual metrics
        for metric_name, values in self.metrics.items():
            if values:  # Only save if we have values
                df = pd.DataFrame({
                    'step': range(len(values)),
                    metric_name: values
                })
                filename = save_path / f"{metric_name}.csv"
                df.to_csv(filename, index=False)
                if self.verbose > 0:
                    print(f"Saved {metric_name} data to {filename}")
        
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
            filename = save_path / "metrics_summary.csv"
            df_summary.to_csv(filename)
            if self.verbose > 0:
                print(f"Saved summary statistics to {filename}") 
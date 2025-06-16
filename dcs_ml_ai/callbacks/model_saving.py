"""
Callbacks for model saving and management.

This module provides callbacks for saving models at various points during training,
with a focus on robustness and clear logging.
"""

from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import os
from typing import Optional
from datetime import datetime

class FinalModelCallback(BaseCallback):
    """
    Callback for saving the final model after training completes.
    
    This callback ensures the final model is saved to a specified location
    with proper error handling and logging. Supports versioning through run IDs
    and can save additional training artifacts.
    
    Args:
        save_path: Directory where the model should be saved
        model_name: Name of the model file (without extension)
        run_id: Unique identifier for this training run. If None, uses timestamp
        verbose: Verbosity level (0: no output, 1: info messages)
    """
    def __init__(
        self,
        save_path: str,
        model_name: str,
        run_id: Optional[str] = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.model_name = model_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _on_training_end(self) -> None:
        """
        Called at the end of training.
        Saves the final model to the specified path and logs the event.
        """
        try:
            # Create save directory with run ID
            run_dir = self.save_path / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct full save path
            full_path = run_dir / f"{self.model_name}.zip"
            
            # Save the model
            self.model.save(str(full_path))
            
            # Log to TensorBoard
            self.logger.record("custom/final_model_saved", self.num_timesteps)
            self.logger.dump(self.num_timesteps)
            
            if self.verbose > 0:
                print(f"\nFinal model saved to: {full_path}")
                print(f"Run ID: {self.run_id}")
            
            # Save any additional artifacts
            self.save_extra_artifacts(run_dir)
                
        except Exception as e:
            if self.verbose > 0:
                print(f"\nError saving final model: {str(e)}")
            # Re-raise the exception to ensure training script is aware of the failure
            raise
            
    def save_extra_artifacts(self, save_dir: Path) -> None:
        """
        Save additional training artifacts alongside the model.
        
        This is a placeholder method that can be extended to save:
        - Hyperparameters
        - Environment configurations
        - Preprocessing components
        - Training statistics
        - Custom metrics
        
        Args:
            save_dir: Directory where artifacts should be saved
        """
        # TODO: Implement artifact saving
        # Example structure:
        # - save_dir/
        #   ├── model.zip
        #   ├── hyperparameters.json
        #   ├── env_config.json
        #   └── training_stats.csv
        pass 
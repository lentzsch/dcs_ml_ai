from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomMetricCallback(BaseCallback):
    """
    Custom callback for tracking domain-specific metrics.
    Currently a placeholder for future implementation of:
    - Thrust usage metrics
    - Fuel efficiency metrics
    - Landing behavior metrics
    - Other domain-specific performance indicators
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {
            'thrust_usage': [],
            'fuel_efficiency': [],
            'landing_quality': []
        }

    def _on_step(self) -> bool:
        """
        Called at each step during training.
        TODO: Implement actual metric collection:
        - Get thrust values from env.get_attr('thrust')
        - Calculate fuel efficiency from state/action data
        - Analyze landing behavior from episode data
        """
        # Placeholder metrics for demonstration
        if self.locals.get('dones', [False])[0]:  # Episode ended
            # Example of how metrics would be collected
            # thrust = self.training_env.get_attr('thrust')[0]
            # fuel = self.training_env.get_attr('fuel')[0]
            # landing = self.training_env.get_attr('landing_status')[0]
            
            # For now, just log dummy data
            self.metrics['thrust_usage'].append(np.random.random())
            self.metrics['fuel_efficiency'].append(np.random.random())
            self.metrics['landing_quality'].append(np.random.random())
            
            # Log to tensorboard
            for metric_name, value in self.metrics.items():
                if value:  # Only log if we have values
                    self.logger.record(f"custom/{metric_name}", value[-1])
        
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout.
        TODO: Implement aggregation of metrics over the rollout
        """
        # Reset metrics for next rollout
        for key in self.metrics:
            self.metrics[key] = [] 
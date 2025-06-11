import numpy as np
import gymnasium as gym

class BasicEnvWrapper(gym.Wrapper):
    """
    Base environment wrapper for consistent preprocessing and safety.
    Ensures action values are well-formed, clipped, and in correct dtype/shape.
    """
    def __init__(self, env_name: str, render_mode: str = "rgb_array", **kwargs):
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
        super().__init__(env)
        self.action_space = env.action_space  # Store action space for bounds checking

    def reset(self, **kwargs):
        """Forward the reset call directly."""
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        """
        Sanitize and validate action before passing to the environment.
        Ensures type and bounds are compatible with expected Box2D input.
        """
        action = np.array(action, dtype=np.float32)  # Ensure float32 dtype
        action = np.clip(action, self.action_space.low, self.action_space.high)  # Clip to valid range
        return self.env.step(action)
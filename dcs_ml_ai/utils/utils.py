"""
Utility functions for the DCS ML AI project.
"""

from typing import Any, Optional
import gymnasium as gym

def unwrap_env(env: Any) -> gym.Env:
    """
    Recursively unwraps a Gym environment to access the innermost core env.
    
    Handles common wrappers like VecEnv, Monitor, RecordVideo, etc.
    Will safely handle both single environments and vectorized environments.
    
    Args:
        env: The environment to unwrap. Can be a VecEnv, wrapped env, or raw env.
        
    Returns:
        The innermost environment after removing all wrappers.
        
    Example:
        >>> env = gym.make("LunarLander-v2")
        >>> env = gym.wrappers.Monitor(env, "videos")
        >>> env = gym.wrappers.RecordVideo(env, "videos")
        >>> core_env = unwrap_env(env)  # Gets the original LunarLander env
    """
    # Handle vectorized environments
    if hasattr(env, "envs"):
        env = env.envs[0]
    
    # Recursively unwrap any .env attributes
    while hasattr(env, "env"):
        env = env.env
        
    return env 
# Entry point for training the DCS ML agent

import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium as gym

from dcs_ml_ai.envs.wrappers import BasicEnvWrapper
from dcs_ml_ai.callbacks.metrics import CustomMetricCallback
from scripts.save_last_session import save_last_session

def clean_video_dir(video_dir):
    """Delete all files in the specified video directory."""
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            file_path = os.path.join(video_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

def make_env(env_id, video_folder=None, record_video=False):
    """Factory function to initialize the environment with optional video recording."""
    def _init():
        env = BasicEnvWrapper(env_id, render_mode="rgb_array" if record_video else None, continuous=True)
        if record_video:
            from gymnasium.wrappers import RecordVideo
            env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)
        return env
    return _init

def main():
    # Parse CLI args
    parser = argparse.ArgumentParser(description="Train the DCS ML AI agent.")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: silent, 1: info, 2: debug)")
    args = parser.parse_args()

    env_id = "LunarLanderContinuous-v3"
    env_name = env_id.split("-")[0].lower()
    
    # Create environment-specific directories
    base_dir = Path(".")
    video_dir = base_dir / "videos" / env_name
    model_dir = base_dir / "models" / env_name
    best_model_dir = model_dir / "best_model"
    checkpoint_dir = model_dir / "checkpoints"
    eval_logs_dir = best_model_dir / "eval_logs"
    
    # Ensure directories exist
    for directory in [video_dir, best_model_dir, checkpoint_dir, eval_logs_dir]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            raise

    # Initialize environment directly first to check action space
    base_env = gym.make(env_id, render_mode="rgb_array" if args.record_video else None, continuous=True)
    print(f"Action space: {base_env.action_space}")
    print(f"Action space low: {base_env.action_space.low}")
    print(f"Action space high: {base_env.action_space.high}")
    base_env.close()

    if args.record_video:
        clean_video_dir(video_dir)

    # Create vectorized training environment
    env = DummyVecEnv([make_env(env_id, video_dir, args.record_video)])
    env = VecMonitor(env)

    # Create evaluation environment with error handling
    try:
        eval_env = DummyVecEnv([make_env(env_id)])
        eval_env = VecMonitor(eval_env)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(eval_logs_dir),
            eval_freq=10_000,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
            warn=False
        )
    except Exception as e:
        print(f"Warning: Failed to create evaluation environment: {e}")
        print("Continuing without evaluation callback...")
        eval_callback = None

    # Initialize callbacks
    callbacks = []
    
    # Add evaluation callback if available
    if eval_callback is not None:
        callbacks.append(eval_callback)
    
    # Add checkpoint callback
    callbacks.append(CheckpointCallback(
        save_freq=100_000,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_checkpoint"
    ))
    
    # Add custom metrics callback with specified verbosity
    callbacks.append(CustomMetricCallback(verbose=args.verbose))

    # Train model with proper hyperparameters for Lunar Lander
    model = PPO(
        "MlpPolicy",
        env,
        verbose=args.verbose,
        tensorboard_log=str(base_dir / "tensorboard_logs" / env_name),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

    try:
        model.learn(total_timesteps=500_000, callback=callbacks)
    finally:
        env.close()
        if eval_callback is not None:
            eval_env.close()
        model.save(str(model_dir / f"ppo_{env_name}_final"))

    print("\nTraining complete.")

    # Prompt to save videos
    if args.record_video:
        save = input("Save this training session's videos? (y/n): ").lower()
        if save == 'y':
            print(f"Saving videos for this session...")
            save_last_session(video_dir, env_name)

if __name__ == "__main__":
    main()
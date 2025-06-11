# Entry point for training the DCS ML agent

import os
import shutil
import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import gymnasium as gym

from dcs_ml_ai.envs.wrappers import BasicEnvWrapper
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
    args = parser.parse_args()

    env_id = "LunarLanderContinuous-v3"
    env_name = env_id.split("-")[0].lower()
    video_folder = os.path.join("videos", env_name)

    # Prepare video folder if recording is enabled
    if args.record_video:
        os.makedirs(video_folder, exist_ok=True)
        clean_video_dir(video_folder)

    # Wrap the environment using DummyVecEnv for compatibility with SB3
    env = DummyVecEnv([make_env(env_id, video_folder, args.record_video)])
    env = VecMonitor(env)  # Monitor for tracking reward and episode length

    # Initialize PPO model with recommended hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

    # Run training
    try:
        model.learn(total_timesteps=10_000)
    finally:
        env.close()
        model.save(f"ppo_{env_name}")

    print("\nTraining complete.")

    # Optionally save videos from this session
    if args.record_video:
        save = input("Save this training session's videos? (y/n): ").lower()
        if save == 'y':
            print(f"Saved videos for this session to a timestamped folder in {os.path.dirname(video_folder)}")
            save_last_session(video_folder, env_name)

if __name__ == "__main__":
    main()
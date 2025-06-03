# Entry point for training the DCS ML agent

import os
import shutil
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import RecordVideo

from dcs_ml_ai.envs.wrappers import BasicEnvWrapper
from scripts.save_last_session import save_last_session

def clean_video_dir(video_dir):
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            file_path = os.path.join(video_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

def main():
    env_id = "CartPole-v1"
    env_name = env_id.split("-")[0].lower()  # -> 'cartpole'
    video_folder = os.path.join("videos", env_name)

    # Clean video directory
    os.makedirs(video_folder, exist_ok=True)
    clean_video_dir(video_folder)

    env = BasicEnvWrapper(env_id, render_mode="rgb_array")
    env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)
    env = DummyVecEnv([lambda: env])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs")
    model.learn(total_timesteps=10_000)
    env.close()
    model.save(f"ppo_{env_name}")

    print("\nTraining complete.")

    # Prompt to save session
    save = input("Save this training session's videos? (y/n): ").lower()
    if save == 'y':
        save_last_session(video_folder, env_name)

if __name__ == "__main__":
    main()
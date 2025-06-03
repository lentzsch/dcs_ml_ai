# Entry point for training the DCS ML agent

from stable_baselines3 import PPO
from dcs_ml_ai.envs.wrappers import BasicEnvWrapper

# Import necessary wrappers
from gymnasium.wrappers import RecordVideo
import os

def main():
    # Set up video recording path
    video_folder = os.path.join("videos", "cartpole")
    os.makedirs(video_folder, exist_ok=True)

    # Initialize environment
    env = BasicEnvWrapper("CartPole-v1")

    # Wrap with RecordVideo to capture episodes
    env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)  # Record every episode

    # Initialize model with TensorBoard logging
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs"
    )

    # Learn!
    model.learn(total_timesteps=10_000)
    env.close()
    model.save("ppo_cartpole")

if __name__ == "__main__":
    main()


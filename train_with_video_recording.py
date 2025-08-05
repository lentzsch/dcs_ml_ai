#!/usr/bin/env python3
"""
Training script with video recording for noteworthy episodes
"""
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
sys.path.append('.')

from dcs_ml_ai.envs.two_d_flight_env import TwoDFlightEnv

class VideoRecordingCallback(BaseCallback):
    """Callback to save noteworthy episodes as videos"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.crash_count = 0
        self.success_count = 0
        
    def _on_step(self) -> bool:
        # Get current episode info
        if len(self.training_env.buf_rews) > 0:
            episode_reward = self.training_env.buf_rews[0]
            episode_length = self.training_env.buf_lengths[0]
            
            # Check if episode just ended
            if self.training_env.buf_dones[0]:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Determine if this was a noteworthy episode
                if episode_reward < -500:  # Bad episode (crash, fuel depletion)
                    self.crash_count += 1
                    if self.crash_count <= 3:  # Save first few crashes
                        self.training_env.envs[0].save_noteworthy_episode("crash", f"reward_{episode_reward:.0f}")
                        print(f"🎬 Saved crash video (reward: {episode_reward:.0f})")
                
                elif episode_reward > 50:  # Good episode
                    self.success_count += 1
                    if self.success_count <= 3:  # Save first few successes
                        self.training_env.envs[0].save_noteworthy_episode("success", f"reward_{episode_reward:.0f}")
                        print(f"🎬 Saved success video (reward: {episode_reward:.0f})")
                
                elif episode_length > 800:  # Long episode (outlier)
                    self.training_env.envs[0].save_noteworthy_episode("outlier", f"length_{episode_length}")
                    print(f"🎬 Saved outlier video (length: {episode_length})")
        
        return True

def train_with_video_recording():
    """Train with video recording for noteworthy episodes"""
    print("🎬 Training with Video Recording")
    print("Videos will be saved for crashes, successes, and outliers")
    print()
    
    # Create environment with video recording
    env = TwoDFlightEnv(
        render_mode="rgb_array",
        record_video=True,
        max_time_steps=1000
    )
    
    # Create callback for video recording
    video_callback = VideoRecordingCallback()
    
    # Create and train model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="logs/2d_flight_video/"
    )
    
    print("🚀 Starting training with video recording...")
    model.learn(
        total_timesteps=50000,
        callback=video_callback,
        progress_bar=True
    )
    
    print("\n✅ Training completed!")
    print("Check videos/2d_flight/ for saved videos of noteworthy episodes.")

if __name__ == "__main__":
    try:
        train_with_video_recording()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 
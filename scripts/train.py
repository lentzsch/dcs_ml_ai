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
from gymnasium.wrappers import RecordVideo

from dcs_ml_ai.envs.two_d_flight_env import TwoDFlightEnv
from dcs_ml_ai.callbacks.metrics import CustomMetricCallback
from scripts.save_last_session import save_last_session

def clean_video_dir(video_dir):
    """Delete all files in the specified video directory."""
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            file_path = os.path.join(video_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

def make_env(video_folder=None, record_video=False):
    """Factory function to initialize the TwoDFlightEnv with optional video recording."""
    def _init():
        env = TwoDFlightEnv(
            target_airspeed=120.0,
            max_altitude=10000.0,
            max_time_steps=1000,
            fuel_penalty_factor=0.1,
            render_mode="rgb_array" if record_video else None
        )
        if record_video and video_folder:
            env = RecordVideo(env, video_folder, episode_trigger=lambda x: x % 10 == 0)
        return env
    return _init

def main():
    # Parse CLI args
    parser = argparse.ArgumentParser(description="Train the DCS ML AI agent on TwoDFlightEnv.")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: silent, 1: info, 2: debug)")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total training timesteps")
    args = parser.parse_args()

    env_name = "2d_flight"
    
    # Create environment-specific directories
    base_dir = Path(".")
    video_dir = base_dir / "videos" / env_name
    model_dir = base_dir / "models" / env_name
    best_model_dir = model_dir / "best_model"
    checkpoint_dir = model_dir / "checkpoints"
    eval_logs_dir = best_model_dir / "eval_logs"
    logs_dir = base_dir / "logs" / env_name
    
    # Ensure directories exist
    for directory in [video_dir, best_model_dir, checkpoint_dir, eval_logs_dir, logs_dir]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            raise

    # Initialize environment directly first to check action space
    print("Testing TwoDFlightEnv creation...")
    base_env = TwoDFlightEnv(render_mode="rgb_array" if args.record_video else None)
    print(f"✅ Environment created successfully!")
    print(f"Action space: {base_env.action_space}")
    print(f"Action space low: {base_env.action_space.low}")
    print(f"Action space high: {base_env.action_space.high}")
    print(f"Observation space: {base_env.observation_space}")
    base_env.close()

    if args.record_video:
        clean_video_dir(video_dir)

    # Create vectorized training environment
    print("Creating training environment...")
    env = DummyVecEnv([make_env(video_dir, args.record_video)])
    env = VecMonitor(env)

    # Create evaluation environment with error handling
    print("Creating evaluation environment...")
    try:
        eval_env = DummyVecEnv([make_env()])
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

    # Train model with proper hyperparameters for flight control
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=args.verbose,
        tensorboard_log=str(logs_dir),
        learning_rate=3e-4,      # Conservative learning rate for stable flight
        n_steps=2048,            # Sufficient steps for episode completion
        batch_size=64,           # Moderate batch size
        n_epochs=10,             # Multiple epochs for stable learning
        gamma=0.99,              # High discount for long-term planning
        gae_lambda=0.95,         # GAE for variance reduction
        clip_range=0.2,          # Conservative clipping
        ent_coef=0.01,           # Small entropy coefficient for exploration
        vf_coef=0.5,             # Value function coefficient
        max_grad_norm=0.5        # Gradient clipping for stability
    )
    print(f"✅ PPO model created with tensorboard_log={logs_dir}")

    # Start training
    print(f"\n🚀 Starting training for {args.total_timesteps:,} timesteps...")
    print(f"📊 Logs will be saved to: {logs_dir}")
    print(f"💾 Best model will be saved to: {best_model_dir}")
    print(f"🔄 Checkpoints will be saved to: {checkpoint_dir}")
    print(f"🎬 Videos will be saved to: {video_dir}" if args.record_video else "📹 Video recording disabled")
    print()

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    finally:
        print("\n🔄 Cleaning up...")
        env.close()
        if eval_callback is not None:
            eval_env.close()
        
        # Save final model
        final_model_path = model_dir / f"ppo_{env_name}_final"
        model.save(str(final_model_path))
        print(f"💾 Final model saved to: {final_model_path}.zip")

    print("\n✅ Training complete!")
    
    # Display training summary
    print("\n📊 Training Summary:")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Environment: {env_name}")
    
    print("\n📁 Output Structure:")
    print(f"   📊 TensorBoard logs: {logs_dir}")
    print(f"   🏆 Best model: {best_model_dir}/best_model.zip")
    print(f"   💾 Checkpoints: {checkpoint_dir}/")
    print(f"   📈 Evaluation logs: {eval_logs_dir}/")
    print(f"   🎯 Final model: {final_model_path}.zip")
    
    # Instructions for next steps
    print("\n🔧 Next Steps:")
    print(f"   1. View training progress: tensorboard --logdir {logs_dir}")
    print(f"   2. Test best model: python -c \"from stable_baselines3 import PPO; model = PPO.load('{best_model_dir}/best_model'); ...\"")
    print(f"   3. Load checkpoint: python -c \"from stable_baselines3 import PPO; model = PPO.load('{checkpoint_dir}/ppo_checkpoint_XXXX_steps'); ...\")")

    # Handle video saving
    if args.record_video:
        save = input("\n💾 Save this training session's videos? (y/n): ").lower()
        if save == 'y':
            print("Saving videos for this session...")
            save_last_session(video_dir, env_name)
        else:
            print("Videos not saved.")

if __name__ == "__main__":
    main()
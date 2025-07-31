#!/usr/bin/env python3
"""
Training script for the 2D Flight Environment

This script trains a PPO agent on the TwoDFlightEnv with comprehensive logging,
evaluation, and checkpointing capabilities.
"""

import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium as gym

from dcs_ml_ai.envs.two_d_flight_env import TwoDFlightEnv
from dcs_ml_ai.callbacks.metrics import CustomMetricCallback
from scripts.save_last_session import save_last_session


class FlightMetricCallback(CustomMetricCallback):
    """
    Custom metrics callback specifically for the 2D Flight Environment.
    
    Tracks flight-specific metrics:
    - Fuel efficiency
    - Airspeed management
    - Altitude stability
    - Energy management
    """
    
    def __init__(self, verbose: int = 0, run_id: str = None):
        super().__init__(verbose, run_id)
        # Override base metrics for flight-specific ones
        self.metrics = {
            'fuel_efficiency': [],
            'airspeed_error': [],
            'altitude_stability': [],
            'energy_efficiency': [],
            'crash_rate': []
        }
    
    def _on_step(self) -> bool:
        """Collect flight-specific metrics from the environment."""
        self.total_steps += 1
        
        # Check if any environment has completed an episode
        if any(self.locals.get("dones", [])):
            # Get the core flight environment
            from dcs_ml_ai.utils import unwrap_env
            core_env = unwrap_env(self.training_env)
            
            # Extract flight metrics
            if hasattr(core_env, 'fuel'):
                fuel_remaining = core_env.fuel
                fuel_efficiency = fuel_remaining  # Higher remaining fuel = better efficiency
                self.metrics['fuel_efficiency'].append(fuel_efficiency)
                self.logger.record("flight/fuel_efficiency", fuel_efficiency)
            
            if hasattr(core_env, 'airspeed') and hasattr(core_env, 'target_airspeed'):
                airspeed_error = abs(core_env.airspeed - core_env.target_airspeed)
                self.metrics['airspeed_error'].append(airspeed_error)
                self.logger.record("flight/airspeed_error", airspeed_error)
            
            if hasattr(core_env, 'pitch_rate'):
                altitude_stability = 1.0 / (1.0 + abs(core_env.pitch_rate))  # Inverse of pitch rate
                self.metrics['altitude_stability'].append(altitude_stability)
                self.logger.record("flight/altitude_stability", altitude_stability)
            
            # Track crash rate
            if hasattr(core_env, 'terminated') and hasattr(core_env, 'altitude'):
                crashed = core_env.terminated and core_env.altitude <= 0
                crash_rate = 1.0 if crashed else 0.0
                self.metrics['crash_rate'].append(crash_rate)
                self.logger.record("flight/crash_rate", crash_rate)
            
            # Energy efficiency (based on kinetic + potential energy vs fuel used)
            if hasattr(core_env, 'fuel'):
                energy_efficiency = core_env.fuel  # Simplified: fuel remaining indicates efficiency
                self.metrics['energy_efficiency'].append(energy_efficiency)
                self.logger.record("flight/energy_efficiency", energy_efficiency)
        
        return True


def clean_video_dir(video_dir):
    """Delete all files in the specified video directory."""
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            file_path = os.path.join(video_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)


def make_env(record_video=False, video_folder=None, seed=None):
    """Factory function to create TwoDFlightEnv with optional video recording."""
    def _init():
        env = TwoDFlightEnv(
            target_airspeed=120.0,
            max_altitude=10000.0,
            max_time_steps=1000,
            fuel_penalty_factor=0.1,
            render_mode="rgb_array" if record_video else None
        )
        
        if seed is not None:
            env.reset(seed=seed)
        
        if record_video and video_folder:
            from gymnasium.wrappers import RecordVideo
            env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)
        
        return env
    return _init


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train PPO agent on 2D Flight Environment")
    parser.add_argument("--record-video", action="store_true", help="Enable video recording")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: silent, 1: info, 2: debug)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="Checkpoint save frequency")
    args = parser.parse_args()

    env_name = "2d_flight"
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Create directory structure
    base_dir = Path(".")
    video_dir = base_dir / "videos" / env_name
    model_dir = base_dir / "models" / env_name
    best_model_dir = model_dir / "best_model"
    checkpoint_dir = model_dir / "checkpoints"
    eval_logs_dir = best_model_dir / "eval_logs"
    logs_dir = base_dir / "logs" / env_name
    
    # Ensure all directories exist
    for directory in [video_dir, best_model_dir, checkpoint_dir, eval_logs_dir, logs_dir]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            raise

    # Test environment creation
    print("Testing environment creation...")
    test_env = TwoDFlightEnv()
    print(f"✅ Environment created successfully!")
    print(f"Action space: {test_env.action_space}")
    print(f"Observation space: {test_env.observation_space}")
    test_env.close()

    if args.record_video:
        clean_video_dir(video_dir)

    # Create training environment
    print("Creating training environment...")
    env = DummyVecEnv([make_env(args.record_video, video_dir, args.seed)])
    env = VecMonitor(env)

    # Create evaluation environment
    print("Creating evaluation environment...")
    try:
        eval_env = DummyVecEnv([make_env(seed=args.seed + 1000)])  # Different seed for eval
        eval_env = VecMonitor(eval_env)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(eval_logs_dir),
            eval_freq=args.eval_freq,
            n_eval_episodes=5,  # As requested
            deterministic=True,
            render=False,
            warn=False,
            verbose=args.verbose
        )
        print(f"✅ Evaluation callback configured (freq={args.eval_freq}, episodes=5)")
    except Exception as e:
        print(f"⚠️  Warning: Failed to create evaluation environment: {e}")
        print("Continuing without evaluation callback...")
        eval_callback = None

    # Initialize callbacks
    callbacks = []
    
    # Add evaluation callback
    if eval_callback is not None:
        callbacks.append(eval_callback)
    
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_2d_flight_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    print(f"✅ Checkpoint callback configured (freq={args.checkpoint_freq})")
    
    # Add flight-specific metrics callback
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    flight_metrics_callback = FlightMetricCallback(verbose=args.verbose, run_id=run_id)
    callbacks.append(flight_metrics_callback)
    print(f"✅ Flight metrics callback configured (run_id={run_id})")

    # Create PPO model with appropriate hyperparameters for flight control
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=args.verbose,
        tensorboard_log=str(logs_dir),
        seed=args.seed,
        # Flight control specific hyperparameters
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
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False
        )
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
    print(f"   Seed: {args.seed}")
    print(f"   Run ID: {run_id}")
    
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
    print(f"   3. Load checkpoint: python -c \"from stable_baselines3 import PPO; model = PPO.load('{checkpoint_dir}/ppo_2d_flight_checkpoint_XXXX_steps'); ...\"")

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
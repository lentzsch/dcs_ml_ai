#!/usr/bin/env python3
"""
Test dynamic video recording with clear movement
"""
import sys
import time
import numpy as np
sys.path.append('.')

from dcs_ml_ai.envs.two_d_flight_env import TwoDFlightEnv

def test_dynamic_video():
    """Test video recording with dynamic movement"""
    print("🎬 Testing Dynamic Video Recording")
    print("This will create videos with clear movement")
    print()
    
    # Test 1: Oscillating flight pattern
    print("🛫 Test 1: Oscillating Flight Pattern")
    env = TwoDFlightEnv(render_mode="rgb_array", record_video=True, max_time_steps=200)
    obs, info = env.reset()
    
    # Create oscillating pattern
    for step in range(150):
        # Oscillate between climb and dive
        if step % 30 < 15:
            action = [0.5, 0.7]  # Climb
        else:
            action = [-0.3, 0.4]  # Dive
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # Save as oscillating episode
    env.save_noteworthy_episode("oscillating", "climb_dive_pattern")
    env.close()
    
    # Test 2: Spiral pattern
    print("\n🛫 Test 2: Spiral Pattern")
    env = TwoDFlightEnv(render_mode="rgb_array", record_video=True, max_time_steps=200)
    obs, info = env.reset()
    
    # Create spiral pattern
    for step in range(150):
        # Vary pitch in a spiral pattern
        pitch_command = 0.3 * np.sin(step * 0.2) + 0.1 * np.cos(step * 0.1)
        throttle_command = 0.5 + 0.2 * np.sin(step * 0.15)
        action = [pitch_command, throttle_command]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # Save as spiral episode
    env.save_noteworthy_episode("spiral", "varying_pattern")
    env.close()
    
    # Test 3: Aggressive maneuvers
    print("\n🛫 Test 3: Aggressive Maneuvers")
    env = TwoDFlightEnv(render_mode="rgb_array", record_video=True, max_time_steps=200)
    obs, info = env.reset()
    
    # Create aggressive maneuvers
    for step in range(150):
        # Sharp maneuvers every 20 steps
        if step % 20 < 5:
            action = [0.8, 1.0]  # Sharp climb
        elif step % 20 < 10:
            action = [-0.6, 0.3]  # Sharp dive
        elif step % 20 < 15:
            action = [0.0, 0.8]  # Level acceleration
        else:
            action = [0.2, 0.5]  # Gentle climb
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # Save as aggressive episode
    env.save_noteworthy_episode("aggressive", "sharp_maneuvers")
    env.close()
    
    print("\n✅ Dynamic video tests completed!")
    print("Check the videos/2d_flight/ directory for the saved videos.")
    print("These should show clear movement and aircraft dynamics.")

if __name__ == "__main__":
    try:
        test_dynamic_video()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 
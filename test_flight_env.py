#!/usr/bin/env python3
"""
Test script for the 2D Flight Environment

This script instantiates the TwoDFlightEnv and runs a random agent
to verify the environment works correctly and debug the physics simulation.
"""

import numpy as np
import sys
import os

# Add the dcs_ml_ai package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from dcs_ml_ai.envs.two_d_flight_env import TwoDFlightEnv


def test_random_agent(num_episodes: int = 3, max_steps_per_episode: int = 200):
    """Test the environment with a random agent"""
    
    print("🛫 Testing 2D Flight Environment")
    print("=" * 50)
    
    # Create environment
    env = TwoDFlightEnv(
        target_airspeed=120.0,
        max_altitude=10000.0,
        max_time_steps=max_steps_per_episode,
        fuel_penalty_factor=0.1,
        render_mode="human"
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()
    
    for episode in range(num_episodes):
        print(f"🎯 Episode {episode + 1}/{num_episodes}")
        print("-" * 30)
        
        # Reset environment
        observation, info = env.reset(seed=episode)
        total_reward = 0.0
        
        print(f"Initial state:")
        print(f"  Pitch: {np.degrees(observation[0]):.1f}°")
        print(f"  Altitude: {observation[2]:.1f} ft")
        print(f"  Airspeed: {observation[4]:.1f} kts")
        print(f"  Fuel: {observation[5]:.2f}")
        print()
        
        step = 0
        while step < max_steps_per_episode:
            # Generate random action
            action = env.action_space.sample()
            elevator, throttle = action
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Print state every 20 steps or on significant events
            if step % 20 == 0 or terminated or truncated:
                print(f"Step {step:3d}: "
                      f"Elevator={elevator:+.2f}, Throttle={throttle:.2f} | "
                      f"Reward={reward:+.2f} | "
                      f"Alt={observation[2]:6.1f}ft, "
                      f"Speed={observation[4]:5.1f}kts, "
                      f"Fuel={observation[5]:.2f}, "
                      f"Pitch={np.degrees(observation[0]):+5.1f}°")
            
            step += 1
            
            # Check if episode ended
            if terminated or truncated:
                if terminated:
                    if observation[2] <= 0:
                        print("  ❌ Episode terminated: CRASHED (altitude <= 0)")
                    elif observation[5] <= 0:
                        print("  ❌ Episode terminated: FUEL DEPLETED")
                    elif observation[2] > env.max_altitude:
                        print("  ❌ Episode terminated: ALTITUDE TOO HIGH")
                    else:
                        print("  ❌ Episode terminated: OTHER REASON")
                elif truncated:
                    print("  ⏰ Episode truncated: TIME LIMIT REACHED")
                break
        
        print(f"Episode summary:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final airspeed error: {info['airspeed_error']:.1f} kts")
        print(f"  Fuel remaining: {info['fuel_remaining']:.2f}")
        print(f"  Steps completed: {step}")
        
        # Energy analysis
        kinetic_energy = info['energy_state']['kinetic']
        potential_energy = info['energy_state']['potential']
        total_energy = kinetic_energy + potential_energy
        print(f"  Energy state:")
        print(f"    Kinetic: {kinetic_energy/1e6:.2f}M ft-lbf")
        print(f"    Potential: {potential_energy/1e6:.2f}M ft-lbf")
        print(f"    Total: {total_energy/1e6:.2f}M ft-lbf")
        print()
    
    env.close()
    print("✅ Environment test completed!")


def test_action_space_limits():
    """Test the action space limits and edge cases"""
    
    print("🔧 Testing Action Space Limits")
    print("=" * 40)
    
    env = TwoDFlightEnv(max_time_steps=50)
    
    # Test extreme actions
    test_actions = [
        ([-1.0, 0.0], "Full nose down, idle throttle"),
        ([1.0, 1.0], "Full nose up, full throttle"),
        ([0.0, 0.5], "Neutral elevator, half throttle"),
        ([-0.5, 0.8], "Moderate nose down, high throttle")
    ]
    
    for action_vals, description in test_actions:
        print(f"\n🎮 Testing: {description}")
        observation, info = env.reset()
        
        # Apply action for 10 steps
        for step in range(10):
            action = np.array(action_vals, dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            
            if step == 9:  # Print final state
                print(f"  After 10 steps:")
                print(f"    Altitude: {observation[2]:.1f} ft")
                print(f"    Airspeed: {observation[4]:.1f} kts")
                print(f"    Pitch: {np.degrees(observation[0]):+.1f}°")
                print(f"    Vertical speed: {observation[3]:+.1f} ft/s")
                print(f"    Reward: {reward:+.2f}")
            
            if terminated or truncated:
                print(f"    Episode ended at step {step}")
                break
    
    env.close()
    print("\n✅ Action space test completed!")


def test_physics_consistency():
    """Test physics consistency and energy conservation principles"""
    
    print("⚡ Testing Physics Consistency")
    print("=" * 35)
    
    env = TwoDFlightEnv(max_time_steps=100)
    observation, info = env.reset()
    
    initial_energy = info['energy_state']['kinetic'] + info['energy_state']['potential']
    print(f"Initial total energy: {initial_energy/1e6:.2f}M ft-lbf")
    
    # Test energy management scenarios
    scenarios = [
        ("Constant cruise", [0.0, 0.4]),  # level flight, moderate throttle
        ("Climb", [0.2, 0.8]),            # slight climb, high throttle
        ("Dive", [-0.3, 0.2])             # dive, low throttle
    ]
    
    for scenario_name, action_vals in scenarios:
        print(f"\n📊 Scenario: {scenario_name}")
        observation, info = env.reset()
        
        action = np.array(action_vals, dtype=np.float32)
        
        for step in range(30):
            observation, reward, terminated, truncated, info = env.step(action)
            
            if step in [9, 19, 29]:  # Check at intervals
                ke = info['energy_state']['kinetic']
                pe = info['energy_state']['potential']
                total = ke + pe
                print(f"  Step {step+1:2d}: KE={ke/1e6:.1f}M, PE={pe/1e6:.1f}M, "
                      f"Total={total/1e6:.1f}M ft-lbf, Alt={observation[2]:.0f}ft, "
                      f"Speed={observation[4]:.0f}kts")
            
            if terminated or truncated:
                break
    
    env.close()
    print("\n✅ Physics test completed!")


if __name__ == "__main__":
    print("🚁 2D Flight Environment Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        test_random_agent()
        print()
        test_action_space_limits()
        print()
        test_physics_consistency()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
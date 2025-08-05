import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque


class TwoDFlightEnv(gym.Env):
    """
    2D Flight Environment for Energy Management Training
    
    The agent learns to manage aircraft energy using pitch and throttle controls.
    Throttle adds total energy to the system, while pitch exchanges energy between
    altitude (potential) and airspeed (kinetic).
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(
        self, 
        target_airspeed: float = 120.0,  # knots
        max_altitude: float = 10000.0,   # feet
        max_time_steps: int = 1000,
        fuel_penalty_factor: float = 0.1,
        render_mode: Optional[str] = None,
        record_video: bool = False,
        video_dir: str = "videos/2d_flight"
    ):
        super().__init__()
        
        # Environment configuration
        self.target_airspeed = target_airspeed
        self.max_altitude = max_altitude
        self.max_time_steps = max_time_steps
        self.fuel_penalty_factor = fuel_penalty_factor
        self.render_mode = render_mode
        
        # Physical constants
        self.dt = 0.1  # time step (seconds)
        self.g = 32.174  # gravity (ft/s²)
        self.max_thrust = 5000.0  # maximum thrust (lbf)
        self.aircraft_mass = 15000.0  # aircraft mass (lbm)
        self.drag_coefficient = 0.02  # drag coefficient
        self.fuel_consumption_rate = 0.0001  # fuel consumption per throttle unit per step
        
        # Control limits
        self.max_pitch = np.pi / 3  # ±60 degrees
        self.max_pitch_rate = np.pi / 6  # ±30 deg/s
        self.elevator_authority = 0.1  # elevator effectiveness (rad/s² per unit deflection)
        
        # Action space: [elevator_deflection, throttle_setting]
        # elevator_deflection: -1 (nose down) to +1 (nose up)
        # throttle_setting: 0 (idle) to 1 (full throttle)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Observation space: [pitch, pitch_rate, altitude, vertical_speed, airspeed, fuel, throttle, elevator]
        self.observation_space = spaces.Box(
            low=np.array([
                -self.max_pitch,      # pitch (rad)
                -self.max_pitch_rate, # pitch_rate (rad/s)
                0.0,                  # altitude (ft)
                -500.0,               # vertical_speed (ft/s)
                50.0,                 # airspeed (knots)
                0.0,                  # fuel (0-1)
                0.0,                  # throttle (0-1)
                -1.0                  # elevator (-1 to 1)
            ]),
            high=np.array([
                self.max_pitch,       # pitch (rad)
                self.max_pitch_rate,  # pitch_rate (rad/s)
                self.max_altitude,    # altitude (ft)
                500.0,                # vertical_speed (ft/s)
                300.0,                # airspeed (knots)
                1.0,                  # fuel (0-1)
                1.0,                  # throttle (0-1)
                1.0                   # elevator (-1 to 1)
            ]),
            dtype=np.float32
        )
        
        # Initialize state variables directly
        self.pitch = 0.0  # level flight
        self.pitch_rate = 0.0
        self.altitude = 5000.0  # feet
        self.vertical_speed = 0.0  # ft/s
        self.airspeed = self.target_airspeed  # knots
        self.fuel = 1.0  # full tank
        self.throttle = 0.5  # moderate throttle
        self.elevator = 0.0  # neutral elevator
        
        # Episode tracking
        self.time_step = 0
        self.terminated = False
        self.truncated = False
        
        # Visualization state
        self.fig = None
        self.ax = None
        self.aircraft_marker = None
        self.position_history = deque(maxlen=50)  # Trail of recent positions
        self.pitch_history = deque(maxlen=50)     # Trail of recent pitch angles
        self.time_history = deque(maxlen=50)      # Time stamps for trail
        self.visualization_initialized = False
        
        # Video recording state
        self.record_video = record_video
        self.video_dir = video_dir
        self.video_frames = []
        self.episode_count = 0
        self.session_id = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initial state (level flight at moderate altitude and airspeed)
        self.pitch = 0.0  # level flight
        self.pitch_rate = 0.0
        self.altitude = 5000.0  # feet
        self.vertical_speed = 0.0  # ft/s
        self.airspeed = self.target_airspeed  # knots
        self.fuel = 1.0  # full tank
        self.throttle = 0.5  # moderate throttle
        self.elevator = 0.0  # neutral elevator
        
        # Episode tracking
        self.time_step = 0
        self.terminated = False
        self.truncated = False
        
        # Initialize video recording for new episode
        if self.record_video:
            self._initialize_video_recording()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        if self.terminated or self.truncated:
            return self._get_observation(), 0.0, self.terminated, self.truncated, self._get_info()
        
        # Parse action
        elevator_command = np.clip(action[0], -1.0, 1.0)
        throttle_command = np.clip(action[1], 0.0, 1.0)
        
        # Update controls (with some lag/smoothing)
        self.elevator += 0.3 * (elevator_command - self.elevator)  # smooth elevator response
        self.throttle += 0.2 * (throttle_command - self.throttle)  # smooth throttle response
        self.elevator = np.clip(self.elevator, -1.0, 1.0)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        
        # Physics simulation
        self._update_physics()
        
        # Update time
        self.time_step += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        self._check_termination()
        
        # Save video frame if recording
        if self.record_video:
            self._save_video_frame()
        
        return self._get_observation(), reward, self.terminated, self.truncated, self._get_info()
    
    def _update_physics(self):
        """Update aircraft physics based on current state and controls"""
        # Convert airspeed from knots to ft/s
        airspeed_fps = self.airspeed * 1.68781  # knots to ft/s
        
        # Calculate forces
        thrust = self.throttle * self.max_thrust  # thrust force (lbf)
        drag = self.drag_coefficient * airspeed_fps**2  # drag force (lbf)
        
        # Net horizontal force affects airspeed
        net_force = thrust - drag
        airspeed_acceleration = net_force / self.aircraft_mass * self.g  # ft/s²
        
        # Update airspeed
        airspeed_fps += airspeed_acceleration * self.dt
        airspeed_fps = max(airspeed_fps, 50.0 * 1.68781)  # minimum airspeed (50 knots)
        self.airspeed = airspeed_fps / 1.68781  # convert back to knots
        
        # Pitch dynamics (elevator control)
        pitch_acceleration = self.elevator * self.elevator_authority  # rad/s²
        self.pitch_rate += pitch_acceleration * self.dt
        self.pitch_rate = np.clip(self.pitch_rate, -self.max_pitch_rate, self.max_pitch_rate)
        
        # Update pitch
        self.pitch += self.pitch_rate * self.dt
        self.pitch = np.clip(self.pitch, -self.max_pitch, self.max_pitch)
        
        # Calculate vertical speed and altitude
        self.vertical_speed = airspeed_fps * np.sin(self.pitch)
        self.altitude += self.vertical_speed * self.dt
        
        # Energy exchange: higher pitch trades airspeed for altitude
        # This is a simplified model where steep climbs reduce airspeed
        energy_exchange_factor = 0.1
        airspeed_loss_from_climb = energy_exchange_factor * self.vertical_speed * self.dt / 1.68781
        self.airspeed = max(self.airspeed - airspeed_loss_from_climb, 50.0)
        
        # Fuel consumption
        fuel_consumption = self.throttle * self.fuel_consumption_rate
        self.fuel = max(0.0, self.fuel - fuel_consumption)
        
        # Update position history for visualization trail
        current_time = self.time_step * self.dt
        self.position_history.append((current_time, self.altitude))
        self.pitch_history.append(self.pitch)
        self.time_history.append(current_time)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state and energy management"""
        reward = 0.0
        
        # Penalize deviation from target airspeed
        airspeed_error = abs(self.airspeed - self.target_airspeed)
        airspeed_penalty = -airspeed_error * 0.01  # scale penalty
        reward += airspeed_penalty
        
        # Penalize fuel consumption
        fuel_penalty = -self.throttle * self.fuel_penalty_factor
        reward += fuel_penalty
        
        # Small bonus for stable flight (low pitch rate)
        stability_bonus = -abs(self.pitch_rate) * 0.1
        reward += stability_bonus
        
        # Large penalty for crash
        if self.altitude <= 0:
            reward -= 1000.0
        
        # Large penalty for fuel depletion
        if self.fuel <= 0:
            reward -= 500.0
        
        # Small bonus for maintaining reasonable altitude
        if 1000 <= self.altitude <= 8000:
            reward += 0.1
        
        return reward
    
    def _check_termination(self):
        """Check if episode should terminate"""
        # Crash (ground contact)
        if self.altitude <= 0:
            self.terminated = True
        
        # Fuel depletion
        if self.fuel <= 0:
            self.terminated = True
        
        # Altitude too high
        if self.altitude > self.max_altitude:
            self.terminated = True
        
        # Time limit
        if self.time_step >= self.max_time_steps:
            self.truncated = True
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        return np.array([
            self.pitch,
            self.pitch_rate,
            self.altitude,
            self.vertical_speed,
            self.airspeed,
            self.fuel,
            self.throttle,
            self.elevator
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        return {
            'time_step': self.time_step,
            'airspeed_error': abs(self.airspeed - self.target_airspeed),
            'fuel_remaining': self.fuel,
            'altitude': self.altitude,
            'energy_state': {
                'kinetic': 0.5 * self.aircraft_mass * (self.airspeed * 1.68781)**2,
                'potential': self.aircraft_mass * self.g * self.altitude
            }
        }
    
    def render(self):
        """Render the environment using matplotlib visualization"""
        if self.render_mode == "human":
            self._render_matplotlib()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _initialize_visualization(self):
        """Initialize matplotlib visualization components"""
        if self.visualization_initialized:
            return
        
        # Set backend based on render mode
        if self.render_mode == "rgb_array":
            matplotlib.use('Agg')  # Use non-interactive backend for RGB arrays
        else:
            # Try to use interactive backend for human viewing
            try:
                matplotlib.use('TkAgg')
                plt.ion()  # Turn on interactive mode
            except:
                matplotlib.use('Agg')  # Fallback to non-interactive
                
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle('2D Flight Environment - Energy Management Training', fontsize=14, fontweight='bold')
        
        # Set up the plot
        self.ax.set_xlim(0, 100)  # Time axis (seconds)
        self.ax.set_ylim(0, self.max_altitude + 1000)  # Altitude axis (feet)
        self.ax.set_xlabel('Time (seconds)', fontsize=12)
        self.ax.set_ylabel('Altitude (feet)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # Initialize aircraft marker (arrow that will point in pitch direction)
        self.aircraft_triangle = None  # Will be created as a FancyArrowPatch
        self.aircraft_center = None    # Will be created as a Circle patch
        
        # Initialize trail line
        self.trail_line, = self.ax.plot([], [], 'b-', alpha=0.6, linewidth=2, label='Flight Path')
        
        # Initialize pitch vector line (red line showing pitch direction)
        self.pitch_vector, = self.ax.plot([], [], 'r-', linewidth=3, alpha=0.8, label='Pitch Vector')
        
        # Initialize target altitude line
        target_alt = 5000  # Reference altitude
        self.ax.axhline(y=target_alt, color='green', linestyle='--', alpha=0.7, 
                       label=f'Reference Alt ({target_alt} ft)')
        
        # Add a dummy line for aircraft marker legend (since patches don't show in legend easily)
        self.ax.plot([], [], 'r-', linewidth=4, alpha=0.8, label='Aircraft (arrow shows pitch)')
        
        # Add compact legend in upper left
        self.ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
        
                # Initialize text displays for telemetry - repositioned to avoid overlap
        # Telemetry info in upper right corner
        self.telemetry_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes, 
                                         fontsize=10, verticalalignment='top', horizontalalignment='right',
                                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Energy state info in lower left corner
        self.energy_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes,
                                       fontsize=9, verticalalignment='bottom', horizontalalignment='left',
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        plt.tight_layout()
        self.visualization_initialized = True
    
    def _render_matplotlib(self):
        """Render using matplotlib with aircraft visualization and telemetry"""
        if not self.visualization_initialized:
            self._initialize_visualization()
        
        current_time = self.time_step * self.dt
        
        # Update aircraft position
        aircraft_x = current_time
        aircraft_y = self.altitude
        
        # AGGRESSIVE CLEANUP: Remove ALL patches to eliminate any lingering distorted triangles
        # We'll recreate only the patches we want (arrow and center dot)
        patches_to_remove = list(self.ax.patches)  # Copy the list since we'll modify it
        for patch in patches_to_remove:
            patch.remove()
        
        # Clear our references since we removed everything
        self.aircraft_triangle = None
        self.aircraft_center = None
        
        # RESTORE arrow (FancyArrowPatch) - this was NOT the culprit
        # Use an arrow that automatically points in the pitch direction
        arrow_length = 1.5  # seconds (length on time axis)
        arrow_height = 100  # feet (height on altitude axis)
        
        # Calculate arrow end point based on pitch
        arrow_end_x = aircraft_x + arrow_length * np.cos(self.pitch)
        arrow_end_y = aircraft_y + arrow_height * np.sin(self.pitch)
        
        # Create directional arrow from center to nose direction
        self.aircraft_triangle = patches.FancyArrowPatch(
            (aircraft_x, aircraft_y),
            (arrow_end_x, arrow_end_y),
            arrowstyle='->',
            mutation_scale=20,
            color='red',
            linewidth=3,
            alpha=0.8,
            zorder=10
        )
        self.ax.add_patch(self.aircraft_triangle)
        
        # KEEP center dot DISABLED - this was the culprit causing the distorted shape
        # The Circle patch was getting stretched due to coordinate scaling differences
        
        # Update flight path trail
        if len(self.position_history) > 1:
            times = [pos[0] for pos in self.position_history]
            altitudes = [pos[1] for pos in self.position_history]
            self.trail_line.set_data(times, altitudes)
        
        # RESTORE pitch vector line - this was NOT the culprit
        # Update pitch vector (extends from aircraft arrow tip)
        nose_x = arrow_end_x
        nose_y = arrow_end_y
        
        # Extend vector from arrow tip
        vector_length = 150  # feet
        vector_time_scale = 1.5  # seconds (keep vector visible on time axis)
        pitch_end_x = nose_x + vector_time_scale * np.cos(self.pitch)
        pitch_end_y = nose_y + vector_length * np.sin(self.pitch)
        self.pitch_vector.set_data([nose_x, pitch_end_x], [nose_y, pitch_end_y])
        
        # Update telemetry display - compact format
        telemetry_info = (
            f"Time: {current_time:.1f}s\n"
            f"Alt: {self.altitude:.0f} ft\n"
            f"Speed: {self.airspeed:.0f} kts\n"
            f"Target: {self.target_airspeed:.0f} kts\n"
            f"Error: {abs(self.airspeed - self.target_airspeed):.0f} kts\n"
            f"Pitch: {np.degrees(self.pitch):+.0f}°\n"
            f"Rate: {np.degrees(self.pitch_rate):+.0f}°/s\n"
            f"V/S: {self.vertical_speed:+.0f} ft/s\n"
            f"Fuel: {self.fuel:.0%}\n"
            f"Throttle: {self.throttle:.0%}\n"
            f"Elevator: {self.elevator:+.1f}"
        )
        self.telemetry_text.set_text(telemetry_info)
        
        # Update energy display
        kinetic_energy = 0.5 * self.aircraft_mass * (self.airspeed * 1.68781)**2
        potential_energy = self.aircraft_mass * self.g * self.altitude
        total_energy = kinetic_energy + potential_energy
        
        energy_info = (
            f"ENERGY STATE\n"
            f"Kinetic: {kinetic_energy/1e6:.0f}M ft-lbf\n"
            f"Potential: {potential_energy/1e6:.0f}M ft-lbf\n"
            f"Total: {total_energy/1e6:.0f}M ft-lbf\n"
            f"Efficiency: {self.fuel:.0%}"
        )
        self.energy_text.set_text(energy_info)
        
        # Auto-scale x-axis to follow aircraft
        if current_time > 50:
            self.ax.set_xlim(current_time - 50, current_time + 10)
        
        # Auto-scale y-axis if needed
        if self.altitude > self.max_altitude * 0.8:
            self.ax.set_ylim(0, self.altitude + 2000)
        
        # Status indicators
        if self.terminated:
            if self.altitude <= 0:
                self.ax.text(0.5, 0.5, 'CRASHED', transform=self.ax.transAxes,
                           fontsize=20, color='red', weight='bold', ha='center')
            elif self.fuel <= 0:
                self.ax.text(0.5, 0.5, 'FUEL DEPLETED', transform=self.ax.transAxes,
                           fontsize=20, color='orange', weight='bold', ha='center')
        
        plt.draw()
        if self.render_mode != "rgb_array":
            plt.pause(0.01)  # Small pause for animation (only for human viewing)
    
    def _render_rgb_array(self):
        """Render to RGB array for video recording"""
        # Force recreation of the visualization for video recording
        self.visualization_initialized = False
        self._initialize_visualization()
        
        # Force a complete redraw of the matplotlib figure
        self._render_matplotlib()
        
        # Ensure the figure is fully updated
        self.fig.canvas.draw()
        
        # Get the buffer immediately after drawing
        try:
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        except AttributeError:
            # For newer matplotlib versions
            buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, :3]  # Remove alpha channel
            return buf
        
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return buf
    
    def _initialize_video_recording(self):
        """Initialize video recording for a new episode"""
        import os
        from datetime import datetime
        
        # Create video directory if it doesn't exist
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Generate session ID if not already set
        if self.session_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_id = f"2d_flight_{timestamp}"
        
        # Reset frame buffer for new episode
        self.video_frames = []
        self.episode_count += 1
        
        print(f"🎬 Starting video recording for episode {self.episode_count}")
    
    def _save_video_frame(self):
        """Save current frame to video buffer"""
        if self.render_mode == "rgb_array":
            frame = self.render()
            if frame is not None:
                self.video_frames.append(frame)
    
    def _save_video_file(self, episode_info: str = ""):
        """Save recorded frames as video file"""
        if not self.video_frames:
            return
        
        import os
        import cv2
        from datetime import datetime
        
        # Create filename with episode info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{self.episode_count:03d}_{timestamp}"
        if episode_info:
            filename += f"_{episode_info}"
        filename += ".avi"
        
        filepath = os.path.join(self.video_dir, filename)
        
        # Get frame dimensions
        height, width = self.video_frames[0].shape[:2]
        
        # Create video writer with higher frame rate and different codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible codec
        out = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))  # Higher frame rate
        
        # Write frames
        for frame in self.video_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"🎬 Video saved: {filepath}")
        print(f"   Frames: {len(self.video_frames)}, Duration: {len(self.video_frames)/30.0:.1f}s")
    
    def save_noteworthy_episode(self, episode_type: str = "noteworthy", episode_info: str = ""):
        """Save current episode as a noteworthy video"""
        if self.record_video and self.video_frames:
            # Combine episode type and info for filename
            info = f"{episode_type}"
            if episode_info:
                info += f"_{episode_info}"
            
            self._save_video_file(info)
            return True
        return False
    
    def close(self):
        """Close the environment and clean up matplotlib resources"""
        if self.visualization_initialized and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.visualization_initialized = False 
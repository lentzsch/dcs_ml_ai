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
        render_mode: Optional[str] = None
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
        
        # Initialize aircraft marker (triangle pointing in pitch direction)
        self.aircraft_marker = self.ax.scatter([], [], s=200, c='red', marker='^', 
                                             edgecolor='black', linewidth=2, zorder=10)
        
        # Initialize trail line
        self.trail_line, = self.ax.plot([], [], 'b-', alpha=0.6, linewidth=2, label='Flight Path')
        
        # Initialize pitch vector line
        self.pitch_vector, = self.ax.plot([], [], 'r-', linewidth=3, alpha=0.8, label='Pitch Vector')
        
        # Initialize target altitude line
        target_alt = 5000  # Reference altitude
        self.ax.axhline(y=target_alt, color='green', linestyle='--', alpha=0.7, 
                       label=f'Reference Alt ({target_alt} ft)')
        
        # Add legend
        self.ax.legend(loc='upper left', fontsize=10)
        
        # Initialize text displays for telemetry
        self.telemetry_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                         fontsize=11, verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.energy_text = self.ax.text(0.02, 0.75, '', transform=self.ax.transAxes,
                                      fontsize=10, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
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
        
        # Update aircraft marker position
        self.aircraft_marker.set_offsets([[aircraft_x, aircraft_y]])
        
        # Update flight path trail
        if len(self.position_history) > 1:
            times = [pos[0] for pos in self.position_history]
            altitudes = [pos[1] for pos in self.position_history]
            self.trail_line.set_data(times, altitudes)
        
        # Update pitch vector (shows aircraft orientation)
        vector_length = 500  # feet
        pitch_end_x = aircraft_x + vector_length * np.cos(self.pitch) / 100  # Scale for visibility
        pitch_end_y = aircraft_y + vector_length * np.sin(self.pitch)
        self.pitch_vector.set_data([aircraft_x, pitch_end_x], [aircraft_y, pitch_end_y])
        
        # Update telemetry display
        telemetry_info = (
            f"Time: {current_time:.1f}s\n"
            f"Altitude: {self.altitude:.0f} ft\n"
            f"Airspeed: {self.airspeed:.1f} kts\n"
            f"Target: {self.target_airspeed:.0f} kts\n"
            f"Error: {abs(self.airspeed - self.target_airspeed):.1f} kts\n"
            f"Pitch: {np.degrees(self.pitch):+.1f}°\n"
            f"Pitch Rate: {np.degrees(self.pitch_rate):+.1f}°/s\n"
            f"Vertical Speed: {self.vertical_speed:+.0f} ft/s\n"
            f"Fuel: {self.fuel:.1%}\n"
            f"Throttle: {self.throttle:.1%}\n"
            f"Elevator: {self.elevator:+.2f}"
        )
        self.telemetry_text.set_text(telemetry_info)
        
        # Update energy display
        kinetic_energy = 0.5 * self.aircraft_mass * (self.airspeed * 1.68781)**2
        potential_energy = self.aircraft_mass * self.g * self.altitude
        total_energy = kinetic_energy + potential_energy
        
        energy_info = (
            f"ENERGY STATE\n"
            f"Kinetic: {kinetic_energy/1e6:.1f}M ft-lbf\n"
            f"Potential: {potential_energy/1e6:.1f}M ft-lbf\n"
            f"Total: {total_energy/1e6:.1f}M ft-lbf\n"
            f"Efficiency: {self.fuel:.1%}"
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
        plt.pause(0.01)  # Small pause for animation
    
    def _render_rgb_array(self):
        """Render to RGB array for video recording"""
        if not self.visualization_initialized:
            self._initialize_visualization()
        
        self._render_matplotlib()
        
        # Convert plot to RGB array
        self.fig.canvas.draw()
        # Use the correct method for modern matplotlib
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
    
    def close(self):
        """Close the environment and clean up matplotlib resources"""
        if self.visualization_initialized and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.visualization_initialized = False 
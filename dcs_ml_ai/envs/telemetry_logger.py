"""
Telemetry Logger for Flight Environment

This module provides telemetry logging capabilities for the 2D Flight Environment,
with future support for TacView-compatible format exports.

Features:
- CSV logging for immediate analysis
- Structured data format for future ACMI export
- Optional logging (disabled by default for performance)
- Timestamped flight data recording
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np


class TelemetryLogger:
    """
    Logs flight telemetry data in structured format for analysis and replay.
    
    Supports CSV output immediately and prepares structure for future 
    TacView ACMI format export.
    """
    
    def __init__(
        self, 
        enabled: bool = False,
        log_dir: str = "telemetry",
        session_id: Optional[str] = None,
        aircraft_id: str = "AI_PILOT_001"
    ):
        """
        Initialize telemetry logger.
        
        Args:
            enabled: Whether to enable logging (disabled by default for performance)
            log_dir: Directory to save telemetry files
            session_id: Unique session identifier (auto-generated if None)
            aircraft_id: Aircraft identifier for multi-aircraft scenarios
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.aircraft_id = aircraft_id
        
        # Data storage
        self.telemetry_data: List[Dict[str, Any]] = []
        self.csv_file_path: Optional[Path] = None
        self.csv_writer: Optional[csv.DictWriter] = None
        self.csv_file_handle = None
        
        # Initialize if enabled
        if self.enabled:
            self._initialize_logging()
    
    def _initialize_logging(self):
        """Initialize logging infrastructure"""
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up CSV file
        csv_filename = f"flight_telemetry_{self.session_id}.csv"
        self.csv_file_path = self.log_dir / csv_filename
        
        # Open CSV file and set up writer
        self.csv_file_handle = open(self.csv_file_path, 'w', newline='')
        
        # Define CSV columns (TacView-compatible structure)
        self.csv_columns = [
            'timestamp',           # Time in seconds
            'aircraft_id',         # Aircraft identifier
            'longitude',           # Longitude (placeholder for 2D)
            'latitude',            # Latitude (placeholder for 2D) 
            'altitude',            # Altitude in feet
            'pitch',               # Pitch angle in degrees
            'roll',                # Roll angle in degrees (0 for 2D)
            'heading',             # Heading in degrees (placeholder for 2D)
            'airspeed',            # Airspeed in knots
            'vertical_speed',      # Vertical speed in feet/second
            'throttle',            # Throttle setting (0-1)
            'elevator',            # Elevator deflection (-1 to 1)
            'fuel_remaining',      # Fuel remaining (0-1)
            'pitch_rate',          # Pitch rate in degrees/second
            'target_airspeed',     # Target airspeed in knots
            'airspeed_error',      # Airspeed error in knots
            'kinetic_energy',      # Kinetic energy in ft-lbf
            'potential_energy',    # Potential energy in ft-lbf
            'total_energy',        # Total energy in ft-lbf
            'terminated',          # Episode termination flag
            'crashed'              # Crash detection flag
        ]
        
        self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=self.csv_columns)
        self.csv_writer.writeheader()
        
        print(f"📊 Telemetry logging initialized: {self.csv_file_path}")
    
    def log_step(self, env_state: Dict[str, Any], simulation_time: float):
        """
        Log a single simulation step.
        
        Args:
            env_state: Dictionary containing environment state variables
            simulation_time: Current simulation time in seconds
        """
        if not self.enabled:
            return
        
        # Extract state variables with defaults
        altitude = env_state.get('altitude', 0.0)
        pitch = env_state.get('pitch', 0.0)
        airspeed = env_state.get('airspeed', 0.0)
        vertical_speed = env_state.get('vertical_speed', 0.0)
        throttle = env_state.get('throttle', 0.0)
        elevator = env_state.get('elevator', 0.0)
        fuel = env_state.get('fuel', 1.0)
        pitch_rate = env_state.get('pitch_rate', 0.0)
        target_airspeed = env_state.get('target_airspeed', 120.0)
        terminated = env_state.get('terminated', False)
        
        # Calculate derived values
        airspeed_error = abs(airspeed - target_airspeed)
        
        # Energy calculations (if mass provided)
        aircraft_mass = env_state.get('aircraft_mass', 15000.0)  # lbm
        g = env_state.get('g', 32.174)  # ft/s²
        airspeed_fps = airspeed * 1.68781  # knots to ft/s
        
        kinetic_energy = 0.5 * aircraft_mass * (airspeed_fps ** 2)
        potential_energy = aircraft_mass * g * altitude
        total_energy = kinetic_energy + potential_energy
        
        # Crash detection
        crashed = terminated and altitude <= 0
        
        # Create telemetry record
        telemetry_record = {
            'timestamp': simulation_time,
            'aircraft_id': self.aircraft_id,
            'longitude': 0.0,  # Placeholder for 2D environment
            'latitude': simulation_time / 100.0,  # Use time as proxy x-coordinate
            'altitude': altitude,
            'pitch': np.degrees(pitch),
            'roll': 0.0,  # No roll in 2D
            'heading': 90.0,  # Placeholder heading
            'airspeed': airspeed,
            'vertical_speed': vertical_speed,
            'throttle': throttle,
            'elevator': elevator,
            'fuel_remaining': fuel,
            'pitch_rate': np.degrees(pitch_rate),
            'target_airspeed': target_airspeed,
            'airspeed_error': airspeed_error,
            'kinetic_energy': kinetic_energy,
            'potential_energy': potential_energy,
            'total_energy': total_energy,
            'terminated': terminated,
            'crashed': crashed
        }
        
        # Store in memory
        self.telemetry_data.append(telemetry_record)
        
        # Write to CSV immediately for real-time access
        if self.csv_writer:
            self.csv_writer.writerow(telemetry_record)
            self.csv_file_handle.flush()  # Ensure data is written
    
    def export_tacview_acmi(self, output_path: Optional[str] = None) -> str:
        """
        Export telemetry data to TacView ACMI format (future implementation).
        
        Args:
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path to exported ACMI file
            
        Note: This is a stub for future TacView integration.
        """
        if not self.enabled or not self.telemetry_data:
            raise ValueError("No telemetry data available for export")
        
        if output_path is None:
            acmi_filename = f"flight_replay_{self.session_id}.acmi"
            output_path = str(self.log_dir / acmi_filename)
        
        # TODO: Implement actual ACMI format export
        # ACMI format specification: https://www.tacview.net/documentation/acmi/en/
        
        # For now, create a placeholder ACMI file with basic structure
        with open(output_path, 'w') as acmi_file:
            # ACMI header
            acmi_file.write("FileType=text/acmi/tacview\n")
            acmi_file.write("FileVersion=2.1\n")
            acmi_file.write(f"# Generated by DCS ML AI - Session {self.session_id}\n")
            acmi_file.write(f"# Aircraft: {self.aircraft_id}\n")
            acmi_file.write("# Format: 2D Flight Environment Telemetry\n")
            acmi_file.write("\n")
            
            # Object declaration
            acmi_file.write("0,ReferenceTime=2024-01-01T00:00:00Z\n")
            acmi_file.write(f"0,1000000,T=0|0|{self.telemetry_data[0]['altitude']}|0|0|{self.telemetry_data[0]['pitch']},Name={self.aircraft_id},Type=Air+FixedWing\n")
            
            # Data records
            for record in self.telemetry_data:
                timestamp = record['timestamp']
                longitude = record['longitude']
                latitude = record['latitude'] 
                altitude = record['altitude']
                pitch = record['pitch']
                
                # Simple ACMI record format
                acmi_file.write(f"{timestamp:.2f},1000000,T={longitude}|{latitude}|{altitude}|0|0|{pitch}\n")
        
        print(f"📄 ACMI export created: {output_path}")
        print("⚠️  Note: This is a basic ACMI stub. Full TacView compatibility requires future enhancement.")
        
        return output_path
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the flight session.
        
        Returns:
            Dictionary containing flight statistics
        """
        if not self.telemetry_data:
            return {}
        
        # Extract key metrics
        altitudes = [record['altitude'] for record in self.telemetry_data]
        airspeeds = [record['airspeed'] for record in self.telemetry_data]
        fuel_levels = [record['fuel_remaining'] for record in self.telemetry_data]
        airspeed_errors = [record['airspeed_error'] for record in self.telemetry_data]
        
        # Calculate statistics
        stats = {
            'session_id': self.session_id,
            'aircraft_id': self.aircraft_id,
            'total_duration': self.telemetry_data[-1]['timestamp'],
            'total_records': len(self.telemetry_data),
            'altitude_stats': {
                'min': min(altitudes),
                'max': max(altitudes),
                'mean': np.mean(altitudes),
                'std': np.std(altitudes)
            },
            'airspeed_stats': {
                'min': min(airspeeds),
                'max': max(airspeeds),
                'mean': np.mean(airspeeds),
                'std': np.std(airspeeds)
            },
            'fuel_consumed': fuel_levels[0] - fuel_levels[-1],
            'average_airspeed_error': np.mean(airspeed_errors),
            'crashed': any(record['crashed'] for record in self.telemetry_data),
            'completed_successfully': not any(record['terminated'] for record in self.telemetry_data[:-1])
        }
        
        return stats
    
    def close(self):
        """Close telemetry logging and clean up resources"""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            self.csv_file_handle = None
        
        if self.enabled and self.telemetry_data:
            # Print summary
            stats = self.get_summary_stats()
            print(f"\n📊 Telemetry Session Summary:")
            print(f"   Session ID: {stats['session_id']}")
            print(f"   Duration: {stats['total_duration']:.1f}s")
            print(f"   Records: {stats['total_records']}")
            print(f"   Altitude Range: {stats['altitude_stats']['min']:.0f} - {stats['altitude_stats']['max']:.0f} ft")
            print(f"   Avg Airspeed Error: {stats['average_airspeed_error']:.1f} kts")
            print(f"   Fuel Consumed: {stats['fuel_consumed']:.1%}")
            print(f"   Crashed: {'Yes' if stats['crashed'] else 'No'}")
            print(f"   CSV File: {self.csv_file_path}")


# Example usage and testing
if __name__ == "__main__":
    # Demonstration of telemetry logger
    logger = TelemetryLogger(enabled=True, session_id="test_session")
    
    # Simulate some flight data
    for step in range(10):
        sim_time = step * 0.1
        env_state = {
            'altitude': 5000 + step * 10,
            'pitch': np.radians(step * 2),
            'airspeed': 120 + step,
            'vertical_speed': step * 5,
            'throttle': 0.5 + step * 0.05,
            'elevator': step * 0.1,
            'fuel': 1.0 - step * 0.05,
            'pitch_rate': np.radians(1.0),
            'target_airspeed': 120.0,
            'terminated': False
        }
        
        logger.log_step(env_state, sim_time)
    
    # Export and close
    acmi_path = logger.export_tacview_acmi()
    logger.close()
    
    print("\n✅ Telemetry logger test completed!")
    print(f"Files created:")
    print(f"  - CSV: {logger.csv_file_path}")
    print(f"  - ACMI: {acmi_path}")
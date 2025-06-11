# scripts/save_last_logs.py

import os
import shutil
from datetime import datetime
from pathlib import Path

def save_last_logs(log_dir: str, env_name: str):
    """
    Moves the contents of the current training session's log folder into a
    timestamped session folder inside tensorboard_logs/saved/{env_name}/.
    
    Args:
        log_dir (str): Path to the current log directory
        env_name (str): Name of the environment (e.g., 'cartpole', 'lunarlandercontinuous')
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path("tensorboard_logs") / "saved" / env_name / timestamp

    save_dir.mkdir(parents=True, exist_ok=True)

    # Get all log files and directories
    log_items = [
        f for f in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, f))  # Only get directories (PPO_* folders)
    ]

    for item in log_items:
        src = os.path.join(log_dir, item)
        dst = os.path.join(save_dir, item)
        shutil.move(src, dst)

    print(f"Saved session logs to {save_dir}")

if __name__ == "__main__":
    # Example usage
    save_last_logs("tensorboard_logs/cartpole", "cartpole") 
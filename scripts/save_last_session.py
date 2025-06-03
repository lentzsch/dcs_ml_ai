# scripts/save_last_session.py

import os
import shutil
from datetime import datetime

def save_last_session(video_folder: str, env_name: str):
    """
    Moves the contents of the current training session's video folder into a
    timestamped session folder inside videos/saved/{env_name}/.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("videos", "saved", env_name, timestamp)

    os.makedirs(save_dir, exist_ok=True)

    video_files = [
        f for f in os.listdir(video_folder)
        if os.path.isfile(os.path.join(video_folder, f))
    ]

    for f in video_files:
        src = os.path.join(video_folder, f)
        dst = os.path.join(save_dir, f)
        shutil.move(src, dst)  # ✅ This moves instead of copying

    print(f"Saved session videos to {save_dir}")


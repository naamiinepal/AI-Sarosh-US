import os
import re
from pathlib import Path
import os
import re
from pathlib import Path

def get_best_checkpoint(log_dir):
    log_dir = Path(log_dir)

    # Search recursively for any 'checkpoints' folders
    checkpoint_dirs = sorted(log_dir.rglob("checkpoints"), key=os.path.getmtime, reverse=True)

    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoint folders found.")

    latest_checkpoint_dir = checkpoint_dirs[0]

    best_ckpt = None
    best_score = float("-inf")

    # Update regex pattern if your filenames look different
    ckpt_pattern = re.compile(r"epoch(\d+)-val_f1([0-9.]+)\.ckpt")

    for ckpt_file in latest_checkpoint_dir.glob("*.ckpt"):
        match = ckpt_pattern.match(ckpt_file.name)
        if match:
            val_f1 = float(match.group(2))
            if val_f1 > best_score:
                best_score = val_f1
                best_ckpt = ckpt_file

    if best_ckpt is None:
        raise ValueError("No checkpoint matching expected pattern found.")

    return best_ckpt


def get_latest_run_dir(parent_dir):
    parent_dir = os.path.abspath(parent_dir)
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    latest_dir = max(subdirs, key=os.path.getmtime)
    
    return latest_dir
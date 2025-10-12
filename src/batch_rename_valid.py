import os
import shutil
from pathlib import Path

# Source folder (where _test files currently are)
src_dir = Path("data/labeled/asl_alphabet_test")

# Destination base (val folder)
dest_base = Path("data/labeled/val")

moved = 0
for img_path in src_dir.glob("*_test.jpg"):
    label = img_path.stem.split("_")[0]  # extract 'A' from 'A_test'
    new_name = f"{label}00.jpg"
    dest_dir = dest_base / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / new_name
    shutil.move(str(img_path), dest_path)
    moved += 1

print(f"Moved and renamed {moved} files into {dest_base}/<label>/")

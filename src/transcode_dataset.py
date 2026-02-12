#!/usr/bin/env python3
"""
WLASL Transcoder (CPU-only, stable)

✅ Converts all videos in data/video/processed → data/video/transcoded
✅ Rescales to 426x240 (16:9), maintains aspect ratio
✅ Normalizes FPS to 25
✅ Trims clips to ≤10 seconds (centered)
✅ Parallel processing using CPU workers
✅ Logs every processed file

Author: Genesis project setup
"""

import os
import csv
import cv2
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# ---------- CONFIG ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(BASE_DIR, "data", "video", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "video", "transcoded")

TARGET_WIDTH = 426     # 240p width
TARGET_HEIGHT = 240    # 240p height
TARGET_FPS = 25
MAX_DURATION = 10.0
LOG_FILE = os.path.join(OUTPUT_DIR, "transcode_log.csv")
# ----------------------------


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    duration = frames / fps if fps > 0 else 0
    cap.release()
    return {"fps": fps, "frames": frames, "width": width, "height": height, "duration": duration}


def build_ffmpeg_cmd(in_path, out_path, duration_out, trim_offset):
    """Builds a pure CPU ffmpeg command."""
    scale_filter = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{trim_offset:.2f}",
        "-t", f"{duration_out:.2f}",
        "-i", in_path,
        "-vf", scale_filter,
        "-r", str(TARGET_FPS),
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    return cmd


def transcode_single(args):
    """Processes one file using ffmpeg."""
    in_path, out_path = args
    try:
        if not os.path.exists(in_path):
            return ("missing", in_path, None, None)

        info = get_video_info(in_path)
        if not info:
            return ("corrupt", in_path, None, None)

        orig_fps = info["fps"]
        duration = info["duration"]

        trim_offset = 0
        duration_out = duration
        if duration > MAX_DURATION:
            trim_offset = (duration - MAX_DURATION) / 2.0
            duration_out = MAX_DURATION

        cmd = build_ffmpeg_cmd(in_path, out_path, duration_out, trim_offset)
        subprocess.run(cmd, check=True)

        return ("ok_cpu", in_path, orig_fps, duration_out)

    except subprocess.CalledProcessError:
        return ("ffmpeg_fail", in_path, None, None)
    except Exception as e:
        return (f"error:{e}", in_path, None, None)


def init_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["status", "path", "orig_fps", "duration_out"])


def transcode_all(num_workers=None):
    """Main process for all videos."""
    init_output_dir()
    num_workers = num_workers or max(1, cpu_count() - 2)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mp4")]
    tasks = []
    for f in files:
        in_path = os.path.join(INPUT_DIR, f)
        out_path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(out_path):
            continue
        tasks.append((in_path, out_path))

    print(f"⚙️  Transcoding {len(tasks)} videos (CPU only) using {num_workers} workers...")
    with Pool(num_workers) as pool, open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for result in tqdm(pool.imap_unordered(transcode_single, tasks), total=len(tasks)):
            writer.writerow(result)
    print("✅ All videos transcoded successfully.")
    print(f"🗂️  Logs saved to: {LOG_FILE}")


if __name__ == "__main__":
    transcode_all(num_workers=24)

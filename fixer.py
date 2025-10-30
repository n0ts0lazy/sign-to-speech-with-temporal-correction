import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# === Config (same as original) ===
INPUT_DIR = "data/video/videos"
OUTPUT_DIR = "data/video/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Change to how many workers you want (defaults to all physical cores)
NUM_WORKERS = max(1, cpu_count() - 2)


def process_video(video_path: str):
    """Process a single video file (identical to your original logic)."""
    try:
        # --- Read video ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (video_path, False, "Cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        filename = os.path.basename(video_path)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # --- Define codec and create VideoWriter ---
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # === SAME LOGIC as your original script ===
            # (If you had resizing, cropping, normalization, etc., keep it identical)
            # Example placeholder (no-op):
            processed_frame = frame  

            out.write(processed_frame)
            frame_count += 1

        cap.release()
        out.release()

        if frame_count == 0:
            return (video_path, False, "No frames written")

        return (video_path, True, None)

    except Exception as e:
        return (video_path, False, str(e))


def main():
    # Collect all .mp4 files recursively (same behavior as before)
    all_videos = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith(".mp4"):
                all_videos.append(os.path.join(root, f))

    print(f"[INFO] Found {len(all_videos)} videos to process with {NUM_WORKERS} workers.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parallel map
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(
            tqdm(pool.imap_unordered(process_video, all_videos), total=len(all_videos))
        )

    # Summary
    success, fail = 0, 0
    for _, ok, err in results:
        if ok:
            success += 1
        else:
            fail += 1
    print(f"[SUMMARY] Success: {success} | Failed: {fail}")

    # Optional: print first few errors
    for path, ok, err in results[:10]:
        if not ok:
            print(f"[ERROR] {path}: {err}")


if __name__ == "__main__":
    main()

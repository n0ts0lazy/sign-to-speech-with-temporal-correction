import os
import json
import random
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def safe_copy_or_link(src, dst, move=False):
    """Copy, move, or hard-link a file safely."""
    if os.path.exists(dst):
        return True
    try:
        if move:
            shutil.move(src, dst)
        else:
            try:
                os.link(src, dst)  # fast hard link
            except OSError:
                shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"[WARN] Could not copy/move {src}: {e}")
        return False


def force_video_split_with_metadata(
    root_dir="data/video/videos",
    output_root="data/video",
    metadata_path="data/video/WLASL_v0.3.json",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    move=False,
    seed=42,
    num_workers=16
):
    """
    Create train/val/test folders AND update the WLASL metadata JSON to match the new split.
    """

    random.seed(seed)

    # --- Load existing metadata ---
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"[ERROR] Metadata not found: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # --- Gather all video files ---
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"[ERROR] Input video folder not found: {root_dir}")

    video_files = [f for f in os.listdir(root_dir) if f.endswith(".mp4")]
    total = len(video_files)
    if total == 0:
        raise RuntimeError(f"[ERROR] No .mp4 files found in {root_dir}")

    random.shuffle(video_files)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_files = set(video_files[:train_end])
    val_files = set(video_files[train_end:val_end])
    test_files = set(video_files[val_end:])

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    # --- Create output dirs ---
    for split in splits:
        os.makedirs(os.path.join(output_root, split), exist_ok=True)

    print(f"[INFO] Processing {total} videos across splits...")
    print(f"→ train: {len(train_files)} | val: {len(val_files)} | test: {len(test_files)}")
    print(f"→ using {num_workers} threads ({'move' if move else 'copy/hardlink'}) mode")

    # --- Parallel copy/move ---
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for split, files in splits.items():
            split_dir = os.path.join(output_root, split)
            for fname in files:
                src = os.path.join(root_dir, fname)
                dst = os.path.join(split_dir, fname)
                futures.append(executor.submit(safe_copy_or_link, src, dst, move))
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying files", unit="file"):
            pass

    # --- Update JSON split info ---
    print("[INFO] Updating metadata splits...")
    updated_entries = 0
    video_id_to_split = {}
    for split_name, file_set in splits.items():
        for fname in file_set:
            vid = os.path.splitext(fname)[0]
            video_id_to_split[vid] = split_name

    for entry in metadata:
        for inst in entry.get("instances", []):
            vid = inst.get("video_id")
            if vid in video_id_to_split:
                inst["split"] = video_id_to_split[vid]
                updated_entries += 1

    # --- Save updated metadata ---
    updated_path = os.path.join(os.path.dirname(metadata_path), "WLASL_v0.3_updated.json")
    with open(updated_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # --- Save summary JSON ---
    summary = {
        "total_videos": total,
        "train_count": len(train_files),
        "val_count": len(val_files),
        "test_count": len(test_files),
        "updated_instances": updated_entries,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "mode": "move" if move else "copy",
        "workers": num_workers
    }
    summary_path = os.path.join(output_root, "video_split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[INFO] Split complete — {updated_entries} metadata entries updated.")
    print(f"Updated metadata saved as: {updated_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    force_video_split_with_metadata(
        root_dir="data/video/videos",
        output_root="data/video",
        metadata_path="data/video/WLASL_v0.3.json",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        move=False,
        seed=42,
        num_workers=20
    )

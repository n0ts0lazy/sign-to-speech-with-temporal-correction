import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

_USE_DECORD = False
try:
    from decord import VideoReader, cpu as decord_cpu
    _USE_DECORD = True
except ImportError:
    pass


class WLASLVideoDataset(Dataset):
    """
    Loader for WLASL videos with automatic split detection.
    Works when you have folder-split data (train/val/test) AND
    a WLASL-style JSON file containing 'instances' lists.
    """
    def __init__(self, root_dir, metadata_path, split="val", frame_tfms=None, frames_per_clip=16):
        self.root_dir = root_dir
        self.split = split
        self.frame_tfms = frame_tfms
        self.frames_per_clip = frames_per_clip
        self.samples = []
        self.missing = 0

        # --- Load metadata ---
        with open(metadata_path, "r") as f:
            meta = json.load(f)

        # JSON can be list (WLASL) or dict with splits
        if isinstance(meta, dict) and split in meta:
            entries = meta[split]
            print(f"[INFO] Using split key '{split}' with {len(entries)} entries.")
        elif isinstance(meta, list):
            entries = meta
            print(f"[INFO] Using flat WLASL-style JSON with {len(entries)} gloss entries.")
        else:
            raise ValueError(f"Unrecognized metadata format for split '{split}'.")

        # Determine base directory for videos
        split_dir = os.path.join(root_dir, split)
        base_dir = split_dir if os.path.isdir(split_dir) else root_dir
        print(f"[INFO] Using base directory: {base_dir}")

        # --- Build samples from entries ---
        for entry in entries:
            # WLASL format: each gloss has multiple instances
            if "instances" in entry:
                label = entry.get("gloss") or entry.get("label") or 0
                for inst in entry["instances"]:
                    vid = inst.get("video_id")
                    inst_split = inst.get("split", split)

                    # Match only chosen split
                    if self.split and inst_split != self.split:
                        continue

                    filename = f"{vid}.mp4" if vid else None
                    if not filename:
                        self.missing += 1
                        continue

                    full_path = os.path.join(root_dir, inst_split, filename)

                    if os.path.exists(full_path):
                        self.samples.append({"path": full_path, "label": label})
                    else:
                        self.missing += 1
            else:
                # Fallback for simple JSON entries
                label = entry.get("label", 0)
                vid = entry.get("video_id") or entry.get("id")
                if not vid:
                    self.missing += 1
                    continue

                filename = f"{vid}.mp4"
                full_path = os.path.join(base_dir, filename)
                if os.path.exists(full_path):
                    self.samples.append({"path": full_path, "label": label})
                else:
                    self.missing += 1

        self.classes = sorted({s["label"] for s in self.samples})
        print(f"[INFO] Loaded {len(self.samples)} valid videos (skipped {self.missing}). Classes: {len(self.classes)}")

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, total_frames):
        """Uniformly sample frame indices."""
        if total_frames < self.frames_per_clip:
            reps = (self.frames_per_clip // max(1, total_frames)) + 1
            idx = (list(range(total_frames)) * reps)[:self.frames_per_clip]
            return idx
        step = total_frames / self.frames_per_clip
        return [int(i * step) for i in range(self.frames_per_clip)]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path, label = sample["path"], sample["label"]

        # --- Read video ---
        try:
            if _USE_DECORD:
                vr = VideoReader(video_path, ctx=decord_cpu())
                frames = [torch.from_numpy(f.asnumpy()) for f in vr]
                vid = torch.stack(frames).permute(0, 3, 1, 2).float() / 255.0
            else:
                vid, _, _ = read_video(video_path, pts_unit="sec")
                vid = vid.permute(0, 3, 1, 2).float() / 255.0
        except Exception as e:
            print(f"[WARN] Could not read video: {video_path} ({e})")
            return torch.zeros((self.frames_per_clip, 3, 112, 112)), 0

        inds = self._sample_indices(vid.shape[0])
        clip = vid[inds]

        if self.frame_tfms:
            clip = torch.stack([self.frame_tfms(f) for f in clip], dim=0)

        return clip, label

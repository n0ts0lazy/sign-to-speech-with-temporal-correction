import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# =====================================================
# Configuration
# =====================================================
MAX_FRAMES = 16  # can be 8 or 16 depending on your model
RESIZE_HW = (224, 224)  # height, width for MViT
# =====================================================


class WLASLDataset(Dataset):
    def __init__(self, json_path, video_root, split="train", transform=None):
        self.video_root = video_root
        self.transform = transform

        # Load JSON metadata
        with open(json_path, "r") as f:
            all_data = json.load(f)

        self.samples = []
        for gloss_entry in all_data:
            gloss = gloss_entry.get("gloss")
            for inst in gloss_entry.get("instances", []):
                if inst.get("split") == split:
                    vid_id = inst.get("video_id")
                    video_path = os.path.join(video_root, f"{vid_id}.mp4")
                    if os.path.exists(video_path):
                        self.samples.append((video_path, gloss))

        print(f"[WLASL] Split={split} | Videos={len(self.samples)} | Classes≈{len(all_data)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.load_video_frames(video_path)
        return {"video": frames, "label": label}

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame to model expected size (224×224)
            frame = cv2.resize(frame, RESIZE_HW)

            # Convert to tensor [3, H, W]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor)

        cap.release()

        # Handle missing or empty video
        if len(frames) == 0:
            frames = [torch.zeros((3, *RESIZE_HW))]

        num_frames = len(frames)

        # Uniformly sample or pad to fixed length
        if num_frames > MAX_FRAMES:
            indices = torch.linspace(0, num_frames - 1, MAX_FRAMES).long()
            frames = [frames[i] for i in indices]
        elif num_frames < MAX_FRAMES:
            frames += [frames[-1]] * (MAX_FRAMES - num_frames)

        frames = torch.stack(frames)  # [T, 3, H, W]
        return frames


# =====================================================
# Collate Function
# =====================================================
def pad_collate_fn(batch):
    """Combine variable-length videos into a batch tensor"""
    videos = [item["video"] for item in batch]
    labels = [item["label"] for item in batch]

    # Stack along batch dimension
    videos = torch.stack(videos)  # [B, T, 3, H, W] (we’ll fix order below)
    videos = videos.permute(0, 2, 1, 3, 4)  # [B, 3, T, H, W]

    return videos, labels


# =====================================================
# Example Usage (only runs if file executed directly)
# =====================================================
if __name__ == "__main__":
    ds = WLASLDataset(
        json_path="../data/video/WLASL_rebuilt.json",
        video_root="../data/video/transcoded",
        split="train",
    )

    print("Dataset size:", len(ds))
    sample = ds[0]
    print("Single sample shape:", sample["video"].shape)

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)
    videos, labels = next(iter(dl))
    print("Batch shape:", videos.shape)

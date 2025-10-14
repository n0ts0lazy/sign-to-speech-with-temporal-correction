#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
video_model_defination.py

Train ViT (image-based) and MViT (video-based with temporal context windows)
on WLASL under data/video/.

- Default model: mvit (override with --model vit)
- Host-process only (num_workers=0) so it won't spawn subprocesses
- RAM guard: keeps process RSS under a user-set limit (default 58 GB)
- Progress bars, checkpoint naming, CSV logging aligned with train_labeled.py
"""

import os
import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm import tqdm

# ---- Optional RAM monitor ----
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

# Silence torchvision video deprecation spam
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.video")

# Try decord for stable video reading; fall back to torchvision if unavailable
_USE_DECORD = False
try:
    from decord import VideoReader, cpu as decord_cpu
    _USE_DECORD = True
except Exception:
    _USE_DECORD = False
    import torchvision
    from torchvision.io import read_video

import torchvision  # still needed for models and transforms
from torchvision import transforms

# ----------------------------
# Paths & defaults
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "video"
VIDEOS_SUBDIR = "videos"
MANIFEST_JSON = "WLASL_v0.3_updated.json"

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ----------------------------
# Util & seed
# ----------------------------
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_dirs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ---- checkpoint: mirror train_labeled.py style ----
def save_checkpoint(model: nn.Module, accuracy: float, model_name: str, out_dir: Optional[Path] = None):
    if out_dir is None:
        out_dir = CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}_{accuracy:.2f}_{nowstamp()}.pth"
    path = out_dir / filename
    torch.save(model.state_dict(), str(path))
    print(f"Saved checkpoint: {path}")
    return filename, str(path)  # returns name (for CSV) and full path

# ----------------------------
# Dataset
# ----------------------------
class VideoMeta:
    __slots__ = ("path", "label")
    def __init__(self, path: str, label: int):
        self.path = path
        self.label = label

class WLASLDataset(Dataset):
    """
    Supports:
    - Manifest: data_root/WLASL_v0.3.json with files in data_root/videos/<video_id>.<ext>
    - Dir tree: data_root/videos/<class_name>/*.mp4
    """
    def __init__(
        self,
        data_root: Path,
        frames_per_clip: int = 32,
        frame_step: int = 2,
        resize: int = 224,
        center_crop: Optional[int] = None,
        use_manifest: Optional[bool] = None,
    ):
        self.data_root = Path(data_root)
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step

        tfms = [transforms.Resize((resize, resize))]
        if center_crop:
            tfms.append(transforms.CenterCrop(center_crop))
        tfms += [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.frame_tfms = transforms.Compose(tfms)

        if use_manifest is None:
            use_manifest = (self.data_root / MANIFEST_JSON).exists()
        self.use_manifest = use_manifest

        if self.use_manifest:
            self.class_to_idx, self.samples = self._scan_manifest()
        else:
            self.class_to_idx, self.samples = self._scan_dir_tree()

        if len(self.samples) == 0:
            raise RuntimeError("No video samples found. Check data/video layout or manifest.")

    def _scan_dir_tree(self) -> Tuple[Dict[str, int], List[VideoMeta]]:
        videos_root = self.data_root / VIDEOS_SUBDIR
        classes = sorted([p.name for p in videos_root.iterdir() if p.is_dir()])
        class_to_idx = {c: i for i, c in enumerate(classes)}
        items: List[VideoMeta] = []
        for c in classes:
            for vp in (videos_root / c).rglob("*"):
                if vp.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                    items.append(VideoMeta(str(vp), class_to_idx[c]))
        return class_to_idx, items

    def _scan_manifest(self) -> Tuple[Dict[str, int], List[VideoMeta]]:
        with open(self.data_root / MANIFEST_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        glosses = sorted({d.get("gloss", f"cls_{i}") for i, d in enumerate(data)})
        class_to_idx = {g: i for i, g in enumerate(glosses)}

        videos_root = self.data_root / VIDEOS_SUBDIR
        items: List[VideoMeta] = []
        for d in data:
            g = d.get("gloss")
            if g not in class_to_idx:
                continue
            label = class_to_idx[g]
            for inst in d.get("instances", []):
                vid = inst.get("video_id")
                if not vid:
                    continue
                for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                    p = videos_root / f"{vid}{ext}"
                    if p.exists():
                        items.append(VideoMeta(str(p), label))
                        break
        return class_to_idx, items

    def _sample_indices(self, total_frames: int):
        # Uniform subsample using frame_step; pad if short
        needed = self.frames_per_clip * self.frame_step
        if total_frames >= needed:
            import random
            start = 0 if total_frames == needed else random.randint(0, total_frames - needed)
            inds = list(range(start, start + needed, self.frame_step))
        else:
            inds = list(range(0, total_frames, max(1, self.frame_step)))
            while len(inds) < self.frames_per_clip:
                inds.append(min(total_frames - 1, inds[-1] if inds else 0))
            inds = inds[: self.frames_per_clip]
        return inds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        meta = self.samples[idx]

        try:
            if _USE_DECORD:
                vr = VideoReader(meta.path, ctx=decord_cpu(0), num_threads=1)
                total = len(vr)
                if total == 0:
                    raise ValueError("Empty video stream")

                # Lazy decode while iterating
                frames = []
                for f in vr:
                    frames.append(torch.from_numpy(f.asnumpy()))

                vid = torch.stack(frames)  # (T, H, W, C)
                vid = vid.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

            else:
                # Torchvision fallback
                vid, _, _ = read_video(meta.path, pts_unit="sec")
                if vid.numel() == 0:
                    raise ValueError("Empty video stream (torchvision fallback)")
                vid = vid.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

            # Uniform frame sampling
            inds = self._sample_indices(vid.shape[0])
            clip = vid[inds]  # (T, C, H, W)

            # Apply transforms per-frame to save memory
            clip = torch.stack([self.frame_tfms(f) for f in clip], dim=0)  # (T, C, H, W)

            return clip, meta.label

        except Exception as e:
            print(f"[WARN] Skipping video {meta.path}: {e}")
            with open("skipped_videos.log", "a") as logf:
                logf.write(f"{meta.path} - {e}\n")

            return self.__getitem__((idx + 1) % len(self))

# ----------------------------
# Split helpers
# ----------------------------
def stratified_split(samples: List[VideoMeta], n_classes: int, val_ratio: float = 0.2, seed: int = 42):
    import random
    random.seed(seed)
    buckets = {c: [] for c in range(n_classes)}
    for i, m in enumerate(samples):
        buckets[m.label].append(i)
    train_idx, val_idx = [], []
    for c, lst in buckets.items():
        random.shuffle(lst)
        n_val = max(1, int(len(lst) * val_ratio))
        val_idx += lst[:n_val]
        train_idx += lst[n_val:]
    return train_idx, val_idx

# ----------------------------
# Models
# ----------------------------
def build_vit(n_classes: int) -> nn.Module:
    # Robust to different torchvision versions
    ctor = None
    for name in ["vit_b_16", "vit_b_32"]:
        if hasattr(torchvision.models, name):
            ctor = getattr(torchvision.models, name)
            break
    if ctor is None:
        raise RuntimeError("No ViT constructor found in torchvision.models")

    try:
        model = ctor(weights="IMAGENET1K_V1")
    except Exception:
        try:
            model = ctor(weights=None)
        except Exception:
            model = ctor(pretrained=False)

    # Replace head
    if hasattr(model, "heads") and isinstance(model.heads, nn.Sequential):
        # vit_b_* has model.heads.head
        in_f = getattr(model.heads, "head").in_features if hasattr(model.heads, "head") else list(model.heads.children())[-1].in_features
        model.heads = nn.Sequential(nn.Linear(in_f, n_classes))
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    else:
        # fallback: replace last Linear found
        last_linear = None
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is None:
            raise RuntimeError("Could not locate ViT classification head")
        last_linear.out_features = n_classes  # not always safe in-place; but OK for typical ViT
    return model

def build_mvit(n_classes: int) -> nn.Module:
    if not hasattr(torchvision.models, "video"):
        raise RuntimeError("torchvision video models not available (need torchvision >= 0.13)")
    ctor = None
    for name in ["mvit_v2_s", "mvit_v2_l", "mvit_v1_b"]:
        if hasattr(torchvision.models.video, name):
            ctor = getattr(torchvision.models.video, name)
            break
    if ctor is None:
        raise RuntimeError("No MViT constructor found in torchvision.models.video")

    try:
        model = ctor(weights=None)
    except TypeError:
        model = ctor(pretrained=False)

    # Replace classifier head (version safe)
    replaced = False
    for attr in ["fc", "head", "heads", "classifier"]:
        if hasattr(model, attr):
            m = getattr(model, attr)
            if isinstance(m, nn.Linear):
                setattr(model, attr, nn.Linear(m.in_features, n_classes))
                replaced = True
                break
            elif isinstance(m, nn.Sequential):
                layers = list(m.children())
                if layers and isinstance(layers[-1], nn.Linear):
                    layers[-1] = nn.Linear(layers[-1].in_features, n_classes)
                    setattr(model, attr, nn.Sequential(*layers))
                    replaced = True
                    break
    if not replaced:
        # ultimate fallback
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                parent = dict(model.named_modules()).get(parent_name, model)
                setattr(parent, name.split(".")[-1], nn.Linear(module.in_features, n_classes))
                replaced = True
                break
    if not replaced:
        raise RuntimeError("Could not replace MViT classification head")
    return model

def get_model(name: str, n_cls: int) -> Tuple[nn.Module, str]:
    name = name.lower()
    if name == "vit":
        return build_vit(n_cls), "vit"
    elif name == "mvit":
        return build_mvit(n_cls), "mvit"
    else:
        raise ValueError("Unsupported --model (choose vit or mvit)")

# ----------------------------
# Temporal fusion for MViT
# ----------------------------
def temporal_fuse(model: nn.Module, clip_bcthw: torch.Tensor, window: int = 15) -> torch.Tensor:
    """
    Split along time into 15-frame windows; average logits across windows.
    clip_bcthw: (B, C, T, H, W)
    """
    B, C, T, H, W = clip_bcthw.shape
    step = window
    outs = []
    for s in range(0, T, step):
        e = min(T, s + window)
        sub = clip_bcthw[:, :, s:e, :, :]
        if sub.shape[2] < window:  # pad last window to fixed length
            pad_t = window - sub.shape[2]
            sub = torch.cat([sub, sub[:, :, -1:].repeat(1, 1, pad_t, 1, 1)], dim=2)
        out = model(sub)  # logits
        outs.append(out)
    return torch.stack(outs, dim=0).mean(dim=0)

# ----------------------------
# Accuracy
# ----------------------------
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

# ----------------------------
# RAM guard
# ----------------------------
def check_ram_guard(max_gb: float):
    if not _HAS_PSUTIL:
        return
    rss = psutil.Process().memory_info().rss / (1024 ** 3)  # GB
    if rss > max_gb:
        raise MemoryError(f"RAM usage {rss:.2f} GB exceeded limit {max_gb:.2f} GB. "
                          f"Reduce --batch-size or --frames-per-clip.")

# ----------------------------
# Train / Eval (aligned with train_labeled.py)
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    criterion,
    optimizer,
    epoch_idx: int,
    total_epochs: int,
    model_kind: str,
    temporal_window: int,
    max_ram_gb: float,
) -> Tuple[float, float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start = time.time()

    for clips, labels in tqdm(loader, desc=f"Epoch {epoch_idx}/{total_epochs}", unit="batch"):
        check_ram_guard(max_ram_gb)

        labels = labels.to(device, non_blocking=False)

        if model_kind == "vit":
            clips = clips.to(device, non_blocking=False)            # (B, T, C, H, W)
            center = clips[:, clips.shape[1] // 2]                  # (B, C, H, W)
            outputs = model(center)
        else:
            clips = clips.to(device, non_blocking=False)            # (B, T, C, H, W)
            inputs = clips.permute(0, 2, 1, 3, 4).contiguous()      # (B, C, T, H, W)
            outputs = temporal_fuse(model, inputs, temporal_window)

        loss = criterion(outputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(1, total)
    train_acc = 100.0 * correct / max(1, total)
    return avg_loss, train_acc, time.time() - start

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    model_kind: str,
    temporal_window: int,
    max_ram_gb: float,
) -> Tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for clips, labels in tqdm(loader, desc="Validating", unit="batch"):
        check_ram_guard(max_ram_gb)

        labels = labels.to(device, non_blocking=False)

        if model_kind == "vit":
            clips = clips.to(device, non_blocking=False)
            center = clips[:, clips.shape[1] // 2]
            outputs = model(center)
        else:
            clips = clips.to(device, non_blocking=False)
            inputs = clips.permute(0, 2, 1, 3, 4).contiguous()
            outputs = temporal_fuse(model, inputs, temporal_window)

        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(1, total)
    val_acc = 100.0 * correct / max(1, total)
    return avg_loss, val_acc

# ----------------------------
# DataLoaders (host process only)
# ----------------------------
def build_dataloaders(
    data_root: Path,
    batch_size: int,
    frames_per_clip: int,
    frame_step: int,
    resize: int,
    center_crop: Optional[int],
    val_ratio: float,
    seed: int,
):
    ds = WLASLDataset(
        data_root=data_root,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        resize=resize,
        center_crop=center_crop,
    )
    n_cls = len(ds.class_to_idx)
    train_idx, val_idx = stratified_split(ds.samples, n_cls, val_ratio=val_ratio, seed=seed)
    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)

    def collate(batch):
        clips = torch.stack([b[0] for b in batch], dim=0)  # (B, T, C, H, W)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return clips, labels

    # Host-process only: no multiprocessing, no pin_memory
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=collate, persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collate, persistent_workers=False
    )
    return train_loader, val_loader, n_cls, ds.class_to_idx

# ----------------------------
# Orchestration (train loop aligned with train_labeled.py)
# ----------------------------
def main(args):
    ensure_dirs()
    set_seed(args.seed)

    # Device: auto = cuda if available else cpu
    device = torch.device("cuda" if torch.cuda.is_available() and (args.device in ["auto", "cuda"]) else "cpu")
    print("\n[INFO] Using device:", device.type)
    if device.type == "cuda":
        print("CUDA available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA available:", torch.cuda.is_available())

    # Build loaders
    train_loader, val_loader, n_cls, class_to_idx = build_dataloaders(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        resize=args.resize,
        center_crop=args.center_crop,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Model
    model, model_kind = get_model(args.model, n_cls)
    model = model.to(device)

    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train loop (with early stopping like your train_labeled.py):contentReference[oaicite:3]{index=3}
    best_val_acc = 0.0
    patience_counter = 0
    log_rows = []
    log_path = RESULTS_DIR / f"training_log_{args.model}.csv"
    print(f"\nModel Selected: {args.model}")
    print(f"Training for up to {args.epochs} epochs (early stop patience = {args.patience})\n")

    overall_start = time.time()
    last_ckpt_name = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, args.model, train_loader, device, criterion, optimizer,
            epoch, args.epochs, model_kind, args.temporal_window, args.max_ram_gb
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, device, criterion, model_kind, args.temporal_window, args.max_ram_gb
        )

        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s")

        ckpt_name = None
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_name, _ = save_checkpoint(model, val_acc, args.model)  # mirrors your naming:contentReference[oaicite:4]{index=4}
            last_ckpt_name = ckpt_name
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs\n")
                break

        log_rows.append({
            "epoch": epoch,
            "model": args.model,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "loss": train_loss,
            "epoch_time_sec": round(epoch_time, 2),
            "checkpoint": ckpt_name or last_ckpt_name
        })

        # Periodic cleanup to keep RAM usage low
        if device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    total_time = time.time() - overall_start
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    # Write CSV in the same style as train_labeled.py:contentReference[oaicite:5]{index=5}
    import pandas as pd
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Training log saved at: {log_path}")

    # Final cleanup
    try:
        if hasattr(train_loader, "_iterator") and train_loader._iterator is not None:
            train_loader._iterator._shutdown_workers()
        if hasattr(val_loader, "_iterator") and val_loader._iterator is not None:
            val_loader._iterator._shutdown_workers()
    except Exception:
        pass
    if device.type == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    print("All resources cleaned up successfully.")
    sys.stdout.flush()

# ----------------------------
# CLI
# ----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Train ViT/MViT on WLASL with temporal analysis and RAM guard.")
    p.add_argument("--model", choices=["vit", "mvit"], default="mvit",
                   help="Model architecture (default: mvit).")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2, help="Smaller default to keep RAM usage low.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--frames-per-clip", type=int, default=32)
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--center-crop", type=int, default=None)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--temporal-window", type=int, default=15, help="Temporal window (frames) for MViT fusion.")
    p.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs).")
    p.add_argument("--device", default="auto", help="auto|cuda|cpu (auto uses CUDA if available).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-ram-gb", type=float, default=58.0,
                   help="Hard limit for process RAM in GB. Training raises if exceeded.")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    try:
        main(args)
    except MemoryError as me:
        print(f"[RAM GUARD] {me}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

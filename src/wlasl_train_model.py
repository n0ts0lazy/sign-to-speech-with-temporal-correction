#!/usr/bin/env python3

import os
import time
import math
import argparse
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from wlasl_dataset_loader import WLASLDataset, pad_collate_fn
from video_model_definition import create_mvit_model


# ---------------------------
# Utility Functions
# ---------------------------

def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def lr_with_warmup_and_cosine(base_lr, epoch, total_epochs, warmup_epochs=5, start_cosine_at=40, min_lr=1e-6):
    if epoch <= warmup_epochs:
        return base_lr * (max(1, epoch) / warmup_epochs)
    if epoch < start_cosine_at:
        return base_lr
    span = total_epochs - start_cosine_at
    t = min(1.0, (epoch - start_cosine_at) / span)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def load_kinetics_weights(model, device):
    try:
        print("[INFO] Loading Kinetics-400 pretrained weights via Torch Hub...")
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # suppress fork warnings
        pretrained_model = torch.hub.load("facebookresearch/pytorchvideo", "mvit_base_16x4", pretrained=True)
        model_dict = model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"[INFO] Loaded {len(pretrained_dict)} matching pretrained layers.")
    except Exception as e:
        print(f"[WARN] Could not load Kinetics pretrained weights: {e}")


def freeze_backbone_until_epoch(model, epoch, unfreeze_epoch=20):
    should_freeze = epoch <= unfreeze_epoch
    for name, p in model.named_parameters():
        if any(tag in name.lower() for tag in ["head", "classifier", "proj", "fc"]):
            p.requires_grad = True
        else:
            p.requires_grad = not should_freeze


# ---------------------------
# Training Function
# ---------------------------

def train_model(
    json_path="./data/video/WLASL_stratified.json",
    video_root="./data/video/transcoded",
    output_dir="./models/checkpoints",
    results_dir="./results",
    batch_size=6,
    epochs=400,
    lr=3e-4,
    num_workers=24,
    patience=50,
    start_cosine_at=40,
    min_lr=1e-6,
    label_smoothing=0.05,
    warmup_epochs=5,
    freeze_until_epoch=20,
    eval_every=10
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    model_name = "mvit"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"training_log_{model_name}.csv")

    print("[INFO] Loading datasets...")
    train_ds = WLASLDataset(json_path, video_root, split="train")
    val_ds = WLASLDataset(json_path, video_root, split="val")
    test_ds = WLASLDataset(json_path, video_root, split="test")

    if len(test_ds) > 0:
        print(f"[INFO] Merging test split into evaluation pool (+{len(test_ds)} samples)")
        val_ds.samples.extend(test_ds.samples)

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty.")

    all_glosses = sorted(list({s[1].strip().lower() for s in train_ds.samples}))
    gloss_to_idx = {g: i for i, g in enumerate(all_glosses)}
    print(f"[INFO] Found {len(gloss_to_idx)} unique glosses/classes from training.")

    class_counts = np.zeros(len(gloss_to_idx), dtype=np.int64)
    for _, label in train_ds.samples:
        g = label.strip().lower()
        if g in gloss_to_idx:
            class_counts[gloss_to_idx[g]] += 1

    class_weights = np.ones(len(gloss_to_idx), dtype=np.float32)
    for i, c in enumerate(class_counts):
        if c <= 2:
            class_weights[i] = 0.5
    class_weights = class_weights * (len(class_weights) / class_weights.sum())
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[INFO] Applied per-class weighting: {np.sum(class_counts <= 2)} low-data classes down-weighted.")

    def collate_fn_with_labels(batch):
        videos, labels = pad_collate_fn(batch)
        valid_videos, valid_labels = [], []
        for v, l in zip(videos, labels):
            key = l.strip().lower()
            if key in gloss_to_idx:
                valid_videos.append(v)
                valid_labels.append(gloss_to_idx[key])
        if len(valid_videos) == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        return torch.stack(valid_videos), torch.tensor(valid_labels, dtype=torch.long)

    persistent = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn_with_labels,
                              persistent_workers=persistent, pin_memory=True)
    eval_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn_with_labels,
                             persistent_workers=persistent, pin_memory=True)

    num_classes = len(gloss_to_idx)
    model = create_mvit_model(num_classes=num_classes, pretrained=False)
    load_kinetics_weights(model, device)
    model.to(device)

    if hasattr(nn.CrossEntropyLoss, "label_smoothing"):
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_train_acc, best_train_loss = 0.0, float("inf")
    best_model_name = "NA"
    no_improve_epochs = 0

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline='') as f:
            f.write("epoch,train_loss,train_acc,val_acc,elapsed,model_file\n")
            f.flush()
            os.fsync(f.fileno())

    for epoch in range(1, epochs + 1):
        new_lr = lr_with_warmup_and_cosine(lr, epoch, epochs, warmup_epochs, start_cosine_at, min_lr)
        set_lr(optimizer, new_lr)
        freeze_backbone_until_epoch(model, epoch, freeze_until_epoch)

        start_t = time.time()
        print(f"\n🌀 Epoch [{epoch}/{epochs}] — Training (lr={new_lr:.6f})")

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", ncols=100) as pbar:
            for videos, labels in train_loader:
                # --- Strong batch validation ---
                if (
                    videos is None or labels is None or
                    videos.numel() == 0 or labels.numel() == 0 or
                    videos.size(0) != labels.size(0)
                ):
                    continue

                videos, labels = videos.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.numel()

                avg_loss = running_loss / (pbar.n + 1)
                train_acc_live = 100.0 * correct / max(total, 1)
                pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{train_acc_live:.2f}%"})
                pbar.update(1)

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(total, 1)
        elapsed = time.time() - start_t

        # Accuracy sanity check
        if train_acc > 100.0:
            print(f"[WARN] Train accuracy exceeded 100% ({train_acc:.2f}%) — possible dataset inconsistency.")

        # ---- Validation ----
        eval_acc, eval_loss, ran_eval = 0.0, 0.0, False
        if len(val_ds) > 0 and (epoch % eval_every == 0 or epoch == 1 or epoch == epochs):
            ran_eval = True
            print(f"🔍 Validation for epoch {epoch} ...")
            model.eval()
            correct_e, total_e, loss_e_sum = 0, 0, 0.0

            with torch.no_grad():
                for videos, labels in eval_loader:
                    if (
                        videos is None or labels is None or
                        videos.numel() == 0 or labels.numel() == 0 or
                        videos.size(0) != labels.size(0)
                    ):
                        continue
                    videos, labels = videos.to(device), labels.to(device)
                    outputs = model(videos)
                    loss_e = criterion(outputs, labels)
                    loss_e_sum += loss_e.item()
                    _, preds = outputs.max(1)
                    correct_e += preds.eq(labels).sum().item()
                    total_e += labels.numel()

            eval_loss = loss_e_sum / max(1, len(eval_loader))
            eval_acc = 100.0 * correct_e / max(total_e, 1)
            print(f"✅ ValLoss={eval_loss:.4f}, ValAcc={eval_acc:.2f}%")

        # ---- Save if training improved ----
        improved = (train_acc > best_train_acc + 0.05) or (train_loss < best_train_loss - 1e-4)
        if improved:
            best_train_acc, best_train_loss = train_acc, train_loss
            no_improve_epochs = 0
            best_model_name = f"mvit_acc_{train_acc:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save(model.state_dict(), os.path.join(output_dir, best_model_name))
            print(f"💾 Saved model — TrainAcc={train_acc:.2f}% | TrainLoss={train_loss:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"🛑 Early stopping: no training improvement for {patience} epochs. "
                      f"Best TrainAcc={best_train_acc:.2f}%")
                break

        # ---- CSV Logging ----
        val_acc_for_log = eval_acc if ran_eval else 0.0
        with open(csv_path, "a", newline='') as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.2f},{val_acc_for_log:.2f},{elapsed:.1f},{best_model_name}\n")
            f.flush()
            os.fsync(f.fileno())

        print(f"📊 Epoch {epoch}/{epochs} — Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"ValAcc={'N/A' if not ran_eval else f'{eval_acc:.2f}%'} | Time={elapsed:.1f}s")

    print(f"\n🎯 Training complete. Log written to: {csv_path}")


# ---------------------------
# Main Entry
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MViT (final refined)")
    parser.add_argument("--json_path", type=str, default="./data/video/WLASL_stratified.json")
    parser.add_argument("--video_root", type=str, default="./data/video/transcoded")
    parser.add_argument("--output_dir", type=str, default="./models/checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--eval_every", type=int, default=10)
    args = parser.parse_args()

    train_model(
        json_path=args.json_path,
        video_root=args.video_root,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        eval_every=args.eval_every
    )

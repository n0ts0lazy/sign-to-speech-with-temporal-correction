import os
import sys
import time
import argparse
import subprocess
import torch
import pandas as pd
from torch import nn, optim
from tqdm import tqdm

# Ensure project root is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset_loader import get_dataloaders
from src.model_definitions import make_resnet18, make_mobilenet, make_cnn_lstm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(model_name, num_classes):
    name = model_name.lower()
    if name in ["resnet", "resnet18"]:
        return make_resnet18(num_classes)
    elif name in ["mobilenet", "mobilenetv3"]:
        return make_mobilenet(num_classes)
    elif name in ["cnn_lstm", "cnnlstm"]:
        return make_cnn_lstm(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def save_checkpoint(model, accuracy, model_name, out_dir=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if out_dir is None:
        out_dir = os.path.join(project_root, "models", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{accuracy:.2f}_{timestamp}.pth"
    path = os.path.join(out_dir, filename)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")
    return filename, path


def train_model(model, model_name, train_loader, val_loader, device, num_epochs=200, lr=1e-4, patience=8):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    patience_counter = 0
    log_data = []

    log_dir = os.path.join(project_root, "results")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"training_log_{model_name}.csv")

    print(f"\nModel Selected: {model_name}")
    print(f"Training for up to {num_epochs} epochs (early stop patience = {patience})\n")

    start_time = time.time()
    last_checkpoint_name = None

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = running_loss / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        checkpoint_filename = None
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_filename, _ = save_checkpoint(model, val_acc, model_name)
            last_checkpoint_name = checkpoint_filename
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs\n")
                break

        log_data.append({
            "epoch": epoch + 1,
            "model": model_name,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "loss": avg_loss,
            "epoch_time_sec": round(epoch_time, 2),
            "checkpoint": checkpoint_filename or last_checkpoint_name
        })

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    pd.DataFrame(log_data).to_csv(log_path, index=False)
    print(f"Training log saved at: {log_path}")

    # Force DataLoader cleanup to release worker threads
    try:
        if hasattr(train_loader, "_iterator") and train_loader._iterator is not None:
            train_loader._iterator._shutdown_workers()
        if hasattr(val_loader, "_iterator") and val_loader._iterator is not None:
            val_loader._iterator._shutdown_workers()
    except Exception:
        pass

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    print("All resources cleaned up successfully.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on labeled dataset")
    parser.add_argument("--model", type=str, default="resnet", help="Model type: resnet, mobilenet, or cnn_lstm")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print("CUDA available:", torch.cuda.is_available())
        print(f"Using GPU: {gpu_name}")
    else:
        try:
            cpu_info = subprocess.check_output(
                "lscpu | grep 'Model name'", shell=True
            ).decode().strip().split(":")[1].strip()
        except Exception:
            cpu_info = "Unknown CPU"
        print("CUDA available:", torch.cuda.is_available())
        print(f"Using CPU: {cpu_info}")

    set_seed(42)
    train_loader, val_loader, class_names = get_dataloaders(batch_size=args.batch_size)
    model = build_model(args.model, num_classes=len(class_names)).to(device)

    train_model(model, args.model, train_loader, val_loader, device,
                num_epochs=args.epochs, lr=args.lr, patience=args.patience)

    print("\nTraining complete. Program exiting cleanly.")
    sys.stdout.flush()
    os._exit(0)

# Configuration file for paths, hyperparameters, etc.
import os, json
from torchvision import datasets
from wlasl_dataset_loader import WLASLDataset

def generate_asl_labels():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_dir = os.path.join(project_root, "data", "labeled", "train")
    config_dir = os.path.join(project_root, "config")
    os.makedirs(config_dir, exist_ok=True)

    # Load ASL dataset structure
    train_dataset = datasets.ImageFolder(root=train_dir)
    label_map = {i: cls for i, cls in enumerate(train_dataset.classes)}

    out_path = os.path.join(config_dir, "labels_asl.json")
    json.dump(label_map, open(out_path, "w"), indent=2)
    print(f"✅ Saved ASL label map to {out_path}")
    print(f"Classes ({len(label_map)}): {list(label_map.values())[:10]}...")


def generate_wlasl_labels():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(project_root, "data", "video", "WLASL_rebuilt.json")
    config_dir = os.path.join(project_root, "config")
    os.makedirs(config_dir, exist_ok=True)

    # Load WLASL JSON annotations
    with open(json_path, "r") as f:
        all_data = json.load(f)

    label_map = {i: entry["gloss"] for i, entry in enumerate(all_data)}

    out_path = os.path.join(config_dir, "labels_wlasl.json")
    json.dump(label_map, open(out_path, "w"), indent=2)
    print(f"✅ Saved WLASL label map to {out_path}")
    print(f"Classes ({len(label_map)}): {list(label_map.values())[:10]}...")


if __name__ == "__main__":
    generate_asl_labels()
    generate_wlasl_labels()

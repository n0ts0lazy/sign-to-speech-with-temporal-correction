import json
import random
import os
from tqdm import tqdm

def stratified_split(json_path, output_path, seed=42):
    """
    Creates stratified train/val/test splits within the WLASL JSON.
    Ensures each gloss has entries in all three splits if possible.
    """
    random.seed(seed)
    with open(json_path, "r") as f:
        data = json.load(f)

    new_data = []
    split_counts = {"train": 0, "val": 0, "test": 0}

    for gloss_entry in tqdm(data, desc="Splitting glosses"):
        gloss = gloss_entry["gloss"]
        instances = gloss_entry["instances"]
        n = len(instances)

        if n == 0:
            continue

        # Handle small cases explicitly
        if n == 1:
            train_split = val_split = test_split = [instances[0]]
        elif n == 2:
            random.shuffle(instances)
            train_split = [instances[0]]
            val_split = [instances[1]]
            test_split = [instances[0]]  # reuse one for test
        else:
            random.shuffle(instances)
            n_train = max(1, int(0.7 * n))
            n_val = max(1, int(0.2 * n))
            n_test = max(1, n - n_train - n_val)

            train_split = instances[:n_train]
            val_split = instances[n_train:n_train+n_val]
            test_split = instances[n_train+n_val:]

        # Mark the split inside each instance
        for inst in train_split:
            inst["split"] = "train"
            split_counts["train"] += 1
        for inst in val_split:
            inst["split"] = "val"
            split_counts["val"] += 1
        for inst in test_split:
            inst["split"] = "test"
            split_counts["test"] += 1

        new_data.append({
            "gloss": gloss,
            "instances": train_split + val_split + test_split
        })

    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"\n✅ Stratified split complete!")
    print(f"Train: {split_counts['train']}")
    print(f"Val:   {split_counts['val']}")
    print(f"Test:  {split_counts['test']}")
    print(f"Saved new JSON: {output_path}")


if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "../data/video/WLASL_rebuilt.json")
    output_path = os.path.join(base_dir, "../data/video/WLASL_stratified.json")

    # Normalize and expand user paths (safe for WSL, Linux, or Windows)
    input_path = os.path.abspath(os.path.expanduser(input_path))
    output_path = os.path.abspath(os.path.expanduser(output_path))

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output: {output_path}")

    stratified_split(input_path, output_path)

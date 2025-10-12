import os, random, shutil
from pathlib import Path

# Original validation data of dataset is too small hence taking 10% of training data as validation data
train_dir = Path("./data/labeled/train")
val_dir = Path("./data/labeled/val")
val_ratio = 0.1  # 10% of data per class

os.makedirs(val_dir, exist_ok=True)

for class_dir in train_dir.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*.jpg"))
    random.shuffle(images)
    n_val = max(1, int(len(images) * val_ratio))
    val_class_dir = val_dir / class_dir.name
    val_class_dir.mkdir(parents=True, exist_ok=True)

    for img in images[:n_val]:
        shutil.move(str(img), val_class_dir / img.name)

print("Created validation split in data/labeled/val/")


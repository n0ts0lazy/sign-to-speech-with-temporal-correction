import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(batch_size=32, num_workers=2, image_size=224):
    """
    Returns PyTorch dataloaders for training and validation datasets.
    Automatically detects project root so it works from any directory
    (src scripts, notebooks, or external imports).
    """

    # --- Try to detect the project root dynamically ---
    # (works whether running from /src, /notebooks, or project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = os.path.join(project_root, "data", "labeled")

    # Fallback: if not found, try one level higher (common when using Jupyter)
    if not os.path.exists(data_root):
        project_root = os.path.abspath(os.path.join(project_root, ".."))
        data_root = os.path.join(project_root, "data", "labeled")

    # Now resolve all four paths explicitly
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    unlabeled_dir = os.path.join(project_root, "data", "unlabeled")  # optional, future use
    checkpoints_dir = os.path.join(project_root, "models", "checkpoints")

    # --- Sanity checks ---
    for path_name, path_value in {
        "Train directory": train_dir,
        "Validation directory": val_dir
    }.items():
        if not os.path.exists(path_value):
            raise FileNotFoundError(
                f"{path_name} not found at: {path_value}\n"
                "Make sure your dataset is extracted correctly under 'data/labeled/'."
            )

    # --- Normalization (ImageNet mean/std for pretrained models)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # --- Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- Datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # --- Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_names = train_dataset.classes

    # --- Summary printout ---
    print(f" Dataset paths resolved:")
    print(f"   Project root     : {project_root}")
    print(f"   Train directory  : {train_dir}")
    print(f"   Validation dir   : {val_dir}")
    print(f"\nLoaded {len(class_names)} classes.")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Classes: {class_names}")

    return train_loader, val_loader, class_names

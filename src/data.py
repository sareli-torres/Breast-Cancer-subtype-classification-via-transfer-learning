"""
Data loading for BreaKHis: reads the frozen metadata.csv and serves images to PyTorch.

Typical use in train.py:

    from data import get_loader, class_weights, NUM_CLASSES

    train_loader = get_loader(META_CSV, DATA_ROOT, split="train", mags=40, task="binary")
    val_loader   = get_loader(META_CSV, DATA_ROOT, split="val",   mags=40, task="binary")
    weights      = class_weights(META_CSV, mags=40, task="binary")

    # Evaluate the trained model on the test patients at every magnification
    for m in [40, 100, 200, 400]:
        test_loader = get_loader(META_CSV, DATA_ROOT, split="test", mags=m, task="binary")

Quick check in Colab:
    !python src/data.py --meta "{DRIVE_ROOT}/data/metadata.csv" --data_root /content/BreaKHis_v1
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

TASKS = {"binary": "binary_label", "8class": "subtype_label"}
NUM_CLASSES = {"binary": 2, "8class": 8}
CLASS_NAMES = {
    "binary": ["benign", "malignant"],
    "8class": ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",
               "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"],
}
ALL_MAGS = [40, 100, 200, 400]

# ImageNet statistics: the pretrained backbones expect inputs normalised this way
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_mags(mags) -> list:
    """Accepts 40, "40", "40x", "all", or a list of those. Returns e.g. [40]."""
    if isinstance(mags, (list, tuple)):
        return sorted({m for x in mags for m in parse_mags(x)})
    s = str(mags).lower().rstrip("x")
    if s == "all":
        return ALL_MAGS
    m = int(s)
    if m not in ALL_MAGS:
        raise ValueError(f"Magnification must be one of {ALL_MAGS} or 'all', got {mags}")
    return [m]


def get_transforms(train: bool, img_size: int = 224) -> transforms.Compose:
    """Same preprocessing for every model, as fixed in the README.

    Train: resize + random flips, small rotation and colour jitter (H&E stain varies between slides).
    Val/test: resize only, so evaluation is deterministic.
    """
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def load_split(meta_csv, split: str, mags="all") -> pd.DataFrame:
    """Rows of metadata.csv for one split and the chosen magnification(s)."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train, val or test, got {split}")
    meta = pd.read_csv(meta_csv)
    rows = meta[(meta["split"] == split) & (meta["magnification"].isin(parse_mags(mags)))]
    return rows.reset_index(drop=True)


class BreakHisDataset(Dataset):
    """Returns (image tensor, label). The rows are kept in .meta, so for Grad-CAM
    or error analysis you can look up the patient, subtype and file of item i with ds.meta.iloc[i]."""

    def __init__(self, meta: pd.DataFrame, data_root, task: str, transform=None):
        if task not in TASKS:
            raise ValueError(f"task must be one of {list(TASKS)}, got {task}")
        self.meta = meta
        self.data_root = Path(data_root)
        self.labels = meta[TASKS[task]].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        image = Image.open(self.data_root / self.meta.at[i, "path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[i]


def get_loader(meta_csv, data_root, split: str, mags, task: str,
               batch_size: int = 32, num_workers: int = 2, img_size: int = 224) -> DataLoader:
    """DataLoader for one split. Training data is shuffled and augmented; val/test are not."""
    train = split == "train"
    ds = BreakHisDataset(load_split(meta_csv, split, mags), data_root, task,
                         get_transforms(train, img_size))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def class_weights(meta_csv, mags, task: str) -> torch.Tensor:
    """Weights for nn.CrossEntropyLoss(weight=...), from the TRAIN images of the chosen magnification(s).

    weight_c = n_images / (n_classes * n_images_of_class_c), so rare classes count more.
    """
    labels = load_split(meta_csv, "train", mags)[TASKS[task]]
    counts = labels.value_counts().reindex(range(NUM_CLASSES[task]), fill_value=0)
    if (counts == 0).any():
        missing = [CLASS_NAMES[task][c] for c in counts[counts == 0].index]
        raise ValueError(f"No training images for {missing} at magnification {mags}")
    return torch.tensor((len(labels) / (NUM_CLASSES[task] * counts)).values, dtype=torch.float32)


def main():
    """Sanity check: sizes of every split, one batch, and the class weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Path to metadata.csv")
    parser.add_argument("--data_root", required=True, help="Folder where BreaKHis_v1 was extracted")
    args = parser.parse_args()

    print("=== Images per split and magnification ===")
    for split in ["train", "val", "test"]:
        sizes = {m: len(load_split(args.meta, split, m)) for m in ALL_MAGS}
        print(f"{split:5s} " + "  ".join(f"{m}X: {n:4d}" for m, n in sizes.items())
              + f"  all: {sum(sizes.values())}")

    for task in TASKS:
        loader = get_loader(args.meta, args.data_root, "train", 40, task, batch_size=8, num_workers=0)
        images, labels = next(iter(loader))
        print(f"\n=== Task {task} (train, 40X) ===")
        print(f"Batch: images {tuple(images.shape)}, labels {labels.tolist()}")
        w = class_weights(args.meta, 40, task)
        for name, value in zip(CLASS_NAMES[task], w.tolist()):
            print(f"  weight {name:20s} {value:.2f}")

    print("\nAll good ✓")


if __name__ == "__main__":
    main()

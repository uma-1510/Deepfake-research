"""
data_loader.py
CelebDF dataset loader — extracts frames from video, preprocesses for model input.

CelebDF structure expected:
  CelebDF/
  ├── YouTube-real/videos/*.mp4        (real videos)
  ├── Celeb-synthesis/videos/*.mp4     (deepfake videos)
  └── List_of_testing_videos.txt       (official test split)
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ── Preprocessing pipeline (ImageNet normalization, matches EfficientNet expects) ──
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Same pipeline but without normalization — used by FFT detector which needs raw pixels
RAW_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def extract_frames(video_path: str, output_dir: str, max_frames: int = 30) -> list[str]:
    """
    Extract evenly-spaced frames from a video file.
    Returns list of saved frame paths.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    # Sample evenly across the video duration
    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    saved_paths = []
    video_name = Path(video_path).stem

    os.makedirs(output_dir, exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_path = os.path.join(output_dir, f"{video_name}_frame{idx:05d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


def build_dataset_from_celebdf(celeb_root: str, output_dir: str, max_frames: int = 30):
    """
    Walks the CelebDF directory, extracts frames from all videos,
    saves real frames to output_dir/real/ and fake frames to output_dir/fake/.
    """
    real_video_dir = os.path.join(celeb_root, "YouTube-real", "videos")
    fake_video_dir = os.path.join(celeb_root, "Celeb-synthesis", "videos")

    real_out = os.path.join(output_dir, "real")
    fake_out = os.path.join(output_dir, "fake")

    for label, src_dir, dst_dir in [
        ("REAL", real_video_dir, real_out),
        ("FAKE", fake_video_dir, fake_out),
    ]:
        videos = list(Path(src_dir).glob("*.mp4"))
        print(f"\n[{label}] Found {len(videos)} videos in {src_dir}")
        for video_path in tqdm(videos, desc=f"Extracting {label} frames"):
            extract_frames(str(video_path), dst_dir, max_frames=max_frames)

    real_count = len(list(Path(real_out).glob("*.jpg")))
    fake_count = len(list(Path(fake_out).glob("*.jpg")))
    print(f"\n✓ Extracted {real_count} real frames → {real_out}")
    print(f"✓ Extracted {fake_count} fake frames → {fake_out}")


# ── PyTorch Dataset ───────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Loads preprocessed frames from data_dir/real/ and data_dir/fake/.
    Labels: 0 = real, 1 = fake.
    """

    def __init__(self, data_dir: str, transform=None, split: str = "all", val_ratio: float = 0.2):
        self.transform = transform or TRANSFORM
        self.samples = []  # list of (path, label)

        real_dir = Path(data_dir) / "real"
        fake_dir = Path(data_dir) / "fake"

        real_paths = sorted(real_dir.glob("*.jpg"))
        fake_paths = sorted(fake_dir.glob("*.jpg"))

        all_samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

        # Deterministic train/val split
        np.random.seed(42)
        indices = np.random.permutation(len(all_samples))
        split_idx = int(len(indices) * (1 - val_ratio))

        if split == "train":
            selected = [all_samples[i] for i in indices[:split_idx]]
        elif split == "val":
            selected = [all_samples[i] for i in indices[split_idx:]]
        else:
            selected = all_samples

        self.samples = selected
        print(f"[DeepfakeDataset] split={split} | {len(self.samples)} samples "
              f"({sum(1 for _, l in self.samples if l == 0)} real, "
              f"{sum(1 for _, l in self.samples if l == 1)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label, str(path)


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """Returns train and val DataLoaders."""
    train_ds = DeepfakeDataset(data_dir, split="train")
    val_ds   = DeepfakeDataset(data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ── CLI entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CelebDF frames")
    parser.add_argument("--celeb_root", required=True, help="Path to CelebDF root directory")
    parser.add_argument("--output_dir", default="./data", help="Where to save extracted frames")
    parser.add_argument("--max_frames", type=int, default=30, help="Max frames to extract per video")
    args = parser.parse_args()

    build_dataset_from_celebdf(args.celeb_root, args.output_dir, args.max_frames)
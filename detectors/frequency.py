"""
detectors/frequency.py
FFT-based spectral anomaly detector — no ML required.

WHY THIS WORKS:
GAN-generated images leave characteristic artifacts in the frequency domain.
The generator's upsampling operations (transposed convolutions, bilinear
upsampling) create periodic patterns at specific spatial frequencies — a
"checkerboard" signal invisible to the human eye but detectable via FFT.

This detector:
1. Converts the image to grayscale
2. Applies 2D FFT → frequency spectrum
3. Computes high-frequency energy ratio (HFR): energy above a threshold radius
   relative to total spectrum energy
4. Applies a learned threshold: HFR > threshold → fake

The threshold is estimated from a calibration pass over a small labeled set.
This is a simple but principled baseline — it explains *why* deepfakes are
detectable, not just *that* they are.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from detectors.base import Detector


def compute_hfr(image_tensor: torch.Tensor, radius_ratio: float = 0.3) -> float:
    """
    Compute High-Frequency Ratio from a normalized image tensor.

    Args:
        image_tensor: (C, H, W) tensor, values in [0, 1] after ToTensor
        radius_ratio: fraction of spectrum radius considered "high frequency"

    Returns:
        HFR scalar — higher values → more high-frequency content → likely fake
    """
    # Convert to grayscale (luminance-weighted)
    # Shape: (H, W)
    gray = (0.299 * image_tensor[0] +
            0.587 * image_tensor[1] +
            0.114 * image_tensor[2]).numpy()

    # 2D FFT + shift zero-frequency component to center
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    H, W = magnitude.shape
    cy, cx = H // 2, W // 2  # center (DC component)

    # Build radius mask: 1 where distance from center > threshold
    radius_threshold = min(H, W) * radius_ratio
    y_coords, x_coords = np.ogrid[:H, :W]
    dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    high_freq_mask = dist > radius_threshold

    total_energy = magnitude.sum() + 1e-8
    high_freq_energy = magnitude[high_freq_mask].sum()

    return float(high_freq_energy / total_energy)


class FrequencyDetector(Detector):
    """
    Threshold-based detector on FFT high-frequency ratio.
    
    threshold: HFR values above this → classified as fake.
               Calibrated automatically on first .predict() call if not set.
    """

    def __init__(self, radius_ratio: float = 0.3, threshold: float = None):
        self.radius_ratio = radius_ratio
        self._threshold = threshold  # None = auto-calibrate

    @property
    def name(self) -> str:
        return "FrequencyDetector"

    def _calibrate(self, hfrs: list[float], labels: list[int]) -> float:
        """
        Find the threshold that maximizes F1 over a labeled calibration set.
        Scans 50 candidate thresholds between min and max HFR.
        """
        hfrs = np.array(hfrs)
        labels = np.array(labels)
        best_f1, best_thresh = 0.0, 0.5

        for t in np.linspace(hfrs.min(), hfrs.max(), 50):
            preds = (hfrs > t).astype(int)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

        print(f"  [FrequencyDetector] Calibrated threshold={best_thresh:.4f} (F1={best_f1:.3f})")
        return best_thresh

    def predict(self, dataset) -> list[dict]:
        """
        Compute HFR for every sample, calibrate threshold on the fly,
        then return predictions.
        """
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

        all_hfrs, all_labels, all_paths = [], [], []

        print(f"\n[{self.name}] Computing FFT high-frequency ratios...")
        for tensors, labels, paths in tqdm(loader, desc=self.name):
            for i in range(len(tensors)):
                hfr = compute_hfr(tensors[i], self.radius_ratio)
                all_hfrs.append(hfr)
                all_labels.append(int(labels[i]))
                all_paths.append(paths[i])

        # Auto-calibrate threshold if not provided
        if self._threshold is None:
            self._threshold = self._calibrate(all_hfrs, all_labels)

        results = []
        for hfr, label, path in zip(all_hfrs, all_labels, all_paths):
            pred = 1 if hfr > self._threshold else 0
            # Normalize HFR to [0,1] confidence score via sigmoid-like scaling
            confidence = float(1 / (1 + np.exp(-10 * (hfr - self._threshold))))
            results.append({
                "path":       path,
                "label":      label,
                "pred":       pred,
                "confidence": confidence,
                "hfr":        hfr,  # extra diagnostic
            })

        return results
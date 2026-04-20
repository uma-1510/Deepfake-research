"""
detectors/base.py
Abstract base class for all deepfake detectors.
Every detector must implement .predict() and expose a .name attribute.
"""

from abc import ABC, abstractmethod
import numpy as np


class Detector(ABC):
    """
    Interface every detector must conform to.
    
    predict() takes a list of (image_tensor, label, path) tuples
    and returns a list of dicts with prediction metadata.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name shown in benchmark reports."""
        pass

    @abstractmethod
    def predict(self, dataset) -> list[dict]:
        """
        Run inference on the full dataset.

        Args:
            dataset: DeepfakeDataset instance (or any iterable yielding
                     (tensor, label, path) tuples)

        Returns:
            List of prediction dicts, one per sample:
            {
                "path":       str,         # source image path
                "label":      int,         # ground truth (0=real, 1=fake)
                "pred":       int,         # predicted class (0 or 1)
                "confidence": float,       # probability of fake (0.0–1.0)
            }
        """
        pass
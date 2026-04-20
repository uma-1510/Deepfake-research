"""
detectors/model_based.py
Fine-tuned EfficientNet-B0 deepfake detector.

Loads weights from a checkpoint produced by train.py and runs inference.
EfficientNet-B0 is chosen for:
- Strong baseline accuracy on image classification
- Compound scaling (depth + width + resolution balanced together)
- Efficient enough to run on a laptop GPU or CPU for demo purposes
- Well-understood architecture, easy to explain in interviews
"""

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from detectors.base import Detector


def build_efficientnet(num_classes: int = 1) -> nn.Module:
    """
    EfficientNet-B0 with the final classifier replaced for binary classification.
    
    The original classifier outputs 1000 ImageNet classes.
    We replace it with a single output neuron + sigmoid for fake probability.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier: original in_features=1280
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


class ModelBasedDetector(Detector):
    """
    Deepfake detector using a fine-tuned EfficientNet-B0 model.
    
    Args:
        model_path: path to .pth checkpoint saved by train.py
        device:     'cuda', 'mps', or 'cpu' — auto-detected if None
    """

    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or self._auto_device()
        self.model = self._load_model()

    @property
    def name(self) -> str:
        return "ModelBasedDetector (EfficientNet-B0)"

    def _auto_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> nn.Module:
        model = build_efficientnet(num_classes=1)
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Handle both raw state_dict and wrapped checkpoint formats
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        print(f"[ModelBasedDetector] Loaded checkpoint from {self.model_path} → {self.device}")
        return model

    @torch.no_grad()
    def predict(self, dataset) -> list[dict]:
        """Run inference on the full dataset."""
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        sigmoid = nn.Sigmoid()
        results = []

        print(f"\n[{self.name}] Running inference...")
        for tensors, labels, paths in tqdm(loader, desc="ModelBased"):
            tensors = tensors.to(self.device)
            logits = self.model(tensors).squeeze(1)     # (B,)
            probs  = sigmoid(logits).cpu().numpy()      # fake probability

            for prob, label, path in zip(probs, labels.numpy(), paths):
                results.append({
                    "path":       path,
                    "label":      int(label),
                    "pred":       1 if prob >= 0.5 else 0,
                    "confidence": float(prob),
                })

        return results
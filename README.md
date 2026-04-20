# Deepfake Detection Benchmark Suite
**Clark University Independent Research | CelebDF-V2 Dataset**

A systematic evaluation framework comparing deepfake detection approaches — from a classical signal-processing baseline (FFT spectral analysis) to a fine-tuned neural network (EfficientNet-B0 via supervised fine-tuning).

---

## Project Structure

```
deepfake_eval/
├── data/
│   ├── real/               ← extracted real frames (.jpg)
│   └── fake/               ← extracted deepfake frames (.jpg)
├── celebdf/                ← raw CelebDF videos (delete after extraction to save space)
│   ├── YouTube-real/
│   │   └── videos/         ← real .mp4 files (590 videos)
│   └── Celeb-synthesis/
│       └── videos/         ← fake .mp4 files (5,639 videos)
├── checkpoints/
│   ├── best_model.pth      ← saved after training
│   └── training_history.json
├── results/
│   ├── benchmark_report.md
│   └── benchmark_report.json
├── detectors/
│   ├── base.py             ← abstract Detector interface
│   ├── frequency.py        ← FFT spectral anomaly detector (no ML)
│   └── model_based.py      ← fine-tuned EfficientNet-B0
├── data_loader.py          ← frame extraction + PyTorch Dataset
├── train.py                ← two-phase supervised fine-tuning
├── evaluate.py             ← runs all detectors, computes metrics
├── metrics.py              ← accuracy, precision, recall, F1, AUC-ROC
└── report.py               ← JSON + markdown benchmark report
```

---

## Requirements

Python 3.10 or 3.11 recommended.

**Mac (Apple Silicon M1/M2/M3):**
```bash
pip install torch torchvision
pip install opencv-python scikit-learn Pillow numpy tqdm
```

**Linux/Windows with NVIDIA GPU:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python scikit-learn Pillow numpy tqdm
```

**Linux/Windows CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python scikit-learn Pillow numpy tqdm
```

Device is auto-detected at runtime — CUDA → MPS → CPU in that order. No config needed.

---

## Step-by-Step: Running from Scratch

### Step 1 — Get the dataset

Request CelebDF-V2 access at: https://github.com/yuezunli/celeb-deepfakeforensics

Alternatively, a pre-extracted image version is available on Kaggle (search "Celeb-DF v2"). If using the Kaggle image version, skip to Step 3.

Once downloaded, organize videos like this:
```
celebdf/
├── YouTube-real/
│   └── videos/      ← real .mp4 files (590 videos)
└── Celeb-synthesis/
    └── videos/      ← fake .mp4 files (5,639 videos)
```

### Step 2 — Extract frames

**Extract fake frames — use only 30 videos to save disk space:**
```bash
python - <<'EOF'
from data_loader import extract_frames
from pathlib import Path
from tqdm import tqdm

src = Path("celebdf/Celeb-synthesis/videos")
videos = list(src.glob("*.mp4"))[:30]   # 30 videos = ~900 frames
print(f"Extracting from {len(videos)} fake videos")
for v in tqdm(videos, desc="Extracting FAKE frames"):
    extract_frames(str(v), "data/fake", max_frames=30)
EOF
```

**Extract real frames:**
```bash
mkdir -p data/real

python - <<'EOF'
from data_loader import extract_frames
from pathlib import Path
from tqdm import tqdm

src = Path("celebdf/YouTube-real/videos")
videos = list(src.glob("*.mp4"))
print(f"Extracting from {len(videos)} real videos")
for v in tqdm(videos, desc="Extracting REAL frames"):
    extract_frames(str(v), "data/real", max_frames=30)
EOF
```

**Verify frame counts:**
```bash
ls data/real | wc -l    # expect ~200-590
ls data/fake | wc -l    # expect ~900
```

**Balance the dataset** — cap fake frames to avoid severe class imbalance:
```bash
# keep ~500 fake frames (roughly 2x real count)
ls data/fake/*.jpg | tail -n +501 | xargs rm
ls data/real | wc -l
ls data/fake | wc -l
```

**Free up disk space** — raw videos no longer needed after extraction:
```bash
rm -rf celebdf/
pip cache purge
df -h .    # verify at least 2GB free before training
```

### Step 3 — Train the model

**Mac (CPU/MPS):**
```bash
python train.py \
  --data_dir ./data \
  --epochs 10 \
  --batch_size 8 \
  --num_workers 0
```

**GPU machine:**
```bash
python train.py \
  --data_dir ./data \
  --epochs 10 \
  --batch_size 32 \
  --num_workers 4
```

Expected output:
```
============================================================
  Deepfake Detection — Supervised Fine-Tuning
  Device: mps | Epochs: 10 | Batch: 8
============================================================
[DeepfakeDataset] split=train | 573 samples (170 real, 403 fake)
[DeepfakeDataset] split=val   | 144 samples (48 real, 96 fake)

Phase 1: Warmup (3 epochs) — training classifier head only

Epoch 01/10  train_loss=0.6507  train_acc=0.6614  val_auc=0.6910
  ✓ New best AUC=0.6910 — checkpoint saved to checkpoints/best_model.pth
Epoch 02/10  train_loss=0.5969  train_acc=0.7068  val_auc=0.8316
Epoch 03/10  train_loss=0.5756  train_acc=0.7086  val_auc=0.8600

Phase 2: Full fine-tuning — unfreezing backbone

Epoch 04/10  train_loss=0.4909  train_acc=0.7417  val_auc=0.9985
  ✓ New best AUC=0.9985 — checkpoint saved to checkpoints/best_model.pth
...
✓ Training complete. Best val AUC: 1.0000
✓ Checkpoint: checkpoints/best_model.pth
```

The AUC jump at epoch 4 is the two-phase fine-tuning effect — the backbone unfreezes and the model begins learning spatial deepfake artifacts beyond what the classifier head alone captures.

Training time: ~30-50 minutes on Mac CPU/MPS with ~700 samples.

### Step 4 — Run the benchmark

```bash
python evaluate.py \
  --data_dir ./data \
  --model_path checkpoints/best_model.pth
```

Expected output:
```
===========================================================================
  BENCHMARK RESULTS
===========================================================================
  Detector                       Accuracy  Precision   Recall      F1     AUC
  -------------------------------------------------------------------------
  FrequencyDetector                0.7169     0.7108   1.0000   0.8310  0.3438
  ModelBasedDetector (Efficien     0.9972     0.9960   1.0000   0.9980  1.0000
===========================================================================

Report written to: results/benchmark_report.md
JSON data written: results/benchmark_report.json
```

### Step 5 — View the full report

```bash
cat results/benchmark_report.md
```

---

## Understanding the Results

### FrequencyDetector
- Achieves ~72% accuracy with zero ML — purely FFT spectral analysis
- Perfect recall (catches every fake) but AUC of 0.34 — confidence scores are poorly calibrated
- Demonstrates that GAN upsampling artifacts are real and detectable, but not reliably rankable without learned features

### ModelBasedDetector (EfficientNet-B0)
- Fine-tuned via two-phase supervised learning on CelebDF frames
- Learns spatial artifacts — blending boundaries, skin texture inconsistencies — that FFT misses
- Near-perfect AUC on this dataset

### Important caveat
Results above are on ~700 samples. Val accuracy of 99%+ reflects overfitting on a small dataset, not generalization. The meaningful test is cross-dataset evaluation: train on CelebDF, evaluate on FaceForensics++ without any fine-tuning. That measures whether the detector learned general deepfake artifacts or just memorized this dataset's distribution.

---

## Detectors

### FrequencyDetector (`detectors/frequency.py`)
GAN generators use transposed convolution for upsampling, which creates periodic artifacts at specific spatial frequencies — invisible to the human eye but detectable via 2D FFT. This detector computes a High-Frequency Ratio (HFR) per image and applies a threshold auto-calibrated on the dataset to maximize F1. No training required.

### ModelBasedDetector (`detectors/model_based.py`)
EfficientNet-B0 pretrained on ImageNet, classifier head replaced for binary real/fake output. Fine-tuned in two phases: phase 1 trains only the head with backbone frozen; phase 2 unfreezes all layers with differential learning rates (backbone LR = head LR / 10). Best checkpoint saved by validation AUC-ROC.

---

## Metrics

| Metric | What it measures | Why it matters here |
|--------|-----------------|----------------|
| Accuracy | Overall correct classifications | Misleading on imbalanced data — avoid as primary metric |
| Precision | Of predicted fakes, how many were actually fake | Cost of false alarms |
| Recall | Of actual fakes, how many were caught | Cost of missed detections |
| F1 | Harmonic mean of precision and recall | Balanced view on imbalanced data |
| AUC-ROC | Discrimination ability across all thresholds | Best single metric for this problem |

AUC-ROC is used as the training checkpoint metric — not accuracy — because CelebDF has ~10x more fake samples than real, making accuracy an unreliable signal.

---

## Extending the Framework

To add a new detector, implement the `Detector` interface:

```python
from detectors.base import Detector

class MyDetector(Detector):
    @property
    def name(self) -> str:
        return "MyDetector"

    def predict(self, dataset) -> list[dict]:
        # return list of: {path, label, pred, confidence}
        ...
```

Then register it in `evaluate.py`:
```python
from detectors.my_detector import MyDetector
detectors = [FrequencyDetector(), ModelBasedDetector(...), MyDetector()]
```

---

## Known Limitations

- **Small dataset overfitting** — ~700 samples is insufficient for generalizable results. Full CelebDF-V2 with identity-level splits is needed for rigorous evaluation.
- **No identity-level splitting** — current split is video-level. Frames from the same identity could appear in both train and val, inflating metrics.
- **Frame-level detection only** — temporal inconsistencies across frames (blinking patterns, head pose jitter) are not captured. An LSTM or transformer over frame sequences is the natural extension.
- **Single dataset** — cross-dataset evaluation against FaceForensics++ or DFDC would test real generalization.

---

*Clark University Independent Research — Deepfake Detection on CelebDF-V2*

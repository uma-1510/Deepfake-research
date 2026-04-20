# Deepfake Detection Benchmark Suite
### Built on CelebDF Dataset | Clark University Research

A systematic evaluation framework for comparing deepfake detection algorithms — from classical signal-processing baselines to fine-tuned neural networks.

---

## Project Structure

```
deepfake_eval/
├── data/
│   ├── real/            ← real video frames (from CelebDF YouTube-real)
│   └── fake/            ← deepfake frames (from CelebDF synthesis set)
├── detectors/
│   ├── base.py          ← abstract Detector interface
│   ├── frequency.py     ← FFT-based spectral anomaly detector
│   └── model_based.py   ← fine-tuned EfficientNet-B0 detector
├── data_loader.py       ← CelebDF frame extraction + preprocessing
├── train.py             ← supervised fine-tuning pipeline
├── evaluate.py          ← benchmarks all detectors, computes metrics
├── metrics.py           ← accuracy, precision, recall, F1, AUC-ROC
└── report.py            ← JSON + markdown benchmark report
```

---

## Setup

```bash
pip install torch torchvision efficientnet-pytorch scikit-learn opencv-python numpy pillow tqdm
```

## Usage

### 1. Prepare your CelebDF data
Point `DATA_ROOT` in `data_loader.py` to your CelebDF directory. The loader expects:
```
CelebDF/
├── YouTube-real/videos/
├── Celeb-synthesis/videos/
└── List_of_testing_videos.txt
```

### 2. Extract frames
```bash
python data_loader.py --celeb_root /path/to/CelebDF --output_dir ./data --max_frames 30
```

### 3. Fine-tune the model
```bash
python train.py --data_dir ./data --epochs 10 --batch_size 32 --lr 1e-4
# Saves best model to: checkpoints/best_model.pth
```

### 4. Run the benchmark
```bash
python evaluate.py --data_dir ./data --model_path checkpoints/best_model.pth
```

### 5. View the report
Results are written to `results/benchmark_report.md` and `results/benchmark_report.json`.

---

## Detectors

### FrequencyDetector
Uses Fast Fourier Transform (FFT) to analyze spectral anomalies. GAN-generated faces leave characteristic high-frequency artifacts invisible to the human eye but detectable in the frequency domain. No ML required — serves as an interpretable baseline.

### ModelBasedDetector  
EfficientNet-B0 fine-tuned on CelebDF via supervised learning. Pre-trained on ImageNet, then adapted for the real/fake binary classification task. Learns spatial artifacts — blending boundaries, unnatural skin textures, temporal inconsistencies at the frame level.

---

## Metrics
| Metric | What it measures |
|--------|-----------------|
| Accuracy | Overall correct classifications |
| Precision | Of predicted fakes, how many were actually fake |
| Recall | Of actual fakes, how many did we catch |
| F1 | Harmonic mean of precision and recall |
| AUC-ROC | Model's ability to discriminate at all thresholds |

---

## Results (example output)
```
╔══════════════════════════════════════════════════════╗
║         DEEPFAKE DETECTION BENCHMARK RESULTS         ║
╠══════════════╦══════════╦═══════════╦════════╦═══════╣
║ Detector     ║ Accuracy ║ Precision ║ Recall ║ F1    ║
╠══════════════╬══════════╬═══════════╬════════╬═══════╣
║ Frequency    ║  0.623   ║   0.601   ║ 0.587  ║ 0.594 ║
║ ModelBased   ║  0.891   ║   0.887   ║ 0.904  ║ 0.895 ║
╚══════════════╩══════════╩═══════════╩════════╩═══════╝
```
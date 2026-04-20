"""
evaluate.py
Main benchmark runner — loads all detectors, runs them on the dataset,
computes metrics, and generates the comparison report.

Usage:
    # Both detectors
    python evaluate.py --data_dir ./data --model_path checkpoints/best_model.pth

    # Frequency detector only (no trained model needed)
    python evaluate.py --data_dir ./data --frequency_only
"""

import argparse
import os
from pathlib import Path

from data_loader import DeepfakeDataset, RAW_TRANSFORM, TRANSFORM
from detectors.frequency import FrequencyDetector
from detectors.model_based import ModelBasedDetector
from metrics import compute_all_metrics, find_worst_cases
from report import generate_report


def run_benchmark(args):
    print("\n" + "=" * 60)
    print("  DEEPFAKE DETECTION BENCHMARK")
    print("  Dataset: CelebDF  |  Clark University Research")
    print("=" * 60)

    results = {}

    # ── Frequency Detector ─────────────────────────────────────────────────────
    # Uses RAW_TRANSFORM (no normalization) — FFT works on raw pixel values
    freq_dataset = DeepfakeDataset(args.data_dir, transform=RAW_TRANSFORM, split="all")
    freq_detector = FrequencyDetector()
    freq_preds = freq_detector.predict(freq_dataset)

    labels      = [p["label"]      for p in freq_preds]
    predictions = [p["pred"]       for p in freq_preds]
    confidences = [p["confidence"] for p in freq_preds]

    results[freq_detector.name] = {
        "metrics":     compute_all_metrics(labels, predictions, confidences),
        "worst_cases": find_worst_cases(freq_preds),
        "total_samples": len(freq_preds),
    }
    print(f"\n✓ {freq_detector.name}: accuracy={results[freq_detector.name]['metrics']['accuracy']:.4f}")

    # ── Model-Based Detector ───────────────────────────────────────────────────
    if not args.frequency_only:
        if not args.model_path or not os.path.exists(args.model_path):
            print(f"\n[WARNING] Model checkpoint not found at '{args.model_path}'")
            print("  Run train.py first, or use --frequency_only to skip model inference.\n")
        else:
            model_dataset = DeepfakeDataset(args.data_dir, transform=TRANSFORM, split="all")
            model_detector = ModelBasedDetector(model_path=args.model_path)
            model_preds = model_detector.predict(model_dataset)

            labels      = [p["label"]      for p in model_preds]
            predictions = [p["pred"]       for p in model_preds]
            confidences = [p["confidence"] for p in model_preds]

            results[model_detector.name] = {
                "metrics":       compute_all_metrics(labels, predictions, confidences),
                "worst_cases":   find_worst_cases(model_preds),
                "total_samples": len(model_preds),
            }
            print(f"✓ {model_detector.name}: accuracy={results[model_detector.name]['metrics']['accuracy']:.4f}")

    # ── Generate Report ────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_report(results, output_dir=str(output_dir))

    print(f"\n{'='*60}")
    print(f"  Report written to: {output_dir}/benchmark_report.md")
    print(f"  JSON data written: {output_dir}/benchmark_report.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection Benchmark")
    parser.add_argument("--data_dir",       default="./data",                    help="Dir with real/ and fake/ subdirs")
    parser.add_argument("--model_path",     default="./checkpoints/best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--output_dir",     default="./results",                 help="Where to write report files")
    parser.add_argument("--frequency_only", action="store_true",                 help="Run only the frequency detector")
    args = parser.parse_args()

    run_benchmark(args)
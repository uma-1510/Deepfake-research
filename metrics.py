"""
metrics.py
Standard binary classification metrics for deepfake detection evaluation.

All functions take:
  labels:      list or array of ground truth labels  (0=real, 1=fake)
  predictions: list or array of predicted labels     (0=real, 1=fake)
  confidences: list or array of fake probabilities   (float 0.0–1.0)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def compute_all_metrics(labels, predictions, confidences) -> dict:
    """
    Compute the full suite of evaluation metrics.
    
    Returns a dict with:
      accuracy, precision, recall, f1, auc_roc,
      true_positives, true_negatives, false_positives, false_negatives,
      false_positive_rate, false_negative_rate,
      roc_curve (for plotting)
    """
    labels      = np.array(labels)
    predictions = np.array(predictions)
    confidences = np.array(confidences)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    # Core metrics
    accuracy  = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall    = recall_score(labels, predictions, zero_division=0)
    f1        = f1_score(labels, predictions, zero_division=0)

    # AUC-ROC — requires probability scores, not hard predictions
    try:
        auc = roc_auc_score(labels, confidences)
        fpr_curve, tpr_curve, thresholds = roc_curve(labels, confidences)
    except ValueError:
        auc = 0.0
        fpr_curve, tpr_curve, thresholds = [], [], []

    return {
        "accuracy":           round(float(accuracy),  4),
        "precision":          round(float(precision), 4),
        "recall":             round(float(recall),    4),
        "f1":                 round(float(f1),        4),
        "auc_roc":            round(float(auc),       4),
        "true_positives":     int(tp),
        "true_negatives":     int(tn),
        "false_positives":    int(fp),
        "false_negatives":    int(fn),
        "false_positive_rate": round(float(fp / (fp + tn + 1e-8)), 4),
        "false_negative_rate": round(float(fn / (fn + tp + 1e-8)), 4),
        "total_samples":      int(len(labels)),
        "roc_curve": {
            "fpr": fpr_curve.tolist() if len(fpr_curve) else [],
            "tpr": tpr_curve.tolist() if len(tpr_curve) else [],
        },
    }


def find_worst_cases(predictions_list: list[dict], n: int = 5) -> dict:
    """
    Find the samples where the model was most confidently wrong.
    Useful for error analysis in interviews — shows you understand failure modes.

    Returns:
      worst_false_positives: real images predicted as fake with highest confidence
      worst_false_negatives: fake images predicted as real with lowest confidence
    """
    false_positives = [
        p for p in predictions_list
        if p["label"] == 0 and p["pred"] == 1
    ]
    false_negatives = [
        p for p in predictions_list
        if p["label"] == 1 and p["pred"] == 0
    ]

    # Sort FP by confidence descending (most confidently wrong)
    false_positives.sort(key=lambda x: x["confidence"], reverse=True)
    # Sort FN by confidence ascending (most confidently missed)
    false_negatives.sort(key=lambda x: x["confidence"])

    return {
        "worst_false_positives": false_positives[:n],
        "worst_false_negatives": false_negatives[:n],
    }
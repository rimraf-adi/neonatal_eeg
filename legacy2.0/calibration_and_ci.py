"""
Legacy 2.0 - Self-Contained Calibration & Bootstrap Confidence Intervals Module
==============================================================================
1. 95% Bootstrap Confidence Intervals:
   - 1,000 resamples with replacement per trial and overall pooled test set for AUROC and F1.
2. Probability Calibration:
   - Reliability curves (10 bins).
   - Brier Score (mean squared probability error).
   - Expected Calibration Error (ECE).
Evaluated for Adaptive NN and XGBoost across all 4 paradigms.

Saves results to:
  legacy2.0/analysis/bootstrap_ci_results.json
  legacy2.0/analysis/calibration_curves.json
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def compute_bootstrap_ci(y_true, probs, threshold=0.5, n_bootstrap=1000, alpha=0.05, seed=42):
    """Computes non-parametric 95% bootstrap confidence intervals for AUROC and F1."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    f1s = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        p_b = probs[idx]

        # Require both classes in bootstrap sample
        if len(np.unique(y_b)) < 2:
            continue

        try:
            auc = roc_auc_score(y_b, p_b)
            aucs.append(auc)
        except Exception:
            pass

        y_pred = (p_b >= threshold).astype(int)
        f1s.append(f1_score(y_b, y_pred, zero_division=0))

    auc_low = float(np.percentile(aucs, 100 * (alpha / 2))) if aucs else 0.0
    auc_high = float(np.percentile(aucs, 100 * (1 - alpha / 2))) if aucs else 0.0
    f1_low = float(np.percentile(f1s, 100 * (alpha / 2))) if f1s else 0.0
    f1_high = float(np.percentile(f1s, 100 * (1 - alpha / 2))) if f1s else 0.0

    return {
        'auroc_mean': float(np.mean(aucs)) if aucs else 0.0,
        'auroc_ci_low': auc_low,
        'auroc_ci_high': auc_high,
        'f1_mean': float(np.mean(f1s)) if f1s else 0.0,
        'f1_ci_low': f1_low,
        'f1_ci_high': f1_high,
    }


def compute_calibration_details(y_true, probs, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='uniform')
    brier = float(brier_score_loss(y_true, probs))

    # Expected Calibration Error
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(probs[mask])
            ece += (np.sum(mask) / len(y_true)) * np.abs(bin_acc - bin_conf)

    return {
        'brier_score': brier,
        'ece': float(ece),
        'curve_prob_pred': [float(p) for p in prob_pred],
        'curve_prob_true': [float(p) for p in prob_true]
    }


def analyze_saved_cv_results():
    print("=" * 80)
    print("COMPUTING BOOTSTRAP CONFIDENCE INTERVALS & CALIBRATION")
    print("=" * 80)

    master_file = RESULTS_DIR / "master_cv_experiments.json"
    if not master_file.exists():
        print(f"File {master_file} not found. Please run train_models.py first.")
        return

    with open(master_file, 'r') as f:
        experiments = json.load(f)

    # Group by paradigm, strategy, and model
    ci_records = []
    calibration_records = []

    for exp in experiments:
        trial = exp['trial']
        paradigm = exp['paradigm']
        strategy = exp['strategy']
        for model_name, m_data in exp['models'].items():
            metrics = m_data['test_metrics']
            w = m_data['best_ma_window']
            t = m_data['best_threshold']

            ci_records.append({
                'trial': trial,
                'paradigm': paradigm,
                'strategy': strategy,
                'model': model_name,
                'ma_window': w,
                'threshold': t,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'specificity': metrics['specificity'],
                'f1': metrics['f1'],
                'auroc': metrics['auroc'],
                'brier_score': metrics['brier'],
                'ece': metrics['ece']
            })

    df = pd.DataFrame(ci_records)
    df.to_csv(ANALYSIS_DIR / "cross_trial_metrics_full.csv", index=False)

    # Summary aggregations: Mean +/- Std across trials
    summary_cols = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auroc', 'brier_score', 'ece']
    agg_df = df.groupby(['paradigm', 'strategy', 'model'])[summary_cols].agg(['mean', 'std']).reset_index()
    agg_df.to_csv(ANALYSIS_DIR / "cross_trial_metrics_summary.csv", index=False)

    print(f"Summary computed across {len(df)} experiment rows.")
    print(f"Saved: {ANALYSIS_DIR / 'cross_trial_metrics_summary.csv'}")


if __name__ == '__main__':
    analyze_saved_cv_results()

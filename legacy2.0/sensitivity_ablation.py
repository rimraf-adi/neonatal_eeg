"""
Legacy 2.0 - Self-Contained Sensitivity & Ablation Sweep Module
==============================================================
Systematically ablates:
1. Seizure to Non-Seizure Balancing Ratios:
   r in {1:1, 1.5:1, 2:1, 3:1, 4:1} vs WeightedRandomSampler (no downsampling) vs Natural Baseline.
2. PCA Dimensionality:
   k in {5, 10, 20, raw (no PCA)}.

Saves comprehensive results to:
  legacy2.0/analysis/ratio_ablation_results.json
  legacy2.0/analysis/pca_ablation_results.json
  legacy2.0/analysis/ablation_summary.csv
"""

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("sklearnex (oneDAL) acceleration active for scikit-learn models.")
except Exception:
    pass

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import torch

from train_models import (
    load_split_dataframe, sweep_ma_and_threshold, compute_metrics, moving_average,
    PARADIGMS, SPLITS_PATH, RESULTS_DIR
)

ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
TABLES_DIR = Path(__file__).resolve().parent / "tables"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

RATIOS = [1.0, 1.5, 2.0, 3.0, 4.0]
PCA_DIMS = [5, 10, 20, 'raw']


def evaluate_classifier_on_data(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """Fits XGBoost classifier and evaluates with validation-tuned MA window & threshold."""
    spw = float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1)
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric='logloss',
        tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu',
        random_state=seed
    )
    clf.fit(X_train, y_train)
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_res = sweep_ma_and_threshold(y_val, val_probs)
    best_w = val_res['f1']['w']
    best_t = val_res['f1']['t']

    test_probs = clf.predict_proba(X_test)[:, 1]
    from train_models import moving_average
    test_probs_smooth = moving_average(test_probs, best_w)
    return compute_metrics(y_test, test_probs_smooth, best_t), best_w, best_t


def run_ratio_ablation(splits):
    out_file = ANALYSIS_DIR / "ratio_ablation_results.json"
    if out_file.exists():
        print(f"\n>>> Ratio ablation already complete ({out_file.name}). Skipping. <<<")
        return

    print("=" * 80)
    print("RUNNING SEIZURE / NON-SEIZURE RATIO ABLATION")
    print(f"Ratios: {RATIOS} + WeightedRandomSampler + Natural")
    print("=" * 80)

    ratio_results = []
    # Test on a representative subset of 5 trials (or all 10)
    eval_trials = list(range(min(5, len(splits))))

    for paradigm in ['spectral_slope', 'fusion']:
        for r in RATIOS:
            trial_metrics = []
            for t_idx in eval_trials:
                split = splits[t_idx]
                train_df = load_split_dataframe(split['train_idx'], paradigm, apply_filter=True, target_ratio=r)
                val_df = load_split_dataframe(split['val_idx'], paradigm, apply_filter=True, target_ratio=r)
                test_df = load_split_dataframe(split['test_idx'], paradigm, apply_filter=True, target_ratio=r)

                if train_df.empty or val_df.empty or test_df.empty:
                    continue

                feat_cols = [c for c in train_df.columns if c not in ['label', 'channel']]
                X_tr, y_tr = train_df[feat_cols].values, train_df['label'].values
                X_v, y_v = val_df[feat_cols].values, val_df['label'].values
                X_te, y_te = test_df[feat_cols].values, test_df['label'].values

                imputer = SimpleImputer(strategy='mean')
                X_tr = imputer.fit_transform(X_tr)
                X_v = imputer.transform(X_v)
                X_te = imputer.transform(X_te)

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_v = scaler.transform(X_v)
                X_te = scaler.transform(X_te)

                pca = PCA(n_components=min(10, X_tr.shape[1]))
                X_tr = pca.fit_transform(X_tr)
                X_v = pca.transform(X_v)
                X_te = pca.transform(X_te)

                m, w, t = evaluate_classifier_on_data(X_tr, y_tr, X_v, y_v, X_te, y_te, seed=42 + t_idx)
                trial_metrics.append(m)

            if trial_metrics:
                mean_f1 = np.mean([m['f1'] for m in trial_metrics])
                mean_auc = np.mean([m['auroc'] for m in trial_metrics])
                mean_rec = np.mean([m['recall'] for m in trial_metrics])
                mean_spec = np.mean([m['specificity'] for m in trial_metrics])
                record = {
                    'paradigm': paradigm,
                    'ratio': f"{r:.1f}:1",
                    'f1': float(mean_f1),
                    'auroc': float(mean_auc),
                    'recall': float(mean_rec),
                    'specificity': float(mean_spec),
                    'raw_trials': trial_metrics
                }
                ratio_results.append(record)
                print(f"[{paradigm}] Ratio {r:.1f}:1 -> F1: {mean_f1:.4f} | AUROC: {mean_auc:.4f} | Rec: {mean_rec:.4f} | Spec: {mean_spec:.4f}")

    with open(ANALYSIS_DIR / "ratio_ablation_results.json", 'w') as f:
        json.dump(ratio_results, f, indent=2)


def run_pca_ablation(splits):
    out_file = ANALYSIS_DIR / "pca_ablation_results.json"
    sum_file = ANALYSIS_DIR / "pca_ablation_summary.csv"
    if out_file.exists() and sum_file.exists():
        print(f"\n>>> PCA dimensionality ablation already complete ({out_file.name}). Skipping. <<<")
        try:
            sum_df = pd.read_csv(sum_file)
            md_table = "# PCA Dimensionality Ablation (Spectral Slope, All Classifiers)\n\n"
            md_table += sum_df[['PCA_Components', 'Model', 'AUROC', 'F1', 'Recall', 'Specificity']].to_markdown(index=False)
            with open(TABLES_DIR / "pca_ablation_table.md", 'w') as f:
                f.write(md_table)
        except Exception as e:
            print(f"Markdown table write warning: {e}")
        return

    print("\n" + "=" * 80)
    print("RUNNING PCA DIMENSIONALITY ABLATION (ALL 4 CLASSIFIERS)")
    print("Spectral Slope Paradigm: k in {5, 10, raw}")
    print("=" * 80)

    eval_trials = list(range(min(5, len(splits))))
    pca_dims = [5, 10, 'raw']
    pca_results = []
    summary_rows = []

    # Cache split data so we only load it once per trial!
    cached_splits = []
    for t_idx in eval_trials:
        split = splits[t_idx]
        train_df = load_split_dataframe(split['train_idx'], 'spectral_slope', apply_filter=True, target_ratio=2.0)
        val_df = load_split_dataframe(split['val_idx'], 'spectral_slope', apply_filter=True, target_ratio=2.0)
        test_df = load_split_dataframe(split['test_idx'], 'spectral_slope', apply_filter=True, target_ratio=2.0)
        feat_cols = [c for c in train_df.columns if c not in ['label', 'channel']]
        cached_splits.append((train_df, val_df, test_df, feat_cols))

    for k in pca_dims:
        print(f"\n--- Testing PCA Component Setting: {k} ---")
        trial_records = []
        for t_idx in eval_trials:
            train_df, val_df, test_df, feat_cols = cached_splits[t_idx]
            X_tr, y_tr = train_df[feat_cols].values, train_df['label'].values
            X_v, y_v = val_df[feat_cols].values, val_df['label'].values
            X_te, y_te = test_df[feat_cols].values, test_df['label'].values

            imputer = SimpleImputer(strategy='mean')
            X_tr = imputer.fit_transform(X_tr)
            X_v = imputer.transform(X_v)
            X_te = imputer.transform(X_te)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_v = scaler.transform(X_v)
            X_te = scaler.transform(X_te)

            if k != 'raw':
                n_comp = min(int(k), X_tr.shape[1])
                pca = PCA(n_components=n_comp)
                X_tr = pca.fit_transform(X_tr)
                X_v = pca.transform(X_v)
                X_te = pca.transform(X_te)

            m_all = evaluate_all_classifiers_on_features(X_tr, y_tr, X_v, y_v, X_te, y_te, seed=42 + t_idx)
            trial_records.append({'trial': t_idx + 1, 'models': m_all})

        pca_results.append({
            'paradigm': 'spectral_slope',
            'pca_components': str(k),
            'trials': trial_records
        })

        for model_name in ['AdaptiveNN', 'XGBoost', 'LogisticRegression', 'RandomForest']:
            f1s = [tr['models'][model_name]['f1'] for tr in trial_records if tr['models'].get(model_name)]
            aucs = [tr['models'][model_name]['auroc'] for tr in trial_records if tr['models'].get(model_name)]
            recs = [tr['models'][model_name]['recall'] for tr in trial_records if tr['models'].get(model_name)]
            specs = [tr['models'][model_name]['specificity'] for tr in trial_records if tr['models'].get(model_name)]

            if f1s:
                summary_rows.append({
                    'PCA_Components': str(k),
                    'Model': model_name,
                    'AUROC': f"{np.mean(aucs):.3f} +/- {np.std(aucs):.3f}",
                    'F1': f"{np.mean(f1s):.3f} +/- {np.std(f1s):.3f}",
                    'Recall': f"{np.mean(recs):.3f} +/- {np.std(recs):.3f}",
                    'Specificity': f"{np.mean(specs):.3f} +/- {np.std(specs):.3f}",
                    'mean_auroc': float(np.mean(aucs)),
                    'mean_f1': float(np.mean(f1s))
                })
                print(f"  {model_name:18s} | F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f} | AUROC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

    with open(ANALYSIS_DIR / "pca_ablation_results.json", 'w') as f:
        json.dump(pca_results, f, indent=2)

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(ANALYSIS_DIR / "pca_ablation_summary.csv", index=False)

    md_table = "# PCA Dimensionality Ablation (Spectral Slope, All Classifiers)\n\n"
    md_table += sum_df[['PCA_Components', 'Model', 'AUROC', 'F1', 'Recall', 'Specificity']].to_markdown(index=False)
    with open(TABLES_DIR / "pca_ablation_table.md", 'w') as f:
        f.write(md_table)


def evaluate_all_classifiers_on_features(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """Evaluates all 4 classifiers (Adaptive NN, XGBoost, LogReg, RF) on the given feature matrix."""
    from sklearn.utils.class_weight import compute_class_weight
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from train_models import AdaptiveNeuralNet, EEGDataset, train_neural_net, predict_nn, DEVICE

    results = {}

    # 1. Adaptive NN (PyTorch MLP on CUDA)
    try:
        train_ds = EEGDataset(X_train, y_train)
        val_ds = EEGDataset(X_val, y_val)
        test_ds = EEGDataset(X_test, y_test)

        cls_w = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, dtype=torch.float32).to(DEVICE))
        train_loader = DataLoader(train_ds, batch_size=32768, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=65536, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=65536, shuffle=False)

        mlp = AdaptiveNeuralNet(input_dim=X_train.shape[1]).to(DEVICE)
        optimizer = optim.Adam(mlp.parameters(), lr=0.003)
        mlp = train_neural_net(mlp, train_loader, val_loader, criterion, optimizer, epochs=25, patience=6)

        val_probs_nn = predict_nn(mlp, val_loader)
        val_res_nn = sweep_ma_and_threshold(y_val, val_probs_nn)
        test_probs_nn = predict_nn(mlp, test_loader)
        test_probs_smooth_nn = moving_average(test_probs_nn, val_res_nn['f1']['w'])
        results['AdaptiveNN'] = compute_metrics(y_test, test_probs_smooth_nn, val_res_nn['f1']['t'])
    except Exception as e:
        print(f"    AdaptiveNN Error: {e}")
        results['AdaptiveNN'] = None

    # 2. XGBoost (CUDA)
    try:
        spw = float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1)
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric='logloss',
            tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=seed
        )
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_res = sweep_ma_and_threshold(y_val, val_probs)
        test_probs = clf.predict_proba(X_test)[:, 1]
        test_probs_smooth = moving_average(test_probs, val_res['f1']['w'])
        results['XGBoost'] = compute_metrics(y_test, test_probs_smooth, val_res['f1']['t'])
    except Exception as e:
        print(f"    XGBoost Error: {e}")
        results['XGBoost'] = None

    # 3. Logistic Regression
    try:
        lr = LogisticRegression(class_weight='balanced', max_iter=500, random_state=seed)
        lr.fit(X_train, y_train)
        val_probs_lr = lr.predict_proba(X_val)[:, 1]
        val_res_lr = sweep_ma_and_threshold(y_val, val_probs_lr)
        test_probs_lr = lr.predict_proba(X_test)[:, 1]
        test_probs_smooth_lr = moving_average(test_probs_lr, val_res_lr['f1']['w'])
        results['LogisticRegression'] = compute_metrics(y_test, test_probs_smooth_lr, val_res_lr['f1']['t'])
    except Exception as e:
        print(f"    LogReg Error: {e}")
        results['LogisticRegression'] = None

    # 4. Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=seed)
        rf.fit(X_train, y_train)
        val_probs_rf = rf.predict_proba(X_val)[:, 1]
        val_res_rf = sweep_ma_and_threshold(y_val, val_probs_rf)
        test_probs_rf = rf.predict_proba(X_test)[:, 1]
        test_probs_smooth_rf = moving_average(test_probs_rf, val_res_rf['f1']['w'])
        results['RandomForest'] = compute_metrics(y_test, test_probs_smooth_rf, val_res_rf['f1']['t'])
    except Exception as e:
        print(f"    RandomForest Error: {e}")
        results['RandomForest'] = None

    return results


def run_frequency_feature_ablation(splits):
    print("\n" + "=" * 80)
    print("RUNNING FEATURE-WISE ABLATION FOR FREQUENCY FEATURES (ALL 4 CLASSIFIERS)")
    print("Isolated Sub-Bands & Feature Types vs. Full Baseline")
    print("=" * 80)

    eval_trials = list(range(min(5, len(splits))))
    ablation_configs = [
        ('Full (12 feats)', None),
        ('Delta only (3 feats)', ['delta_slope', 'delta_intercept', 'delta_midband']),
        ('Theta only (3 feats)', ['theta_slope', 'theta_intercept', 'theta_midband']),
        ('Alpha only (3 feats)', ['alpha_slope', 'alpha_intercept', 'alpha_midband']),
        ('Beta only (3 feats)', ['beta_slope', 'beta_intercept', 'beta_midband']),
        ('Slope only (4 feats)', ['delta_slope', 'theta_slope', 'alpha_slope', 'beta_slope']),
        ('Intercept only (4 feats)', ['delta_intercept', 'theta_intercept', 'alpha_intercept', 'beta_intercept']),
        ('Midband only (4 feats)', ['delta_midband', 'theta_midband', 'alpha_midband', 'beta_midband']),
    ]

    # Pre-cache the 5 trials in memory to avoid redundant disk I/O
    print("Pre-caching spectral slope evaluation trials in memory...")
    cached_splits = []
    for t_idx in eval_trials:
        split = splits[t_idx]
        tr_df = load_split_dataframe(split['train_idx'], 'spectral_slope', apply_filter=True, target_ratio=2.0)
        va_df = load_split_dataframe(split['val_idx'], 'spectral_slope', apply_filter=True, target_ratio=2.0)
        te_df = load_split_dataframe(split['test_idx'], 'spectral_slope', apply_filter=True, target_ratio=2.0)
        cached_splits.append((tr_df, va_df, te_df))

    all_ablation_results = []
    summary_rows = []

    for cfg_name, cols_to_use in ablation_configs:
        print(f"\n--- Testing Configuration: {cfg_name} ---")
        trial_records = []

        for t_idx in eval_trials:
            train_df, val_df, test_df = cached_splits[t_idx]

            if train_df.empty or val_df.empty or test_df.empty:
                continue

            if cols_to_use is None:
                feat_cols = [c for c in train_df.columns if c not in ['label', 'channel']]
            else:
                feat_cols = [c for c in cols_to_use if c in train_df.columns]

            X_tr, y_tr = train_df[feat_cols].values, train_df['label'].values
            X_v, y_v = val_df[feat_cols].values, val_df['label'].values
            X_te, y_te = test_df[feat_cols].values, test_df['label'].values

            imputer = SimpleImputer(strategy='mean')
            X_tr = imputer.fit_transform(X_tr)
            X_v = imputer.transform(X_v)
            X_te = imputer.transform(X_te)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_v = scaler.transform(X_v)
            X_te = scaler.transform(X_te)

            m_all = evaluate_all_classifiers_on_features(X_tr, y_tr, X_v, y_v, X_te, y_te, seed=42 + t_idx)
            trial_records.append({'trial': t_idx + 1, 'models': m_all})

        all_ablation_results.append({
            'configuration': cfg_name,
            'features': feat_cols,
            'n_features': len(feat_cols),
            'trials': trial_records
        })

        # Summarize across trials for each model
        for model_name in ['AdaptiveNN', 'XGBoost', 'LogisticRegression', 'RandomForest']:
            f1s = [tr['models'][model_name]['f1'] for tr in trial_records if tr['models'].get(model_name)]
            aucs = [tr['models'][model_name]['auroc'] for tr in trial_records if tr['models'].get(model_name)]
            recs = [tr['models'][model_name]['recall'] for tr in trial_records if tr['models'].get(model_name)]
            specs = [tr['models'][model_name]['specificity'] for tr in trial_records if tr['models'].get(model_name)]

            if f1s:
                summary_rows.append({
                    'Configuration': cfg_name,
                    'Features': len(feat_cols),
                    'Model': model_name,
                    'AUROC': f"{np.mean(aucs):.3f} +/- {np.std(aucs):.3f}",
                    'F1': f"{np.mean(f1s):.3f} +/- {np.std(f1s):.3f}",
                    'Recall': f"{np.mean(recs):.3f} +/- {np.std(recs):.3f}",
                    'Specificity': f"{np.mean(specs):.3f} +/- {np.std(specs):.3f}",
                    'mean_auroc': float(np.mean(aucs)),
                    'mean_f1': float(np.mean(f1s))
                })
                print(f"  {model_name:18s} | F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f} | AUROC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

    # Save JSON results
    with open(ANALYSIS_DIR / "frequency_feature_ablation.json", 'w') as f:
        json.dump(all_ablation_results, f, indent=2)

    # Save CSV and Markdown table
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(ANALYSIS_DIR / "frequency_feature_ablation_summary.csv", index=False)

    display_cols = ['Configuration', 'Features', 'Model', 'AUROC', 'F1', 'Recall', 'Specificity']
    md_table = "# Frequency Feature Ablation Study (All Classifiers)\n\n"
    md_table += "Systematic evaluation of individual sub-bands and parameter types versus the full 12-feature baseline:\n\n"
    try:
        md_table += sum_df[display_cols].to_markdown(index=False)
    except Exception:
        md_table += sum_df[display_cols].to_csv(sep='|', index=False)
    with open(TABLES_DIR / "frequency_feature_ablation_table.md", 'w') as f:
        f.write(md_table)

    print(f"\nSaved feature ablation results to {ANALYSIS_DIR / 'frequency_feature_ablation_summary.csv'}")
    print(f"Saved Markdown table to {TABLES_DIR / 'frequency_feature_ablation_table.md'}")


def main():
    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)
    run_ratio_ablation(splits)
    run_pca_ablation(splits)
    run_frequency_feature_ablation(splits)


if __name__ == '__main__':
    main()

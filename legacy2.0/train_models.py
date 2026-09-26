"""
Legacy 2.0 - Self-Contained Model Training & Cross-Validation Module
===================================================================
Executes 10-trial cross-validation across 4 feature paradigms:
  1. Spectral Slope (12 features -> PCA 10)
  2. DWT Wavelets (16 features -> PCA 10)
  3. EMD (16 features -> PCA 10)
  4. Feature Fusion (44 features -> PCA 10)

Evaluates 4 classifiers per trial:
  1. Adaptive NN (PyTorch MLP with BatchNorm, Dropout, Early Stopping, CUDA acceleration)
  2. Logistic Regression (L2, class_weight='balanced')
  3. Random Forest (100 trees, balanced subsampling)
  4. XGBoost (100 rounds, GPU histogram tree method)

Implements two decoupled data sampling strategies:
  - Strategy 1: Offline Temporal Nearest-Neighbor Downsampling (r=2.0)
  - Strategy 2: Online Weighted Random Sampling (no downsampling, 100% data, WeightedRandomSampler)

Computes comprehensive metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, AUROC,
Brier score, ECE, optimal MA window and threshold.
"""

import os
import sys
import json
import time
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Directory setup
ROOT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT_DIR / "legacy2.0" / "features"
RESULTS_DIR = ROOT_DIR / "legacy2.0" / "results"
SPLITS_PATH = ROOT_DIR / "patient_splits.json"

PARADIGMS = ['spectral_slope', 'wavelet', 'emd', 'fusion']
MA_WINDOW_SIZES = list(range(1, 21))
PROB_THRESHOLDS = np.arange(0.05, 0.96, 0.01)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# PyTorch Dataset & MLP Architecture
# ============================================================================

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AdaptiveNeuralNet(nn.Module):
    def __init__(self, input_dim=10):
        super(AdaptiveNeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Temporal Downsampling & Data Ingestion
# ============================================================================

def apply_adaptive_filter(df, target_ratio=2.0):
    """
    Vectorized temporal nearest-neighbor class balancing:
    1. Keeps all seizure epochs (label=1).
    2. Selects N non-seizure epochs (label=0) temporally closest to seizures.
       N = target_ratio * num_seizures.
    Uses O(N log M) np.searchsorted for instant evaluation.
    """
    labels = df['label'].values
    seizure_indices = np.where(labels == 1)[0]
    num_seizures = len(seizure_indices)

    if num_seizures == 0:
        return df.iloc[[]].copy()

    non_seizure_indices = np.where(labels == 0)[0]
    if len(non_seizure_indices) == 0:
        return df.iloc[seizure_indices].copy()

    # O(N log M) distance calculation
    idx = np.searchsorted(seizure_indices, non_seizure_indices)
    idx_left = np.clip(idx - 1, 0, num_seizures - 1)
    idx_right = np.clip(idx, 0, num_seizures - 1)
    min_distances = np.minimum(
        np.abs(non_seizure_indices - seizure_indices[idx_left]),
        np.abs(non_seizure_indices - seizure_indices[idx_right])
    )

    n_to_select = int(target_ratio * num_seizures)
    n_to_select = min(n_to_select, len(non_seizure_indices))

    if n_to_select < len(non_seizure_indices):
        closest_part = np.argpartition(min_distances, n_to_select - 1)[:n_to_select]
        selected_ns = non_seizure_indices[closest_part]
    else:
        selected_ns = non_seizure_indices

    indices_to_keep = np.sort(np.concatenate([seizure_indices, selected_ns]))
    return df.iloc[indices_to_keep].copy()


def load_split_dataframe(patient_ids, paradigm, apply_filter=True, target_ratio=2.0):
    """Loads and concatenates patient CSVs for a split."""
    feat_dir = FEATURES_DIR / paradigm
    frames = []
    for pid in patient_ids:
        csv_file = feat_dir / f"patient_{pid:03d}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            if apply_filter:
                df = apply_adaptive_filter(df, target_ratio=target_ratio)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# Post-Processing & Sweep Utilities
# ============================================================================

def moving_average(probs, window_size):
    if window_size <= 1:
        return probs
    return np.convolve(probs, np.ones(window_size) / window_size, mode='same')


def compute_metrics(y_true, probs_smoothed, threshold):
    y_pred = (probs_smoothed >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / max(tn + fp, 1)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, probs_smoothed)
    except Exception:
        auroc = 0.5
    brier = brier_score_loss(y_true, probs_smoothed)

    # Expected Calibration Error (ECE) with 10 bins
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        mask = (probs_smoothed >= bins[i]) & (probs_smoothed < bins[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(probs_smoothed[mask])
            ece += (np.sum(mask) / len(y_true)) * np.abs(bin_acc - bin_conf)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'auroc': float(auroc),
        'brier': float(brier),
        'ece': float(ece),
        'confusion_matrix': cm.tolist()
    }


def sweep_ma_and_threshold(y_true, probs):
    """Vectorized MA window (1-20) and threshold (0.05-0.95) sweep."""
    y_true = np.asarray(y_true, dtype=int)
    pos = int(np.sum(y_true == 1))
    neg = len(y_true) - pos

    best_results = {
        'f1': {'val': -1.0, 'w': 1, 't': 0.5, 'metrics': None},
        'auroc': {'val': -1.0, 'w': 1, 't': 0.5, 'metrics': None}
    }

    for w in MA_WINDOW_SIZES:
        p_smooth = moving_average(probs, w) if w > 1 else probs
        try:
            auroc = float(roc_auc_score(y_true, p_smooth))
        except Exception:
            auroc = 0.5
        brier = float(brier_score_loss(y_true, p_smooth))

        if auroc > best_results['auroc']['val']:
            best_results['auroc']['val'] = auroc
            best_results['auroc']['w'] = w

        for t in PROB_THRESHOLDS:
            pred_pos = (p_smooth >= t)
            tp = int(np.count_nonzero(pred_pos & (y_true == 1)))
            fp = int(np.count_nonzero(pred_pos & (y_true == 0)))
            fn = pos - tp
            tn = neg - fp

            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
            acc = (tp + tn) / max(pos + neg, 1)

            if f1 > best_results['f1']['val']:
                best_results['f1'] = {
                    'val': float(f1),
                    'w': w,
                    't': round(float(t), 2),
                    'metrics': {
                        'accuracy': float(acc),
                        'precision': float(prec),
                        'recall': float(rec),
                        'specificity': float(spec),
                        'f1': float(f1),
                        'auroc': float(auroc),
                        'brier': float(brier),
                        'confusion_matrix': [[tn, fp], [fn, tp]]
                    }
                }
    return best_results


# ============================================================================
# Neural Net Training Loop
# ============================================================================

def train_neural_net(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=10):
    best_val_loss = float('inf')
    patience_cnt = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                out = model(X_b)
                val_loss += criterion(out, y_b).item()

        val_loss /= max(len(val_loader), 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    return model


def predict_nn(model, data_loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_b, _ in data_loader:
            X_b = X_b.to(DEVICE)
            out = model(X_b)
            probs = torch.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


# ============================================================================
# Single Trial Execution
# ============================================================================

def run_single_trial(trial_idx, split_info, paradigm, strategy='downsample_2x', n_pca=10):
    """
    Executes a single cross-validation trial for a given paradigm and strategy.
    strategy: 'downsample_2x' (Strategy 1) or 'weighted_sampler' (Strategy 2)
    """
    seed = 42 + trial_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_pids = split_info['train_idx']
    val_pids = split_info['val_idx']
    test_pids = split_info['test_idx']

    use_downsample = (strategy == 'downsample_2x')
    target_r = 2.0 if use_downsample else 1.0

    train_df = load_split_dataframe(train_pids, paradigm, apply_filter=use_downsample, target_ratio=target_r)
    val_df = load_split_dataframe(val_pids, paradigm, apply_filter=use_downsample, target_ratio=target_r)
    test_df = load_split_dataframe(test_pids, paradigm, apply_filter=use_downsample, target_ratio=target_r)

    if train_df.empty or val_df.empty or test_df.empty:
        print(f"Warning: Empty split for Trial {trial_idx+1}, Paradigm {paradigm}")
        return None

    feat_cols = [c for c in train_df.columns if c not in ['label', 'channel']]

    X_train, y_train = train_df[feat_cols].values, train_df['label'].values
    X_val, y_val = val_df[feat_cols].values, val_df['label'].values
    X_test, y_test = test_df[feat_cols].values, test_df['label'].values

    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # PCA reduction
    n_comp = min(n_pca, X_train.shape[1])
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    trial_results = {
        'trial': trial_idx + 1,
        'paradigm': paradigm,
        'strategy': strategy,
        'pca_components': n_comp,
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'models': {}
    }

    # ------------------------------------------------------------------------
    # 1. Adaptive NN (PyTorch MLP)
    # ------------------------------------------------------------------------
    train_ds = EEGDataset(X_train_pca, y_train)
    val_ds = EEGDataset(X_val_pca, y_val)
    test_ds = EEGDataset(X_test_pca, y_test)
    BATCH_SIZE = 32768
    if strategy == 'weighted_sampler':
        class_counts = Counter(y_train)
        weights = [1.0 / max(class_counts[y], 1) for y in y_train]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
    else:
        cls_w = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, dtype=torch.float32).to(DEVICE))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=65536, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=65536, shuffle=False)

    mlp = AdaptiveNeuralNet(input_dim=n_comp).to(DEVICE)
    optimizer = optim.Adam(mlp.parameters(), lr=0.003)
    mlp = train_neural_net(mlp, train_loader, val_loader, criterion, optimizer, epochs=50, patience=10)

    # Evaluate validation for optimal MA window & threshold
    val_probs_nn = predict_nn(mlp, val_loader)
    val_sweep_nn = sweep_ma_and_threshold(y_val, val_probs_nn)
    best_w = val_sweep_nn['f1']['w']
    best_t = val_sweep_nn['f1']['t']

    # Apply optimal hyperparameters on test set
    test_probs_nn = predict_nn(mlp, test_loader)
    test_probs_smooth_nn = moving_average(test_probs_nn, best_w)
    test_metrics_nn = compute_metrics(y_test, test_probs_smooth_nn, best_t)

    trial_results['models']['AdaptiveNN'] = {
        'best_ma_window': best_w,
        'best_threshold': best_t,
        'test_metrics': test_metrics_nn
    }

    # ------------------------------------------------------------------------
    # 2. Logistic Regression
    # ------------------------------------------------------------------------
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)
    lr.fit(X_train_pca, y_train)
    val_probs_lr = lr.predict_proba(X_val_pca)[:, 1]
    val_sweep_lr = sweep_ma_and_threshold(y_val, val_probs_lr)
    test_probs_lr = lr.predict_proba(X_test_pca)[:, 1]
    test_probs_smooth_lr = moving_average(test_probs_lr, val_sweep_lr['f1']['w'])
    trial_results['models']['LogisticRegression'] = {
        'best_ma_window': val_sweep_lr['f1']['w'],
        'best_threshold': val_sweep_lr['f1']['t'],
        'test_metrics': compute_metrics(y_test, test_probs_smooth_lr, val_sweep_lr['f1']['t'])
    }

    # ------------------------------------------------------------------------
    # 3. Random Forest
    # ------------------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=seed)
    rf.fit(X_train_pca, y_train)
    val_probs_rf = rf.predict_proba(X_val_pca)[:, 1]
    val_sweep_rf = sweep_ma_and_threshold(y_val, val_probs_rf)
    test_probs_rf = rf.predict_proba(X_test_pca)[:, 1]
    test_probs_smooth_rf = moving_average(test_probs_rf, val_sweep_rf['f1']['w'])
    trial_results['models']['RandomForest'] = {
        'best_ma_window': val_sweep_rf['f1']['w'],
        'best_threshold': val_sweep_rf['f1']['t'],
        'test_metrics': compute_metrics(y_test, test_probs_smooth_rf, val_sweep_rf['f1']['t'])
    }

    # ------------------------------------------------------------------------
    # 4. XGBoost
    # ------------------------------------------------------------------------
    spw = float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric='logloss',
        tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu',
        random_state=seed
    )
    xgb_clf.fit(X_train_pca, y_train)
    val_probs_xgb = xgb_clf.predict_proba(X_val_pca)[:, 1]
    val_sweep_xgb = sweep_ma_and_threshold(y_val, val_probs_xgb)
    test_probs_xgb = xgb_clf.predict_proba(X_test_pca)[:, 1]
    test_probs_smooth_xgb = moving_average(test_probs_xgb, val_sweep_xgb['f1']['w'])
    trial_results['models']['XGBoost'] = {
        'best_ma_window': val_sweep_xgb['f1']['w'],
        'best_threshold': val_sweep_xgb['f1']['t'],
        'test_metrics': compute_metrics(y_test, test_probs_smooth_xgb, val_sweep_xgb['f1']['t'])
    }

    return trial_results


# ============================================================================
# Master Cross-Validation Runner
# ============================================================================

def run_all_cross_validations():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)

    all_experiments = []
    strategies = ['downsample_2x', 'weighted_sampler']

    print("=" * 80)
    print("LEGACY 2.0: 10-TRIAL CROSS-VALIDATION MATRIX")
    print(f"Paradigms: {PARADIGMS}")
    print(f"Strategies: {strategies}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    start_time = time.time()

    for paradigm in PARADIGMS:
        for strategy in strategies:
            strat_name = "Downsample (2:1)" if strategy == 'downsample_2x' else "WeightedRandomSampler"
            out_file = RESULTS_DIR / f"{paradigm}_{strategy}_cv_results.json"
            if out_file.exists():
                print(f"\n>>> Paradigm: {paradigm.upper()} | Strategy: {strat_name} already completed. Skipping. <<<")
                with open(out_file, 'r') as f:
                    paradigm_results = json.load(f)
                all_experiments.extend(paradigm_results)
                continue

            print(f"\n>>> Running Paradigm: {paradigm.upper()} | Strategy: {strat_name} <<<")
            paradigm_results = []

            for trial_idx in range(len(splits)):
                t0 = time.time()
                print(f"  [Trial {trial_idx+1:02d}/10] Training 4 models...", end='', flush=True)
                res = run_single_trial(trial_idx, splits[trial_idx], paradigm, strategy=strategy)
                if res is not None:
                    paradigm_results.append(res)
                    all_experiments.append(res)
                    dt = time.time() - t0
                    f1_nn = res['models']['AdaptiveNN']['test_metrics']['f1']
                    auc_nn = res['models']['AdaptiveNN']['test_metrics']['auroc']
                    f1_xgb = res['models']['XGBoost']['test_metrics']['f1']
                    auc_xgb = res['models']['XGBoost']['test_metrics']['auroc']
                    print(f" Done ({dt:.1f}s) | MLP [F1={f1_nn:.3f}, AUC={auc_nn:.3f}] | XGB [F1={f1_xgb:.3f}, AUC={auc_xgb:.3f}]")
                    try:
                        from telegram_notifier import notify_rich_benchmark
                        notify_rich_benchmark(paradigm, strategy, trial_idx, len(splits), res['models'], dt)
                    except Exception:
                        pass

            # Save per-paradigm / strategy JSON
            out_file = RESULTS_DIR / f"{paradigm}_{strategy}_cv_results.json"
            with open(out_file, 'w') as f:
                json.dump(paradigm_results, f, indent=2)

            # Telegram update
            try:
                from telegram_notifier import notify_status
                mean_f1_nn = np.mean([r['models']['AdaptiveNN']['test_metrics']['f1'] for r in paradigm_results])
                mean_auc_nn = np.mean([r['models']['AdaptiveNN']['test_metrics']['auroc'] for r in paradigm_results])
                mean_f1_xgb = np.mean([r['models']['XGBoost']['test_metrics']['f1'] for r in paradigm_results])
                mean_auc_xgb = np.mean([r['models']['XGBoost']['test_metrics']['auroc'] for r in paradigm_results])
                notify_status(f"10-Trial CV Complete: {paradigm.upper()} ({strat_name})", {
                    "Adaptive NN (MLP)": f"F1={mean_f1_nn:.3f} | AUC={mean_auc_nn:.3f}",
                    "XGBoost": f"F1={mean_f1_xgb:.3f} | AUC={mean_auc_xgb:.3f}",
                    "Trials": f"{len(paradigm_results)}/10"
                })
            except Exception:
                pass

    # Save master raw summary JSON
    with open(RESULTS_DIR / "master_cv_experiments.json", 'w') as f:
        json.dump(all_experiments, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"All 10-trial cross-validations completed in {elapsed/60:.2f} minutes!")
    print(f"Results saved to {RESULTS_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    run_all_cross_validations()

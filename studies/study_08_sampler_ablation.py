"""
Study 8: WeightedRandomSampler Ratio Ablation
===============================================

Trains the MLP architecture from freq2.py using PyTorch's
WeightedRandomSampler at 6 different seizure:non-seizure sampling
ratios across all 10 trials. No dataset pre-filtering — the full
unfiltered dataset is used with sampling controlling class balance.

Sampling ratios tested:
  1. 50:50 (dominant) — equal class sampling
  2. 60:40 — slight seizure over-representation
  3. 40:60 — mild non-seizure bias
  4. 30:70 — moderate natural shift
  5. 20:80 — near-natural
  6. Natural (unweighted shuffle) — baseline

Output:
  study_results/08_sampler_ablation/
    sampler_results.csv
    sampler_summary.txt
    recall_vs_ratio.png
    f1_vs_ratio.png
    metrics_heatmap.png

Usage:
  uv run python -m studies.study_08_sampler_ablation
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score,
)
import plotly.graph_objects as go
import plotly.io as pio

from studies.utils import (
    load_patient_splits, load_patient_features, get_feature_columns,
    preprocess_features,
    check_cache, ensure_dir, print_header, print_subheader,
    RESULTS_BASE, SPECTRAL_CACHE_DIR,
)

OUTPUT_DIR = os.path.join(RESULTS_BASE, '08_sampler_ablation')

# Sampling ratios: (name, seizure_weight_multiplier)
# None means no weighted sampling (natural distribution with shuffle)
SAMPLING_RATIOS = [
    ('50:50',    1.0),    # Equal weight -> 50:50 effective sampling
    ('60:40',    1.5),    # 1.5x seizure -> ~60:40
    ('40:60',    0.67),   # 0.67x seizure -> ~40:60
    ('30:70',    0.43),   # ~30:70
    ('20:80',    0.25),   # ~20:80
    ('Natural',  None),   # No sampler, regular shuffle
]


# =============================================================================
# MLP Architecture  (exact replica of freq2.py NeuralNet)
# =============================================================================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNet(nn.Module):
    """Exact replica of freq2.py NeuralNet."""
    def __init__(self, input_dim):
        super().__init__()
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
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Training  (adapted from freq2.py)
# =============================================================================

def get_sample_weights(labels, seizure_weight_mult):
    """Compute 1D sample weights tensor for WeightedRandomSampler / multinomial."""
    if seizure_weight_mult is None:
        return torch.ones(len(labels), dtype=torch.float64)
    class_counts = np.bincount(labels)
    base_weights = 1.0 / class_counts.astype(float)
    base_weights[1] *= seizure_weight_mult
    base_weights = base_weights / base_weights.sum()
    return torch.tensor(base_weights[labels], dtype=torch.float64)


def train_model(model, X_train_t, y_train_t, sample_weights, X_val_t, y_val_t, criterion, optimizer, device, epochs=50, patience=10, batch_size=1024):
    """Train with early stopping on GPU tensors."""
    best_val_loss = float('inf')
    patience_counter = 0
    num_samples = min(100000, len(y_train_t))

    for epoch in range(epochs):
        model.train()
        indices = torch.multinomial(sample_weights, num_samples, replacement=True).to(device)
        epoch_X = X_train_t[indices]
        epoch_y = y_train_t[indices]

        perm = torch.randperm(num_samples, device=device)
        epoch_X = epoch_X[perm]
        epoch_y = epoch_y[perm]

        for i in range(0, num_samples, batch_size):
            xb = epoch_X[i:i+batch_size]
            yb = epoch_y[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for i in range(0, len(X_val_t), 8192):
                xb = X_val_t[i:i+8192]
                yb = y_val_t[i:i+8192]
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model


def evaluate_model(model, X_t, y_true, device, batch_size=8192):
    """Get predictions and compute metrics using fast GPU batched inference."""
    model.eval()
    probs = []

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size]
            outputs = model(xb)
            p = torch.softmax(outputs, dim=1)[:, 1]
            probs.append(p.cpu().numpy())

    y_prob = np.concatenate(probs) if len(probs) > 0 else np.zeros(len(y_true))
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.0

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auroc': auroc,
    }


def main():
    print_header("STUDY 8: WEIGHTED RANDOM SAMPLER RATIO ABLATION")
    check_cache()
    ensure_dir(OUTPUT_DIR)

    splits = load_patient_splits()
    print(f"Loaded {len(splits)} trial splits")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_results = []

    for trial_idx, split_info in enumerate(splits):
        trial_num = split_info.get('trial', trial_idx + 1)
        print_subheader(f"Trial {trial_num}/10")

        # Load ALL data (no adaptive filtering)
        train_df = load_patient_features(split_info['train_idx'], SPECTRAL_CACHE_DIR)
        val_df = load_patient_features(split_info['val_idx'], SPECTRAL_CACHE_DIR)
        test_df = load_patient_features(split_info['test_idx'], SPECTRAL_CACHE_DIR)

        feature_cols = get_feature_columns(train_df)

        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['label'].values

        X_train, X_val, X_test, _, _, _ = preprocess_features(
            X_train, X_val, X_test, apply_pca=True, pca_components=10
        )

        n_features = X_train.shape[1]

        # Report class distribution
        n1 = np.sum(y_train == 1)
        n0 = np.sum(y_train == 0)
        print(f"  Train: {len(y_train)} total (seizure={n1}, non-seizure={n0}, "
              f"ratio={n0/max(n1,1):.1f}:1)")

        # Load tensors directly to device
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

        for ratio_name, seizure_mult in SAMPLING_RATIOS:
            print(f"  Ratio {ratio_name}...", end=' ', flush=True)

            # Set seeds
            torch.manual_seed(42 + trial_idx)
            np.random.seed(42 + trial_idx)

            sample_weights = get_sample_weights(y_train, seizure_mult)

            # Build model
            model = NeuralNet(n_features).to(device)

            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
            )
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train on GPU
            model = train_model(model, X_train_t, y_train_t, sample_weights,
                                X_val_t, y_val_t, criterion, optimizer,
                                device, epochs=50, patience=10, batch_size=1024)

            # Evaluate
            val_metrics = evaluate_model(model, X_val_t, y_val, device)
            test_metrics = evaluate_model(model, X_test_t, y_test, device)

            result = {
                'trial': trial_num,
                'ratio': ratio_name,
            }
            for k, v in val_metrics.items():
                result[f'val_{k}'] = v
            for k, v in test_metrics.items():
                result[f'test_{k}'] = v

            all_results.append(result)
            print(f"Val F1={val_metrics['f1']:.4f} AUROC={val_metrics['auroc']:.4f} | "
                  f"Test F1={test_metrics['f1']:.4f} AUROC={test_metrics['auroc']:.4f}", flush=True)

    # -- Save results --------------------------------------------------
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'sampler_results.csv'), index=False)

    # -- Summary -------------------------------------------------------
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
    summary_rows = []

    for ratio_name, _ in SAMPLING_RATIOS:
        ratio_df = df_results[df_results['ratio'] == ratio_name]
        row = {'ratio': ratio_name}
        for split in ['val', 'test']:
            for m in metrics:
                col = f'{split}_{m}'
                row[f'{col}_mean'] = ratio_df[col].mean()
                row[f'{col}_std'] = ratio_df[col].std()
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    summary_path = os.path.join(OUTPUT_DIR, 'sampler_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 110 + "\n")
        f.write("STUDY 8: WEIGHTED RANDOM SAMPLER RATIO ABLATION — SUMMARY\n")
        f.write("=" * 110 + "\n\n")
        f.write("MLP architecture: Input->128->BN->ReLU->Drop(0.3)->64->BN->ReLU->Drop(0.3)->32->BN->ReLU->Drop(0.3)->2\n")
        f.write("Uses WeightedRandomSampler (not dataset filtering). Full dataset preserved.\n")
        f.write("50:50 = dominant config (prevents classifier collapse).\n\n")

        for split in ['val', 'test']:
            f.write(f"\n{'-' * 100}\n")
            f.write(f"  {split.upper()} SET (mean ± std across 10 trials)\n")
            f.write(f"{'-' * 100}\n\n")

            f.write(f"{'Ratio':<12s}")
            for m in metrics:
                f.write(f"  {m.upper():<18s}")
            f.write("\n")
            f.write("-" * 105 + "\n")

            for _, row in df_summary.iterrows():
                marker = " <" if row['ratio'] == '50:50' else ""
                f.write(f"{row['ratio']:<12s}")
                for m in metrics:
                    mean = row[f'{split}_{m}_mean']
                    std = row[f'{split}_{m}_std']
                    f.write(f"  {mean:.4f} ± {std:.4f}   ")
                f.write(f"{marker}\n")

        f.write(f"\n\n{'=' * 80}\n")
        f.write("INTERPRETATION\n")
        f.write(f"{'=' * 80}\n\n")

        best_f1_ratio = df_summary.loc[df_summary['test_f1_mean'].idxmax(), 'ratio']
        best_f1_val = df_summary['test_f1_mean'].max()
        nat_recall = df_summary[df_summary['ratio'] == 'Natural']['test_recall_mean'].values
        balanced_recall = df_summary[df_summary['ratio'] == '50:50']['test_recall_mean'].values

        f.write(f"Best test F1: {best_f1_ratio} (F1 = {best_f1_val:.4f})\n\n")

        if len(nat_recall) > 0 and len(balanced_recall) > 0:
            recall_drop = balanced_recall[0] - nat_recall[0]
            f.write(f"50:50 recall = {balanced_recall[0]:.4f}\n")
            f.write(f"Natural recall = {nat_recall[0]:.4f}\n")
            f.write(f"Recall improvement from balanced sampling: {recall_drop:+.4f}\n\n")

            if nat_recall[0] < 0.3:
                f.write("[WARN]  Natural ratio yields near-zero recall -> classifier collapse confirmed.\n")
                f.write("    WeightedRandomSampler at 50:50 is ESSENTIAL for seizure detection.\n")

    print(f"Saved summary: {summary_path}")

    # -- Line plots ----------------------------------------------------
    ratio_order = [r[0] for r in SAMPLING_RATIOS]

    for metric in ['recall', 'f1', 'auroc']:
        fig = go.Figure()
        for split, color in [('val', '#3498db'), ('test', '#e74c3c')]:
            means = [df_summary[df_summary['ratio'] == r][f'{split}_{metric}_mean'].values[0]
                     for r in ratio_order]
            stds = [df_summary[df_summary['ratio'] == r][f'{split}_{metric}_std'].values[0]
                    for r in ratio_order]
            fig.add_trace(go.Scatter(
                x=ratio_order, y=means,
                error_y=dict(type='data', array=stds, visible=True),
                mode='lines+markers', name=split.title(),
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))

        fig.update_layout(
            title=f"Sampling Ratio vs {metric.upper()} (mean ± std across 10 trials)",
            xaxis_title="Sampling Ratio (Seizure:Non-Seizure)",
            yaxis_title=metric.upper(),
            template="plotly_white",
            font=dict(color="black", size=14),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        chart_path = os.path.join(OUTPUT_DIR, f'{metric}_vs_ratio.png')
        pio.write_image(fig, chart_path, width=1000, height=600, scale=2)
        print(f"Saved: {chart_path}")

    print_header("STUDY 8 COMPLETE")
    print(f"  Results: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

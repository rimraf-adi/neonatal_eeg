"""
Study 1: Classifier Complexity Ladder
======================================

Trains 4 classical ML models on spectral slope features across all
10 patient-split trials. If the simplest model (Logistic Regression)
achieves high AUROC, it proves the features are doing the work, not
the model.

Models:
  1. Logistic Regression (linear, class_weight='balanced')
  2. Linear SVM (linear kernel, class_weight='balanced')
  3. Random Forest (100 trees, class_weight='balanced')
  4. K-Nearest Neighbors (k=5, uniform weights)

Output:
  study_results/01_classifier_ladder/
    per_trial_results.csv
    summary_table.txt
    auroc_comparison.png
    f1_comparison.png

Usage:
  uv run python -m studies.study_01_classifier_ladder
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
try:
    import cuml
    from cuml.svm import SVC as CumlSVC
    from cuml.ensemble import RandomForestClassifier as CumlRF
    from cuml.neighbors import KNeighborsClassifier as CumlKNN
    HAS_CUML = True
    print("Using GPU-accelerated cuML models")
except ImportError:
    HAS_CUML = False
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score,
)
import plotly.graph_objects as go
import plotly.io as pio

from studies.utils import (
    load_patient_splits, prepare_trial_data,
    check_cache, ensure_dir, print_header, print_subheader,
    RESULTS_BASE,
)

OUTPUT_DIR = os.path.join(RESULTS_BASE, '01_classifier_ladder')

# Model definitions
MODELS = {
    'Logistic Regression': lambda: LogisticRegression(
        max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=42
    ),
    'Linear SVM': lambda: (
        CumlSVC(kernel='linear', class_weight='balanced') if HAS_CUML
        else LinearSVC(dual=False, class_weight='balanced', max_iter=2000, random_state=42)
    ),
    'Random Forest': lambda: (
        CumlRF(n_estimators=100, max_depth=16, random_state=42) if HAS_CUML
        else RandomForestClassifier(n_estimators=100, max_depth=16, class_weight='balanced', random_state=42, n_jobs=-1)
    ),
    'KNN (k=5)': lambda: (
        CumlKNN(n_neighbors=5) if HAS_CUML
        else KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    ),
}


def subsample_balanced(X, y, max_samples=50000, random_state=42):
    """Subsample dataset while preserving class ratio for instance-based models."""
    if len(y) <= max_samples:
        return X, y
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    samples_per_class = max_samples // len(classes)
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, min(len(c_idx), samples_per_class), replace=False)
        indices.extend(chosen)
    rng.shuffle(indices)
    return X[indices], y[indices]


def evaluate_model(model, X_test, y_test):
    """Evaluate a fitted model on test data."""
    y_pred = model.predict(X_test)

    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        auroc = roc_auc_score(y_test, y_prob)
    except Exception:
        auroc = 0.0

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auroc': auroc,
    }


def save_bar_chart(df_summary, metric, output_path):
    """Save a grouped bar chart for a metric across models."""
    fig = go.Figure()

    for _, row in df_summary.iterrows():
        fig.add_trace(go.Bar(
            name=row['model'],
            x=[row['model']],
            y=[row[f'{metric}_mean']],
            error_y=dict(type='data', array=[row[f'{metric}_std']], visible=True),
            text=[f"{row[f'{metric}_mean']:.4f}"],
            textposition='outside',
        ))

    fig.update_layout(
        title=f"Classifier Ladder: {metric.upper()} Comparison (mean ± std across 10 trials)",
        yaxis_title=metric.upper(),
        xaxis_title="Model",
        template="plotly_white",
        font=dict(color="black", size=14),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    pio.write_image(fig, output_path, width=1000, height=600, scale=2)
    print(f"  Saved: {output_path}")


def main():
    print_header("STUDY 1: CLASSIFIER COMPLEXITY LADDER")
    check_cache()
    ensure_dir(OUTPUT_DIR)

    splits = load_patient_splits()
    print(f"Loaded {len(splits)} trial splits")

    # -- Run all models across all trials ------------------------------
    all_results = []

    for trial_idx, split_info in enumerate(splits):
        trial_num = split_info.get('trial', trial_idx + 1)
        print_subheader(f"Trial {trial_num}/10")

        # Load data (no PCA — raw features)
        data = prepare_trial_data(split_info, apply_pca=False)
        print(f"  Train: {len(data['y_train'])} samples "
              f"(seizure={data['train_counts'].get(1,0)}, non-seizure={data['train_counts'].get(0,0)})")
        print(f"  Val:   {len(data['y_val'])} samples")
        print(f"  Test:  {len(data['y_test'])} samples")
        print(f"  Features: {data['X_train'].shape[1]} dimensions")

        for model_name, model_factory in MODELS.items():
            print(f"  Training {model_name}...", end=' ', flush=True)

            model = model_factory()
            if 'KNN' in model_name and not HAS_CUML:
                X_tr, y_tr = subsample_balanced(data['X_train'], data['y_train'], max_samples=50000)
                X_v, y_v = subsample_balanced(data['X_val'], data['y_val'], max_samples=10000)
                X_te, y_te = subsample_balanced(data['X_test'], data['y_test'], max_samples=10000)
                model.fit(X_tr, y_tr)
                val_metrics = evaluate_model(model, X_v, y_v)
                test_metrics = evaluate_model(model, X_te, y_te)
            else:
                model.fit(data['X_train'], data['y_train'])
                val_metrics = evaluate_model(model, data['X_val'], data['y_val'])
                test_metrics = evaluate_model(model, data['X_test'], data['y_test'])

            result = {
                'trial': trial_num,
                'model': model_name,
            }
            for k, v in val_metrics.items():
                result[f'val_{k}'] = v
            for k, v in test_metrics.items():
                result[f'test_{k}'] = v

            all_results.append(result)
            print(f"Val AUROC={val_metrics['auroc']:.4f}, "
                  f"Test AUROC={test_metrics['auroc']:.4f}")

    # -- Save per-trial results ----------------------------------------
    df_results = pd.DataFrame(all_results)
    per_trial_path = os.path.join(OUTPUT_DIR, 'per_trial_results.csv')
    df_results.to_csv(per_trial_path, index=False)
    print(f"\nSaved per-trial results: {per_trial_path}")

    # -- Compute summary statistics ------------------------------------
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
    summary_rows = []

    for model_name in MODELS.keys():
        model_df = df_results[df_results['model'] == model_name]
        row = {'model': model_name}
        for split in ['val', 'test']:
            for m in metrics:
                col = f'{split}_{m}'
                row[f'{col}_mean'] = model_df[col].mean()
                row[f'{col}_std'] = model_df[col].std()
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # -- Write summary table --------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, 'summary_table.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("STUDY 1: CLASSIFIER COMPLEXITY LADDER — SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write("Goal: If Logistic Regression achieves AUROC > 0.80, the spectral slope\n")
        f.write("features are linearly separable — the features do the work, not the model.\n\n")

        for split in ['val', 'test']:
            f.write(f"\n{'-' * 80}\n")
            f.write(f"  {split.upper()} SET RESULTS (mean ± std across 10 trials)\n")
            f.write(f"{'-' * 80}\n\n")

            f.write(f"{'Model':<25s}")
            for m in metrics:
                f.write(f"  {m.upper():<18s}")
            f.write("\n")
            f.write(f"{'-' * 25}")
            for _ in metrics:
                f.write(f"  {'-' * 18}")
            f.write("\n")

            for _, row in df_summary.iterrows():
                f.write(f"{row['model']:<25s}")
                for m in metrics:
                    mean = row[f'{split}_{m}_mean']
                    std = row[f'{split}_{m}_std']
                    f.write(f"  {mean:.4f} ± {std:.4f}   ")
                f.write("\n")

        f.write(f"\n\n{'=' * 80}\n")
        f.write("INTERPRETATION\n")
        f.write(f"{'=' * 80}\n\n")

        lr_auroc = df_summary[df_summary['model'] == 'Logistic Regression']['test_auroc_mean'].values[0]
        if lr_auroc > 0.80:
            f.write(f"[PASS] Logistic Regression test AUROC = {lr_auroc:.4f} (> 0.80)\n")
            f.write(f"   -> The spectral slope features are LINEARLY SEPARABLE.\n")
            f.write(f"   -> The log-PSD line fitting captures seizure dynamics effectively.\n")
            f.write(f"   -> A simple linear decision boundary suffices.\n")
        else:
            f.write(f"[WARN]  Logistic Regression test AUROC = {lr_auroc:.4f} (< 0.80)\n")
            f.write(f"   -> Features may require non-linear modelling for full potential.\n")

    print(f"Saved summary: {summary_path}")

    # -- Generate charts -----------------------------------------------
    for metric in ['auroc', 'f1']:
        for split in ['val', 'test']:
            chart_path = os.path.join(OUTPUT_DIR, f'{split}_{metric}_comparison.png')
            chart_df = df_summary.copy()
            chart_df_renamed = chart_df.rename(columns={
                f'{split}_{metric}_mean': f'{metric}_mean',
                f'{split}_{metric}_std': f'{metric}_std',
            })
            save_bar_chart(chart_df_renamed, metric, chart_path)

    print_header("STUDY 1 COMPLETE")
    print(f"  Results: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

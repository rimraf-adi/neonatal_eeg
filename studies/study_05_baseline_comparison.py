"""
Study 5: Baseline Feature Comparison
======================================

Compares spectral slope features against 4 simpler baseline feature sets
on the exact same pipeline (same patient splits, same preprocessing, same
Logistic Regression model).

Baseline feature sets (extracted by cache_features.py):
  1. Raw Band Power (area under PSD per band) — 4 feats/ch x 18 ch = 72
  2. Hjorth Parameters (Activity, Mobility, Complexity) — 3 feats/ch x 18 ch = 54
  3. Time-domain Stats (variance, RMS, line length, zero crossings) — 4 feats/ch x 18 ch = 72
  4. Spectral Entropy (Shannon entropy per band) — 4 feats/ch x 18 ch = 72

If spectral slope consistently beats all baselines, the log-PSD linear
fitting extracts information that simpler features miss.

Output:
  study_results/05_baseline_comparison/
    comparison_results.csv
    comparison_summary.txt
    comparison_chart.png

Usage:
  uv run python -m studies.study_05_baseline_comparison
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    RESULTS_BASE, SPECTRAL_CACHE_DIR, BASELINE_CACHE_DIR,
    BASELINE_FEATURE_SETS,
)

OUTPUT_DIR = os.path.join(RESULTS_BASE, '05_baseline_comparison')


def run_comparison_trial(split_info, feature_dir, feature_cols_fn):
    """Run Logistic Regression for one feature set on one trial.

    Args:
        split_info: dict with train_idx, val_idx, test_idx
        feature_dir: directory containing patient CSVs for this feature set
        feature_cols_fn: function(df) -> list[str] of feature column names

    Returns:
        dict with metrics, or None on failure
    """
    try:
        train_df = load_patient_features(split_info['train_idx'], feature_dir)
        val_df = load_patient_features(split_info['val_idx'], feature_dir)
        test_df = load_patient_features(split_info['test_idx'], feature_dir)
    except (ValueError, FileNotFoundError):
        return None

    feature_cols = feature_cols_fn(train_df)
    if not feature_cols:
        return None

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values.astype(int)
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values.astype(int)

    X_train, X_val, X_test, _, _, _ = preprocess_features(X_train, X_val, X_test)

    model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)

    results = {'n_features': len(feature_cols)}
    for split_name, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        try:
            y_prob = model.predict_proba(X)[:, 1]
            auroc = roc_auc_score(y, y_prob)
        except Exception:
            auroc = 0.0

        results[f'{split_name}_accuracy'] = accuracy_score(y, y_pred)
        results[f'{split_name}_precision'] = precision_score(y, y_pred, zero_division=0)
        results[f'{split_name}_recall'] = recall_score(y, y_pred, zero_division=0)
        results[f'{split_name}_f1'] = f1_score(y, y_pred, zero_division=0)
        results[f'{split_name}_auroc'] = auroc

    return results


def main():
    print_header("STUDY 5: BASELINE FEATURE COMPARISON")
    check_cache()
    ensure_dir(OUTPUT_DIR)

    # Check baseline cache exists
    for feat_name in BASELINE_FEATURE_SETS:
        baseline_dir = os.path.join(BASELINE_CACHE_DIR, feat_name)
        if not os.path.exists(baseline_dir):
            print(f"ERROR: Baseline feature cache not found: {baseline_dir}")
            print(f"Run: uv run python -m studies.cache_features")
            sys.exit(1)

    splits = load_patient_splits()
    print(f"Loaded {len(splits)} trial splits")

    # Feature set definitions: (name, feature_dir, column_selector_fn)
    feature_sets = [
        (
            'Spectral Slope',
            SPECTRAL_CACHE_DIR,
            lambda df: get_feature_columns(df),
        ),
    ]

    # Add baseline feature sets
    for feat_name, feat_config in BASELINE_FEATURE_SETS.items():
        col_names = feat_config['columns']
        feature_sets.append((
            feat_name.replace('_', ' ').title(),
            os.path.join(BASELINE_CACHE_DIR, feat_name),
            lambda df, cols=col_names: [c for c in df.columns if c in cols],
        ))

    # -- Run all feature sets across all trials ------------------------
    all_results = []

    for feat_set_name, feat_dir, col_fn in feature_sets:
        print_subheader(f"Feature Set: {feat_set_name}")

        for trial_idx, split_info in enumerate(splits):
            trial_num = split_info.get('trial', trial_idx + 1)
            print(f"  Trial {trial_num}...", end=' ', flush=True)

            result = run_comparison_trial(split_info, feat_dir, col_fn)
            if result is None:
                print("SKIP (missing data)")
                continue

            result['trial'] = trial_num
            result['feature_set'] = feat_set_name
            all_results.append(result)

            print(f"AUROC val={result['val_auroc']:.4f} test={result['test_auroc']:.4f} "
                  f"({result['n_features']} feats)")

    # -- Save per-trial results ----------------------------------------
    df_results = pd.DataFrame(all_results)
    per_trial_path = os.path.join(OUTPUT_DIR, 'comparison_results.csv')
    df_results.to_csv(per_trial_path, index=False)
    print(f"\nSaved per-trial results: {per_trial_path}")

    # -- Compute summary -----------------------------------------------
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
    summary_rows = []

    for feat_set_name, _, _ in feature_sets:
        fs_df = df_results[df_results['feature_set'] == feat_set_name]
        if len(fs_df) == 0:
            continue
        row = {
            'feature_set': feat_set_name,
            'n_features': int(fs_df['n_features'].iloc[0]),
        }
        for split in ['val', 'test']:
            for m in metrics:
                col = f'{split}_{m}'
                row[f'{col}_mean'] = fs_df[col].mean()
                row[f'{col}_std'] = fs_df[col].std()
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # -- Write summary -------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, 'comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("STUDY 5: BASELINE FEATURE COMPARISON — SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write("All feature sets evaluated with Logistic Regression (class_weight='balanced')\n")
        f.write("across the same 10-trial patient-level CV splits.\n\n")

        for split in ['val', 'test']:
            f.write(f"\n{'-' * 90}\n")
            f.write(f"  {split.upper()} SET (mean ± std across 10 trials)\n")
            f.write(f"{'-' * 90}\n\n")

            f.write(f"{'Feature Set':<25s} {'#Feats':<8s}")
            for m in ['auroc', 'f1', 'recall', 'precision']:
                f.write(f"  {m.upper():<18s}")
            f.write("\n")
            f.write("-" * 100 + "\n")

            for _, row in df_summary.iterrows():
                f.write(f"{row['feature_set']:<25s} {row['n_features']:<8d}")
                for m in ['auroc', 'f1', 'recall', 'precision']:
                    mean = row[f'{split}_{m}_mean']
                    std = row[f'{split}_{m}_std']
                    f.write(f"  {mean:.4f} ± {std:.4f}   ")
                f.write("\n")

        f.write(f"\n\n{'=' * 80}\n")
        f.write("INTERPRETATION\n")
        f.write(f"{'=' * 80}\n\n")

        ss_row = df_summary[df_summary['feature_set'] == 'Spectral Slope']
        if len(ss_row) > 0:
            ss_auroc = ss_row['test_auroc_mean'].values[0]
            beats_all = True
            for _, row in df_summary.iterrows():
                if row['feature_set'] == 'Spectral Slope':
                    continue
                if row['test_auroc_mean'] >= ss_auroc:
                    beats_all = False
                    f.write(f"[WARN]  {row['feature_set']} matches/exceeds Spectral Slope "
                            f"(AUROC {row['test_auroc_mean']:.4f} vs {ss_auroc:.4f})\n")

            if beats_all:
                f.write(f"[PASS] Spectral Slope (AUROC = {ss_auroc:.4f}) outperforms ALL baselines.\n")
                f.write(f"   -> Log-PSD linear fitting extracts information simpler features miss.\n")

    print(f"Saved summary: {summary_path}")

    # -- Chart ---------------------------------------------------------
    chart_path = os.path.join(OUTPUT_DIR, 'comparison_chart.png')
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']

    fig = go.Figure()
    for i, (_, row) in enumerate(df_summary.iterrows()):
        fig.add_trace(go.Bar(
            name=row['feature_set'],
            x=[row['feature_set']],
            y=[row['test_auroc_mean']],
            error_y=dict(type='data', array=[row['test_auroc_std']], visible=True),
            marker_color=colors[i % len(colors)],
            text=[f"{row['test_auroc_mean']:.4f}"],
            textposition='outside',
        ))

    fig.update_layout(
        title="Baseline Feature Comparison: Test AUROC (mean ± std, Logistic Regression)",
        yaxis_title="AUROC",
        xaxis_title="Feature Set",
        template="plotly_white",
        font=dict(color="black", size=13),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_tickangle=-20,
    )
    pio.write_image(fig, chart_path, width=1000, height=600, scale=2)
    print(f"Saved chart: {chart_path}")

    print_header("STUDY 5 COMPLETE")
    print(f"  Results: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

"""
Study 2: Feature Ablation Study
=================================

Systematically drops one band or one feature type at a time and trains
Logistic Regression on each ablated feature set. Measures performance
degradation to prove each component contributes non-redundant information.

Ablation configurations:
  1. Full set (all bands x all types) — baseline
  2. Drop Delta
  3. Drop Theta
  4. Drop Alpha
  5. Drop Beta
  6. Slope only (drop intercept & midband)
  7. Intercept only (drop slope & midband)
  8. Midband only (drop slope & intercept)

Output:
  study_results/02_feature_ablation/
    ablation_results.csv
    ablation_summary.txt
    ablation_chart.png

Usage:
  uv run python -m studies.study_02_feature_ablation
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
    load_patient_splits, load_patient_features,
    get_feature_columns, preprocess_features,
    check_cache, ensure_dir, print_header, print_subheader,
    RESULTS_BASE, BAND_NAMES, FEATURE_TYPES, SPECTRAL_CACHE_DIR,
)

OUTPUT_DIR = os.path.join(RESULTS_BASE, '02_feature_ablation')

# Ablation configurations: (name, bands_to_keep, types_to_keep)
ABLATIONS = [
    ('Full (baseline)', BAND_NAMES, FEATURE_TYPES),
    ('Drop Delta',      [b for b in BAND_NAMES if b != 'delta'], FEATURE_TYPES),
    ('Drop Theta',      [b for b in BAND_NAMES if b != 'theta'], FEATURE_TYPES),
    ('Drop Alpha',      [b for b in BAND_NAMES if b != 'alpha'], FEATURE_TYPES),
    ('Drop Beta',       [b for b in BAND_NAMES if b != 'beta'],  FEATURE_TYPES),
    ('Slope only',      BAND_NAMES, ['slope']),
    ('Intercept only',  BAND_NAMES, ['intercept']),
    ('Midband only',    BAND_NAMES, ['midband']),
]


def run_ablation_trial(split_info, bands, feature_types):
    """Run Logistic Regression for one ablation configuration on one trial."""
    train_df = load_patient_features(split_info['train_idx'], SPECTRAL_CACHE_DIR)
    val_df = load_patient_features(split_info['val_idx'], SPECTRAL_CACHE_DIR)
    test_df = load_patient_features(split_info['test_idx'], SPECTRAL_CACHE_DIR)

    feature_cols = get_feature_columns(train_df, bands=bands, feature_types=feature_types)

    if not feature_cols:
        return None

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

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
    print_header("STUDY 2: FEATURE ABLATION STUDY")
    check_cache()
    ensure_dir(OUTPUT_DIR)

    splits = load_patient_splits()
    print(f"Loaded {len(splits)} trial splits")

    # -- Run all ablations across all trials ---------------------------
    all_results = []

    for abl_name, abl_bands, abl_types in ABLATIONS:
        print_subheader(f"Ablation: {abl_name}")
        print(f"  Bands: {abl_bands}")
        print(f"  Types: {abl_types}")

        for trial_idx, split_info in enumerate(splits):
            trial_num = split_info.get('trial', trial_idx + 1)
            print(f"  Trial {trial_num}...", end=' ', flush=True)

            result = run_ablation_trial(split_info, abl_bands, abl_types)
            if result is None:
                print("SKIP (no features)")
                continue

            result['trial'] = trial_num
            result['ablation'] = abl_name
            result['bands'] = ','.join(abl_bands)
            result['types'] = ','.join(abl_types)
            all_results.append(result)

            print(f"AUROC val={result['val_auroc']:.4f} test={result['test_auroc']:.4f} "
                  f"({result['n_features']} feats)")

    # -- Save per-trial results ----------------------------------------
    df_results = pd.DataFrame(all_results)
    per_trial_path = os.path.join(OUTPUT_DIR, 'ablation_results.csv')
    df_results.to_csv(per_trial_path, index=False)
    print(f"\nSaved per-trial results: {per_trial_path}")

    # -- Compute summary & delta from baseline -------------------------
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
    summary_rows = []

    baseline_means = {}
    for abl_name, _, _ in ABLATIONS:
        abl_df = df_results[df_results['ablation'] == abl_name]
        row = {
            'ablation': abl_name,
            'n_features': int(abl_df['n_features'].iloc[0]) if len(abl_df) > 0 else 0,
        }
        for split in ['val', 'test']:
            for m in metrics:
                col = f'{split}_{m}'
                row[f'{col}_mean'] = abl_df[col].mean()
                row[f'{col}_std'] = abl_df[col].std()

                if abl_name == 'Full (baseline)':
                    baseline_means[col] = row[f'{col}_mean']

                row[f'{col}_delta'] = row[f'{col}_mean'] - baseline_means.get(col, 0.0)

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # -- Write summary table -------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, 'ablation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("STUDY 2: FEATURE ABLATION STUDY — SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write("Goal: Prove each band/parameter contributes non-redundant information.\n")
        f.write("If dropping a band causes a large AUROC drop, it is essential.\n\n")

        for split in ['val', 'test']:
            f.write(f"\n{'-' * 90}\n")
            f.write(f"  {split.upper()} SET (mean across 10 trials)\n")
            f.write(f"{'-' * 90}\n\n")

            f.write(f"{'Ablation':<25s} {'#Feats':<8s}")
            for m in ['auroc', 'f1']:
                f.write(f"  {m.upper():<12s} {'DeltaAUROC' if m == 'auroc' else 'DeltaF1':<10s}")
            f.write("\n")
            f.write("-" * 80 + "\n")

            for _, row in df_summary.iterrows():
                f.write(f"{row['ablation']:<25s} {row['n_features']:<8d}")
                for m in ['auroc', 'f1']:
                    mean = row[f'{split}_{m}_mean']
                    delta = row[f'{split}_{m}_delta']
                    delta_str = f"{delta:+.4f}" if row['ablation'] != 'Full (baseline)' else "  —"
                    f.write(f"  {mean:.4f}       {delta_str:<10s}")
                f.write("\n")

        # Interpretation
        f.write(f"\n\n{'=' * 80}\n")
        f.write("INTERPRETATION\n")
        f.write(f"{'=' * 80}\n\n")

        # Find biggest drop
        non_baseline = df_summary[df_summary['ablation'] != 'Full (baseline)']
        if len(non_baseline) > 0:
            worst_drop_idx = non_baseline['test_auroc_delta'].idxmin()
            worst = non_baseline.loc[worst_drop_idx]
            f.write(f"Largest AUROC drop: '{worst['ablation']}' "
                    f"(Delta = {worst['test_auroc_delta']:+.4f})\n")
            f.write(f"  -> This component contributes the most discriminative information.\n\n")

            # Check if all drops are meaningful
            all_drops = non_baseline[non_baseline['ablation'].str.startswith('Drop')]['test_auroc_delta']
            if (all_drops < -0.01).all():
                f.write("All band drops cause > 0.01 AUROC decrease -> all bands contribute.\n")
            else:
                redundant = non_baseline[non_baseline['test_auroc_delta'] >= -0.01]['ablation'].tolist()
                f.write(f"Possibly redundant: {redundant}\n")

    print(f"Saved summary: {summary_path}")

    # -- Generate chart -----------------------------------------------
    chart_path = os.path.join(OUTPUT_DIR, 'ablation_chart.png')
    fig = go.Figure()

    for _, row in df_summary.iterrows():
        color = '#2ecc71' if row['ablation'] == 'Full (baseline)' else '#e74c3c'
        fig.add_trace(go.Bar(
            name=row['ablation'],
            x=[row['ablation']],
            y=[row['test_auroc_mean']],
            error_y=dict(type='data', array=[row['test_auroc_std']], visible=True),
            marker_color=color,
            text=[f"{row['test_auroc_mean']:.4f}"],
            textposition='outside',
        ))

    fig.update_layout(
        title="Feature Ablation: Test AUROC (mean ± std across 10 trials)",
        yaxis_title="AUROC",
        xaxis_title="Ablation Configuration",
        template="plotly_white",
        font=dict(color="black", size=13),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_tickangle=-30,
    )

    pio.write_image(fig, chart_path, width=1200, height=600, scale=2)
    print(f"Saved chart: {chart_path}")

    print_header("STUDY 2 COMPLETE")
    print(f"  Results: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

"""
Study 4: Fisher's Discriminant Ratio
======================================

Computes Fisher's Discriminant Ratio for every spectral slope feature
across all 10 trials (training split only). Ranks features and identifies
top discriminative bandxchannelxparameter combinations.

Fisher Ratio: J_i = (mu_1 - mu_0)^2 / (sigma^2_1 + sigma^2_0)

Unlike t-tests, this does NOT inflate with sample size.

Output:
  study_results/04_fisher_ratio/
    fisher_scores_all.csv
    fisher_scores_per_trial.csv
    top30_fisher.png
    fisher_by_band.png
    fisher_by_type.png
    fisher_summary.txt

Usage:
  uv run python -m studies.study_04_fisher_ratio
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from studies.utils import (
    load_patient_splits, load_patient_features, get_feature_columns,
    preprocess_features,
    check_cache, ensure_dir, print_header, print_subheader,
    RESULTS_BASE, BAND_NAMES, FEATURE_TYPES, SPECTRAL_CACHE_DIR,
)

OUTPUT_DIR = os.path.join(RESULTS_BASE, '04_fisher_ratio')


def compute_fisher_ratio(X, y):
    """Compute Fisher's Discriminant Ratio for each feature.

    J_i = (mu_1 - mu_0)^2 / (sigma^2_1 + sigma^2_0)

    Args:
        X: (n_samples, n_features) array
        y: (n_samples,) label array (0 or 1)

    Returns:
        np.ndarray of shape (n_features,)
    """
    mask_0 = y == 0
    mask_1 = y == 1

    mu_0 = np.nanmean(X[mask_0], axis=0)
    mu_1 = np.nanmean(X[mask_1], axis=0)
    var_0 = np.nanvar(X[mask_0], axis=0)
    var_1 = np.nanvar(X[mask_1], axis=0)

    denom = var_0 + var_1
    denom[denom == 0] = 1e-10  # prevent division by zero

    fisher = (mu_1 - mu_0) ** 2 / denom
    return fisher


def parse_feature_name(feat_name):
    """Parse a feature name like 'delta_slope' into (band, type) components."""
    for band in BAND_NAMES:
        for ftype in FEATURE_TYPES:
            if f'{band}_{ftype}' in feat_name:
                return band, ftype
    return 'unknown', 'unknown'


def main():
    print_header("STUDY 4: FISHER'S DISCRIMINANT RATIO")
    check_cache()
    ensure_dir(OUTPUT_DIR)

    splits = load_patient_splits()
    print(f"Loaded {len(splits)} trial splits")

    # -- Compute Fisher ratios across all trials -----------------------
    per_trial_results = []

    for trial_idx, split_info in enumerate(splits):
        trial_num = split_info.get('trial', trial_idx + 1)
        print_subheader(f"Trial {trial_num}/10")

        train_df = load_patient_features(split_info['train_idx'], SPECTRAL_CACHE_DIR)
        feature_cols = get_feature_columns(train_df)

        X = train_df[feature_cols].values
        y = train_df['label'].values

        # Impute NaN before computing Fisher ratio
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        fisher_scores = compute_fisher_ratio(X, y)

        for i, col in enumerate(feature_cols):
            per_trial_results.append({
                'trial': trial_num,
                'feature': col,
                'fisher_ratio': fisher_scores[i],
            })

        top5 = np.argsort(fisher_scores)[::-1][:5]
        print(f"  Top 5 features by Fisher ratio:")
        for rank, idx in enumerate(top5):
            print(f"    {rank+1}. {feature_cols[idx]}: J = {fisher_scores[idx]:.4f}")

    # -- Save per-trial results ----------------------------------------
    df_per_trial = pd.DataFrame(per_trial_results)
    df_per_trial.to_csv(os.path.join(OUTPUT_DIR, 'fisher_scores_per_trial.csv'), index=False)

    # -- Aggregate across trials ---------------------------------------
    df_agg = df_per_trial.groupby('feature').agg(
        fisher_mean=('fisher_ratio', 'mean'),
        fisher_std=('fisher_ratio', 'std'),
        fisher_min=('fisher_ratio', 'min'),
        fisher_max=('fisher_ratio', 'max'),
    ).reset_index()

    # Parse band/type from feature names
    df_agg['band'] = df_agg['feature'].apply(lambda x: parse_feature_name(x)[0])
    df_agg['type'] = df_agg['feature'].apply(lambda x: parse_feature_name(x)[1])

    df_agg = df_agg.sort_values('fisher_mean', ascending=False)
    df_agg.to_csv(os.path.join(OUTPUT_DIR, 'fisher_scores_all.csv'), index=False)

    # -- Top 30 bar chart ----------------------------------------------
    top30 = df_agg.head(30)
    fig = px.bar(
        top30, x='feature', y='fisher_mean',
        error_y='fisher_std',
        color='band',
        title="Top 30 Features by Fisher's Discriminant Ratio (mean across 10 trials)",
        labels={'fisher_mean': "Fisher Ratio (J)", 'feature': 'Feature'},
        color_discrete_map={
            'delta': '#e74c3c', 'theta': '#3498db', 'alpha': '#2ecc71', 'beta': '#f39c12'
        },
    )
    fig.update_layout(
        template="plotly_white", font=dict(color="black", size=12),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_tickangle=-45,
    )
    pio.write_image(fig, os.path.join(OUTPUT_DIR, 'top30_fisher.png'), width=1400, height=700, scale=2)
    print(f"Saved: top30_fisher.png")

    # -- Fisher by band (box plot) -------------------------------------
    fig_band = px.box(
        df_agg, x='band', y='fisher_mean', color='band',
        title="Fisher Ratio Distribution by Frequency Band",
        labels={'fisher_mean': 'Fisher Ratio (J)', 'band': 'Frequency Band'},
        color_discrete_map={
            'delta': '#e74c3c', 'theta': '#3498db', 'alpha': '#2ecc71', 'beta': '#f39c12'
        },
    )
    fig_band.update_layout(
        template="plotly_white", font=dict(color="black", size=14),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    pio.write_image(fig_band, os.path.join(OUTPUT_DIR, 'fisher_by_band.png'), width=800, height=600, scale=2)
    print(f"Saved: fisher_by_band.png")

    # -- Fisher by type (box plot) -------------------------------------
    fig_type = px.box(
        df_agg, x='type', y='fisher_mean', color='type',
        title="Fisher Ratio Distribution by Feature Type",
        labels={'fisher_mean': 'Fisher Ratio (J)', 'type': 'Feature Type'},
    )
    fig_type.update_layout(
        template="plotly_white", font=dict(color="black", size=14),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    pio.write_image(fig_type, os.path.join(OUTPUT_DIR, 'fisher_by_type.png'), width=800, height=600, scale=2)
    print(f"Saved: fisher_by_type.png")

    # -- Summary text --------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, 'fisher_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STUDY 4: FISHER'S DISCRIMINANT RATIO — SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Fisher Ratio J = (mu_1 - mu_0)^2 / (sigma^2_1 + sigma^2_0)\n")
        f.write("Higher J = better class separation for that feature.\n")
        f.write("Unlike t-tests, Fisher ratio does NOT inflate with sample size.\n\n")

        f.write(f"Total features analyzed: {len(df_agg)}\n\n")

        f.write("TOP 30 FEATURES BY FISHER RATIO\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rank':<6s} {'Feature':<30s} {'Band':<8s} {'Type':<12s} {'J (mean±std)':<18s}\n")
        f.write("-" * 60 + "\n")
        for rank, (_, row) in enumerate(top30.iterrows()):
            f.write(f"{rank+1:<6d} {row['feature']:<30s} {row['band']:<8s} {row['type']:<12s} "
                    f"{row['fisher_mean']:.4f} ± {row['fisher_std']:.4f}\n")

        f.write(f"\n\nFISHER RATIO BY BAND (mean of means)\n")
        f.write("-" * 40 + "\n")
        band_means = df_agg.groupby('band')['fisher_mean'].mean().sort_values(ascending=False)
        for band, mean_j in band_means.items():
            f.write(f"  {band:<10s}: J = {mean_j:.4f}\n")

        f.write(f"\nFISHER RATIO BY TYPE (mean of means)\n")
        f.write("-" * 40 + "\n")
        type_means = df_agg.groupby('type')['fisher_mean'].mean().sort_values(ascending=False)
        for ftype, mean_j in type_means.items():
            f.write(f"  {ftype:<12s}: J = {mean_j:.4f}\n")

        f.write(f"\n\n{'=' * 80}\n")
        f.write("INTERPRETATION\n")
        f.write(f"{'=' * 80}\n\n")
        best_band = band_means.index[0]
        f.write(f"Most discriminative band: {best_band} (J = {band_means.iloc[0]:.4f})\n")
        best_type = type_means.index[0]
        f.write(f"Most discriminative type: {best_type} (J = {type_means.iloc[0]:.4f})\n")

    print(f"Saved summary: {summary_path}")

    print_header("STUDY 4 COMPLETE")
    print(f"  Results: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

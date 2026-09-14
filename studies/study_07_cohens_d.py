"""
Study 7: Cohen's d Effect Size
================================

Computes Cohen's d effect size for every spectral slope feature across
10 trials (training split). Categorizes features by effect size magnitude.

d = (mu_1 - mu_0) / sigma_pooled
sigma_pooled = sqrt((sigma_1^2 + sigma_0^2) / 2)

Unlike p-values, Cohen's d does NOT inflate with sample size.

Categories:
  |d| > 0.8  -> Large effect
  0.5-0.8    -> Medium effect
  0.2-0.5    -> Small effect
  < 0.2      -> Negligible

Output:
  study_results/07_cohens_d/
    cohens_d_all.csv
    cohens_d_per_trial.csv
    cohens_d_distribution.png
    cohens_d_top30.png
    cohens_d_by_band.png
    effect_size_summary.txt

Usage:
  uv run python -m studies.study_07_cohens_d
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.impute import SimpleImputer

from studies.utils import (
    load_patient_splits, load_patient_features, get_feature_columns,
    check_cache, ensure_dir, print_header, print_subheader,
    RESULTS_BASE, BAND_NAMES, FEATURE_TYPES, SPECTRAL_CACHE_DIR,
)

OUTPUT_DIR = os.path.join(RESULTS_BASE, '07_cohens_d')


def compute_cohens_d(X, y):
    """Compute Cohen's d for each feature.

    d = (mu_1 - mu_0) / sigma_pooled
    sigma_pooled = sqrt((sigma_1^2 + sigma_0^2) / 2)

    Args:
        X: (n_samples, n_features) array
        y: (n_samples,) label array

    Returns:
        np.ndarray of shape (n_features,)
    """
    mask_0 = y == 0
    mask_1 = y == 1

    mu_0 = np.nanmean(X[mask_0], axis=0)
    mu_1 = np.nanmean(X[mask_1], axis=0)
    var_0 = np.nanvar(X[mask_0], axis=0)
    var_1 = np.nanvar(X[mask_1], axis=0)

    sigma_pooled = np.sqrt((var_0 + var_1) / 2)
    sigma_pooled[sigma_pooled == 0] = 1e-10

    d = (mu_1 - mu_0) / sigma_pooled
    return d


def categorize_d(d_value):
    """Categorize effect size magnitude."""
    abs_d = abs(d_value)
    if abs_d >= 0.8:
        return 'Large'
    elif abs_d >= 0.5:
        return 'Medium'
    elif abs_d >= 0.2:
        return 'Small'
    else:
        return 'Negligible'


def parse_feature_name(feat_name):
    """Parse a feature name into (band, type)."""
    for band in BAND_NAMES:
        for ftype in FEATURE_TYPES:
            if f'{band}_{ftype}' in feat_name:
                return band, ftype
    return 'unknown', 'unknown'


def main():
    print_header("STUDY 7: COHEN'S d EFFECT SIZE")
    check_cache()
    ensure_dir(OUTPUT_DIR)

    splits = load_patient_splits()
    print(f"Loaded {len(splits)} trial splits")

    # -- Compute Cohen's d across all trials ---------------------------
    per_trial_results = []

    for trial_idx, split_info in enumerate(splits):
        trial_num = split_info.get('trial', trial_idx + 1)
        print_subheader(f"Trial {trial_num}/10")

        train_df = load_patient_features(split_info['train_idx'], SPECTRAL_CACHE_DIR)
        feature_cols = get_feature_columns(train_df)

        X = train_df[feature_cols].values
        y = train_df['label'].values

        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        d_values = compute_cohens_d(X, y)

        for i, col in enumerate(feature_cols):
            per_trial_results.append({
                'trial': trial_num,
                'feature': col,
                'cohens_d': d_values[i],
                'abs_d': abs(d_values[i]),
                'category': categorize_d(d_values[i]),
            })

        # Top 5
        top5 = np.argsort(np.abs(d_values))[::-1][:5]
        print(f"  Top 5 by |d|:")
        for rank, idx in enumerate(top5):
            print(f"    {rank+1}. {feature_cols[idx]}: d = {d_values[idx]:+.4f} "
                  f"({categorize_d(d_values[idx])})")

    # -- Save per-trial results ----------------------------------------
    df_per_trial = pd.DataFrame(per_trial_results)
    df_per_trial.to_csv(os.path.join(OUTPUT_DIR, 'cohens_d_per_trial.csv'), index=False)

    # -- Aggregate across trials ---------------------------------------
    df_agg = df_per_trial.groupby('feature').agg(
        d_mean=('cohens_d', 'mean'),
        d_std=('cohens_d', 'std'),
        abs_d_mean=('abs_d', 'mean'),
        abs_d_std=('abs_d', 'std'),
    ).reset_index()

    df_agg['band'] = df_agg['feature'].apply(lambda x: parse_feature_name(x)[0])
    df_agg['type'] = df_agg['feature'].apply(lambda x: parse_feature_name(x)[1])
    df_agg['category'] = df_agg['abs_d_mean'].apply(categorize_d)
    df_agg = df_agg.sort_values('abs_d_mean', ascending=False)
    df_agg.to_csv(os.path.join(OUTPUT_DIR, 'cohens_d_all.csv'), index=False)

    # -- Category counts -----------------------------------------------
    category_counts = df_agg['category'].value_counts()

    # -- Distribution histogram ----------------------------------------
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_agg['d_mean'], nbinsx=40,
        marker_color='#3498db', opacity=0.8,
        name="Cohen's d",
    ))

    for threshold, label, color in [
        (0.8, '|d|=0.8 (Large)', '#e74c3c'),
        (-0.8, '', '#e74c3c'),
        (0.5, '|d|=0.5 (Medium)', '#f39c12'),
        (-0.5, '', '#f39c12'),
        (0.2, '|d|=0.2 (Small)', '#95a5a6'),
        (-0.2, '', '#95a5a6'),
    ]:
        fig_hist.add_vline(
            x=threshold, line_dash="dash", line_color=color,
            annotation_text=label if label else None,
            annotation_position="top" if threshold > 0 else None,
        )

    fig_hist.update_layout(
        title="Distribution of Cohen's d Effect Sizes (mean across 10 trials)",
        xaxis_title="Cohen's d",
        yaxis_title="Number of Features",
        template="plotly_white",
        font=dict(color="black", size=14),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    pio.write_image(fig_hist, os.path.join(OUTPUT_DIR, 'cohens_d_distribution.png'),
                    width=1000, height=600, scale=2)
    print(f"Saved: cohens_d_distribution.png")

    # -- Top 30 bar chart ----------------------------------------------
    top30 = df_agg.head(30)
    fig_top = px.bar(
        top30, x='feature', y='d_mean',
        error_y='d_std', color='band',
        title="Top 30 Features by |Cohen's d| (mean across 10 trials)",
        labels={'d_mean': "Cohen's d", 'feature': 'Feature'},
        color_discrete_map={
            'delta': '#e74c3c', 'theta': '#3498db', 'alpha': '#2ecc71', 'beta': '#f39c12'
        },
    )
    fig_top.update_layout(
        template="plotly_white", font=dict(color="black", size=12),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_tickangle=-45,
    )
    pio.write_image(fig_top, os.path.join(OUTPUT_DIR, 'cohens_d_top30.png'),
                    width=1400, height=700, scale=2)
    print(f"Saved: cohens_d_top30.png")

    # -- By band violin ------------------------------------------------
    fig_band = px.violin(
        df_agg, x='band', y='abs_d_mean', color='band',
        box=True, points='all',
        title="|Cohen's d| Distribution by Frequency Band",
        labels={'abs_d_mean': "|Cohen's d|", 'band': 'Frequency Band'},
        color_discrete_map={
            'delta': '#e74c3c', 'theta': '#3498db', 'alpha': '#2ecc71', 'beta': '#f39c12'
        },
    )
    fig_band.update_layout(
        template="plotly_white", font=dict(color="black", size=14),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    pio.write_image(fig_band, os.path.join(OUTPUT_DIR, 'cohens_d_by_band.png'),
                    width=800, height=600, scale=2)
    print(f"Saved: cohens_d_by_band.png")

    # -- Summary text --------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, 'effect_size_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STUDY 7: COHEN'S d EFFECT SIZE — SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("d = (mu_1 - mu_0) / sigma_pooled\n")
        f.write("sigma_pooled = sqrt((sigma_1^2 + sigma_0^2) / 2)\n\n")
        f.write("Unlike p-values, Cohen's d does NOT inflate with sample size.\n")
        f.write("With thousands of epochs, p < 0.001 for nearly everything.\n")
        f.write("Effect size tells you how MEANINGFULLY different distributions are.\n\n")

        f.write(f"Total features analyzed: {len(df_agg)}\n\n")

        f.write("EFFECT SIZE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for cat in ['Large', 'Medium', 'Small', 'Negligible']:
            count = category_counts.get(cat, 0)
            pct = 100.0 * count / len(df_agg)
            f.write(f"  {cat:<12s}: {count:4d} features ({pct:5.1f}%)\n")

        f.write(f"\n\nTOP 30 FEATURES BY |COHEN'S d|\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rank':<6s} {'Feature':<30s} {'Band':<8s} {'Type':<12s} "
                f"{'d (mean±std)':<20s} {'Category':<12s}\n")
        f.write("-" * 70 + "\n")
        for rank, (_, row) in enumerate(top30.iterrows()):
            f.write(f"{rank+1:<6d} {row['feature']:<30s} {row['band']:<8s} {row['type']:<12s} "
                    f"{row['d_mean']:+.4f} ± {row['d_std']:.4f}   {row['category']:<12s}\n")

        f.write(f"\n\nEFFECT SIZE BY BAND (mean |d|)\n")
        f.write("-" * 40 + "\n")
        band_means = df_agg.groupby('band')['abs_d_mean'].mean().sort_values(ascending=False)
        for band, mean_d in band_means.items():
            f.write(f"  {band:<10s}: |d| = {mean_d:.4f}\n")

        f.write(f"\nEFFECT SIZE BY TYPE (mean |d|)\n")
        f.write("-" * 40 + "\n")
        type_means = df_agg.groupby('type')['abs_d_mean'].mean().sort_values(ascending=False)
        for ftype, mean_d in type_means.items():
            f.write(f"  {ftype:<12s}: |d| = {mean_d:.4f}\n")

        f.write(f"\n\n{'=' * 80}\n")
        f.write("INTERPRETATION\n")
        f.write(f"{'=' * 80}\n\n")

        n_large = category_counts.get('Large', 0)
        n_medium = category_counts.get('Medium', 0)
        if n_large + n_medium > len(df_agg) * 0.3:
            f.write(f"[PASS] {n_large + n_medium}/{len(df_agg)} features ({100*(n_large+n_medium)/len(df_agg):.0f}%) "
                    f"have medium-to-large effect sizes.\n")
            f.write(f"   -> Spectral slope features show strong, clinically meaningful\n")
            f.write(f"     separation between seizure and non-seizure distributions.\n")
        else:
            f.write(f"[WARN]  Only {n_large + n_medium}/{len(df_agg)} features have medium-to-large effects.\n")
            f.write(f"   -> Most features have weak individual effect sizes, but may\n")
            f.write(f"     combine to produce strong classification performance.\n")

    print(f"Saved summary: {summary_path}")

    print_header("STUDY 7 COMPLETE")
    print(f"  Results: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

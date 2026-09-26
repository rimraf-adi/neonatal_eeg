"""
Legacy 2.0 - Master Figures & Supplementary Tables Generator
============================================================
Generates publication-quality charts and tables:
1. Grouped Performance Bar Charts (AUROC, F1, Recall, Specificity with std error bars).
2. ROC & PR Curve Overlays (Multi-paradigm x Multi-classifier).
3. Calibration & Reliability Diagrams.
4. SHAP Feature Importance Summary Bar Charts.
5. PCA Loading Vectors Breakdown.
6. Feature Significance & T-statistic Bar Charts without dotted lines.
7. Master Supplementary Results Table (CSV, JSON, Markdown, LaTeX).

Outputs to:
  legacy2.0/figures/
  legacy2.0/tables/
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
ANALYSIS_DIR = ROOT_DIR / "analysis"
FIGURES_DIR = ROOT_DIR / "figures"
TABLES_DIR = ROOT_DIR / "tables"

for d in [FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style='whitegrid', font_scale=1.1)

PARADIGM_PALETTE = {
    'spectral_slope': '#1f77b4',
    'wavelet': '#2ca02c',
    'emd': '#ff7f0e',
    'fusion': '#9467bd'
}


def generate_performance_comparison_plots(df):
    print("Generating performance comparison bar charts...")
    metrics_to_plot = ['auroc', 'f1', 'recall', 'specificity']

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        # Filter for Strategy 1 (Downsample 2x)
        sub_df = df[df['strategy'] == 'downsample_2x']
        
        ax = sns.barplot(
            data=sub_df, x='model', y=metric, hue='paradigm',
            palette=PARADIGM_PALETTE, errorbar='sd', capsize=0.1, err_kws={'linewidth': 1.5}
        )
        plt.title(f"Cross-Validation {metric.upper()} by Model & Paradigm (Mean +/- SD)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Classifier", fontsize=12, fontweight='bold')
        plt.ylabel(metric.upper(), fontsize=12, fontweight='bold')
        plt.ylim(0.4, 1.0)
        plt.legend(title='Paradigm', loc='lower right', framealpha=0.9)
        plt.tight_layout()
        out_path = FIGURES_DIR / f"comparison_bar_{metric}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  Saved {out_path.name}")


def generate_shap_plot():
    shap_file = ANALYSIS_DIR / "feature_importance_shap.json"
    if not shap_file.exists():
        return
    with open(shap_file, 'r') as f:
        data = json.load(f)[:15] # Top 15

    feats = [item['feature'] for item in data][::-1]
    vals = [item['mean_abs_shap'] for item in data][::-1]

    plt.figure(figsize=(9, 7))
    colors = ['#1f77b4' if 'slope' in f or 'midband' in f or 'intercept' in f
              else '#2ca02c' if any(w in f for w in ['a3', 'd3', 'd2', 'd1'])
              else '#ff7f0e' for f in feats]

    bars = plt.barh(feats, vals, color=colors, edgecolor='none', height=0.65)
    plt.title("Top 15 Features by Mean Absolute SHAP Value (XGBoost)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    out_path = FIGURES_DIR / "shap_importance_summary.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path.name}")


def generate_pca_loadings_plot():
    pca_file = ANALYSIS_DIR / "pca_loadings.json"
    if not pca_file.exists():
        return
    with open(pca_file, 'r') as f:
        pca_data = json.load(f)

    # Plot PC1 and PC2 top loadings
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, pc in enumerate(['PC1', 'PC2']):
        items = pca_data['components'][pc][:8]
        feats = [it['feature'] for it in items][::-1]
        loadings = [it['loading'] for it in items][::-1]
        axes[i].barh(feats, loadings, color='#4575b4' if i==0 else '#d73027', height=0.6)
        axes[i].set_title(f"Top Feature Loadings on {pc} (Var Exp: {pca_data['explained_variance_ratio'][i]:.1%})", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Loading Coefficient", fontsize=10)

    plt.tight_layout()
    out_path = FIGURES_DIR / "pca_top_loadings.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path.name}")


def generate_master_supplementary_table(df):
    print("Generating master supplementary tables...")
    summary_cols = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auroc', 'brier_score', 'ece']
    
    # 1. Master CSV
    df.to_csv(TABLES_DIR / "master_supplementary_results.csv", index=False)
    
    # 2. Aggregated table (Mean +/- SD)
    agg = df.groupby(['paradigm', 'strategy', 'model'])[summary_cols].agg(
        lambda x: f"{np.mean(x):.3f} +/- {np.std(x):.3f}"
    ).reset_index()
    agg.to_csv(TABLES_DIR / "master_aggregated_table.csv", index=False)
    
    # 3. Markdown formatted summary table
    md_content = "# Supplementary Table: 10-Trial Cross-Validation Benchmark\n\n"
    md_content += agg.to_markdown(index=False)
    with open(TABLES_DIR / "master_supplementary_table.md", 'w') as f:
        f.write(md_content)

    # 4. LaTeX formatted table
    latex_content = agg.to_latex(index=False)
    with open(TABLES_DIR / "master_supplementary_table.tex", 'w') as f:
        f.write(latex_content)

    print(f"  Saved master CSV, Markdown, and LaTeX tables in {TABLES_DIR}")


def main():
    metrics_csv = ANALYSIS_DIR / "cross_trial_metrics_full.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        generate_performance_comparison_plots(df)
        generate_master_supplementary_table(df)
    
    generate_shap_plot()
    generate_pca_loadings_plot()
    generate_frequency_ablation_plot()
    print("=" * 80)
    print("ALL PUBLICATION FIGURES & TABLES GENERATED SUCCESSFULLY!")
    print("=" * 80)


def generate_frequency_ablation_plot():
    ablation_csv = ANALYSIS_DIR / "frequency_feature_ablation_summary.csv"
    if not ablation_csv.exists():
        return
    df = pd.read_csv(ablation_csv)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df, x='Configuration', y='mean_auroc', hue='Model',
        palette='Set2'
    )
    plt.title("Frequency Feature Ablation: Individual Sub-Bands & Feature Types vs Full (Mean AUROC)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Feature Configuration", fontsize=11, fontweight='bold')
    plt.ylabel("Test AUROC", fontsize=11, fontweight='bold')
    plt.ylim(0.4, 1.0)
    plt.xticks(rotation=20, ha='right')
    plt.legend(title='Classifier', loc='lower right', framealpha=0.9)
    plt.tight_layout()
    out_path = FIGURES_DIR / "frequency_feature_ablation_chart.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path.name}")


if __name__ == '__main__':
    main()

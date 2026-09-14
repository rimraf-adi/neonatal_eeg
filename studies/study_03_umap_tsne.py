"""
Study 3: UMAP / t-SNE Visualization
=====================================

Projects spectral slope features into 2D using UMAP and t-SNE.
Creates scatter plots colored by seizure/non-seizure.
Computes Silhouette Score and Davies-Bouldin Index.

If distinct clusters appear without any model, the features
inherently separate seizure from background EEG.

Output:
  study_results/03_umap_tsne/
    umap_scatter.png
    tsne_scatter.png
    umap_scatter.html (interactive)
    tsne_scatter.html (interactive)
    clustering_metrics.txt
    clustering_metrics_per_trial.csv

Usage:
  uv run --with umap-learn python -m studies.study_03_umap_tsne
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.io as pio

try:
    import umap
except ImportError:
    print("ERROR: umap-learn not installed.")
    print("Run: uv run --with umap-learn python -m studies.study_03_umap_tsne")
    sys.exit(1)

from studies.utils import (
    load_patient_splits, prepare_trial_data,
    check_cache, ensure_dir, print_header, print_subheader,
    RESULTS_BASE,
)

OUTPUT_DIR = os.path.join(RESULTS_BASE, '03_umap_tsne')

# Max samples per class for visualization (to keep plots readable)
MAX_SAMPLES_PER_CLASS = 5000


def subsample_balanced(X, y, max_per_class=MAX_SAMPLES_PER_CLASS):
    """Subsample to at most max_per_class per class for visualization."""
    indices = []
    for label in np.unique(y):
        label_idx = np.where(y == label)[0]
        if len(label_idx) > max_per_class:
            label_idx = np.random.choice(label_idx, max_per_class, replace=False)
        indices.extend(label_idx)
    indices = np.array(sorted(indices))
    return X[indices], y[indices]


def create_scatter(embedding, labels, method_name, output_dir):
    """Create and save scatter plot for a 2D embedding."""
    df_plot = pd.DataFrame({
        'dim1': embedding[:, 0],
        'dim2': embedding[:, 1],
        'label': ['Seizure' if l == 1 else 'Non-Seizure' for l in labels],
    })

    fig = px.scatter(
        df_plot, x='dim1', y='dim2', color='label',
        color_discrete_map={'Seizure': '#e74c3c', 'Non-Seizure': '#3498db'},
        title=f"{method_name} Projection of Spectral Slope Features",
        labels={'dim1': f'{method_name} Dim 1', 'dim2': f'{method_name} Dim 2'},
        opacity=0.5,
        category_orders={'label': ['Non-Seizure', 'Seizure']},
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(color="black", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title="Class",
    )
    fig.update_traces(marker=dict(size=4))

    # Save static PNG
    png_path = os.path.join(output_dir, f'{method_name.lower()}_scatter.png')
    pio.write_image(fig, png_path, width=1000, height=800, scale=2)
    print(f"  Saved: {png_path}")

    # Save interactive HTML
    html_path = os.path.join(output_dir, f'{method_name.lower()}_scatter.html')
    fig.write_html(html_path)
    print(f"  Saved: {html_path}")


def main():
    print_header("STUDY 3: UMAP / t-SNE VISUALIZATION")
    check_cache()
    ensure_dir(OUTPUT_DIR)

    splits = load_patient_splits()

    # -- Compute clustering metrics across all trials ------------------
    trial_metrics = []

    for trial_idx, split_info in enumerate(splits):
        trial_num = split_info.get('trial', trial_idx + 1)
        print_subheader(f"Trial {trial_num}/10 — Clustering Metrics")

        data = prepare_trial_data(split_info, apply_pca=False)
        X_train, y_train = data['X_train'], data['y_train']

        # Subsample for speed
        X_sub, y_sub = subsample_balanced(X_train, y_train)
        print(f"  Samples: {len(y_sub)} (seizure={np.sum(y_sub==1)}, non-seizure={np.sum(y_sub==0)})")

        sil = silhouette_score(X_sub, y_sub, sample_size=min(5000, len(y_sub)))
        dbi = davies_bouldin_score(X_sub, y_sub)

        trial_metrics.append({
            'trial': trial_num,
            'silhouette_score': sil,
            'davies_bouldin_index': dbi,
            'n_samples': len(y_sub),
        })

        print(f"  Silhouette Score: {sil:.4f}")
        print(f"  Davies-Bouldin:   {dbi:.4f}")

    # Save per-trial metrics
    df_metrics = pd.DataFrame(trial_metrics)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'clustering_metrics_per_trial.csv'), index=False)

    # -- Generate UMAP & t-SNE plots for Trial 1 (representative) -----
    print_subheader("Generating UMAP & t-SNE Projections (Trial 1)")

    data = prepare_trial_data(splits[0], apply_pca=False)
    X_train, y_train = data['X_train'], data['y_train']

    X_sub, y_sub = subsample_balanced(X_train, y_train)
    print(f"  Visualization samples: {len(y_sub)}")

    # UMAP
    print(f"  Computing UMAP embedding...", flush=True)
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_embedding = umap_reducer.fit_transform(X_sub)
    create_scatter(umap_embedding, y_sub, 'UMAP', OUTPUT_DIR)

    # t-SNE
    print(f"  Computing t-SNE embedding...", flush=True)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_embedding = tsne.fit_transform(X_sub)
    create_scatter(tsne_embedding, y_sub, 'tSNE', OUTPUT_DIR)

    # -- Write clustering metrics summary ------------------------------
    summary_path = os.path.join(OUTPUT_DIR, 'clustering_metrics.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STUDY 3: UMAP / t-SNE — CLUSTERING METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Metrics computed on raw (non-PCA) spectral slope features.\n")
        f.write("Silhouette Score: [-1, 1], higher = better cluster separation.\n")
        f.write("Davies-Bouldin Index: [0, ∞), lower = better cluster separation.\n\n")

        f.write(f"{'Trial':<8s} {'Silhouette':<14s} {'Davies-Bouldin':<16s} {'N Samples':<12s}\n")
        f.write("-" * 50 + "\n")
        for _, row in df_metrics.iterrows():
            f.write(f"{int(row['trial']):<8d} {row['silhouette_score']:<14.4f} "
                    f"{row['davies_bouldin_index']:<16.4f} {int(row['n_samples']):<12d}\n")

        f.write(f"\n{'-' * 50}\n")
        f.write(f"{'Mean':<8s} {df_metrics['silhouette_score'].mean():<14.4f} "
                f"{df_metrics['davies_bouldin_index'].mean():<16.4f}\n")
        f.write(f"{'Std':<8s} {df_metrics['silhouette_score'].std():<14.4f} "
                f"{df_metrics['davies_bouldin_index'].std():<16.4f}\n")

        f.write(f"\n\n{'=' * 80}\n")
        f.write("INTERPRETATION\n")
        f.write(f"{'=' * 80}\n\n")

        mean_sil = df_metrics['silhouette_score'].mean()
        if mean_sil > 0.25:
            f.write(f"[PASS] Mean Silhouette = {mean_sil:.4f} (> 0.25)\n")
            f.write(f"   -> Features show meaningful cluster structure.\n")
        elif mean_sil > 0.0:
            f.write(f"[WARN]  Mean Silhouette = {mean_sil:.4f} (0 < x < 0.25)\n")
            f.write(f"   -> Weak but present cluster structure.\n")
        else:
            f.write(f"[FAIL] Mean Silhouette = {mean_sil:.4f} (≤ 0)\n")
            f.write(f"   -> No meaningful cluster separation in feature space.\n")

    print(f"Saved summary: {summary_path}")

    print_header("STUDY 3 COMPLETE")
    print(f"  Results: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

"""
Export all possible dashboard screenshots as static PNG images.
Generates every combination of dataset, split, metric, and trial mode.

Usage:
    python export_screenshots.py
    
Output:
    ./dashboard_screenshots/
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import glob
import json
import numpy as np

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR = "./dashboard_screenshots"
PLOTLY_TEMPLATE = "plotly_white"
SCALE = 3  # Higher = better resolution

RESULTS_DIRS = {
    "Frequency Features": "./adaptive_nn_results",
    "EMD Features": "./adaptive_emd_results",
    "PCA Frequency Features": "./pca_adaptive_nn_results",
    "PCA EMD Features": "./pca_adaptive_emd_results"
}

METRIC_OPTIONS = ["F1", "AUROC", "Precision", "Recall", "Accuracy"]

DATASET_MAP = {
    "Frequency Features": "freq",
    "EMD Features": "emd",
    "PCA Frequency Features": "pca_freq",
    "PCA EMD Features": "pca_emd"
}

LIGHT_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black", size=14),
    xaxis=dict(
        title_font=dict(color="black", size=16),
        tickfont=dict(color="black", size=14),
        color="black",
        gridcolor="#e0e0e0",
    ),
    yaxis=dict(
        title_font=dict(color="black", size=16),
        tickfont=dict(color="black", size=14),
        color="black",
        gridcolor="#e0e0e0",
    ),
)


# ============================================================================
# Data Loading (same logic as dashboard.py)
# ============================================================================
def load_data(base_dir, dataset_name):
    data = []
    search_pattern = os.path.join(base_dir, "trial_*", "detailed", "*", "ma*", "*.txt")
    files = glob.glob(search_pattern)

    for filepath in files:
        try:
            parts = filepath.split(os.sep)
            try:
                detailed_idx = parts.index("detailed")
            except ValueError:
                continue

            trial_str = parts[detailed_idx - 1]
            split = parts[detailed_idx + 1]

            match_trial = re.search(r'trial_(\d+)', trial_str)
            trial_num = int(match_trial.group(1)) if match_trial else -1

            with open(filepath, 'r') as f:
                content = f.read()

            ma_match = re.search(r'MA Window:\s*(\d+)', content)
            thresh_match = re.search(r'Threshold:\s*([\d\.]+)', content)
            prec_match = re.search(r'Precision:\s*([\d\.]+)', content)
            rec_match = re.search(r'Recall:\s*([\d\.]+)', content)
            f1_match = re.search(r'F1 Score:\s*([\d\.]+)', content)
            acc_match = re.search(r'Accuracy:\s*([\d\.]+)', content)
            auc_match = re.search(r'AUROC:\s*([\d\.]+)', content)

            if all([ma_match, thresh_match, prec_match, rec_match, f1_match, acc_match, auc_match]):
                entry = {
                    "Dataset": dataset_name,
                    "Trial": trial_num,
                    "Split": split.capitalize(),
                    "MA_Window": int(ma_match.group(1)),
                    "Threshold": float(thresh_match.group(1)),
                    "Precision": float(prec_match.group(1)),
                    "Recall": float(rec_match.group(1)),
                    "F1": float(f1_match.group(1)),
                    "Accuracy": float(acc_match.group(1)),
                    "AUROC": float(auc_match.group(1))
                }
                data.append(entry)
        except Exception:
            continue

    return pd.DataFrame(data)


def load_ttest_data(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def safe_name(s):
    """Make a filesystem-safe name."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(s)).lower().strip('_')


# ============================================================================
# Chart Generators
# ============================================================================
def make_heatmap(df_viz, metric, dataset, split, trial_label):
    heatmap_data = df_viz.pivot(index="MA_Window", columns="Threshold", values=metric)

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Probability Threshold", y="MA Window", color=metric),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        origin="lower"
    )
    fig.update_layout(
        **LIGHT_LAYOUT,
        title=f"Heatmap: {metric} — {dataset} / {split} / {trial_label}",
        width=1200, height=700
    )
    return fig


def make_boxplot(df_filtered, metric, dataset, split):
    # 1. Identify Top 5 Configs based on average metric
    # We need to group by MA_Window and Threshold first to find best configs
    df_avg = df_filtered.groupby(["MA_Window", "Threshold"])[metric].mean().reset_index()
    top5_configs = df_avg.sort_values(metric, ascending=False).head(5)

    # 2. Filter original data to only these configs
    top5_keys = set(zip(top5_configs["MA_Window"], top5_configs["Threshold"]))
    
    df_top5_dist = df_filtered[
        df_filtered.apply(lambda row: (row["MA_Window"], row["Threshold"]) in top5_keys, axis=1)
    ].copy()

    # 3. Create readable label
    df_top5_dist["Config"] = df_top5_dist.apply(
        lambda x: f"MA:{x['MA_Window']} Th:{x['Threshold']:.2f}", axis=1
    )

    if df_top5_dist.empty:
        return None

    fig = px.box(
        df_top5_dist, 
        x="Config",
        y=metric, 
        points="all",
        title=f"{metric} Distribution (Top 5 Configs) — {dataset} / {split}",
        color="Config",
        color_discrete_sequence=["#FF0000", "#FF4500", "#FF8C00", "#FFA500", "#FFD700"]
    )
    fig.update_layout(
        **LIGHT_LAYOUT, 
        width=800, height=600,
        showlegend=False
    )
    return fig


def make_line_chart(df_viz, metric, ma_windows, dataset, split, trial_label):
    df_line = df_viz[df_viz["MA_Window"].isin(ma_windows)]
    if df_line.empty:
        return None

    fig = px.line(
        df_line, x="Threshold", y=metric, color="MA_Window",
        markers=True,
        title=f"{metric} vs Threshold — {dataset} / {split} / {trial_label}",
        color_discrete_sequence=["#d62728", "#e45a33", "#f4a041", "#ffc857", "#cc5500", "#b22222", "#ff6347"]
    )
    fig.update_layout(**LIGHT_LAYOUT, width=1200, height=600)
    return fig


def make_ttest_chart(ttest_data, feat_key, dataset, trial=None):
    if feat_key not in ttest_data:
        return None

    feat_results = ttest_data[feat_key]
    plot_data = []

    if trial is not None:
        trial_res = next((t for t in feat_results if t['trial'] == trial), None)
        if not trial_res:
            return None
        for feat, res in trial_res['results'].items():
            plot_data.append({"Feature": feat, "-log10(p)": res['-log10p']})
        title_suffix = f"Trial {trial}"
    else:
        all_feats = {}
        for t in feat_results:
            for f, r in t['results'].items():
                if f not in all_feats:
                    all_feats[f] = []
                all_feats[f].append(r['-log10p'])
        for f, vals in all_feats.items():
            plot_data.append({"Feature": f, "-log10(p)": sum(vals) / len(vals)})
        title_suffix = "Averaged"

    if not plot_data:
        return None

    df_ttest = pd.DataFrame(plot_data).sort_values("-log10(p)", ascending=False)

    fig = px.bar(
        df_ttest, x="Feature", y="-log10(p)",
        title=f"Feature Significance — {dataset} / {title_suffix}",
        color="-log10(p)", color_continuous_scale="OrRd",
    )
    fig.update_layout(**LIGHT_LAYOUT, width=1400, height=600)
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#8b4513", annotation_text="p=0.05")
    fig.add_hline(y=-np.log10(0.001), line_dash="dash", line_color="#cc5500", annotation_text="p=0.001")
    return fig


def make_top5_table_chart(df_viz, metric, dataset, split, trial_label):
    """Create a table figure of the top 5 configurations by a given metric (Accuracy or F1 only)."""
    top5 = df_viz.sort_values(metric, ascending=False).head(5)
    cols = ["MA_Window", "Threshold", "Accuracy", "F1"]
    cols = [c for c in cols if c in top5.columns]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in cols],
            fill_color="#f8f9fa",
            font=dict(color="black", size=13),
            align="center"
        ),
        cells=dict(
            values=[top5[c].round(4) for c in cols],
            fill_color="white",
            font=dict(color="black", size=12),
            align="center",
            line=dict(color="#dddddd")
        )
    )])
    fig.update_layout(
        title=f"Top 5 by {metric} — {dataset} / {split} / {trial_label}",
        paper_bgcolor="white", width=1200, height=350
    )
    return fig


# ============================================================================
# Main Export
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all data
    all_frames = []
    for name, path in RESULTS_DIRS.items():
        if os.path.exists(path):
            df = load_data(path, name)
            if not df.empty:
                all_frames.append(df)

    if not all_frames:
        print("❌ No data found. Run training scripts first.")
        return

    df_all = pd.concat(all_frames, ignore_index=True)

    # Load t-test data
    ttest_path = os.path.join(os.path.dirname(__file__), "ttest_results", "trial_ttests.json")
    ttest_data = load_ttest_data(ttest_path)

    count = 0
    datasets = df_all["Dataset"].unique()
    splits = df_all["Split"].unique()

    for dataset in datasets:
        for split in splits:
            df_filt = df_all[
                (df_all["Dataset"] == dataset) &
                (df_all["Split"] == split)
            ]
            if df_filt.empty:
                continue

            # Filter to threshold range 0.49–0.55
            min_t, max_t = df_filt["Threshold"].min(), df_filt["Threshold"].max()
            lo = max(min_t, 0.49)
            hi = min(max_t, 0.55)
            df_filt = df_filt[(df_filt["Threshold"] >= lo) & (df_filt["Threshold"] <= hi)]
            if df_filt.empty:
                continue

            ds_safe = safe_name(dataset)
            sp_safe = safe_name(split)
            prefix = f"{ds_safe}__{sp_safe}"

            for metric in METRIC_OPTIONS:
                m_safe = safe_name(metric)

                # ---- Averaged across trials ----
                df_avg = df_filt.groupby(["MA_Window", "Threshold"])[METRIC_OPTIONS].mean().reset_index()

                # Heatmap
                fig = make_heatmap(df_avg, metric, dataset, split, "Avg")
                path = os.path.join(OUTPUT_DIR, f"{prefix}__{m_safe}__heatmap_avg.png")
                fig.write_image(path, scale=SCALE)
                count += 1

                # Box plot
                fig = make_boxplot(df_filt, metric, dataset, split)
                if fig:
                    path = os.path.join(OUTPUT_DIR, f"{prefix}__{m_safe}__boxplot_avg.png")
                    fig.write_image(path, scale=SCALE)
                    count += 1

                # Line chart (top MA windows)
                top_mas = df_avg.sort_values(metric, ascending=False)["MA_Window"].unique()[:4].tolist()
                fig = make_line_chart(df_avg, metric, top_mas, dataset, split, "Avg")
                if fig:
                    path = os.path.join(OUTPUT_DIR, f"{prefix}__{m_safe}__line_avg.png")
                    fig.write_image(path, scale=SCALE)
                    count += 1

                # Top 5 tables (Accuracy and F1 only)
                for rank_metric in ["Accuracy", "F1"]:
                    rm_safe = safe_name(rank_metric)
                    fig = make_top5_table_chart(df_avg, rank_metric, dataset, split, "Avg")
                    path = os.path.join(OUTPUT_DIR, f"{prefix}__top5_{rm_safe}_avg.png")
                    fig.write_image(path, scale=SCALE)
                    count += 1

                # ---- Per trial ----
                for trial in sorted(df_filt["Trial"].unique()):
                    df_trial = df_filt[df_filt["Trial"] == trial]
                    t_label = f"Trial_{trial:02d}"
                    t_safe = safe_name(t_label)

                    # Heatmap
                    try:
                        fig = make_heatmap(df_trial, metric, dataset, split, t_label)
                        path = os.path.join(OUTPUT_DIR, f"{prefix}__{m_safe}__heatmap_{t_safe}.png")
                        fig.write_image(path, scale=SCALE)
                        count += 1
                    except Exception:
                        pass

                    # Line chart
                    try:
                        trial_mas = df_trial.sort_values(metric, ascending=False)["MA_Window"].unique()[:4].tolist()
                        fig = make_line_chart(df_trial, metric, trial_mas, dataset, split, t_label)
                        if fig:
                            path = os.path.join(OUTPUT_DIR, f"{prefix}__{m_safe}__line_{t_safe}.png")
                            fig.write_image(path, scale=SCALE)
                            count += 1
                    except Exception:
                        pass

            # ---- T-Test charts ----
            if ttest_data:
                feat_key = DATASET_MAP.get(dataset)
                if feat_key:
                    # Averaged
                    fig = make_ttest_chart(ttest_data, feat_key, dataset)
                    if fig:
                        path = os.path.join(OUTPUT_DIR, f"{prefix}__ttest_avg.png")
                        fig.write_image(path, scale=SCALE)
                        count += 1

                    # Per trial
                    for trial in sorted(df_filt["Trial"].unique()):
                        fig = make_ttest_chart(ttest_data, feat_key, dataset, trial=trial)
                        if fig:
                            t_safe = safe_name(f"trial_{trial:02d}")
                            path = os.path.join(OUTPUT_DIR, f"{prefix}__ttest_{t_safe}.png")
                            fig.write_image(path, scale=SCALE)
                            count += 1

            print(f"  ✅ {dataset} / {split} done")

    print(f"\n🎉 Exported {count} screenshots to {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()

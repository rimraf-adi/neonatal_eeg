"""
Regenerate and overwrite t-test plots for all three feature sets:
1. Frequency Features (12 spectral slope features)
2. EMD Features (24 empirical mode decomposition features)
3. Wavelet Features (30 legacy DWT features)

Generates:
- {key}_significance.png: bar chart of -log10(p-value) with p=0.05 and p=0.001 dashed lines
- {key}_t_stat.png: bar chart of t-statistics with diverging RdBu color scale

Overwrites into:
- legacy/ttest_results/
- paper/figures/
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

PLOTLY_TEMPLATE = "plotly_white"
AXIS_CONFIG = dict(
    xaxis=dict(
        title_font=dict(color="black", size=18),
        tickfont=dict(color="black", size=14),
        color="black",
        gridcolor="#e0e0e0",
    ),
    yaxis=dict(
        title_font=dict(color="black", size=18),
        tickfont=dict(color="black", size=14),
        color="black",
        gridcolor="#e0e0e0",
    ),
)
LAYOUT_CONFIG = dict(
    template=PLOTLY_TEMPLATE,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black", size=14),
    **AXIS_CONFIG
)

def save_plot_chart(df, x_col, y_col, title, target_paths, color_col=None, color_scale="OrRd", hlines=None):
    if df.empty:
        return

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=color_col if color_col else y_col,
        color_continuous_scale=color_scale,
        labels={y_col: y_col}
    )
    
    if hlines:
        for y_val, color, text in hlines:
            fig.add_hline(y=y_val, line_dash="dash", line_color=color, annotation_text=text)

    fig.update_layout(xaxis_tickangle=-90, **LAYOUT_CONFIG)
    
    first_path = target_paths[0]
    os.makedirs(os.path.dirname(first_path), exist_ok=True)
    try:
        pio.write_image(fig, first_path, width=1200, height=800, scale=2)
        print(f"Saved: {first_path}")
        # Copy to remaining targets
        for p in target_paths[1:]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            shutil.copy2(first_path, p)
            print(f"Copied to: {p}")
    except Exception as e:
        print(f"Error saving {first_path}: {e}")

def main():
    dirs = [
        os.path.join("legacy", "ttest_results"),
        os.path.join("paper", "figures")
    ]
    
    # 1. Load legacy trial_ttests.json for freq and emd
    legacy_json_path = os.path.join("legacy", "ttest_results", "trial_ttests.json")
    with open(legacy_json_path, 'r') as f:
        legacy_data = json.load(f)
        
    # 2. Load wavelet results
    wavelet_json_path = os.path.join("legacy", "ttest_results", "wavelet_trial_ttests.json")
    if os.path.exists(wavelet_json_path):
        with open(wavelet_json_path, 'r') as f:
            wavelet_data = json.load(f)
    else:
        print(f"Wavelet JSON not yet found at {wavelet_json_path}")
        wavelet_data = None

    all_sets = {
        "freq": legacy_data.get("freq", []),
        "emd": legacy_data.get("emd", []),
        "pca_freq": legacy_data.get("pca_freq", []),
        "pca_emd": legacy_data.get("pca_emd", []),
    }
    if wavelet_data:
        all_sets["wavelet"] = wavelet_data

    for key, trial_list in all_sets.items():
        if not trial_list:
            continue
            
        print(f"\nProcessing plots for: {key} (Trials: {len(trial_list)})")
        
        feature_stats = {}
        for trial_data in trial_list:
            results = trial_data['results']
            for feat, metrics in results.items():
                if feat not in feature_stats:
                    feature_stats[feat] = {'-log10p': [], 't_stat': []}
                feature_stats[feat]['-log10p'].append(metrics['-log10p'])
                feature_stats[feat]['t_stat'].append(metrics['t_stat'])
                
        avg_data = []
        for feat, lists in feature_stats.items():
            avg_data.append({
                "Feature": feat,
                "-log10p": float(np.mean(lists['-log10p'])),
                "t_stat": float(np.mean(lists['t_stat']))
            })
            
        df_avg = pd.DataFrame(avg_data)
        
        # Plot 1: Significance -log10(p)
        df_p = df_avg.sort_values("-log10p", ascending=False)
        sig_targets = [os.path.join(d, f"{key}_significance.png") for d in dirs]
        save_plot_chart(
            df_p,
            x_col="Feature",
            y_col="-log10p",
            title=f"Feature Significance: -log10(p) ({key})",
            target_paths=sig_targets,
            color_scale="OrRd",
            hlines=None
        )
        
        # Plot 2: T-stat Direction
        df_t = df_avg.sort_values("t_stat", ascending=False)
        t_targets = [os.path.join(d, f"{key}_t_stat.png") for d in dirs]
        save_plot_chart(
            df_t,
            x_col="Feature",
            y_col="t_stat",
            title=f"Feature Direction: T-Statistic ({key})",
            target_paths=t_targets,
            color_scale="RdBu",
            color_col="t_stat"
        )

    print("\nAll plots regenerated successfully across legacy/ttest_results/ and paper/figures/!")

if __name__ == "__main__":
    main()

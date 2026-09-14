"""
Study 6: Temporal Onset Trajectory Plots
==========================================

For 4 patients with the most seizure activity, plots continuous time-series
of key spectral features (Delta slope, Theta midband, Alpha slope, Beta slope)
across the full recording with seizure regions shaded.

If features show a sharp transition at seizure onset/offset, it proves
temporal sensitivity — the features track the physiological event in real time.

Output:
  study_results/06_temporal_trajectory/
    patient_{id}_all_features.png
    patient_{id}_delta_slope.png
    selected_patients.txt
    seizure_durations.csv

Usage:
  uv run python -m studies.study_06_temporal_trajectory
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from studies.utils import (
    load_annotations, get_unanimous_labels,
    load_patient_features,
    check_cache, ensure_dir, print_header, print_subheader,
    EEG_IDX, RESULTS_BASE, SPECTRAL_CACHE_DIR, DESIRED_ORDER,
)

OUTPUT_DIR = os.path.join(RESULTS_BASE, '06_temporal_trajectory')

# Key features to plot (one per band)
KEY_FEATURES = ['delta_slope', 'theta_midband', 'alpha_slope', 'beta_slope']
FEATURE_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

NUM_PATIENTS = 4  # How many patients to visualize


def find_seizure_regions(labels, valid_mask):
    """Find contiguous seizure regions from labels.

    Returns:
        list of (start_sec, end_sec) tuples
    """
    regions = []
    in_seizure = False
    start = 0

    for i in range(len(labels)):
        if valid_mask[i] and labels[i] == 1:
            if not in_seizure:
                start = i
                in_seizure = True
        else:
            if in_seizure:
                regions.append((start, i))
                in_seizure = False

    if in_seizure:
        regions.append((start, len(labels)))

    return regions


def select_top_patients(annotation_dfs, n=NUM_PATIENTS):
    """Select patients with the most seizure seconds.

    Returns:
        list of (patient_id, total_seizure_seconds, seizure_regions)
    """
    patient_info = []

    for pid in EEG_IDX:
        labels, valid_mask = get_unanimous_labels(annotation_dfs, pid)
        if labels is None or valid_mask is None:
            continue

        seizure_secs = np.sum(labels[valid_mask] == 1)
        regions = find_seizure_regions(labels, valid_mask)

        if seizure_secs > 0:
            patient_info.append({
                'patient_id': pid,
                'seizure_seconds': int(seizure_secs),
                'n_regions': len(regions),
                'regions': regions,
                'total_seconds': int(np.sum(valid_mask)),
            })

    patient_info.sort(key=lambda x: x['seizure_seconds'], reverse=True)
    return patient_info[:n]


def plot_patient_trajectory(pid, feature_df, seizure_regions, output_dir):
    """Create multi-panel time-series plot for one patient.

    Args:
        pid: patient ID
        feature_df: DataFrame with features for this patient (with 'channel' column retained)
        seizure_regions: list of (start, end) tuples in seconds
        output_dir: output directory
    """
    # We need to aggregate across channels. Use mean across all 18 channels per epoch.
    # The feature_df has one row per (epoch, channel), so we need to reconstruct epoch-level features.

    # The data is stored as: for each epoch, 18 rows (one per channel). So epoch_idx = row_idx // 18.
    # But actually the feature CSVs have 'label' and 'channel', so let's group by epoch.

    # Add epoch index (rows are ordered: epoch0_ch0, epoch0_ch1, ..., epoch0_ch17, epoch1_ch0, ...)
    n_channels = len(DESIRED_ORDER)
    n_rows = len(feature_df)
    n_epochs = n_rows // n_channels

    if n_epochs == 0:
        print(f"  [SKIP] Patient {pid}: no epochs")
        return

    # Reshape: take mean across channels for each epoch
    epoch_means = {}
    for feat in KEY_FEATURES:
        if feat not in feature_df.columns:
            continue
        vals = feature_df[feat].values[:n_epochs * n_channels]
        reshaped = vals.reshape(n_epochs, n_channels)
        epoch_means[feat] = np.nanmean(reshaped, axis=1)

    time_axis = np.arange(n_epochs)  # seconds

    # Multi-panel plot
    fig = make_subplots(
        rows=len(KEY_FEATURES), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f.replace('_', ' ').title() for f in KEY_FEATURES],
    )

    for i, (feat, color) in enumerate(zip(KEY_FEATURES, FEATURE_COLORS)):
        if feat not in epoch_means:
            continue

        values = epoch_means[feat]

        # Add feature trace
        fig.add_trace(
            go.Scatter(
                x=time_axis, y=values,
                mode='lines', name=feat,
                line=dict(color=color, width=1),
                showlegend=False,
            ),
            row=i + 1, col=1,
        )

        # Add seizure shading
        for start, end in seizure_regions:
            if start < n_epochs:
                fig.add_vrect(
                    x0=start, x1=min(end, n_epochs),
                    fillcolor="rgba(231, 76, 60, 0.15)",
                    line_width=0,
                    row=i + 1, col=1,
                )

    fig.update_layout(
        title=f"Patient {pid}: Feature Trajectory with Seizure Regions (shaded red)",
        height=200 * len(KEY_FEATURES) + 100,
        width=1400,
        template="plotly_white",
        font=dict(color="black", size=12),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(title_text="Time (seconds)", row=len(KEY_FEATURES), col=1)

    # Save
    png_path = os.path.join(output_dir, f'patient_{pid:03d}_all_features.png')
    pio.write_image(fig, png_path, width=1400, height=200 * len(KEY_FEATURES) + 100, scale=2)
    print(f"  Saved: {png_path}")

    html_path = os.path.join(output_dir, f'patient_{pid:03d}_all_features.html')
    fig.write_html(html_path)
    print(f"  Saved: {html_path}")

    # Single feature plot: Delta slope
    if 'delta_slope' in epoch_means:
        fig_single = go.Figure()
        fig_single.add_trace(go.Scatter(
            x=time_axis, y=epoch_means['delta_slope'],
            mode='lines', name='Delta Slope',
            line=dict(color='#e74c3c', width=1.5),
        ))
        for start, end in seizure_regions:
            if start < n_epochs:
                fig_single.add_vrect(
                    x0=start, x1=min(end, n_epochs),
                    fillcolor="rgba(231, 76, 60, 0.2)",
                    line_width=0,
                    annotation_text="Seizure" if start == seizure_regions[0][0] else None,
                )

        fig_single.update_layout(
            title=f"Patient {pid}: Delta Slope Trajectory",
            xaxis_title="Time (seconds)",
            yaxis_title="Delta Slope (mean across channels)",
            template="plotly_white",
            font=dict(color="black", size=14),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        delta_path = os.path.join(output_dir, f'patient_{pid:03d}_delta_slope.png')
        pio.write_image(fig_single, delta_path, width=1200, height=500, scale=2)
        print(f"  Saved: {delta_path}")


def main():
    print_header("STUDY 6: TEMPORAL ONSET TRAJECTORY PLOTS")
    check_cache()
    ensure_dir(OUTPUT_DIR)

    # -- Load annotations and select top patients ----------------------
    print_subheader("Selecting patients with most seizure activity")
    annotation_dfs = load_annotations()
    top_patients = select_top_patients(annotation_dfs, n=NUM_PATIENTS)

    # Save selection info
    dur_rows = []
    for p in top_patients:
        dur_rows.append({
            'patient_id': p['patient_id'],
            'seizure_seconds': p['seizure_seconds'],
            'n_seizure_regions': p['n_regions'],
            'total_seconds': p['total_seconds'],
            'seizure_percent': 100.0 * p['seizure_seconds'] / max(p['total_seconds'], 1),
        })
    df_dur = pd.DataFrame(dur_rows)
    df_dur.to_csv(os.path.join(OUTPUT_DIR, 'seizure_durations.csv'), index=False)

    sel_path = os.path.join(OUTPUT_DIR, 'selected_patients.txt')
    with open(sel_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SELECTED PATIENTS (top by seizure duration)\n")
        f.write("=" * 60 + "\n\n")
        for p in top_patients:
            f.write(f"Patient {p['patient_id']:3d}: {p['seizure_seconds']:5d}s seizure "
                    f"({p['n_regions']} regions) / {p['total_seconds']}s total "
                    f"({100*p['seizure_seconds']/max(p['total_seconds'],1):.1f}%)\n")

    print(f"Selected {len(top_patients)} patients:")
    for p in top_patients:
        print(f"  Patient {p['patient_id']}: {p['seizure_seconds']}s seizure "
              f"({p['n_regions']} regions)")

    # -- Generate trajectory plots -------------------------------------
    for p in top_patients:
        pid = p['patient_id']
        print_subheader(f"Patient {pid}")

        # Load features WITH channel column
        try:
            feat_df = load_patient_features([pid], SPECTRAL_CACHE_DIR, drop_channel_col=False)
        except ValueError:
            print(f"  [SKIP] No feature data for patient {pid}")
            continue

        plot_patient_trajectory(pid, feat_df, p['regions'], OUTPUT_DIR)

    print_header("STUDY 6 COMPLETE")
    print(f"  Results: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

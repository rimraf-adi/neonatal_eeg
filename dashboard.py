import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import glob
import json
import numpy as np

# ============================================================================
# Configuration & Setup
# ============================================================================
st.set_page_config(layout="wide", page_title="EEG Analysis Dashboard")

RESULTS_DIRS = {
    "Frequency Features": "./adaptive_nn_results",
    "EMD Features": "./adaptive_emd_results"
}

# ============================================================================
# Data Loading
# ============================================================================
@st.cache_data
def load_ttest_data(filepath):
    """Load pre-calculated t-test results."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

@st.cache_data
def load_data(base_dir, dataset_name):
    """
    Crawl directory structure and load all result files.
    Structure: trial_XX/detailed/[validation|test]/maXX/maXX_threshX.XX.txt
    """
    data = []
    
    # improved glob pattern to catch all result files
    search_pattern = os.path.join(base_dir, "trial_*", "detailed", "*", "ma*", "*.txt")
    files = glob.glob(search_pattern)
    
    if not files:
        # Try fallback structure (some older results might be flat or different)
        # But based on current execution, they should be in trial_XX
        pass

    for filepath in files:
        try:
            # Parse path components
            # Expected parsed path: .../trial_00/detailed/validation/ma05/ma05_thresh0.50.txt
            parts = filepath.split(os.sep)
            
            # Find key components relative to "detailed"
            try:
                detailed_idx = parts.index("detailed")
            except ValueError:
                continue
                
            trial_str = parts[detailed_idx - 1] # trial_XX
            split = parts[detailed_idx + 1]     # validation or test
            
            # extract trial number
            match_trial = re.search(r'trial_(\d+)', trial_str)
            trial_num = int(match_trial.group(1)) if match_trial else -1
            
            # Parse file content
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Extract metrics using regex
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
                    "Split": split.capitalize(), # Validation/Test
                    "MA_Window": int(ma_match.group(1)),
                    "Threshold": float(thresh_match.group(1)),
                    "Precision": float(prec_match.group(1)),
                    "Recall": float(rec_match.group(1)),
                    "F1": float(f1_match.group(1)),
                    "Accuracy": float(acc_match.group(1)),
                    "AUROC": float(auc_match.group(1))
                }
                data.append(entry)
        except Exception as e:
            # st.error(f"Error parsing {filepath}: {e}")
            continue
            
    return pd.DataFrame(data)

# ============================================================================
# Main Dashboard Logic
# ============================================================================

st.title("🧠 EEG Biomarker Analysis Dashboard")

# 1. Load Data
all_data_frames = []
for name, path in RESULTS_DIRS.items():
    if os.path.exists(path):
        df = load_data(path, name)
        if not df.empty:
            all_data_frames.append(df)
    else:
        st.warning(f"Directory not found: {path} (Try running the training script first)")

if not all_data_frames:
    st.error("No data found! Please run the training scripts to generate results.")
    st.stop()

df_all = pd.concat(all_data_frames, ignore_index=True)

# 2. Sidebar Filters
st.sidebar.header("Filters")

# Reload button to clear cache and refresh data
if st.sidebar.button("🔄 Reload Data"):
    st.cache_data.clear()
    st.rerun()

# Dataset Selection
dataset_options = df_all["Dataset"].unique()
selected_dataset = st.sidebar.selectbox("Select Feature Set", dataset_options)

# Split Selection
split_options = df_all["Split"].unique()
selected_split = st.sidebar.radio("Select Data Split", split_options, index=0) # Default to Validation usually

# Filter Data
df_filtered = df_all[
    (df_all["Dataset"] == selected_dataset) & 
    (df_all["Split"] == selected_split)
]

# Threshold Filter
st.sidebar.markdown("---")
min_thresh, max_thresh = df_filtered["Threshold"].min(), df_filtered["Threshold"].max()
# Default to 0.4-0.7 if within range, else full range
default_min = 0.4 if min_thresh <= 0.4 <= max_thresh else min_thresh
default_max = 0.7 if min_thresh <= 0.7 <= max_thresh else max_thresh

selected_threshold_range = st.sidebar.slider(
    "Filter Threshold Range", 
    min_value=float(min_thresh), 
    max_value=float(max_thresh), 
    value=(float(default_min), float(default_max)),
    step=0.01
)

df_filtered = df_filtered[
    (df_filtered["Threshold"] >= selected_threshold_range[0]) & 
    (df_filtered["Threshold"] <= selected_threshold_range[1])
]

# Metric Selection
metric_options = ["F1", "AUROC", "Precision", "Recall", "Accuracy"]
selected_metric = st.sidebar.selectbox("Select Metric to Visualize", metric_options)

# Trial Aggregation
trial_mode = st.sidebar.radio("Trial Mode", ["Average across all trials", "Specific Trial"])
selected_trial = None
if trial_mode == "Specific Trial":
    trial_options = sorted(df_filtered["Trial"].unique())
    selected_trial = st.sidebar.selectbox("Select Trial", trial_options)
    df_viz = df_filtered[df_filtered["Trial"] == selected_trial]
    st.sidebar.markdown(f"**Viewing results for Trial {selected_trial}**")
else:
    # Aggregate
    df_viz = df_filtered.groupby(["MA_Window", "Threshold"])[metric_options].mean().reset_index()
    st.sidebar.markdown(f"**Viewing averaged results across {df_filtered['Trial'].nunique()} trials**")


# ============================================================================
# Visualizations
# ============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"🔥 Heatmap: {selected_metric} ")
    st.markdown(f"Optimal **Moving Average Window** vs **Probability Threshold**")
    
    # Pivot for heatmap
    heatmap_data = df_viz.pivot(index="MA_Window", columns="Threshold", values=selected_metric)
    
    # Find best value coordinates
    best_val = df_viz[selected_metric].max()
    best_row = df_viz[df_viz[selected_metric] == best_val].iloc[0]
    
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Probability Threshold", y="MA Window", color=selected_metric),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        aspect="auto",
        color_continuous_scale="Viridis",
        origin="lower" # Make sure MA=1 is at bottom or top typically? Standard is Y upwards.
    )
    
    # Annotate best point
    fig_heat.add_annotation(
        x=best_row["Threshold"],
        y=best_row["MA_Window"],
        text="★ Best",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
        font=dict(color="red", size=14)
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.info(
        f"**Best {selected_metric}: {best_val:.4f}**\n\n"
        f"MA Window: {int(best_row['MA_Window'])}, "
        f"Threshold: {best_row['Threshold']:.2f}"
    )

with col2:
    st.subheader("Results Distribution")
    
    if trial_mode == "Average across all trials":
        st.markdown("Variance across trials for the **Best Configuration** found above.")
        # Get data for best config across all trials
        best_ma = best_row["MA_Window"]
        best_thresh = best_row["Threshold"]
        
        df_best_config = df_filtered[
            (df_filtered["MA_Window"] == best_ma) &
            (df_filtered["Threshold"] == best_thresh)
        ]
        
        if not df_best_config.empty:
            fig_box = px.box(
                df_best_config, 
                y=selected_metric, 
                points="all",
                title=f"{selected_metric} Stability (MA={best_ma}, Th={best_thresh:.2f})",
                color="Dataset" # dummy color just for visuals
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.write(df_best_config[["Trial", selected_metric]].sort_values("Trial").set_index("Trial"))
    else:
        st.markdown("Detailed metrics for this trial.")
        st.dataframe(df_viz.sort_values(selected_metric, ascending=False).head(20))


# ============================================================================
# Detailed Line Analysis
# ============================================================================
st.markdown("---")
st.subheader("📈 Detailed Performance Lines")

# Dropdown to select specific MA windows to compare
ma_options = sorted(df_viz["MA_Window"].unique())
default_mas = [best_row["MA_Window"], 1, 10, 20]
selected_mas = st.multiselect(
    "Select MA Windows to compare", 
    ma_options, 
    default=[ma for ma in default_mas if ma in ma_options]
)

if selected_mas:
    df_line = df_viz[df_viz["MA_Window"].isin(selected_mas)]
    
    fig_line = px.line(
        df_line, 
        x="Threshold", 
        y=selected_metric, 
        color="MA_Window",
        markers=True,
        title=f"{selected_metric} vs Threshold (across selected MA Windows)"
    )
    st.plotly_chart(fig_line, use_container_width=True)

# ============================================================================
# Comparison Table (Manual entry for now or calculated)
# ============================================================================
if trial_mode == "Average across all trials":
    st.markdown("---")
    st.subheader(f"🏆 Top 5 Configurations ({selected_metric})")
    st.dataframe(
        df_viz.sort_values(selected_metric, ascending=False).head(5)
        .style.format({
            "Threshold": "{:.2f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "Accuracy": "{:.4f}",
            "AUROC": "{:.4f}"
        })
    )

# ============================================================================
# T-Test Significance Analysis
# ============================================================================
st.markdown("---")
st.subheader("🔬 Feature Significance (Welch's T-Test)")
st.markdown("Analysis of which features significantly differ between **Seizure (1)** and **Non-Seizure (0)** states using the training split for each trial.")

ttest_path = "/Users/adityakinjawadekar/Documents/eeg/biomarker/ttest_results/trial_ttests.json"
ttest_data = load_ttest_data(ttest_path)

if ttest_data:
    # Mapping dashboard dataset names to ttest keys
    dataset_map = {
        "Frequency Features": "freq",
        "EMD Features": "emd"
    }
    
    feat_key = dataset_map.get(selected_dataset)
    
    if feat_key and feat_key in ttest_data:
        feat_results = ttest_data[feat_key]
        
        # Prepare plotting data
        plot_data = []
        
        if trial_mode == "Specific Trial" and selected_trial is not None:
            # Find exact trial results
            trial_res = next((t for t in feat_results if t['trial'] == selected_trial), None)
            if trial_res:
                for feat, res in trial_res['results'].items():
                    plot_data.append({
                        "Feature": feat,
                        "t_stat": res['t_stat'],
                        "p_val": res['p_val'],
                        "-log10(p)": res['-log10p']
                    })
                title_suffix = f" (Trial {selected_trial})"
            else:
                st.warning(f"No t-test results found for Trial {selected_trial}")
        else:
            # Average across all trials
            all_feats = {}
            for t in feat_results:
                for f, r in t['results'].items():
                    if f not in all_feats:
                        all_feats[f] = []
                    all_feats[f].append(r['-log10p'])
            
            for f, vals in all_feats.items():
                plot_data.append({
                    "Feature": f,
                    "-log10(p)": sum(vals) / len(vals)
                })
            title_suffix = " (Averaged across all trials)"

        if plot_data:
            df_ttest = pd.DataFrame(plot_data).sort_values("-log10(p)", ascending=False)
            
            fig_ttest = px.bar(
                df_ttest,
                x="Feature",
                y="-log10(p)",
                title=f"Feature Significance: -log10(p-value) {title_suffix}",
                color="-log10(p)",
                color_continuous_scale="Reds",
                labels={"-log10(p)": "-log10(p)"}
            )
            
            # Threshold line for p=0.05
            fig_ttest.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="black", annotation_text="p=0.05")
            # Threshold line for p=0.001
            fig_ttest.add_hline(y=-np.log10(0.001), line_dash="dash", line_color="blue", annotation_text="p=0.001")

            st.plotly_chart(fig_ttest, use_container_width=True)
            
            with st.expander("View Data Table"):
                st.dataframe(df_ttest)
    else:
        st.info(f"T-test results not found for feature set: {selected_dataset}")
else:
    st.warning("T-test result file not found. Please run the `run_ttests.py` script.")

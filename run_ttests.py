import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio

# Helper for consistent Plotly layout
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

def save_plot_chart(df, x_col, y_col, title, filename, color_col=None, color_scale="OrRd", hlines=None):
    """Generates and saves a bar chart using Plotly."""
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
    
    # Add horizontal lines if specified (e.g., significance thresholds)
    if hlines:
        for y_val, color, text in hlines:
            fig.add_hline(y=y_val, line_dash="dash", line_color=color, annotation_text=text)

    fig.update_layout(xaxis_tickangle=-90, **LAYOUT_CONFIG)
    
    # Save
    try:
        pio.write_image(fig, filename, width=1200, height=800, scale=2)
        print(f"Saved plot: {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")

def run_ttests():
    results_dir = "/Users/adityakinjawadekar/Documents/eeg/biomarker/ttest_results"
    os.makedirs(results_dir, exist_ok=True)
    
    splits_path = "patient_splits.json" # Assumes in current directory now
    if not os.path.exists(splits_path):
        print(f"Error: {splits_path} not found.")
        return

    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    feature_configs = {
        "freq": {
            "dir": "/Users/adityakinjawadekar/Documents/eeg/biomarker/freq_features_updated",
            "features": [
                'delta_slope', 'delta_intercept', 'delta_midband',
                'theta_slope', 'theta_intercept', 'theta_midband',
                'alpha_slope', 'alpha_intercept', 'alpha_midband',
                'beta_slope', 'beta_intercept', 'beta_midband'
            ]
        },
        "emd": {
            "dir": "/Users/adityakinjawadekar/Documents/eeg/biomarker/emd_features_updated",
            "features": [
                'imf1_energy', 'imf1_wiener_entropy', 'imf1_skewness', 'imf1_kurtosis', 'imf1_psd_rms', 'imf1_std',
                'imf2_energy', 'imf2_wiener_entropy', 'imf2_skewness', 'imf2_kurtosis', 'imf2_psd_rms', 'imf2_std',
                'imf3_energy', 'imf3_wiener_entropy', 'imf3_skewness', 'imf3_kurtosis', 'imf3_psd_rms', 'imf3_std',
                'imf4_energy', 'imf4_wiener_entropy', 'imf4_skewness', 'imf4_kurtosis', 'imf4_psd_rms', 'imf4_std'
            ]
        }
    }
    
    all_results = {}

    for feat_type, config in feature_configs.items():
        print(f"Processing {feat_type} features...")
        feat_dir = config["dir"]
        feature_list = config["features"]
        
        # Pre-load all patients to save time
        patient_cache = {}
        patient_files = glob(os.path.join(feat_dir, "patient_*.csv"))
        for pfile in tqdm(patient_files, desc=f"Caching {feat_type} patients"):
            pid = int(os.path.basename(pfile).split('_')[1].split('.')[0])
            try:
                df = pd.read_csv(pfile)
                # Basic cleaning
                if 'channel' in df.columns:
                    df = df.drop(columns=['channel'])
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                patient_cache[pid] = df
            except Exception as e:
                print(f"Error loading {pfile}: {e}")
            
        type_results = []
        pca_type_results = []
        
        for trial_info in tqdm(splits, desc=f"Evaluating trials for {feat_type}"):
            trial_id = trial_info['trial']
            train_idx = trial_info['train_idx']
            
            # Combine training data for this trial
            trial_dfs = [patient_cache[pid] for pid in train_idx if pid in patient_cache]
            if not trial_dfs:
                continue
            
            combined_df = pd.concat(trial_dfs, ignore_index=True)
            
            # 1. Standard Feature T-Tests
            seizure = combined_df[combined_df['label'] == 1]
            non_seizure = combined_df[combined_df['label'] == 0]
            
            trial_ttest = {}
            for feat in feature_list:
                if feat not in combined_df.columns:
                    continue
                
                s_vals = seizure[feat].values
                ns_vals = non_seizure[feat].values
                
                if len(s_vals) < 2 or len(ns_vals) < 2:
                    t_stat, p_val = np.nan, np.nan
                else:
                    t_stat, p_val = stats.ttest_ind(s_vals, ns_vals, equal_var=False)
                
                trial_ttest[feat] = {
                    "t_stat": t_stat if not np.isnan(t_stat) else 0.0,
                    "p_val": p_val if not np.isnan(p_val) else 1.0,
                    "-log10p": -np.log10(p_val) if (not np.isnan(p_val) and p_val > 0) else (100.0 if p_val == 0 else 0.0)
                }
            
            type_results.append({
                "trial": trial_id,
                "results": trial_ttest
            })
            
            # 2. PCA Component T-Tests
            # Prepare data
            X = combined_df[feature_list].values
            y = combined_df['label'].values
            
            # Impute & Scale (match training pipeline)
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()
            
            X = imputer.fit_transform(X)
            X = scaler.fit_transform(X)
            
            # Apply PCA
            n_components = min(10, X.shape[1])
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            
            # Create DF for PCA components
            pca_cols = [f"PC{i+1}" for i in range(n_components)]
            df_pca = pd.DataFrame(X_pca, columns=pca_cols)
            df_pca['label'] = y
            
            seizure_pca = df_pca[df_pca['label'] == 1]
            non_seizure_pca = df_pca[df_pca['label'] == 0]
            
            trial_pca_ttest = {}
            for col in pca_cols:
                s_vals = seizure_pca[col].values
                ns_vals = non_seizure_pca[col].values
                
                if len(s_vals) < 2 or len(ns_vals) < 2:
                    t_stat, p_val = np.nan, np.nan
                else:
                    t_stat, p_val = stats.ttest_ind(s_vals, ns_vals, equal_var=False)
                
                trial_pca_ttest[col] = {
                    "t_stat": t_stat if not np.isnan(t_stat) else 0.0,
                    "p_val": p_val if not np.isnan(p_val) else 1.0,
                    "-log10p": -np.log10(p_val) if (not np.isnan(p_val) and p_val > 0) else (100.0 if p_val == 0 else 0.0)
                }
            
            pca_type_results.append({
                "trial": trial_id,
                "results": trial_pca_ttest
            })
            
        all_results[feat_type] = type_results
        all_results[f"pca_{feat_type}"] = pca_type_results

    # Save to JSON
    output_path = os.path.join(results_dir, "trial_ttests.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {output_path}")

    # ============================================================================
    # Visualization & Export
    # ============================================================================
    print("Generating visualizations...")
    
    for key, trial_list in all_results.items():
        if not trial_list:
            continue
            
        # Aggregate across trials
        # Structure: list of dicts with 'results' -> dict of feats -> stats
        
        # Collect all stats
        feature_stats = {}
        for trial_data in trial_list:
            results = trial_data['results']
            for feat, metrics in results.items():
                if feat not in feature_stats:
                    feature_stats[feat] = {'-log10p': [], 't_stat': []}
                feature_stats[feat]['-log10p'].append(metrics['-log10p'])
                feature_stats[feat]['t_stat'].append(metrics['t_stat'])
        
        # Calculate means
        avg_data = []
        for feat, lists in feature_stats.items():
            avg_data.append({
                "Feature": feat,
                "-log10p": np.mean(lists['-log10p']),
                "t_stat": np.mean(lists['t_stat'])
            })
            
        df_avg = pd.DataFrame(avg_data)
        
        # Plot 1: -log10(p) Significance
        df_p = df_avg.sort_values("-log10p", ascending=False)
        save_plot_chart(
            df_p, 
            x_col="Feature", 
            y_col="-log10p", 
            title=f"Feature Significance: -log10(p) ({key})",
            filename=os.path.join(results_dir, f"{key}_significance.png"),
            color_scale="OrRd",
            hlines=[
                (-np.log10(0.05), "#8b4513", "p=0.05"),
                (-np.log10(0.001), "#cc5500", "p=0.001")
            ]
        )
        
        # Plot 2: T-Statistic Direction
        df_t = df_avg.sort_values("t_stat", ascending=False)
        save_plot_chart(
            df_t, 
            x_col="Feature", 
            y_col="t_stat", 
            title=f"Feature Direction: T-Statistic ({key})",
            filename=os.path.join(results_dir, f"{key}_t_stat.png"),
            color_scale="RdBu", # Diverging scale for +/- values
            color_col="t_stat"
        )

if __name__ == "__main__":
    run_ttests()

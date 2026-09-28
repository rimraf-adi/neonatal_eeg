import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Point to legacy feature caches
LEGACY_CACHE_DIR = BASE_DIR.parent / "study_results" / "feature_cache"

PARADIGMS = {
    'spectral_slope': ('Frequency Features', LEGACY_CACHE_DIR / "spectral_slope"),
    'wavelet': ('Wavelet Features', LEGACY_CACHE_DIR / "legacy_wavelets"),
    'emd': ('EMD Features', LEGACY_CACHE_DIR / "baselines" / "emd")
}

def compute_ttests(df):
    """Computes t-statistic and -log10(p-value) for each feature."""
    features = [c for c in df.columns if c not in ('label', 'channel')]
    
    seizure_mask = (df['label'] == 1)
    seizure_df = df[seizure_mask]
    non_seizure_df = df[~seizure_mask]
    
    results = []
    for feat in features:
        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(
            seizure_df[feat].dropna(), 
            non_seizure_df[feat].dropna(), 
            equal_var=False,
            nan_policy='omit'
        )
        
        # Handle exact zero p-value from underflow
        if p_val == 0:
            p_val = 1e-300
            
        neg_log_p = -np.log10(p_val)
        
        results.append({
            'Feature': feat,
            'T-Statistic': t_stat,
            '-log10(p)': neg_log_p
        })
        
    return pd.DataFrame(results)

def plot_paradigm(paradigm_key, paradigm_name, paradigm_path):
    print(f"Processing {paradigm_name}...")
    
    # Load all CSVs for this paradigm
    csv_files = glob.glob(str(paradigm_path / "patient_*.csv"))
    if not csv_files:
        print(f"No CSV files found for {paradigm_key} at {paradigm_path}. Skipping.")
        return
        
    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))
        
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_all)} samples for {paradigm_key}.")
    
    # Compute stats
    stats_df = compute_ttests(df_all)
    
    # Sort by absolute T-statistic for better visualization
    stats_df['abs_t'] = stats_df['T-Statistic'].abs()
    stats_df = stats_df.sort_values(by='abs_t', ascending=True)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(stats_df)*0.3)))
    
    # Subplot 1: T-Statistic
    sns.barplot(
        data=stats_df, 
        y='Feature', 
        x='T-Statistic', 
        ax=axes[0], 
        color='skyblue'
    )
    axes[0].set_title("T-Statistic")
    axes[0].set_xlabel("t-statistic")
    axes[0].set_ylabel("")
    
    # Subplot 2: -log10(p-value)
    sns.barplot(
        data=stats_df, 
        y='Feature', 
        x='-log10(p)', 
        ax=axes[1], 
        color='salmon'
    )
    axes[1].set_title("-log10(p-value)")
    axes[1].set_xlabel("-log10(p)")
    axes[1].set_ylabel("")
    axes[1].set_yticks([]) # Hide y-labels for the second plot since they align
    
    # Super title
    fig.suptitle(paradigm_name, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = FIGURES_DIR / f"ttest_{paradigm_key}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path.name}")

if __name__ == "__main__":
    for key, (name, path) in PARADIGMS.items():
        plot_paradigm(key, name, path)

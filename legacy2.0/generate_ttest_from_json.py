import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
BASE_DIR = Path("d:/neonatal/legacy2.0")
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
TTEST_DIR = Path("d:/neonatal/legacy/ttest_results")

def aggregate_trials(trial_list):
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
            "-log10p": np.mean(lists['-log10p']),
            "t_stat": np.mean(lists['t_stat'])
        })
    return pd.DataFrame(avg_data)

def generate_figure(df, title, filename):
    df_t = df.sort_values(by='t_stat', ascending=False)
    df_p = df.sort_values(by='-log10p', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, len(df)*0.3)))
    fig.suptitle(title, fontsize=20, y=0.98)
    
    # Plot T-Statistic
    sns.barplot(
        data=df_t,
        x='t_stat',
        y='Feature',
        hue='t_stat',
        palette="vlag",
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title("T-Statistic", fontsize=16)
    axes[0].set_xlabel("T-Statistic", fontsize=14)
    axes[0].set_ylabel("Feature", fontsize=14)
    axes[0].tick_params(axis='y', labelsize=10)
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)
    axes[0].axvline(0, color='black', linewidth=0.8)
    
    # Plot -log10(p-value)
    sns.barplot(
        data=df_p,
        x='-log10p',
        y='Feature',
        hue='-log10p',
        palette="Reds",
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title("-log10(p-value)", fontsize=16)
    axes[1].set_xlabel("-log10(p-value)", fontsize=14)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis='y', labelsize=10)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=3.0)
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

def main():
    # Load JSON files
    with open(TTEST_DIR / "trial_ttests.json", 'r') as f:
        trial_ttests = json.load(f)
    
    with open(TTEST_DIR / "wavelet_trial_ttests.json", 'r') as f:
        wavelet_trial_ttests = json.load(f)
        
    # Aggregate
    freq_df = aggregate_trials(trial_ttests['freq'])
    emd_df = aggregate_trials(trial_ttests['emd'])
    wavelet_df = aggregate_trials(wavelet_trial_ttests)
    
    # Generate Figures
    generate_figure(freq_df, "Frequency Features", "ttest_spectral_slope.png")
    generate_figure(wavelet_df, "Wavelet Features", "ttest_wavelet.png")
    generate_figure(emd_df, "EMD Features", "ttest_emd.png")

if __name__ == "__main__":
    main()

"""
Extract Legacy Wavelet Features & Run Welch's T-Tests Before PCA
================================================================
Recreates the exact legacy DWT features from legacy/v1.0.py:
- db4 wavelet, 4 decomposition levels -> [a4, d4, d3, d2, d1]
- 6 statistical features per subband:
  [energy, wiener_entropy, skewness, kurtosis, psd_rms, std]
- 30 features per channel
- Evaluated on all 39 annotated neonatal seizure patients in dataset/
- Evaluates Welch's unequal variance t-test across 10 cross-validation trials
  matching legacy/run_ttests.py
"""

import os
import sys
import time
import json
import csv
import pywt
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from studies.utils import (
    EEG_IDX, FS, DESIRED_ORDER,
    DATA_DIR, load_edf_epochs, load_annotations, get_unanimous_labels
)

CACHE_DIR = os.path.join("study_results", "feature_cache", "legacy_wavelets")
RESULTS_DIR = os.path.join("legacy", "ttest_results")
SPLITS_PATH = "patient_splits.json"

SUBBANDS = ['a4', 'd4', 'd3', 'd2', 'd1']
METRICS = ['energy', 'wiener_entropy', 'skewness', 'kurtosis', 'psd_rms', 'std']
FEATURE_COLUMNS = [f"{sb}_{m}" for sb in SUBBANDS for m in METRICS]

def get_header():
    return ['label', 'channel'] + FEATURE_COLUMNS

def extract_wavelet_features_epoch_batch(eeg_valid, eps=1e-12):
    """
    Vectorized extraction of 30 DWT subband features for (n_epochs, n_channels, 256)
    Exact replica of legacy/v1.0.py _calc_features() on pywt.wavedec(sig, 'db4', level=4).
    """
    # coeffs: [a4, d4, d3, d2, d1]
    coeffs = pywt.wavedec(eeg_valid, 'db4', level=4, axis=-1)
    
    all_feats = []
    for c in coeffs:
        p = c ** 2
        # 1. energy
        energy = np.sum(p, axis=-1)
        # 2. wiener_entropy
        mean_p = np.mean(p, axis=-1)
        geom_mean = np.exp(np.mean(np.log(p + eps), axis=-1))
        wiener = np.where(mean_p >= eps, geom_mean / mean_p, 0.0)
        # 3. std
        std = np.std(c, axis=-1)
        # 4. skewness & kurtosis (Fisher-Pearson / Fisher excess matching scipy.stats)
        mean = np.mean(c, axis=-1, keepdims=True)
        diff = c - mean
        var = np.mean(diff ** 2, axis=-1, keepdims=True)
        std_safe = np.sqrt(np.maximum(var, eps))
        skew = np.mean(diff ** 3, axis=-1) / (std_safe.squeeze(-1) ** 3)
        kurt = (np.mean(diff ** 4, axis=-1) / (std_safe.squeeze(-1) ** 4)) - 3.0
        # 5. psd_rms
        psd_rms = np.sqrt(mean_p)
        
        all_feats.extend([energy, wiener, skew, kurt, psd_rms, std])
        
    return np.stack(all_feats, axis=-1) # (n_epochs, n_channels, 30)

def extract_and_cache_all_patients():
    os.makedirs(CACHE_DIR, exist_ok=True)
    anno_dfs = load_annotations()
    
    total_patients = len(EEG_IDX)
    print(f"Starting feature extraction for {total_patients} annotated patients...")
    start_time = time.time()
    
    for idx_num, pid in enumerate(EEG_IDX, 1):
        out_csv = os.path.join(CACHE_DIR, f"patient_{pid:03d}.csv")
        if os.path.exists(out_csv):
            print(f"[{idx_num}/{total_patients}] Patient {pid:03d} already cached. Skipping.")
            continue
            
        edf_path = os.path.join(DATA_DIR, f"eeg{pid}.edf")
        if not os.path.exists(edf_path):
            print(f"[{idx_num}/{total_patients}] Patient {pid:03d}: EDF not found ({edf_path})!")
            continue
            
        labels, valid_mask = get_unanimous_labels(anno_dfs, pid)
        if labels is None or valid_mask is None:
            print(f"[{idx_num}/{total_patients}] Patient {pid:03d}: Missing annotations!")
            continue
            
        t0 = time.time()
        eeg = load_edf_epochs(edf_path)
        if eeg is None:
            print(f"[{idx_num}/{total_patients}] Patient {pid:03d}: Failed to load EDF!")
            continue
            
        n_epochs = min(eeg.shape[0], len(labels), len(valid_mask))
        valid_idx = np.where(valid_mask[:n_epochs])[0]
        if len(valid_idx) == 0:
            print(f"[{idx_num}/{total_patients}] Patient {pid:03d}: No valid unanimous epochs!")
            continue
            
        eeg_valid = eeg[valid_idx] # (n_valid, n_ch, 256)
        labels_valid = labels[valid_idx]
        
        feat_mat = extract_wavelet_features_epoch_batch(eeg_valid) # (n_valid, n_ch, 30)
        
        # Write CSV
        n_valid, n_ch, n_feats = feat_mat.shape
        ch_names = DESIRED_ORDER[:n_ch]
        
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(get_header())
            for ep_i in range(n_valid):
                lbl = labels_valid[ep_i]
                for ch_i, ch_name in enumerate(ch_names):
                    row = [lbl, ch_name] + feat_mat[ep_i, ch_i].tolist()
                    writer.writerow(row)
                    
        elapsed = time.time() - t0
        print(f"[{idx_num}/{total_patients}] Patient {pid:03d}: {n_valid} epochs ({n_valid * n_ch} rows) extracted & saved in {elapsed:.2f}s")
        
    print(f"\nAll patients cached successfully in {time.time() - start_time:.2f}s!")

def run_wavelet_ttests():
    print("\n" + "="*80)
    print("RUNNING WELCH'S T-TESTS ACROSS 10 CROSS-VALIDATION TRIALS (BEFORE PCA)")
    print("="*80)
    
    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)
        
    # Pre-cache patient data
    patient_cache = {}
    for pid in EEG_IDX:
        pfile = os.path.join(CACHE_DIR, f"patient_{pid:03d}.csv")
        if os.path.exists(pfile):
            df = pd.read_csv(pfile)
            if 'channel' in df.columns:
                df = df.drop(columns=['channel'])
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            patient_cache[pid] = df
            
    print(f"Loaded {len(patient_cache)} patients into memory for trial evaluation.")
    
    trial_results = []
    
    for trial_info in splits:
        trial_id = trial_info['trial']
        train_idx = trial_info['train_idx']
        
        trial_dfs = [patient_cache[pid] for pid in train_idx if pid in patient_cache]
        if not trial_dfs:
            continue
            
        combined_df = pd.concat(trial_dfs, ignore_index=True)
        seizure = combined_df[combined_df['label'] == 1]
        non_seizure = combined_df[combined_df['label'] == 0]
        
        trial_ttest = {}
        for feat in FEATURE_COLUMNS:
            s_vals = seizure[feat].values
            ns_vals = non_seizure[feat].values
            
            if len(s_vals) < 2 or len(ns_vals) < 2:
                t_stat, p_val = np.nan, np.nan
            else:
                t_stat, p_val = stats.ttest_ind(s_vals, ns_vals, equal_var=False)
                
            trial_ttest[feat] = {
                "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
                "p_val": float(p_val) if not np.isnan(p_val) else 1.0,
                "-log10p": float(-np.log10(p_val)) if (not np.isnan(p_val) and p_val > 0) else (100.0 if p_val == 0 else 0.0)
            }
            
        trial_results.append({
            "trial": trial_id,
            "results": trial_ttest
        })
        
    # Save to JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "wavelet_trial_ttests.json")
    with open(out_json, 'w') as f:
        json.dump(trial_results, f, indent=4)
    print(f"Saved trial t-test results to {out_json}")
    
    # Compute aggregate summary statistics
    feature_stats = {}
    for t_data in trial_results:
        res = t_data['results']
        for feat, metrics in res.items():
            if feat not in feature_stats:
                feature_stats[feat] = {'t_stat': [], 'p_val': [], '-log10p': []}
            feature_stats[feat]['t_stat'].append(metrics['t_stat'])
            feature_stats[feat]['p_val'].append(metrics['p_val'])
            feature_stats[feat]['-log10p'].append(metrics['-log10p'])
            
    summary_rows = []
    for feat, d in feature_stats.items():
        summary_rows.append({
            'Feature': feat,
            'Subband': feat.split('_')[0],
            'Metric': '_'.join(feat.split('_')[1:]),
            'Mean_t_stat': np.mean(d['t_stat']),
            'Min_t_stat': np.min(d['t_stat']),
            'Max_t_stat': np.max(d['t_stat']),
            'Mean_p_val': np.mean(d['p_val']),
            'Mean_neg_log10p': np.mean(d['-log10p']),
            'Sig_trials_count': sum(1 for p in d['p_val'] if p < 0.05)
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary_sorted = df_summary.sort_values(by='Mean_neg_log10p', ascending=False)
    
    summary_csv = os.path.join(RESULTS_DIR, "wavelet_ttest_summary.csv")
    df_summary_sorted.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to {summary_csv}")
    
    print("\n=== TOP 30 LEGACY WAVELET FEATURES SORTED BY SIGNIFICANCE ===")
    header_fmt = "%-22s | %10s | %12s | %14s | %15s | %10s"
    print(header_fmt % ("Feature", "Subband", "Mean t-stat", "Mean p-val", "Mean -log10(p)", "Sig Trials"))
    print("-" * 92)
    for _, r in df_summary_sorted.iterrows():
        print("%-22s | %10s | %12.4f | %14.4e | %15.2f | %10d" % (
            r['Feature'], r['Subband'], r['Mean_t_stat'], r['Mean_p_val'], r['Mean_neg_log10p'], int(r['Sig_trials_count'])
        ))

if __name__ == "__main__":
    extract_and_cache_all_patients()
    run_wavelet_ttests()

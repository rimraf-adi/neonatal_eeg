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

def run_ttests():
    results_dir = "/Users/adityakinjawadekar/Documents/eeg/biomarker/ttest_results"
    os.makedirs(results_dir, exist_ok=True)
    
    splits_path = "patient_splits.json" # Assumes in current directory now
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    feature_configs = {
        "freq": {
            "dir": "/Users/adityakinjawadekar/Documents/eeg/biomarker/freq_features_updated",
            "features": [
                'delta_slope', 'delta_intercept', 'delta_midband',
                'theta_slope', 'theta_intercept', 'theta_midband',
                'alpha_slope', 'alpha_intercept', 'alpha_midband',
                'beta_slope', 'beta_intercept', 'beta_midband',
                'gamma_slope', 'gamma_intercept', 'gamma_midband'
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

if __name__ == "__main__":
    run_ttests()

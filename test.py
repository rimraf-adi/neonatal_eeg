import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import glob

DATA_DIR = "freq_features"
OUTPUT_DIR = "ttest_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
FEATURES = ["slope", "intercept", "midband"]

def get_feature_columns():
    cols = []
    for band in BANDS:
        for feat in FEATURES:
            cols.append(f"{band}_{feat}")
    return cols

def run_ttest(seizure_data, nonseizure_data, feature_cols):
    results = {}
    for col in feature_cols:
        seizure_vals = seizure_data[col].dropna().values
        nonseizure_vals = nonseizure_data[col].dropna().values
        if len(seizure_vals) > 1 and len(nonseizure_vals) > 1:
            t_stat, p_val = ttest_ind(seizure_vals, nonseizure_vals)
            if p_val > 0:
                log_p = -np.log10(p_val)
            else:
                log_p = np.inf
            results[col] = {"t_statistic": t_stat, "p_value": p_val, "neg_log10_p": log_p}
        else:
            results[col] = {"t_statistic": np.nan, "p_value": np.nan, "neg_log10_p": np.nan}
    return results

def main():
    feature_cols = get_feature_columns()
    patient_files = sorted(glob.glob(os.path.join(DATA_DIR, "patient_*.csv")))
    all_data = []
    
    with open(os.path.join(OUTPUT_DIR, "patient_wise_ttest.txt"), "w") as f:
        for pfile in patient_files:
            patient_id = os.path.basename(pfile).replace(".csv", "")
            df = pd.read_csv(pfile)
            
            seizure_data = df[df["label"] == 1]
            nonseizure_data = df[df["label"] == 0]
            
            if len(seizure_data) > 1 and len(nonseizure_data) > 1:
                results = run_ttest(seizure_data, nonseizure_data, feature_cols)
                f.write(f"=== {patient_id} ===\n")
                f.write("Feature\tT-Statistic\tP-Value\t-log10(P)\n")
                for feat, vals in results.items():
                    f.write(f"{feat}\t{vals['t_statistic']:.6f}\t{vals['p_value']:.2e}\t{vals['neg_log10_p']:.2f}\n")
                f.write("\n")
            
            all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_seizure = combined_df[combined_df["label"] == 1]
    combined_nonseizure = combined_df[combined_df["label"] == 0]
    combined_results = run_ttest(combined_seizure, combined_nonseizure, feature_cols)
    
    with open(os.path.join(OUTPUT_DIR, "combined_ttest.txt"), "w") as f:
        f.write("Feature\tT-Statistic\tP-Value\t-log10(P)\n")
        for feat, vals in combined_results.items():
            f.write(f"{feat}\t{vals['t_statistic']:.6f}\t{vals['p_value']:.2e}\t{vals['neg_log10_p']:.2f}\n")

if __name__ == "__main__":
    main()
import os
import glob
import re
import pandas as pd

# Define results directories
RESULTS_DIRS = {
    "PCA Frequency Features": "./pca_adaptive_nn_results",
    "PCA EMD Features": "./pca_adaptive_emd_results"
}

def parse_metrics(filepath):
    """
    Parse a single result file for metrics.
    Returns a dictionary of metrics or None if parsing fails.
    """
    try:
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
            return {
                "MA_Window": int(ma_match.group(1)),
                "Threshold": float(thresh_match.group(1)),
                "Precision": float(prec_match.group(1)),
                "Recall": float(rec_match.group(1)),
                "F1": float(f1_match.group(1)),
                "Accuracy": float(acc_match.group(1)),
                "AUROC": float(auc_match.group(1))
            }
    except Exception as e:
        # print(f"Error parsing {filepath}: {e}")
        pass
    return None

def get_trial_metrics(base_dir, dataset_name, split="validation"):
    """
    Load data specifically for MA 20 and Threshold 0.49 for a given split.
    Target path: trial_*/detailed/{split}/ma20/ma20_thresh0.49.txt
    """
    data = []
    
    # Construct the specific search pattern
    search_pattern = os.path.join(base_dir, "trial_*", "detailed", split, "ma20", "ma20_thresh0.49.txt")
    files = glob.glob(search_pattern)
    
    for filepath in files:
        # Extract trial number from path
        try:
            parts = filepath.split(os.sep)
            detailed_idx = parts.index("detailed")
            trial_str = parts[detailed_idx - 1]
            match_trial = re.search(r'trial_(\d+)', trial_str)
            trial_num = int(match_trial.group(1)) if match_trial else -1
            
            metrics = parse_metrics(filepath)
            if metrics:
                entry = {
                    "Dataset": dataset_name,
                    "Trial": trial_num,
                    "Split": split,
                    **metrics
                }
                data.append(entry)
        except ValueError:
            continue
            
    return data

def get_test_metrics_for_trial(base_dir, trial_num):
    """
    Fetch test metrics for a specific trial.
    Path: .../trial_{trial_num}/detailed/test/ma20/ma20_thresh0.49.txt
    """
    # Pattern to find the specific trial folder. Since we don't know the exact path structure (e.g. if trial_0 or trial_00),
    # we can construct it based on how we found it, or just glob for that specific trial folder.
    # Assuming standard format 'trial_{trial_num}'
    
    # Construct path directly with zero padding for single digit trials
    # The folders are named trial_00, trial_01, etc.
    test_path = os.path.join(base_dir, f"trial_{trial_num:02d}", "detailed", "test", "ma20", "ma20_thresh0.49.txt")
    
    if os.path.exists(test_path):
        return parse_metrics(test_path)
    
    return None

def main():
    print("Processing PCA Frequency and PCA EMD Features for MA=20, Threshold=0.49...")
    
    final_output = []
    
    output_file = "top5_pca_metrics_val_test.txt"
    with open(output_file, "w") as f_out:
        
        for name, path in RESULTS_DIRS.items():
            if not os.path.exists(path):
                print(f"Warning: Directory not found: {path}")
                continue
                
            print(f"\nAnalying {name}...")
            f_out.write(f"\n{'='*60}\n")
            f_out.write(f"Dataset: {name}\n")
            f_out.write(f"{'='*60}\n\n")
            
            # Get Validation Data
            val_data = get_trial_metrics(path, name, split="validation")
            
            if not val_data:
                print(f"  No validation data found for {name}")
                f_out.write("  No validation data found.\n")
                continue
                
            df_val = pd.DataFrame(val_data)
            
            # Sort by Accuracy (descending)
            df_sorted = df_val.sort_values(by=["Accuracy", "F1"], ascending=[False, False])
            
            # Get Top 5
            top_5 = df_sorted.head(5)
            
            print(f"  Top 5 Trials (Validation Acc): {top_5['Trial'].tolist()}")
            
            # Prepare Table Headers
            headers = ["Rank", "Trial", "Val Acc", "Val F1", "Val Prec", "Val Rec", "Val AUC", "Test Acc", "Test F1", "Test Prec", "Test Rec", "Test AUC"]
            
            # Create a list of lists for tabulate or manual formatting
            table_rows = []
            
            for rank, (idx, row) in enumerate(top_5.iterrows(), 1):
                trial = int(row['Trial'])
                
                # Fetch Test Metrics
                test_metrics = get_test_metrics_for_trial(path, trial)
                
                if test_metrics:
                    test_acc = f"{test_metrics['Accuracy']:.4f}"
                    test_f1 = f"{test_metrics['F1']:.4f}"
                    test_prec = f"{test_metrics['Precision']:.4f}"
                    test_rec = f"{test_metrics['Recall']:.4f}"
                    test_auc = f"{test_metrics['AUROC']:.4f}"
                else:
                    test_acc = "N/A"
                    test_f1 = "N/A"
                    test_prec = "N/A"
                    test_rec = "N/A"
                    test_auc = "N/A"

                table_rows.append([
                    rank,
                    trial,
                    f"{row['Accuracy']:.4f}",
                    f"{row['F1']:.4f}",
                    f"{row['Precision']:.4f}",
                    f"{row['Recall']:.4f}",
                    f"{row['AUROC']:.4f}",
                    test_acc,
                    test_f1,
                    test_prec,
                    test_rec,
                    test_auc
                ])

            # Formatting table
            # Calculate column widths
            col_widths = [len(h) for h in headers]
            for row in table_rows:
                for i, val in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(val)))
            
            # Print Header
            header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
            separator = "-+-".join("-" * w for w in col_widths) # simple separator
            
            f_out.write(f"{header_str}\n")
            f_out.write(f"{separator}\n")
            
            for row in table_rows:
                row_str = " | ".join(f"{str(val):<{w}}" for val, w in zip(row, col_widths))
                f_out.write(f"{row_str}\n")
            
            f_out.write("\n")

    print(f"\nDetailed analysis saved to {output_file}")

if __name__ == "__main__":
    main()

"""
Adaptive Nearest Neighbor EEG Analysis with EMD Features

This script:
1. Implements adaptive temporal downsampling to balance classes
2. Keeps all seizure epochs (1s)
3. Adaptively selects N non-seizure epochs (0s) with TARGET_RATIO × seizures
4. Uses EMD (Empirical Mode Decomposition) features
5. Trains a neural network classifier
6. Performs MA (1-20) and threshold (0.05-0.95) sweeps
7. Saves detailed results and trained models
"""

import torch
import torch.nn as nn
import json
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
)
import random
from collections import Counter

# ============================================================================
# Configuration
# ============================================================================
FEATURE_DIR = '/Users/adityakinjawadekar/Documents/eeg/biomarker/emd_features_updated'
RESULTS_DIR = './pca_adaptive_emd_results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
DETAILED_DIR = os.path.join(RESULTS_DIR, 'detailed')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DETAILED_DIR, exist_ok=True)

# All patient indices
EEG_IDX = [1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 31, 34,
           36, 38, 39, 40, 41, 44, 47, 50, 51, 52, 62, 63, 66, 67, 69, 73, 75,
           76, 77, 78, 79]

# Moving average window sizes to test
MA_WINDOW_SIZES = list(range(1, 21))  # 1 to 20 (1 = no smoothing)

# Probability thresholds to test
PROB_THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# PCA Configuration - max 10 components
PCA_MAX_COMPONENTS = 10
PCA_VARIANCE_THRESHOLD = 0.95  # Keep 95% variance or max 10 components, whichever is smaller

# Target ratio for class balancing (0s:1s ratio)
# This ensures 0s > 1s while avoiding severe imbalance
# Valid range: 1.5 to 3.0 (i.e., 1:1.5 to 1:3)
TARGET_RATIO = 2.0  # Select 2x non-seizure epochs for each seizure epoch

# ============================================================================
# Adaptive Nearest Neighbor Filtering
# ============================================================================
def apply_adaptive_filter(df):
    """
    Filter dataframe to balance classes with controlled ratio:
    1. Keep all seizure epochs (label=1)
    2. Adaptively select non-seizure epochs (label=0) closest to seizures
       such that 0s = TARGET_RATIO × 1s (ensuring 0s > 1s)
    
    Strategy: 
    - Find all seizure indices
    - For all non-seizure indices, compute minimum distance to any seizure
    - Select N non-seizure epochs with smallest distances, where N = TARGET_RATIO × num_seizures
    
    Args:
        df: DataFrame with 'label' column
    
    Returns:
        Filtered DataFrame with controlled class ratio
    """
    labels = df['label'].values
    indices_to_keep = set()
    
    # Find all seizure epoch indices
    seizure_indices = np.where(labels == 1)[0]
    num_seizures = len(seizure_indices)
    
    if num_seizures == 0:
        # No seizures, return empty dataframe
        print("    WARNING: No seizures found in this patient data")
        return df.iloc[[]].copy()
    
    # Always keep all seizure epochs
    indices_to_keep.update(seizure_indices)
    
    # Find all non-seizure epoch indices
    non_seizure_indices = np.where(labels == 0)[0]
    
    if len(non_seizure_indices) == 0:
        # No non-seizure epochs, just return seizure epochs
        print("    WARNING: No non-seizure epochs found in this patient data")
        return df.iloc[list(indices_to_keep)].copy()
    
    # For each non-seizure epoch, find distance to nearest seizure
    min_distances = np.zeros(len(non_seizure_indices))
    for i, ns_idx in enumerate(non_seizure_indices):
        # Distance to nearest seizure
        min_distances[i] = np.min(np.abs(ns_idx - seizure_indices))
    
    # Select N non-seizure epochs with smallest distances
    # where N = TARGET_RATIO × number of seizures (ensuring 0s > 1s)
    n_to_select = int(TARGET_RATIO * num_seizures)
    n_to_select = min(n_to_select, len(non_seizure_indices))  # Can't select more than available
    
    # Ensure we always have 0s > 1s (minimum ratio of 1.1 if not enough non-seizures)
    if n_to_select <= num_seizures and len(non_seizure_indices) > num_seizures:
        n_to_select = min(int(1.5 * num_seizures), len(non_seizure_indices))
    
    # Get indices of N closest non-seizures
    if n_to_select < len(non_seizure_indices):
        closest_indices = np.argpartition(min_distances, n_to_select-1)[:n_to_select]
    else:
        # Keep all non-seizures if there are fewer than needed
        closest_indices = np.arange(len(non_seizure_indices))
    
    # Add selected non-seizure epochs to keep set
    indices_to_keep.update(non_seizure_indices[closest_indices])
    
    # Filter dataframe
    indices_to_keep = sorted(list(indices_to_keep))
    filtered_df = df.iloc[indices_to_keep].copy()
    
    return filtered_df


def load_patient_data(idx_list, mode, apply_filter=False):
    """Load patient data from CSV files with optional adaptive filtering."""
    frames = []
    for i in idx_list:
        filename = os.path.join(FEATURE_DIR, f'patient_{i:03d}.csv')
        if os.path.exists(filename):
            print(f'[{mode}] Processing file {filename}')
            df = pd.read_csv(filename)
            
            # Apply adaptive filtering if specified
            if apply_filter:
                original_size = len(df)
                original_1s = (df['label'] == 1).sum()
                original_0s = (df['label'] == 0).sum()
                
                df = apply_adaptive_filter(df)
                
                filtered_size = len(df)
                filtered_1s = (df['label'] == 1).sum()
                filtered_0s = (df['label'] == 0).sum()
                
                print(f'  Adaptive filter: {original_size} -> {filtered_size} epochs')
                print(f'    Before: 1s={original_1s}, 0s={original_0s} (ratio: {original_0s/max(original_1s,1):.2f})')
                print(f'    After:  1s={filtered_1s}, 0s={filtered_0s} (ratio: {filtered_0s/max(filtered_1s,1):.2f})')
            
            frames.append(df)
        else:
            print(f'[{mode}] Warning: File {filename} not found.')
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_feature_columns(df):
    """Get all EMD feature columns (excluding label, channel, and psd_rms features)."""
    all_cols = df.columns.tolist()
    # Exclude label, channel, and psd_rms features
    feature_cols = [c for c in all_cols if c not in ['label', 'channel'] and 'psd_rms' not in c]
    return feature_cols


# ============================================================================
# Dataset and Model
# ============================================================================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Training and Evaluation
# ============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    """Train the model with early stopping."""
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  Early stopping at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}')
    
    return model


def get_predictions(model, data_loader, device):
    """Get model predictions and probabilities."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def moving_average(probabilities, window_size):
    """Apply moving average smoothing to probabilities."""
    if window_size == 1:
        return probabilities
    
    smoothed = np.convolve(probabilities, np.ones(window_size)/window_size, mode='same')
    return smoothed


def evaluate_with_threshold(y_true, probs_smoothed, threshold):
    """Evaluate predictions with a specific probability threshold."""
    y_pred = (probs_smoothed >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    try:
        auroc = roc_auc_score(y_true, probs_smoothed)
    except:
        auroc = 0.0
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'auroc': auroc
    }


def save_detailed_results(results, ma_window, threshold, output_dir):
    """Save detailed results for a specific MA window and threshold."""
    filename = os.path.join(output_dir, f'ma{ma_window:02d}_thresh{threshold:.2f}.txt')
    
    with open(filename, 'w') as f:
        f.write(f"MA Window: {ma_window}\n")
        f.write(f"Threshold: {threshold:.2f}\n")
        f.write(f"\nConfusion Matrix:\n{results['confusion_matrix']}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"AUROC: {results['auroc']:.4f}\n")


def save_best_results(best_results, output_file, dataset_name="Test"):
    """Save best results across all MA windows and thresholds."""
    with open(output_file, 'w') as f:
        f.write(f"Best Results Across All MA Windows and Thresholds ({dataset_name} Set)\n")
        f.write("=" * 80 + "\n\n")
        
        for metric in ['precision', 'recall', 'f1', 'auroc']:
            f.write(f"\nBest {metric.upper()}:\n")
            f.write(f"  MA Window: {best_results[metric]['ma_window']}\n")
            f.write(f"  Threshold: {best_results[metric]['threshold']:.2f}\n")
            f.write(f"  {metric.capitalize()}: {best_results[metric]['value']:.4f}\n")
            f.write(f"  Precision: {best_results[metric]['precision']:.4f}\n")
            f.write(f"  Recall: {best_results[metric]['recall']:.4f}\n")
            f.write(f"  F1: {best_results[metric]['f1']:.4f}\n")
            f.write(f"  AUROC: {best_results[metric]['auroc']:.4f}\n")
            f.write(f"  Confusion Matrix:\n{best_results[metric]['confusion_matrix']}\n")


# ============================================================================
# Main Training
# ============================================================================
import shutil

def run_trial(trial_num, base_results_dir, split_info):
    """Run a single trial of the training pipeline using pre-defined splits."""
    
    # Create trial-specific directories
    trial_dir = os.path.join(base_results_dir, f'trial_{trial_num:02d}')
    models_dir = os.path.join(trial_dir, 'models')
    detailed_dir = os.path.join(trial_dir, 'detailed')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(detailed_dir, exist_ok=True)
    
    print("\n" + "#" * 80)
    print(f"TRIAL {trial_num+1}/10")
    print("#" * 80)
    
    # Set seed for this trial
    current_seed = 42 + trial_num
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    
    print(f"\nStrategy: Keep all seizures (1s) and adaptively select")
    print(f"closest non-seizures (0s) with target ratio of {TARGET_RATIO}:1")
    print(f"This ensures 0s > 1s for each patient with controlled imbalance")
    print(f"Using EMD (Empirical Mode Decomposition) features\n")
    
    # Use provided splits
    train_idx = split_info['train_idx']
    val_idx = split_info['val_idx']
    test_idx = split_info['test_idx']
    
    print(f"Train patients ({len(train_idx)}): {sorted(train_idx)}")
    print(f"Val patients ({len(val_idx)}): {sorted(val_idx)}")
    print(f"Test patients ({len(test_idx)}): {sorted(test_idx)}\n")
    
    # Load data with adaptive filtering
    print(f"Loading training data (Seed {current_seed})...")
    train_df = load_patient_data(train_idx, 'TRAIN', apply_filter=True)
    if train_df.empty:
        print("ERROR: No training data available")
        return
    
    # print("\nLoading validation data with adaptive filtering...")
    val_df = load_patient_data(val_idx, 'VAL', apply_filter=True)
    if val_df.empty:
        print("ERROR: No validation data available")
        return
    
    # print("\nLoading test data with adaptive filtering...")
    test_df = load_patient_data(test_idx, 'TEST', apply_filter=True)
    if test_df.empty:
        print("ERROR: No test data available")
        return
    
    # Check class distribution
    # print(f"\n{'='*80}")
    # print("Final Class Distribution:")
    # print(f"{'='*80}")
    train_counts = Counter(train_df['label'])
    val_counts = Counter(val_df['label'])
    test_counts = Counter(test_df['label'])
    
    # print(f"Train: 1s={train_counts[1]}, 0s={train_counts[0]}, Ratio={train_counts[0]/max(train_counts[1],1):.3f}")
    # print(f"Val:   1s={val_counts[1]}, 0s={val_counts[0]}, Ratio={val_counts[0]/max(val_counts[1],1):.3f}")
    # print(f"Test:  1s={test_counts[1]}, 0s={test_counts[0]}, Ratio={test_counts[0]/max(test_counts[1],1):.3f}")
    
    # Get EMD feature columns
    feature_cols = get_feature_columns(train_df)
    # print(f"\nNumber of EMD features: {len(feature_cols)}")
    # print(f"(Excluding 'label', 'channel', and 'psd_rms' features)")
    
    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Apply PCA dimensionality reduction - always use max components
    n_components = min(PCA_MAX_COMPONENTS, X_train.shape[1])
    
    print(f"  PCA: Reducing from {X_train.shape[1]} to {n_components} components")
    
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    
    # Create datasets and dataloaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cpu")
    
    model = NeuralNet(input_dim=n_components).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print(f"Training Model (Trial {trial_num+1})...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50)
    
    # Save model
    model_path = os.path.join(models_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'imputer': imputer,
        'pca': pca,
        'feature_cols': feature_cols,
        'n_pca_components': n_components
    }, model_path)
    
    # Get predictions on validation and test sets
    print("Evaluating on Validation and Test Sets...")
    
    y_val_true, y_val_pred, val_probs = get_predictions(model, val_loader, device)
    y_test_true, y_test_pred, test_probs = get_predictions(model, test_loader, device)
    
    # Define sweep function for both datasets
    def run_sweep(y_true, probs, dataset_name):
        # print(f"\nPerforming MA and threshold sweeps on {dataset_name} set...")
        best_results = {
            'precision': {'value': 0, 'ma_window': 0, 'threshold': 0},
            'recall': {'value': 0, 'ma_window': 0, 'threshold': 0},
            'f1': {'value': 0, 'ma_window': 0, 'threshold': 0},
            'auroc': {'value': 0, 'ma_window': 0, 'threshold': 0}
        }
        
        # Create dataset-specific detailed directory
        dataset_detailed_dir = os.path.join(detailed_dir, dataset_name.lower())
        os.makedirs(dataset_detailed_dir, exist_ok=True)
        
        for ma_window in MA_WINDOW_SIZES:
            # Create subdirectory for this MA window
            ma_dir = os.path.join(dataset_detailed_dir, f'ma{ma_window:02d}')
            os.makedirs(ma_dir, exist_ok=True)
            
            # Apply moving average
            probs_smoothed = moving_average(probs, ma_window)
            
            for threshold in PROB_THRESHOLDS:
                # Evaluate with this threshold
                results = evaluate_with_threshold(y_true, probs_smoothed, threshold)
                
                # Save detailed results
                save_detailed_results(results, ma_window, threshold, ma_dir)
                
                # Update best results
                for metric in ['precision', 'recall', 'f1', 'auroc']:
                    if results[metric] > best_results[metric]['value']:
                        best_results[metric] = {
                            'value': results[metric],
                            'ma_window': ma_window,
                            'threshold': threshold,
                            'precision': results['precision'],
                            'recall': results['recall'],
                            'f1': results['f1'],
                            'auroc': results['auroc'],
                            'confusion_matrix': results['confusion_matrix']
                        }
            
            # print(f"  [{dataset_name}] Completed MA window {ma_window}")
        
        return best_results
    
    # Run sweep on validation set
    val_best_results = run_sweep(y_val_true, val_probs, 'Validation')
    
    # Run sweep on test set
    test_best_results = run_sweep(y_test_true, test_probs, 'Test')
    
    # Save best results for both sets
    val_best_file = os.path.join(trial_dir, 'best_results_val.txt')
    test_best_file = os.path.join(trial_dir, 'best_results_test.txt')
    save_best_results(val_best_results, val_best_file, 'Validation')
    save_best_results(test_best_results, test_best_file, 'Test')
    
    # Save summary
    summary_file = os.path.join(trial_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Adaptive Nearest Neighbor Class Balancing - Trial {trial_num+1}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Seed: {current_seed}\n")
        f.write(f"Strategy: Adaptive selection with target ratio {TARGET_RATIO}:1 (0s:1s)\n")
        f.write("Ensures 0s > 1s for each patient with controlled imbalance\n")
        f.write("Using EMD (Empirical Mode Decomposition) features\n\n")
        f.write(f"Training samples: {len(y_train)}\n")
        f.write(f"Validation samples: {len(y_val)}\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"\nClass distribution (Train): {Counter(y_train)}\n")
        f.write(f"Class distribution (Val): {Counter(y_val)}\n")
        f.write(f"Class distribution (Test): {Counter(y_test)}\n")
        f.write(f"\nClass ratios (0s/1s):\n")
        f.write(f"  Train: {train_counts[0]/max(train_counts[1],1):.3f}\n")
        f.write(f"  Val:   {val_counts[0]/max(val_counts[1],1):.3f}\n")
        f.write(f"  Test:  {test_counts[0]/max(test_counts[1],1):.3f}\n")
        f.write(f"\nNumber of EMD features: {len(feature_cols)}\n")
        f.write(f"MA windows tested: {len(MA_WINDOW_SIZES)}\n")
        f.write(f"Thresholds tested: {len(PROB_THRESHOLDS)}\n")
        
        f.write(f"\n\n" + "="*80 + "\n")
        f.write("VALIDATION SET BEST RESULTS\n")
        f.write("="*80 + "\n")
        for metric in ['precision', 'recall', 'f1', 'auroc']:
            f.write(f"Best {metric}: {val_best_results[metric]['value']:.4f} "
                    f"(MA={val_best_results[metric]['ma_window']}, "
                    f"thresh={val_best_results[metric]['threshold']:.2f})\n")
        
        f.write(f"\n" + "="*80 + "\n")
        f.write("TEST SET BEST RESULTS\n")
        f.write("="*80 + "\n")
        for metric in ['precision', 'recall', 'f1', 'auroc']:
            f.write(f"Best {metric}: {test_best_results[metric]['value']:.4f} "
                    f"(MA={test_best_results[metric]['ma_window']}, "
                    f"thresh={test_best_results[metric]['threshold']:.2f})\n")
    
    print(f"Trial {trial_num+1} completed. Results saved in {trial_dir}")

def main():
    """Main execution function."""
    print("=" * 80)
    print("Adaptive Nearest Neighbor Class Balancing with EMD Features - 10 Fold/Trial CV")
    print("=" * 80)
    
    # 1. Clear current results
    if os.path.exists(RESULTS_DIR):
        print(f"Clearing existing results directory: {RESULTS_DIR}")
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 2. Load patient splits
    # Or use the absolute path provided by user if preferred/safer:
    # 2. Load patient splits
    splits_file = 'patient_splits.json'
    
    if not os.path.exists(splits_file):
        print(f"ERROR: Splits file not found at {splits_file}")
        return

    with open(splits_file, 'r') as f:
        all_splits = json.load(f)
    
    print(f"Loaded {len(all_splits)} splits from {splits_file}")

    # 3. Run trials based on loaded splits
    for i, split_info in enumerate(all_splits):
        run_trial(i, RESULTS_DIR, split_info)
    
    print(f"\n{'='*80}")
    print(f"All {len(all_splits)} trials completed!")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"{'='*80}")


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    main()
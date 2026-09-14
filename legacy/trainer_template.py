import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter

# Patient indices
eeg_idx = [1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 31, 34,
           36, 38, 39, 40, 41, 44, 47, 50, 51, 52, 62, 63, 66, 67, 69, 73, 75,
           76, 77, 78, 79]

train_count = int(len(eeg_idx) * 0.75)
train_idx = random.sample(eeg_idx, train_count)
test_idx = [i for i in eeg_idx if i not in train_idx]

feature_dir = './emd/emd_features'

# Load data
def load_patient_data(idx_list, mode):
    frames = []
    for i in idx_list:
        filename = os.path.join(feature_dir, f'patient_{i:03d}.csv')
        if os.path.exists(filename):
            print(f'[{mode}] Processing file {filename}')
            df = pd.read_csv(filename)
            frames.append(df)
        else:
            print(f'[{mode}] Warning: File {filename} not found.')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

train_df = load_patient_data(train_idx, 'TRAIN')
test_df = load_patient_data(test_idx, 'TEST')

# Feature selection - drop psd_rms features for all IMFs
all_cols = train_df.columns.tolist()
feature_cols = [c for c in all_cols if c not in ['label', 'channel'] and 'psd_rms' not in c]
print(f"Dropped psd_rms columns. Remaining features: {feature_cols}")

# Preprocessing
y_train = train_df['label'].values
y_test = test_df['label'].values

train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy='median')
train_df[feature_cols] = imputer.fit_transform(train_df[feature_cols])
test_df[feature_cols] = imputer.transform(test_df[feature_cols])

X_train = train_df[feature_cols].astype('float32').values
X_test = test_df[feature_cols].astype('float32').values

print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Features count: {len(feature_cols)}")

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weights: {class_weights_tensor}")
print(f"Class distribution: {Counter(y_train)}")


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model_name, results_file, model_file, cm_file, curves_file,
                       pca_info=None):
    """Train model and save results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = EEGDataset(X_train_data, y_train_data)
    test_dataset = EEGDataset(X_test_data, y_test_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = NeuralNet(X_train_data.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
    
    best_val_loss = float('inf')
    patience = 20
    epochs_no_improve = 0
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"Input dimension: {X_train_data.shape[1]}")
    print(f"{'='*50}\n")
    
    for epoch in range(100):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.4f}")

        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_file)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load(model_file))
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, axis=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)

    auroc = roc_auc_score(y_test_data, all_probs)
    cm = confusion_matrix(y_test_data, all_preds)
    report = classification_report(y_test_data, all_preds, target_names=['nonseiz', 'seiz'])

    print(f"\n{model_name} Results:")
    print(f"AUROC: {auroc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    # Save results to text file
    with open(results_file, 'w') as f:
        f.write(f"{model_name} Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Train patients: {train_idx}\n")
        f.write(f"Test patients: {test_idx}\n\n")
        f.write(f"Features used: {feature_cols}\n\n")
        f.write(f"Input dimension: {X_train_data.shape[1]}\n\n")
        if pca_info:
            f.write(f"PCA components: {pca_info['n_components']}\n")
            f.write(f"PCA explained variance ratio: {pca_info['explained_variance_ratio']}\n")
            f.write(f"Total explained variance: {pca_info['total_variance']:.4f}\n\n")
        f.write(f"AUROC: {auroc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"Results saved to {results_file}")

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['nonseiz', 'seiz'],
                yticklabels=['nonseiz', 'seiz'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()

    # Training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_file)
    plt.close()

    return auroc, cm, report


# ========== Model 1: Without PCA ==========
print("\n" + "=" * 60)
print("MODEL 1: EMD Features WITHOUT PCA")
print("=" * 60)

auroc_no_pca, cm_no_pca, report_no_pca = train_and_evaluate(
    X_train_scaled, X_test_scaled, y_train, y_test,
    model_name="EMD Features (No PCA)",
    results_file="emd_no_pca_results.txt",
    model_file="best_model_no_pca.pt",
    cm_file="emd_no_pca_confusion_matrix.png",
    curves_file="emd_no_pca_training_curves.png"
)


# ========== Model 2: With PCA (10 components) ==========
print("\n" + "=" * 60)
print("MODEL 2: EMD Features WITH PCA (10 components)")
print("=" * 60)

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

pca_info = {
    'n_components': 10,
    'explained_variance_ratio': pca.explained_variance_ratio_,
    'total_variance': sum(pca.explained_variance_ratio_)
}

auroc_pca, cm_pca, report_pca = train_and_evaluate(
    X_train_pca, X_test_pca, y_train, y_test,
    model_name="EMD Features with PCA (10 components)",
    results_file="emd_pca10_results.txt",
    model_file="best_model_pca10.pt",
    cm_file="emd_pca10_confusion_matrix.png",
    curves_file="emd_pca10_training_curves.png",
    pca_info=pca_info
)


# ========== Summary ==========
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model 1 (No PCA) - AUROC: {auroc_no_pca:.4f}")
print(f"Model 2 (PCA 10) - AUROC: {auroc_pca:.4f}")
print("\nResults saved to:")
print("  - emd_no_pca_results.txt")
print("  - emd_pca10_results.txt")
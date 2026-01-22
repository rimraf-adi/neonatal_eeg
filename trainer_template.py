import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import random

eeg_idx = [1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 31, 34,
           36, 38, 39, 40, 41, 44, 47, 50, 51, 52, 62, 63, 66, 67, 69, 73, 75,
           76, 77, 78, 79]

train_count = int(len(eeg_idx) * 0.75)
train_idx = random.sample(eeg_idx, train_count)
test_idx = [i for i in eeg_idx if i not in train_idx]

train_frames = []
test_frames = []

feature_dir = './stft_features'

def load_patient_data(idx_list, frames_list, mode):
    for i in idx_list:
        filename = os.path.join(feature_dir, f'patient_{i:03d}.csv')
        if os.path.exists(filename):
            print(f'[{mode}] Processing file {filename}')
            df = pd.read_csv(filename)
            frames_list.append(df)
        else:
            print(f'[{mode}] Warning: File {filename} not found.')

load_patient_data(train_idx, train_frames, 'TRAIN')
train_df = pd.concat(train_frames, ignore_index=True)

load_patient_data(test_idx, test_frames, 'TEST')
test_df = pd.concat(test_frames, ignore_index=True)


all_cols = train_df.columns.tolist()
feature_cols = [c for c in all_cols if c not in ['label', 'channel']]

X_train = train_df[feature_cols]
y_train = train_df['label']

X_test = test_df[feature_cols]
y_test = test_df['label']

print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Features count: {len(feature_cols)}")


features = features[:-1]



train_df[features] = train_df[features].replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy='median')
train_df[features] = imputer.fit_transform(train_df[features])

test_df[features] = test_df[features].replace([np.inf, -np.inf], np.nan)
test_df[features] = imputer.transform(test_df[features])

X_train = train_df[features].astype('float32').values
X_test = test_df[features].astype('float32').values

# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(train_df['label'])
# y_test = label_encoder.transform(test_df['label'])


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#SMOTE
X_train_resampled = X_train_scaled
y_train_resampled = y_train

print(f"Train size : {X_train_resampled.shape[0]}")

#class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weights: {class_weights_tensor}")


import torch
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

from collections import Counter
print(f"Class distribution: {Counter(y_train_resampled)}")


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = EEGDataset(X_train_scaled, y_train)
test_dataset = EEGDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(X_train_scaled.shape[1]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

best_val_loss = float('inf')
patience = 20
epochs_no_improve = 0
history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

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
        torch.save(model.state_dict(), "best_model.pt")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break


model.load_state_dict(torch.load("best_model.pt"))

model.eval()
all_preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, axis=1).cpu().numpy()
        all_preds.extend(preds)

cm = confusion_matrix(y_test, all_preds)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, all_preds, target_names=['nonseiz', 'seiz']))

# Confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['nonseiz', 'seiz'],
            yticklabels=['nonseiz', 'seiz'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

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
plt.show()
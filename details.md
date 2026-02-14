# Neonatal EEG Seizure Biomarker Detection

## 1. Introduction

This repository implements a complete machine learning pipeline for the automated detection of neonatal seizures from electroencephalogram (EEG) recordings. The system processes raw multi-channel EEG data stored in European Data Format (EDF), extracts domain-specific features, and trains neural network classifiers to discriminate between ictal (seizure) and non-ictal (background) epochs. Two complementary feature-engineering paradigms are employed: **spectral-slope frequency features** and **Empirical Mode Decomposition (EMD) features**. Both paradigms are further augmented with **Principal Component Analysis (PCA)** for dimensionality reduction, yielding four experimental configurations evaluated across 10-trial cross-validation.

---

## 2. Dataset

### 2.1 Raw Data

The dataset comprises **79 EDF recordings** stored in the `data/` directory, each representing a continuous neonatal EEG recording. Of these, **39 patients** (indices: 1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 31, 34, 36, 38, 39, 40, 41, 44, 47, 50, 51, 52, 62, 63, 66, 67, 69, 73, 75, 76, 77, 78, 79) are used for analysis, selected based on the availability of complete annotation data. The recordings are sampled at **256 Hz**.

### 2.2 Channel Configuration

Raw EEG signals are converted to an **18-channel bipolar montage** using the standard 10-20 electrode system. The bipolar derivation pairs are organised in the following clinical order:

| Chain | Channels |
|---|---|
| Right Parasagittal | Fp2-F4, F4-C4, C4-P4, P4-O2 |
| Left Parasagittal | Fp1-F3, F3-C3, C3-P3, P3-O1 |
| Right Temporal | Fp2-F8, F8-T4, T4-T6, T6-O2 |
| Left Temporal | Fp1-F7, F7-T3, T3-T5, T5-O1 |
| Midline | Fz-Cz, Cz-Pz |

Signal preprocessing includes channel normalisation, removal of non-EEG channels (ECG, respiratory effort), and bipolar re-referencing via MNE-Python.

### 2.3 Annotations

Seizure annotations are provided by **three independent expert annotators** in CSV format (`annotations_2017_A_fixed.csv`, `annotations_2017_B.csv`, `annotations_2017_C.csv`). The final ground-truth labels are computed using a **unanimous agreement criterion**: only epochs where all three annotators concur on the label (seizure or non-seizure) are retained for analysis. A supplementary annotation file (`annot.xlsx`) provides per-channel seizure/non-seizure labels with start and stop times.

### 2.4 Epoching

Continuous EEG recordings are segmented into **1-second fixed-length epochs** (256 samples per epoch) with no overlap, using `mne.make_fixed_length_epochs`.

---

## 3. Feature Extraction

Feature extraction is performed per-channel, per-epoch, and each patient's features are saved as an independent CSV file. Two feature sets are computed:

### 3.1 Spectral Slope Features (`freq_features_updated/`)

Extracted by `v2.1.py` (vectorised implementation) and earlier iterations (`v1.3.py`). For each epoch on each channel, the following procedure is applied:

1. **Bandpass filtering**: A 4th-order Butterworth filter isolates each of four frequency bands:
   - **Delta** (0.5–4 Hz)
   - **Theta** (4–8 Hz)
   - **Alpha** (8–13 Hz)
   - **Beta** (13–30 Hz)
2. **Power Spectral Density (PSD) estimation**: Welch's method with a segment length equal to the sampling rate.
3. **Spectral slope computation**: A first-degree polynomial fit (`np.polyfit`) is applied to the log-transformed PSD within each band, yielding:
   - **Slope**: gradient of the spectral roll-off
   - **Intercept**: y-intercept of the fitted line
   - **Midband power**: predicted power at the band's geometric centre frequency

This produces **12 features per channel** (3 features × 4 bands).

### 3.2 Empirical Mode Decomposition Features (`emd_features_updated/`)

Extracted by `emd/v2.2.py` using the `pyemdcpp` library. For each epoch on each channel:

1. **EMD decomposition**: The signal is decomposed into Intrinsic Mode Functions (IMFs). The first 4 IMFs are retained.
2. **Per-IMF feature computation**: For each of the 4 IMFs, the following 6 features are calculated:
   - **Energy**: Sum of squared amplitudes
   - **Wiener entropy**: Ratio of geometric to arithmetic mean of the squared signal, quantifying spectral flatness
   - **Skewness**: Third standardised moment (asymmetry of the distribution)
   - **Kurtosis**: Fourth standardised moment (tailedness of the distribution)
   - **Standard deviation**: Measure of signal spread

This produces **24 features per channel** (6 features × 4 IMFs).

### 3.3 Earlier Feature Extraction Variants

The repository preserves earlier iterations of the feature extraction pipeline:

- `v1.0.py`: Time-domain and nonlinear features including Lyapunov exponents (Rosenstein's method), Hurst exponent (R/S analysis), Katz fractal dimension, wavelet-derived features, and spectral entropy.
- `v1.2.py`: Short-Time Fourier Transform (STFT)-based frequency features.
- `v1.3.py`: Initial spectral slope implementation using `sklearn.LinearRegression`.

---

## 4. Classification Pipeline

### 4.1 Class Balancing — Adaptive Nearest Neighbour Filtering

Neonatal EEG data is inherently severely imbalanced, with non-seizure epochs vastly outnumbering seizure epochs. Both classifiers (`emd2.py`, `freq2.py`) implement an **adaptive nearest-neighbour downsampling** strategy:

1. All seizure epochs (label = 1) are retained.
2. For each non-seizure epoch, the temporal distance to the nearest seizure epoch is computed.
3. The *N* closest non-seizure epochs are selected, where *N* = `TARGET_RATIO` × number of seizure epochs (default `TARGET_RATIO` = 2.0).
4. A minimum ratio guard ensures that non-seizure samples always outnumber seizure samples.

This approach preserves the temporal context surrounding seizure events while achieving a controlled class ratio.

### 4.2 Preprocessing

1. **Missing value imputation**: Mean imputation via `sklearn.impute.SimpleImputer`.
2. **Standardisation**: Zero-mean, unit-variance scaling via `sklearn.preprocessing.StandardScaler`.
3. **Dimensionality reduction (PCA variants)**: PCA is applied to reduce the feature space to a maximum of **10 principal components**, capturing at least 95% of the total variance. Two experimental configurations exist for each feature type: with and without PCA.

### 4.3 Neural Network Architecture

Both feature pipelines share an identical feedforward neural network classifier:

```
Input (D) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
          → Linear(32)  → BatchNorm → ReLU → Dropout(0.3)
          → Linear(2)   → Softmax
```

- **Loss function**: Cross-Entropy Loss with inverse-frequency class weights (`sklearn.utils.class_weight.compute_class_weight`).
- **Optimiser**: Adam (learning rate = 0.001).
- **Batch size**: 64.
- **Early stopping**: Patience of 10–25 epochs on validation loss, maximum 50 epochs.

### 4.4 Post-Processing Sweep

After training, softmax probabilities on the validation and test sets undergo a systematic **hyperparameter sweep**:

- **Moving Average (MA) window sizes**: 1 through 20 (temporal smoothing of prediction probabilities).
- **Decision thresholds**: 0.05 through 0.95 in increments of 0.01.

For each (MA window, threshold) combination, the following metrics are computed:
- **Precision**, **Recall**, **F1 Score**, **Accuracy**, **AUROC**
- **Confusion Matrix**

Results for every combination are saved as individual text files in structured directories (`trial_XX/detailed/[validation|test]/maXX/`).

---

## 5. Experimental Design

### 5.1 Cross-Validation

A **10-trial patient-level cross-validation** strategy is implemented. Patient splits are performed at the subject level to prevent data leakage:

- **Training**: ~70% of patients (28 subjects)
- **Validation**: ~15% of patients (6 subjects)
- **Test**: ~15% of patients (5 subjects)

Each trial uses a unique random seed (42 + trial index). Patient splits are serialised to `patient_splits.json` and reused across feature types to ensure comparability.

### 5.2 Experimental Configurations

Four configurations are evaluated:

| Configuration | Feature Type | PCA | Results Directory |
|---|---|---|---|
| Frequency Features | Spectral slope (12 features) | No | `adaptive_nn_results/` |
| PCA Frequency Features | Spectral slope → PCA (≤10 components) | Yes | `pca_adaptive_nn_results/` |
| EMD Features | EMD-derived (24 features) | No | `adaptive_emd_results/` |
| PCA EMD Features | EMD-derived → PCA (≤10 components) | Yes | `pca_adaptive_emd_results/` |

---

## 6. Statistical Testing

### 6.1 Welch's T-Test

The script `run_ttests.py` performs **Welch's unequal-variance t-test** (`scipy.stats.ttest_ind`, `equal_var=False`) on each feature to assess whether its distribution differs significantly between the seizure and non-seizure classes. Tests are conducted:

- On the raw features in the original feature space.
- On the PCA-transformed principal components.

Results are computed per trial (using only the training split to avoid bias) and stored in `ttest_results/trial_ttests.json`. Feature significance is reported as −log₁₀(*p*-value).

### 6.2 Patient-Wise Testing

The supplementary script `test.py` computes per-patient and combined t-tests on frequency features, saving patient-level results to `ttest_results/patient_wise_ttest.txt` and aggregated results to `ttest_results/combined_ttest.txt`.

---

## 7. Results Dashboard

An interactive **Streamlit dashboard** (`dashboard.py`) provides real-time visualisation of all experimental results. The dashboard supports:

- **Dataset selection**: Toggle between any of the four experimental configurations.
- **Split selection**: View validation or test set results.
- **Trial selection**: Examine individual trials or view aggregated (mean) results across all 10 trials.
- **Heatmaps**: Performance metrics (Precision, Recall, F1, Accuracy, AUROC) as a function of MA window and decision threshold.
- **Distribution plots**: Box plots of metric variance across trials for the best configuration.
- **Line plots**: Detailed performance curves across thresholds for selected MA windows.
- **T-test visualisation**: Bar charts of feature significance (−log₁₀(p)) with interactive drill-down.

The dashboard is launched via:
```bash
uv run --with streamlit --with pandas --with plotly --with regex streamlit run dashboard.py
```

---

## 8. Repository Structure

```
biomarker/
├── data/                        # Raw EDF recordings (79 patients)
├── emd/                         # EMD feature extraction scripts
│   ├── v2.2.py                  #   Current EMD extractor (pyemdcpp)
│   └── emd_features/            #   Legacy EMD feature output
├── emd_features_updated/        # Current EMD features (39 patient CSVs)
├── freq_features_updated/       # Current frequency features (39 patient CSVs)
├── pca_adaptive_emd_results/    # PCA + EMD classification results (10 trials)
├── pca_adaptive_nn_results/     # PCA + Frequency classification results (10 trials)
├── adaptive_emd_results/        # EMD classification results (no PCA)
├── adaptive_nn_results/         # Frequency classification results (no PCA)
├── ttest_results/               # Statistical test outputs
├── main.py                      # EEG preprocessing and bipolar re-referencing
├── emd2.py                      # EMD-based neural network classifier
├── freq2.py                     # Frequency-based neural network classifier
├── run_ttests.py                # Welch's t-test analysis
├── test.py                      # Per-patient t-test analysis
├── dashboard.py                 # Streamlit results dashboard
├── benchmark.py                 # CPU/MPS performance benchmarking
├── trainer_template.py          # Template training script
├── v1.0.py                      # Feature extractor v1 (time-domain/nonlinear)
├── v1.2.py                      # Feature extractor v2 (STFT)
├── v1.3.py                      # Feature extractor v3 (spectral slope)
├── v2.1.py                      # Feature extractor v4 (vectorised spectral slope)
├── annotations_2017_*.csv       # Three-annotator seizure labels
├── annot.xlsx                   # Per-channel annotation spreadsheet
├── patient_splits.json          # Serialised cross-validation splits
├── structure.md                 # Model architecture summary
└── pyproject.toml               # Project dependencies
```

---

## 9. Dependencies

The project is managed with `uv` and requires Python ≥ 3.11. Key dependencies include:

| Package | Purpose |
|---|---|
| `torch` | Neural network training and inference |
| `mne` | EEG data loading and bipolar re-referencing |
| `scikit-learn` | PCA, scaling, imputation, class weighting, metrics |
| `scipy` | Signal processing (Welch, filters), statistical tests |
| `pandas` / `numpy` | Data manipulation and numerical computation |
| `emd-signal` / `pyemdcpp` | Empirical Mode Decomposition |
| `pywavelets` | Wavelet transforms (legacy features) |
| `antropy` | Entropy and complexity measures (legacy features) |
| `streamlit` / `plotly` | Interactive results dashboard |

---

## 10. Execution Workflow

1. **Feature Extraction**: Run `v2.1.py` (frequency features) and `emd/v2.2.py` (EMD features) to generate per-patient CSV files.
2. **Classification**: Execute `freq2.py` and `emd2.py` to train classifiers and perform post-processing sweeps across all 10 trials.
3. **Statistical Analysis**: Run `run_ttests.py` to compute feature significance tests.
4. **Visualisation**: Launch `dashboard.py` via Streamlit to interactively explore results.

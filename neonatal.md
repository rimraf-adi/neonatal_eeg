# Neonatal EEG Seizure Biomarker Detection & Classification Pipeline

## Executive Summary

This repository contains an end-to-end machine learning and signal-processing pipeline for automated **neonatal seizure detection** using multi-channel Electroencephalogram (EEG) recordings. 

The system ingests raw European Data Format (`.edf`) recordings from 79 patients (39 with annotated seizure events), applies an 18-channel bipolar montage, segmenting signals into 1-second fixed epochs. It extracts domain-specific biomarker features using **Spectral Slope Analysis** and **Empirical Mode Decomposition (EMD)**, handles extreme class imbalance via **Adaptive Nearest-Neighbor Temporal Filtering**, and trains a deep Feedforward Neural Network (MLP). Post-processing features a 2D hyperparameter sweep over temporal Moving Average (MA) windows and decision thresholds. A Streamlit interactive dashboard and statistical significance tools (Welch's t-tests) allow full exploration of results.

---

## 1. System Architecture & Data Pipeline

```
┌─────────────────┐    ┌──────────────────────────┐    ┌───────────────────────────┐
│ Raw EDF Signals │───>│ 18-Channel Bipolar       │───>│ 1-Second Non-Overlapping  │
│ (79 Patients)   │    │ Montage Re-referencing   │    │ Epoching (256 Hz)         │
└─────────────────┘    └──────────────────────────┘    └─────────────────────────┬─┘
                                                                                 │
                                                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────┴─┐
│                              Feature Extraction Paradigms                         │
├─────────────────────────────────────────┬────────────────────────────────────────┤
│ Spectral Slope Features (v2.1.py)       │ EMD Features (emd/v2.2.py)             │
│ • Delta, Theta, Alpha, Beta bands       │ • 4 Intrinsic Mode Functions (IMFs)    │
│ • Welch PSD fit (Slope, Intercept, Mid) │ • Energy, Entropy, Skew, Kurtosis, Std │
│ • 12 features/channel (216 total)       │ • 24 features/channel (432 total)      │
└─────────────────────────────────────────┴────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   Adaptive Nearest-Neighbor Class Balancing                      │
│ • Retain 100% seizure epochs (1s)                                                │
│ • Select N temporally nearest non-seizure epochs (0s = 2.0 × 1s)                 │
└──────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     Preprocessing & Dimensionality Reduction                     │
│ • Mean Imputation (SimpleImputer) + Standard Scaling (StandardScaler)           │
│ • Optional PCA (Max 10 components, ≥95% variance)                                │
└──────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           Deep MLP Neural Network                                │
│ Input(D) -> FC(128)-BN-ReLU-Drop(0.3) -> FC(64)-BN-ReLU-Drop(0.3) ->             │
│             FC(32)-BN-ReLU-Drop(0.3)  -> FC(2)-Softmax                           │
│ • Weighted Cross-Entropy Loss | Adam Optimizer (lr=1e-3) | Early Stopping        │
└──────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      Post-Processing & Optimization Sweep                        │
│ • Moving Average (MA) Window Sweep: 1 to 20 seconds                              │
│ • Threshold Sweep: 0.05 to 0.95 (step 0.01)                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        Evaluation & Visualization                                │
│ • 10-Trial Patient-Level Cross-Validation (patient_splits.json)                  │
│ • Welch's T-Test Feature Analysis (run_ttests.py)                                │
│ • Interactive Streamlit Results Dashboard (dashboard.py)                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset & Preprocessing Details

- **Raw Recordings**: 79 continuous EDF EEG recordings sampled at **256 Hz** (`eeg1.edf` to `eeg79.edf`).
- **Patient Cohort**: 39 patients (IDs: 1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 31, 34, 36, 38, 39, 40, 41, 44, 47, 50, 51, 52, 62, 63, 66, 67, 69, 73, 75, 76, 77, 78, 79) possess verified annotation data.
- **18-Channel Bipolar Montage**: Standard 10-20 electrode system converted to bipolar derivations via MNE-Python:
  - *Right Parasagittal*: `Fp2-F4`, `F4-C4`, `C4-P4`, `P4-O2`
  - *Left Parasagittal*: `Fp1-F3`, `F3-C3`, `C3-P3`, `P3-O1`
  - *Right Temporal*: `Fp2-F8`, `F8-T4`, `T4-T6`, `T6-O2`
  - *Left Temporal*: `Fp1-F7`, `F7-T3`, `T3-T5`, `T5-O1`
  - *Midline*: `Fz-Cz`, `Cz-Pz`
- **Annotations**: Derived from 3 independent experts (`annotations_2017_A_fixed.csv`, `annotations_2017_B.csv`, `annotations_2017_C.csv`). Ground truth labels enforce **unanimous consensus** across all three annotators. Channel-specific temporal ranges are specified in `annot.xlsx`.
- **Epoching**: Fixed **1-second non-overlapping epochs** (256 samples/epoch) using `mne.make_fixed_length_epochs`.

---

## 3. Feature Extraction Paradigms

### A. Spectral Slope Frequency Features ([`v2.1.py`](file:///d:/neonatal/neonatal_eeg/v2.1.py))
- **Core Concept**: Log-transformed Power Spectral Density (PSD) line fitting per frequency band.
- **Frequency Bands**: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz).
- **PSD Estimation**: Welch's periodogram with segment length = 256 samples.
- **Polynomial Fit**: Log-log linear regression (`np.polyfit`) per band:
  1. **Slope**: Spectral decay roll-off rate.
  2. **Intercept**: Power axis intercept.
  3. **Midband Power**: Model-predicted power at geometric center frequency.
- **Dimension**: 12 features per channel × 18 channels = **216 features per epoch**.

### B. Empirical Mode Decomposition Features ([`emd/v2.2.py`](file:///d:/neonatal/neonatal_eeg/emd/v2.2.py))
- Decomposes signals using `pyemdcpp` into Intrinsic Mode Functions (IMFs). First 4 IMFs are selected.
- Per-IMF metrics computed:
  1. Energy (sum of squared amplitudes)
  2. Wiener Entropy (ratio of geometric to arithmetic mean of squared signal)
  3. Skewness (3rd standardized moment)
  4. Kurtosis (4th standardized moment)
  5. Standard Deviation
- **Dimension**: 24 features per channel × 18 channels = **432 features per epoch**.

### C. Legacy Extractors
- [`v1.0.py`](file:///d:/neonatal/neonatal_eeg/v1.0.py): Non-linear complexity features (Rosenstein Lyapunov exponents, Hurst exponent via R/S analysis, Katz fractal dimension, wavelet entropy, spectral entropy).
- [`v1.2.py`](file:///d:/neonatal/neonatal_eeg/v1.2.py): Short-Time Fourier Transform (STFT) features.
- [`v1.3.py`](file:///d:/neonatal/neonatal_eeg/v1.3.py): Initial un-vectorized spectral slope fitting using `sklearn.LinearRegression`.

---

## 4. Class Balancing & Machine Learning

### 4.1 Adaptive Nearest-Neighbor Filtering (Legacy)
To address severe imbalance (background epochs >> seizure epochs):
1. Retains **all seizure epochs** (label = 1).
2. Computes the temporal sample index distance from every non-seizure epoch to its nearest seizure event.
3. Selects the $N = \text{TARGET\_RATIO} \times N_{\text{seizure}}$ non-seizure epochs closest in time (default `TARGET_RATIO` = 2.0).
4. Maintains local background context preceding and following seizure events.

### 4.2 WeightedRandomSampler (Recommended for DL)

For deep learning pipelines, use PyTorch's `torch.utils.data.WeightedRandomSampler` instead of pre-filtering the dataset. This approach:

- **Preserves the full dataset** — no epochs are discarded, so the model sees all available data across training
- **Controls class balance at the batch level** — each mini-batch is sampled to approximate the target seizure:non-seizure ratio
- **Prevents classifier collapse** — without balanced sampling, the model learns to predict the majority class (non-seizure) for everything and achieves high accuracy with zero recall

**Implementation**:
```python
from torch.utils.data import WeightedRandomSampler

# Compute per-sample weights (inverse class frequency)
class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels]

# For 50:50 ratio (dominant configuration)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(labels),
    replacement=True
)

train_loader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

### 4.3 Sampling Ratio Ablation Study

Ablate the effective seizure:non-seizure sampling ratio to find the optimal balance point. The **50:50 ratio** is the dominant configuration — it ensures the classifier never collapses to majority-class prediction.

| Sampling Ratio (Seizure : Non-Seizure) | What It Tests |
|---|---|
| **50 : 50** (dominant) | Maximum seizure emphasis — prevents classifier collapse, highest recall |
| 60 : 40 | Slight over-representation of seizure class |
| 40 : 60 | Mild non-seizure bias while maintaining seizure sensitivity |
| 30 : 70 | Closer to natural distribution but still controlled |
| 20 : 80 | Near-natural ratio — tests if features are strong enough to overcome imbalance |
| Natural ratio (unweighted) | Baseline — expect high accuracy, near-zero recall |

**Expected findings**:
- 50:50 should yield the **best F1 / highest recall** — the classifier is forced to learn seizure patterns rather than defaulting to "non-seizure"
- As the ratio shifts toward natural, **accuracy increases but recall drops sharply** — the model increasingly predicts non-seizure for all epochs
- The optimal operating point for clinical use likely lies between 50:50 and 40:60, where recall remains high while precision improves

**Ablation output**: For each ratio, report Precision, Recall, F1, AUROC, and Accuracy across the same 10-trial patient-level CV splits. Present as a table and a line plot (ratio on x-axis, metrics on y-axis).

### 4.4 Model Architecture ([`freq2.py`](file:///d:/neonatal/neonatal_eeg/freq2.py), [`emd2.py`](file:///d:/neonatal/neonatal_eeg/emd2.py))
- **Preprocessing**: `SimpleImputer(strategy='mean')` $\to$ `StandardScaler()`.
- **Dimensionality Reduction**: Optional PCA retaining $\le 10$ components ($\ge 95\%$ variance).
- **MLP Classifier**:
  ```
  Input (D) ──> Linear(128) ──> BatchNorm1d ──> ReLU ──> Dropout(0.3)
            ──> Linear(64)  ──> BatchNorm1d ──> ReLU ──> Dropout(0.3)
            ──> Linear(32)  ──> BatchNorm1d ──> ReLU ──> Dropout(0.3)
            ──> Linear(2)   ──> Softmax
  ```
- **Training Setup**:
  - Loss: Cross-Entropy Loss with inverse-frequency class weights (`sklearn.utils.class_weight`).
  - Sampler: `WeightedRandomSampler` with 50:50 target ratio (see §4.2).
  - Optimizer: Adam ($lr = 0.001$), Batch Size = 64.
  - Early Stopping: Patience of 10–25 epochs on validation loss (max 50 epochs).

### 4.5 Post-Processing Hyperparameter Sweep
- **Temporal Moving Average (MA)**: Probabilities smoothed over 1 to 20-second sliding windows.
- **Decision Threshold Sweep**: Evaluated from 0.05 to 0.95 in 0.01 increments.
- Metrics recorded per grid point: Accuracy, Precision, Recall, F1 Score, AUROC, Confusion Matrix.

---

## 5. Experimental Results & Findings

### Experimental Configurations Evaluated Across 10 Patient-Level Trials
Patient splits ([`patient_splits.json`](file:///d:/neonatal/neonatal_eeg/patient_splits.json)): ~70% Train (28 subjects), ~15% Validation (6 subjects), ~15% Test (5 subjects).

| Configuration | Feature Base | PCA Applied | Output Directory |
|---|---|---|---|
| Frequency Features | Spectral Slope (216-D) | No | `adaptive_nn_results/` |
| PCA Frequency Features | Spectral Slope $\to$ PCA ($\le$10-D) | Yes | `pca_adaptive_nn_results/` |
| EMD Features | EMD (432-D) | No | `adaptive_emd_results/` |
| PCA EMD Features | EMD $\to$ PCA ($\le$10-D) | Yes | `pca_adaptive_emd_results/` |

### Benchmark Highlights (from [`top5_pca_metrics_val_test.txt`](file:///d:/neonatal/neonatal_eeg/top5_pca_metrics_val_test.txt))

- **PCA Frequency Features (Trial 0)**:
  - **Validation Set**: Accuracy = 0.7616, F1 = 0.6884, Precision = 0.6826, Recall = 0.6942, **AUROC = 0.8070**
  - **Test Set**: Accuracy = 0.7381, F1 = 0.7308, Precision = 0.6826, Recall = 0.7864, **AUROC = 0.8576**

- **PCA EMD Features (Trial 0)**:
  - **Validation Set**: Accuracy = 0.6716, F1 = 0.6466, Precision = 0.5461, Recall = 0.7924, **AUROC = 0.7896**
  - **Test Set**: Accuracy = 0.6777, F1 = 0.7028, Precision = 0.6026, Recall = 0.8429, **AUROC = 0.7787**

---

## 6. Feature Set Validation & Quality Methodologies (Beyond t-tests & ANN)

The core feature set is a **parameterization of the 1/f spectral structure**: a line fit to log-PSD within each frequency band, extracting slope, intercept, and midband power. The following 7 methodologies prove this representation captures seizure dynamics *independent of any particular classifier or statistical test*.

### 6.1 Classifier Complexity Ladder

Train a **progression of increasingly simple models** on the same features and same patient splits. If the simplest model already achieves high performance, the features themselves are doing the discriminative work — not the model.

| Model | What It Proves |
|---|---|
| Logistic Regression | Features are **linearly separable** — the log-PSD fit did the real work |
| Linear SVM | Same argument, different margin formulation |
| Random Forest / XGBoost | Features work with non-parametric tree models too |
| k-Nearest Neighbors ($k$=5) | The feature **geometry** itself clusters well in Euclidean space |

**Key threshold**: If Logistic Regression alone achieves AUROC > 0.80, that is a very strong statement: *"our features are so well-constructed that even a linear decision boundary suffices."*

### 6.2 Feature Ablation Study

Systematically **remove one component at a time** and measure performance degradation to prove each band/parameter contributes non-redundant information:

| Ablation | What It Tests |
|---|---|
| Full feature set (Delta + Theta + Alpha + Beta) | Baseline performance |
| Drop Delta band features | Contribution of low-frequency spectral structure |
| Drop Theta band features | Contribution of theta-range dynamics |
| Drop Alpha band features | Contribution of alpha-range dynamics |
| Drop Beta band features | Contribution of high-frequency spectral structure |
| Slope-only (remove intercept & midband) | How much do the secondary parameters add? |
| Intercept-only | Is the intercept alone sufficient? |

If dropping Delta causes the largest performance drop, that is physiologically interpretable — neonatal seizures have strong low-frequency signatures. If all bands contribute, it demonstrates that the multi-band parameterization captures complementary spectral information.

### 6.3 UMAP / t-SNE Visualization

Project the 216-D spectral feature vectors into 2D using **UMAP** (`umap.UMAP()`) and color-code each epoch by seizure (ictal) vs. non-seizure (background):

- **Distinct clusters** → features separate classes *with no model at all*
- **Overlapping clouds** → features lack discriminative geometry

This is the single most visually compelling figure for a paper/thesis — reviewers can assess feature quality at a glance. Quantify with **Silhouette Score** (`sklearn.metrics.silhouette_score`) and **Davies-Bouldin Index** for numeric backing.

### 6.4 Fisher's Discriminant Ratio (Per-Feature Separability)

For each of the 216 features, compute the Fisher score:

$$J_i = \frac{(\mu_{i,\text{seizure}} - \mu_{i,\text{non-seizure}})^2}{\sigma^2_{i,\text{seizure}} + \sigma^2_{i,\text{non-seizure}}}$$

This is cleaner than t-tests because:
- **Does not inflate with sample size** (t-statistics grow with $\sqrt{N}$, Fisher ratio does not)
- Directly measures the **signal-to-noise ratio** of each feature for class discrimination
- Enables ranking all 216 features to identify which *band × channel × parameter* combinations are most discriminative (e.g., "Delta slope on F7-T3 has the highest Fisher score")

### 6.5 Comparison Against Baseline Feature Sets

Show the spectral slope features **outperform simpler alternatives** on the exact same pipeline (same patient splits, same class balancing, same model):

| Baseline Feature Set | What It Tests |
|---|---|
| Raw band power (area under PSD per band) | Does the *shape* of PSD (slope/intercept) add information beyond raw power? |
| Time-domain variance / RMS per channel | Does frequency decomposition help at all? |
| Line length / zero-crossing rate | Standard clinical EEG features |
| Hjorth parameters (Activity, Mobility, Complexity) | Classical EEG descriptors from the literature |
| Spectral entropy per band | Does the parametric fit beat a non-parametric entropy measure? |

If spectral slope features consistently beat these baselines → the log-PSD linear fitting extracts information that simpler features miss. This is the **comparative advantage** argument.

### 6.6 Temporal Onset Trajectory Plots

For 3–4 patients with clear seizure events, plot the **raw feature value over time** (one point per 1-second epoch) for key features like `Delta_slope` or `Theta_midband`:

```
Time (seconds)    ←— pre-ictal —→  ←——— seizure ———→  ←— post-ictal —→
                                   |                  |
Delta_slope:  ─────────────────────┃▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄┃──────────────────
                                   ↑                  ↑
                             seizure onset       seizure offset
```

If the feature shows a **sharp transition** precisely at the expert-annotated seizure onset, that is powerful evidence of temporal sensitivity — the feature *tracks* the physiological event in real time. This also provides **clinical interpretability**: clinicians can understand what the model is detecting.

### 6.7 Cohen's $d$ Effect Size

For each feature, compute the standardized effect size:

$$d = \frac{\mu_{\text{seizure}} - \mu_{\text{non-seizure}}}{\sigma_{\text{pooled}}}$$

Unlike p-values, **Cohen's $d$ does not inflate with sample size**. With thousands of 1-second epochs, nearly any feature will have $p < 0.001$, making p-values uninformative. Effect size tells you how *meaningfully different* the distributions are:

| Cohen's $d$ | Interpretation |
|---|---|
| $d > 0.8$ | Large effect — strong clinical/physiological separation |
| $0.5 < d < 0.8$ | Medium effect — meaningful but modest separation |
| $0.2 < d < 0.5$ | Small effect — detectable but weak separation |
| $d < 0.2$ | Negligible — feature not useful |

Report how many of the 216 features fall into each category.

### Implementation Priority Order

For a paper or thesis, implement in this order:

| Priority | Method | Effort | Impact |
|---|---|---|---|
| 1 | Classifier Complexity Ladder | Low | Strongest argument — proves features, not model |
| 2 | Feature Ablation Study | Low | Proves each band contributes non-redundantly |
| 3 | UMAP Scatter Plot | Low | One figure, visually compelling |
| 4 | Baseline Feature Comparison | Medium | Proves added value over simpler features |
| 5 | Fisher Ratio + Cohen's $d$ | Low | Per-feature quantitative evidence |
| 6 | Temporal Trajectory Plots | Medium | Clinical interpretability story |
| 7 | Mutual Information | Low | Non-parametric dependency measure (bonus) |

---

## 7. Statistical Significance & Dashboard

### Welch's T-Test Analysis ([`run_ttests.py`](file:///d:/neonatal/neonatal_eeg/run_ttests.py), [`test.py`](file:///d:/neonatal/neonatal_eeg/test.py))
- Evaluates feature separation between seizure vs. non-seizure classes per trial using Welch's unequal variance t-test (`scipy.stats.ttest_ind(..., equal_var=False)`).
- Analyzes both raw feature space and principal component space.
- Stores output as $-\log_{10}(p\text{-value})$ in `ttest_results/trial_ttests.json` and patient summaries in `ttest_results/patient_wise_ttest.txt`.

### Interactive Streamlit Dashboard ([`dashboard.py`](file:///d:/neonatal/neonatal_eeg/dashboard.py))
Provides dynamic interactive analysis:
1. Dataset & Split Selector (Frequency / EMD, with or without PCA; Validation / Test).
2. 2D Heatmaps (MA Window vs Threshold for Precision, Recall, F1, Accuracy, AUROC).
3. Trial Variance Boxplots & Performance Metric Threshold Curves.
4. Interactive Feature Significance $-\log_{10}(p)$ Bar Charts.

---

## 8. Repository File Map

| File / Directory | Description |
|---|---|
| [`main.py`](file:///d:/neonatal/neonatal_eeg/main.py) | Ingests EDF recordings, sets 18-channel bipolar montage, epochs data, maps annotations from `annot.xlsx`. |
| [`v2.1.py`](file:///d:/neonatal/neonatal_eeg/v2.1.py) | Current production spectral slope feature extractor (vectorized numpy polyfit across 4 bands). |
| [`emd/v2.2.py`](file:///d:/neonatal/neonatal_eeg/emd/v2.2.py) | Current production EMD feature extractor (`pyemdcpp`, 4 IMFs, 6 metrics/IMF). |
| [`freq2.py`](file:///d:/neonatal/neonatal_eeg/freq2.py) | Neural network training, PCA, adaptive class balancing, & sweep pipeline for frequency features. |
| [`emd2.py`](file:///d:/neonatal/neonatal_eeg/emd2.py) | Neural network training, PCA, adaptive class balancing, & sweep pipeline for EMD features. |
| [`run_ttests.py`](file:///d:/neonatal/neonatal_eeg/run_ttests.py) | Computes per-trial Welch's t-tests on raw and PCA features. |
| [`test.py`](file:///d:/neonatal/neonatal_eeg/test.py) | Computes patient-specific and combined t-tests on frequency features. |
| [`dashboard.py`](file:///d:/neonatal/neonatal_eeg/dashboard.py) | Streamlit dashboard for interactive visualization of model performance and statistical metrics. |
| [`benchmark.py`](file:///d:/neonatal/neonatal_eeg/benchmark.py) | Benchmarks PyTorch training performance on CPU vs MPS/CUDA. |
| [`generate_charts.py`](file:///d:/neonatal/neonatal_eeg/generate_charts.py) | Generates figures and charts for publication/reports. |
| [`export_screenshots.py`](file:///d:/neonatal/neonatal_eeg/export_screenshots.py) | Automated dashboard screenshot generator using Playwright. |
| [`extract_metrics.py`](file:///d:/neonatal/neonatal_eeg/extract_metrics.py) | Extracts top-performing hyperparameter settings across trials. |
| [`trainer_template.py`](file:///d:/neonatal/neonatal_eeg/trainer_template.py) | Modular training script template. |
| `v1.0.py`, `v1.2.py`, `v1.3.py` | Legacy feature extraction scripts (non-linear, STFT, baseline linear regression). |
| `patient_splits.json` | Fixed 10-trial patient-level cross-validation split mapping. |
| `pyproject.toml` | Project configuration and dependencies managed via `uv`. |

---

## 9. How to Run

### Installation & Environment Setup
Managed via `uv` with Python $\ge 3.11$:
```bash
cd neonatal_eeg
# Install dependencies
uv sync
```

### Execution Workflow

1. **Extract Features**:
   ```bash
   uv run python v2.1.py        # Extract Spectral Slope features
   uv run python emd/v2.2.py    # Extract EMD features
   ```

2. **Train Models & Perform Hyperparameter Sweeps**:
   ```bash
   uv run python freq2.py       # Train & sweep Frequency models
   uv run python emd2.py        # Train & sweep EMD models
   ```

3. **Statistical Analysis**:
   ```bash
   uv run python run_ttests.py  # Run Welch's t-tests across trials
   ```

4. **Launch Interactive Dashboard**:
   ```bash
   uv run --with streamlit --with pandas --with plotly --with regex streamlit run dashboard.py
   ```

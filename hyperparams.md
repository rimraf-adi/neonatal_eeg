# Legacy Codebase Hyperparameters Reference

Comprehensive reference guide for all hyperparameters, configuration options, data thresholds, and architectural settings in the legacy neonatal EEG seizure detection codebase ([`legacy/`](file:///d:/neonatal/legacy)).

---

## 1. Raw Signal Preprocessing & Segmentation

| Hyperparameter | Value | Description & Code Source |
|---|---|---|
| **Sampling Frequency ($f_s$)** | `256 Hz` | Standard sampling rate across EDF recordings. |
| **Epoch Duration** | `1.0 s` | Fixed-length segmentation window ([`legacy/main.py#L93`](file:///d:/neonatal/legacy/main.py#L93), [`legacy/v2.1.py#L90`](file:///d:/neonatal/legacy/v2.1.py#L90)). |
| **Epoch Overlap** | `0.0 s` (0%) | Non-overlapping consecutive temporal windows. |
| **Samples per Epoch** | `256` | $256\text{ Hz} \times 1.0\text{ s}$ per epoch per channel. |
| **Bipolar Montage** | 18 Channels | Standard 10–20 electrode montage converted to bipolar derivations. |
| **Clinical Chains** | 18 bipolar pairs | **Right Parasagittal (4)**: `Fp2-F4`, `F4-C4`, `C4-P4`, `P4-O2`<br>**Left Parasagittal (4)**: `Fp1-F3`, `F3-C3`, `C3-P3`, `P3-O1`<br>**Right Temporal (4)**: `Fp2-F8`, `F8-T4`, `T4-T6`, `T6-O2`<br>**Left Temporal (4)**: `Fp1-F7`, `F7-T3`, `T3-T5`, `T5-O1`<br>**Midline (2)**: `Fz-Cz`, `Cz-Pz` |
| **Dropped Non-EEG Channels** | 4 channels | `ECG EKG`, `RESP EFFORT`, `ECG EKG-REF`, `RESP EFFORT-REF`. |
| **Ground Truth Consensus** | Unanimous (3/3) | Epoch retained only if Annotator A, B, and C agree (`s1 == s2 == s3`) in [`legacy/v2.1.py#L136`](file:///d:/neonatal/legacy/v2.1.py#L136). *(Majority vote $\ge 2$ used in earlier `v1.0.py`/`v1.2.py`)*. |

---

## 2. Feature Extraction Hyperparameters

### A. Spectral Slope Frequency Features ([`legacy/v2.1.py`](file:///d:/neonatal/legacy/v2.1.py), [`legacy/v1.3.py`](file:///d:/neonatal/legacy/v1.3.py))
- **Bandpass Filter**: 4th-order zero-phase Butterworth filter (`order = 4`, `sosfiltfilt`).
- **Frequency Bands**:
  - **Delta**: `0.5 – 4.0 Hz`
  - **Theta**: `4.0 – 8.0 Hz`
  - **Alpha**: `8.0 – 12.0 Hz` *(in `v2.1.py`, `v1.3.py`) / `8.0 – 13.0 Hz` (in `v1.2.py` / `details.md`)*
  - **Beta**: `12.0 – 35.0 Hz` *(in `v2.1.py`, `v1.3.py`) / `13.0 – 30.0 Hz` (in `details.md`)*
  - *(Optional Gamma band in dictionary: `35.0 – 100.0 Hz`)*
- **PSD Estimation**: Welch's periodogram with `nperseg = 256` ($1.0\text{ s}$) at $f_s = 256\text{ Hz}$.
- **Log Transformation**: $\log_{10}(\text{PSD} + 10^{-10})$.
- **Spectral Slope Fit**: 1st-degree polynomial line fit (`np.polyfit(log_f, log_p, deg=1)` or `LinearRegression`).
- **Features Extracted per Band**: 3 features:
  1. `slope`: Spectral decay roll-off gradient.
  2. `intercept`: Power axis intercept.
  3. `midband`: Estimated power at the band's geometric midpoint frequency ($f_{\text{mid}} = \frac{\text{low} + \text{high}}{2}$).
- **Total Feature Dimension**: 18 channels $\times$ 4 bands $\times$ 3 features = **216 features per epoch** (12 per channel).

### B. Empirical Mode Decomposition (EMD) Features ([`legacy/emd2.py`](file:///d:/neonatal/legacy/emd2.py), [`legacy/details.md`](file:///d:/neonatal/legacy/details.md))
- **Decomposition Library**: `pyemdcpp` / `emd-signal`.
- **IMFs Retained**: First 4 Intrinsic Mode Functions (`IMF 1–4`).
- **Features Extracted per IMF**: 6 statistical metrics:
  1. `energy`: $\sum x^2$
  2. `wiener_entropy`: $\frac{\exp(\text{mean}(\log(x^2 + 10^{-12})))}{\text{mean}(x^2)}$ (spectral flatness)
  3. `skewness`: Fisher-Pearson standardized skewness (`scipy.stats.skew`)
  4. `kurtosis`: 4th standardized moment (`scipy.stats.kurtosis`)
  5. `std`: Standard deviation $\sigma$
  6. `psd_rms`: $\sqrt{\text{mean}(\text{PSD})}$ *(dropped in downstream classifiers)*
- **Total Feature Dimension**: 18 channels $\times$ 4 IMFs $\times$ 6 features = **432 features per epoch** (24 per channel).

### C. Earlier Feature Extraction Variants ([`legacy/v1.0.py`](file:///d:/neonatal/legacy/v1.0.py), [`legacy/v1.2.py`](file:///d:/neonatal/legacy/v1.2.py))
- **Discrete Wavelet Transform (DWT)**: Daubechies 4 (`db4`), 4 decomposition levels (`wavedec(sig, 'db4', level=4)`) yielding sub-bands `[a4, d4, d3, d2, d1]`.
- **Short-Time Fourier Transform (STFT)**: `scipy.signal.stft(sig, fs=256, nperseg=256)`.
- **Non-Linear Complexity Measures**:
  - *Lyapunov Exponent (Rosenstein)*: Embedding dimension $m = 5$, delay $\tau = 1$, exclusion mean period = 1, iterations $\min(50, N // 10)$.
  - *Hurst Exponent (R/S analysis)*: Chunk sizes starting from `8` doubled up to $N$.
  - *Katz Fractal Dimension*: $D = \frac{\log_{10}(n)}{\log_{10}(d / L) + \log_{10}(n)}$.

---

## 3. Class Balancing (Adaptive Nearest Neighbor Filtering)
*(Defined in [`legacy/freq2.py#L69-L141`](file:///d:/neonatal/legacy/freq2.py#L69-L141), [`legacy/emd2.py#L64-L136`](file:///d:/neonatal/legacy/emd2.py#L64-L136))*

| Hyperparameter | Value | Description |
|---|---|---|
| **Target Ratio (`TARGET_RATIO`)** | `2.0` | Target ratio of non-seizures to seizures ($N_0 : N_1 = 2.0 : 1$). |
| **Permitted Target Range** | `1.5` to `3.0` | Operating bounds for target class ratio. |
| **Seizure Retention** | `100%` | All seizure epochs (label = 1) are kept unconditionally. |
| **Distance Function** | 1D temporal distance | $\min_{j \in S} \|i - j\|$ in epoch index space. |
| **Selection Algorithm** | Nearest $N$ neighbours | `np.argpartition(min_distances, n_to_select - 1)[:n_to_select]`. |
| **Minimum Ratio Guard** | $1.5 \times N_1$ or $1.1 \times N_1$ | Fallback to ensure non-seizure epochs always exceed seizure epochs ($N_0 > N_1$). |

---

## 4. Preprocessing & Dimensionality Reduction (PCA)
*(Defined in [`legacy/freq2.py#L59-L62,L504-L525`](file:///d:/neonatal/legacy/freq2.py#L59-L62,L504-L525))*

| Hyperparameter | Value | Description |
|---|---|---|
| **Missing Value Imputer** | `SimpleImputer(strategy='mean')` | Mean imputation across columns (`median` in `trainer_template.py`). |
| **Feature Scaler** | `StandardScaler()` | Zero-mean, unit-variance standardization. |
| **PCA Maximum Components** | `10` | `PCA_MAX_COMPONENTS = 10`. |
| **PCA Variance Threshold** | `0.95` (95%) | `PCA_VARIANCE_THRESHOLD = 0.95` (retains $\le 10$ components or 95% variance). |

---

## 5. Cross-Validation & Dataset Splits
*(Defined in [`legacy/freq2.py#L457-L470`](file:///d:/neonatal/legacy/freq2.py#L457-L470), [`legacy/patient_splits.json`](file:///d:/neonatal/legacy/patient_splits.json))*

| Hyperparameter | Value | Description |
|---|---|---|
| **Validation Strategy** | 10-Trial Cross-Validation | Independent patient-level random partitioning. |
| **Annotated Cohort** | 39 patients | Patient IDs: `[1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 31, 34, 36, 38, 39, 40, 41, 44, 47, 50, 51, 52, 62, 63, 66, 67, 69, 73, 75, 76, 77, 78, 79]`. |
| **Train Set Size** | 28 patients (~71.8%) | First 28 patients from shuffled indices. |
| **Validation Set Size** | 6 patients (~15.4%) | Middle 6 patients from shuffled indices. |
| **Test Set Size** | 5 patients (~12.8%) | Remaining 5 patients from shuffled indices. |
| **Trial Random Seeds** | $42 + \text{trial\_index}$ | Seeds: `42, 43, 44, 45, 46, 47, 48, 49, 50, 51`. |

---

## 6. Classifier Architecture & Neural Network Settings
*(Defined in [`legacy/freq2.py#L219-L236`](file:///d:/neonatal/legacy/freq2.py#L219-L236))*

### Primary MLP Architecture
```
Input (D) ──> Linear(D, 128) ──> BatchNorm1d(128) ──> ReLU() ──> Dropout(0.3)
          ──> Linear(128, 64) ──> BatchNorm1d(64)  ──> ReLU() ──> Dropout(0.3)
          ──> Linear(64, 32)  ──> BatchNorm1d(32)  ──> ReLU() ──> Dropout(0.3)
          ──> Linear(32, 2)
```

- **Input Dimension ($D$)**:
  - With PCA: $D = 10$
  - Without PCA (Spectral Slope): $D = 216$
  - Without PCA (EMD): $D = 432$
- **Dropout Rate**: `0.3` across all hidden layers.
- **Normalization**: Batch Normalization (`BatchNorm1d`) before activation.

*(Earlier variant in [`legacy/trainer_template.py`](file:///d:/neonatal/legacy/trainer_template.py#L92-L110): Linear 64 $\to$ 32 $\to$ 16 $\to$ 8 $\to$ 2 with Dropouts 0.3, 0.2, 0.1)*

---

## 7. Model Training & Optimization
*(Defined in [`legacy/freq2.py#L245-L290,L531-L549`](file:///d:/neonatal/legacy/freq2.py#L245-L290,L531-L549))*

| Hyperparameter | Value | Description |
|---|---|---|
| **Optimizer** | `torch.optim.Adam` | Adam optimizer. |
| **Learning Rate ($lr$)** | `0.001` ($10^{-3}$) | Initial learning rate. |
| **Batch Size** | `64` | Training / validation / test mini-batch size (`32` in `trainer_template.py`). |
| **Loss Function** | `nn.CrossEntropyLoss` | Weighted cross-entropy using class weights. |
| **Class Weighting** | Balanced inverse frequency | `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`. |
| **Max Epochs** | `50` | Maximum training epochs (`100` in `trainer_template.py`). |
| **Early Stopping Patience** | `10` epochs | Epochs without validation loss improvement before stopping (`20` in `trainer_template.py`). |
| **Early Stopping Metric** | Validation Loss | Monitored with `best_val_loss = float('inf')`. |
| **DataLoader Shuffle** | `True` (train), `False` (val, test) | Mini-batch shuffling behavior. |
| **LR Scheduler** | *None in `freq2.py`* | *(In `trainer_template.py`: `ReduceLROnPlateau(mode='min', factor=0.5, patience=10, min_lr=1e-7)`)* |

---

## 8. Post-Processing & Evaluation Hyperparameters
*(Defined in [`legacy/freq2.py#L53-L57,L381-L433`](file:///d:/neonatal/legacy/freq2.py#L53-L57,L381-L433))*

| Hyperparameter | Grid / Value | Description |
|---|---|---|
| **Moving Average (MA) Windows** | `[1, 2, 3, ..., 20]` | 20 sliding temporal smoothing window sizes (in seconds; 1 = no smoothing). |
| **Smoothing Formula** | Uniform convolution | `np.convolve(probs, np.ones(w)/w, mode='same')`. |
| **Probability Thresholds** | `0.05` to `0.95` (`step = 0.01`) | 91 decision threshold cutoffs (`np.arange(0.05, 0.96, 0.01)`). |
| **Total 2D Grid Points** | **1,820** evaluations | $20 \text{ MA windows} \times 91 \text{ thresholds}$ per trial per split. |
| **Evaluation Metrics** | Precision, Recall, F1, Accuracy, AUROC | Computed per grid point with `zero_division=0` along with full confusion matrices. |

---

## 9. Statistical Significance Testing
*(Defined in [`legacy/run_ttests.py`](file:///d:/neonatal/legacy/run_ttests.py))*

- **Test**: Welch's unequal variance $t$-test (`scipy.stats.ttest_ind(equal_var=False)`).
- **Data Split**: Computed strictly on the training set (`X_train`) to prevent data leakage.
- **Reported Metric**: $-\log_{10}(p\text{-value})$.

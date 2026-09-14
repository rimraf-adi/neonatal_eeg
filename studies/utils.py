"""
Shared utilities for all feature validation studies.

Replicates the exact preprocessing pipeline from v2.1.py:
  - Bipolar montage (18 channels, 10-20 system)
  - MNE-based EDF loading, channel normalization, bipolar re-referencing
  - Fixed-length 1-second epochs (256 Hz, no overlap)
  - 3-annotator unanimous consensus labels
  - Butterworth bandpass filtering per band
  - Welch PSD + np.polyfit spectral slope extraction

All studies import from here to avoid code duplication.
"""

import os
import gc
import csv
import json
import random
import warnings
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from collections import Counter

mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

# =============================================================================
# Constants  (mirrored from v2.1.py / freq2.py)
# =============================================================================

# All 39 annotated patient indices (from freq2.py line 45-47)
EEG_IDX = [
    1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 31, 34,
    36, 38, 39, 40, 41, 44, 47, 50, 51, 52, 62, 63, 66, 67, 69, 73, 75,
    76, 77, 78, 79
]

FS = 256  # Sampling frequency (Hz)

# Frequency bands (from v2.1.py line 197-202)
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta':  (12, 35),
}
BAND_NAMES = list(BANDS.keys())
FEATURE_TYPES = ['slope', 'intercept', 'midband']

# Bipolar pairs (from v2.1.py line 42-52)
BIPOLAR_PAIRS = [
    ('EEG Fp1-REF', 'EEG F7-REF'), ('EEG F7-REF',  'EEG T3-REF'),
    ('EEG T3-REF',  'EEG T5-REF'), ('EEG T5-REF',  'EEG O1-REF'),
    ('EEG Fp1-REF', 'EEG F3-REF'), ('EEG F3-REF',  'EEG C3-REF'),
    ('EEG C3-REF',  'EEG P3-REF'), ('EEG P3-REF',  'EEG O1-REF'),
    ('EEG Fz-REF',  'EEG Cz-REF'), ('EEG Cz-REF',  'EEG Pz-REF'),
    ('EEG Fp2-REF', 'EEG F4-REF'), ('EEG F4-REF',  'EEG C4-REF'),
    ('EEG C4-REF',  'EEG P4-REF'), ('EEG P4-REF',  'EEG O2-REF'),
    ('EEG Fp2-REF', 'EEG F8-REF'), ('EEG F8-REF',  'EEG T4-REF'),
    ('EEG T4-REF',  'EEG T6-REF'), ('EEG T6-REF',  'EEG O2-REF'),
]

# Desired channel order (from v2.1.py line 54-60)
DESIRED_ORDER = [
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fz-Cz', 'Cz-Pz',
]

# Paths
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('NEONATAL_DATA_DIR', r'd:\neonatal\dataset')
FEATURE_CACHE_DIR = os.path.join(_REPO_ROOT, 'study_results', 'feature_cache')
SPECTRAL_CACHE_DIR = os.path.join(FEATURE_CACHE_DIR, 'spectral_slope')
BASELINE_CACHE_DIR = os.path.join(FEATURE_CACHE_DIR, 'baselines')
SPLITS_PATH = os.path.join(_REPO_ROOT, 'patient_splits.json')
RESULTS_BASE = os.path.join(_REPO_ROOT, 'study_results')

ANNOTATION_FILES = [
    os.path.join(DATA_DIR, 'annotations_2017_A_fixed.csv'),
    os.path.join(DATA_DIR, 'annotations_2017_B.csv'),
    os.path.join(DATA_DIR, 'annotations_2017_C.csv'),
]


# =============================================================================
# Channel Mapping  (exact replica of v2.1.py _prepare_channel_mappings)
# =============================================================================

def _prepare_channel_mappings():
    """Build anode/cathode/ch_names arrays from bipolar pairs, reordered."""
    def normalize(ch):
        return ch.strip().upper()

    def pair_name(p):
        left = p[0].replace('EEG ', '').replace('-REF', '')
        right = p[1].replace('EEG ', '').replace('-REF', '')
        return f"{left}-{right}"

    clean_pairs = [(normalize(a), normalize(b)) for a, b in BIPOLAR_PAIRS]
    name_map = {pair_name(p): cp for p, cp in zip(BIPOLAR_PAIRS, clean_pairs)}

    reordered = [name_map[name] for name in DESIRED_ORDER if name in name_map]
    anode = [a for a, _ in reordered]
    cathode = [b for _, b in reordered]

    def pretty(ch):
        return ch.replace('EEG ', '').replace('-REF', '').capitalize()

    ch_names = [f"{pretty(a)}-{pretty(b)}" for a, b in reordered]
    return anode, cathode, ch_names

ANODE, CATHODE, CH_NAMES = _prepare_channel_mappings()


# =============================================================================
# EDF Loading  (exact replica of v2.1.py _get_array)
# =============================================================================

def load_edf_epochs(filepath):
    """Load a single EDF file, apply bipolar montage, return epoched data.

    Replicates v2.1.py _get_array exactly:
      1. read_raw_edf with preload
      2. rename channels to UPPER
      3. drop ECG/RESP channels
      4. set_bipolar_reference with reordered anode/cathode
      5. make_fixed_length_epochs (1s, no overlap)

    Args:
        filepath: Path to .edf file.

    Returns:
        np.ndarray of shape (n_epochs, n_channels, 256) or None on error.
    """
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True)
        raw.rename_channels(lambda ch: ch.upper())

        drop_candidates = ['ECG EKG', 'RESP EFFORT', 'ECG EKG-REF', 'RESP EFFORT-REF']
        raw.drop_channels([c for c in drop_candidates if c in raw.ch_names])

        raw = mne.set_bipolar_reference(
            raw, anode=ANODE, cathode=CATHODE,
            ch_name=CH_NAMES, copy=False
        )

        epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.0, verbose=False)
        data = epochs.get_data(copy=False)
        return data

    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None


# =============================================================================
# Annotation Loading  (exact replica of v2.1.py annotate + _get_patient_annotation)
# =============================================================================

def load_annotations():
    """Load the 3 annotator CSV files from DATA_DIR.

    Returns:
        list[pd.DataFrame]: Three annotation DataFrames.
    """
    dfs = []
    for path in ANNOTATION_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Annotation file not found: {path}")
        dfs.append(pd.read_csv(path))
    return dfs


def get_unanimous_labels(annotation_dfs, patient_id):
    """Get per-second labels where all 3 annotators agree.

    Replicates v2.1.py _get_patient_annotation exactly.

    Args:
        annotation_dfs: List of 3 annotation DataFrames.
        patient_id: 1-indexed patient ID.

    Returns:
        tuple: (labels, valid_mask) arrays, or (None, None) if unavailable.
    """
    col = str(patient_id)
    try:
        s1 = annotation_dfs[0][col].dropna().values
        s2 = annotation_dfs[1][col].dropna().values
        s3 = annotation_dfs[2][col].dropna().values

        min_len = min(len(s1), len(s2), len(s3))
        s1, s2, s3 = s1[:min_len], s2[:min_len], s3[:min_len]

        agreement_mask = (s1 == s2) & (s2 == s3)
        labels = s1.astype(int)

        return labels, agreement_mask
    except KeyError:
        return None, None


# =============================================================================
# Spectral Slope Feature Extraction  (exact replica of v2.1.py)
# =============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs=FS, order=4):
    """4th-order Butterworth bandpass filter.  (v2.1.py line 143-149)"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', analog=False, output='sos')
    return sosfiltfilt(sos, data)


def extract_spectral_slope_batch(signals, fs=FS, bands=None):
    """Vectorized spectral slope features for multiple channels.

    Exact replica of v2.1.py _get_spectral_slope_features_batch.

    Args:
        signals: shape (n_channels, n_samples)
        fs: Sampling frequency.
        bands: dict of {band_name: (low, high)}.

    Returns:
        np.ndarray of shape (n_channels, n_bands * 3)
        Feature order per band: [slope, intercept, midband]
    """
    if bands is None:
        bands = BANDS

    n_channels = signals.shape[0]
    n_bands = len(bands)
    n_feats_per_band = 3  # slope, intercept, midband
    all_feats = np.full((n_channels, n_bands * n_feats_per_band), np.nan)

    for b_idx, (b_name, (low, high)) in enumerate(bands.items()):
        try:
            filtered_sigs = np.array([
                butter_bandpass_filter(sig, low, high, fs, order=4) for sig in signals
            ])

            freqs, psds = welch(filtered_sigs, fs=fs, nperseg=fs, axis=-1)

            idx_band = (freqs >= low) & (freqs <= high)
            f_band = freqs[idx_band]
            p_band = psds[:, idx_band]

            log_f = f_band
            log_p = np.log10(p_band + 1e-10)

            for ch_idx in range(n_channels):
                coeffs = np.polyfit(log_f, log_p[ch_idx], 1)
                slope, intercept = coeffs[0], coeffs[1]
                mid_freq = (low + high) / 2
                midband = slope * mid_freq + intercept

                feat_start = b_idx * n_feats_per_band
                all_feats[ch_idx, feat_start:feat_start + 3] = [slope, intercept, midband]

        except Exception:
            pass  # Features remain as NaN

    return all_feats


def get_spectral_slope_header(bands=None):
    """Build the CSV column header for spectral slope features."""
    if bands is None:
        bands = BANDS
    header = ['label', 'channel']
    for bn in bands.keys():
        for fsn in FEATURE_TYPES:
            header.append(f"{bn}_{fsn}")
    return header


# =============================================================================
# Baseline Feature Extraction  (for Study 5 comparison)
# =============================================================================

def extract_band_power(signals, fs=FS, bands=None):
    """Raw band power (area under PSD per band) for each channel.

    Args:
        signals: shape (n_channels, n_samples)

    Returns:
        np.ndarray of shape (n_channels, n_bands)
    """
    if bands is None:
        bands = BANDS

    n_channels = signals.shape[0]
    n_bands = len(bands)
    features = np.full((n_channels, n_bands), np.nan)

    freqs, psds = welch(signals, fs=fs, nperseg=fs, axis=-1)

    for b_idx, (b_name, (low, high)) in enumerate(bands.items()):
        idx = (freqs >= low) & (freqs <= high)
        if idx.any():
            features[:, b_idx] = np.trapezoid(psds[:, idx], freqs[idx], axis=-1)

    return features


def extract_hjorth_parameters(signals):
    """Hjorth Activity, Mobility, Complexity for each channel.

    Args:
        signals: shape (n_channels, n_samples)

    Returns:
        np.ndarray of shape (n_channels, 3) — [activity, mobility, complexity]
    """
    n_channels = signals.shape[0]
    features = np.full((n_channels, 3), np.nan)

    for ch in range(n_channels):
        x = signals[ch]
        dx = np.diff(x)
        ddx = np.diff(dx)

        var_x = np.var(x)
        var_dx = np.var(dx)
        var_ddx = np.var(ddx)

        activity = var_x
        mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0.0
        complexity = (np.sqrt(var_ddx / var_dx) / mobility) if (var_dx > 0 and mobility > 0) else 0.0

        features[ch] = [activity, mobility, complexity]

    return features


def extract_time_domain_stats(signals):
    """Time-domain statistics for each channel: variance, RMS, line length, zero crossings.

    Args:
        signals: shape (n_channels, n_samples)

    Returns:
        np.ndarray of shape (n_channels, 4)
    """
    n_channels = signals.shape[0]
    features = np.full((n_channels, 4), np.nan)

    for ch in range(n_channels):
        x = signals[ch]
        variance = np.var(x)
        rms = np.sqrt(np.mean(x ** 2))
        line_length = np.sum(np.abs(np.diff(x)))
        zero_crossings = np.sum(np.diff(np.sign(x)) != 0)

        features[ch] = [variance, rms, line_length, zero_crossings]

    return features


def extract_emd(signals, max_imfs=3):
    """Extract energy and variance of the first max_imfs Intrinsic Mode Functions (IMFs).

    Args:
        signals: shape (n_channels, n_samples)

    Returns:
        np.ndarray of shape (n_channels, max_imfs * 2)
    """
    try:
        from PyEMD import EMD
    except ImportError:
        EMD = None

    n_channels = signals.shape[0]
    features = np.full((n_channels, max_imfs * 2), np.nan)

    if EMD is None:
        return features

    emd = EMD()
    emd.FIXE_H = 3
    emd.MAX_ITERATION = 10

    for ch in range(n_channels):
        try:
            imfs = emd.emd(np.array(signals[ch], dtype=np.float64), max_imf=max_imfs)
            f_vals = []
            for i in range(max_imfs):
                if i < imfs.shape[0]:
                    imf = imfs[i]
                    f_vals.extend([np.var(imf), np.sum(imf**2)])
                else:
                    f_vals.extend([0.0, 0.0])
            features[ch] = f_vals
        except Exception:
            pass

    return features


def extract_dwt(signals, level=4, wavelet='db4'):
    """Extract energy of approximation and detail coefficients using DWT.

    Args:
        signals: shape (n_channels, n_samples)

    Returns:
        np.ndarray of shape (n_channels, level + 1)
    """
    try:
        import pywt
    except ImportError:
        pywt = None

    n_channels = signals.shape[0]
    features = np.full((n_channels, level + 1), np.nan)

    if pywt is None:
        return features

    for ch in range(n_channels):
        try:
            coeffs = pywt.wavedec(signals[ch], wavelet, level=level)
            features[ch] = [np.sum(c**2) for c in coeffs]
        except Exception:
            pass

    return features


def extract_spectral_entropy(signals, fs=FS, bands=None):
    """Shannon entropy of normalized PSD per band for each channel.

    Args:
        signals: shape (n_channels, n_samples)

    Returns:
        np.ndarray of shape (n_channels, n_bands)
    """
    if bands is None:
        bands = BANDS

    n_channels = signals.shape[0]
    n_bands = len(bands)
    features = np.full((n_channels, n_bands), np.nan)

    freqs, psds = welch(signals, fs=fs, nperseg=fs, axis=-1)

    for b_idx, (b_name, (low, high)) in enumerate(bands.items()):
        idx = (freqs >= low) & (freqs <= high)
        if idx.any():
            for ch in range(n_channels):
                psd_band = psds[ch, idx]
                psd_norm = psd_band / (np.sum(psd_band) + 1e-10)
                psd_norm = psd_norm[psd_norm > 0]
                features[ch, b_idx] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    return features


# Baseline feature set definitions
BASELINE_FEATURE_SETS = {
    'band_power': {
        'extractor': extract_band_power,
        'columns': [f'{bn}_power' for bn in BANDS.keys()],
    },
    'hjorth': {
        'extractor': extract_hjorth_parameters,
        'columns': ['activity', 'mobility', 'complexity'],
    },
    'time_domain': {
        'extractor': extract_time_domain_stats,
        'columns': ['variance', 'rms', 'line_length', 'zero_crossings'],
    },
    'spectral_entropy': {
        'extractor': extract_spectral_entropy,
        'columns': [f'{bn}_entropy' for bn in BANDS.keys()],
    },
    'emd': {
        'extractor': extract_emd,
        'columns': ['imf1_var', 'imf1_energy', 'imf2_var', 'imf2_energy', 'imf3_var', 'imf3_energy'],
    },
    'dwt': {
        'extractor': extract_dwt,
        'columns': ['dwt_A4', 'dwt_D4', 'dwt_D3', 'dwt_D2', 'dwt_D1'],
    },
}


# =============================================================================
# Patient-level Feature CSV loading  (from cached/extracted files)
# =============================================================================

def load_patient_features(idx_list, feature_dir=None, drop_channel_col=True):
    """Load and concatenate patient feature CSVs.

    Args:
        idx_list: List of patient IDs to load.
        feature_dir: Directory containing patient_XXX.csv files.
        drop_channel_col: If True, drop the 'channel' column.

    Returns:
        pd.DataFrame with columns: label, [feature columns...]
    """
    feat_dir = feature_dir or SPECTRAL_CACHE_DIR
    frames = []
    for i in idx_list:
        filename = os.path.join(feat_dir, f'patient_{i:03d}.csv')
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            if drop_channel_col and 'channel' in df.columns:
                df = df.drop(columns=['channel'])
            df = df.replace([np.inf, -np.inf], np.nan)
            frames.append(df)
        else:
            print(f'  Warning: {filename} not found, skipping.')

    if not frames:
        raise ValueError(f"No patient feature files found in {feat_dir} for IDs: {idx_list}")

    return pd.concat(frames, ignore_index=True)


def load_patient_splits(splits_path=None):
    """Load the 10-trial patient-level cross-validation splits.

    Returns:
        list[dict]: Each dict has 'trial', 'train_idx', 'val_idx', 'test_idx'.
    """
    path = splits_path or SPLITS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Patient splits file not found: {path}\n"
            f"Run freq2.py or emd2.py first to generate patient_splits.json"
        )
    with open(path, 'r') as f:
        return json.load(f)


def get_feature_columns(df, bands=None, feature_types=None):
    """Extract feature column names matching bandxtype patterns.

    Args:
        df: DataFrame with feature columns.
        bands: List of band names (default: BAND_NAMES).
        feature_types: List of feature type names (default: FEATURE_TYPES).

    Returns:
        list[str]: Matching column names.
    """
    bands = bands or BAND_NAMES
    feature_types = feature_types or FEATURE_TYPES

    keywords = [f'{band}_{feat}' for band in bands for feat in feature_types]
    all_cols = df.columns.tolist()
    return [c for c in all_cols if any(key in c for key in keywords)]


# =============================================================================
# Preprocessing Pipeline
# =============================================================================

def preprocess_features(X_train, X_val=None, X_test=None, apply_pca=False, pca_components=10):
    """Standard preprocessing: impute -> scale -> optional PCA.

    Replicates freq2.py line 504-524.

    Returns:
        tuple: (X_train, X_val, X_test, imputer, scaler, pca_or_None)
    """
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    if X_val is not None:
        X_val = imputer.transform(X_val)
    if X_test is not None:
        X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_val is not None:
        X_val = scaler.transform(X_val)
    if X_test is not None:
        X_test = scaler.transform(X_test)

    pca = None
    if apply_pca:
        n_components = min(pca_components, X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        if X_val is not None:
            X_val = pca.transform(X_val)
        if X_test is not None:
            X_test = pca.transform(X_test)

    return X_train, X_val, X_test, imputer, scaler, pca


def prepare_trial_data(split_info, feature_dir=None, bands=None, feature_types=None,
                       apply_pca=False, pca_components=10):
    """Load and preprocess data for a single trial.

    Args:
        split_info: Dict with 'train_idx', 'val_idx', 'test_idx'.
        feature_dir: Directory containing patient CSVs.
        bands: Band names to include.
        feature_types: Feature types to include.
        apply_pca: Whether to apply PCA.
        pca_components: Max PCA components.

    Returns:
        dict with X_train, y_train, X_val, y_val, X_test, y_test,
             feature_cols, imputer, scaler, pca, train_counts, etc.
    """
    feat_dir = feature_dir or SPECTRAL_CACHE_DIR
    train_df = load_patient_features(split_info['train_idx'], feat_dir)
    val_df = load_patient_features(split_info['val_idx'], feat_dir)
    test_df = load_patient_features(split_info['test_idx'], feat_dir)

    feature_cols = get_feature_columns(train_df, bands=bands, feature_types=feature_types)

    if not feature_cols:
        raise ValueError(f"No feature columns found for bands={bands}, types={feature_types}")

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

    X_train, X_val, X_test, imputer, scaler, pca = preprocess_features(
        X_train, X_val, X_test, apply_pca=apply_pca, pca_components=pca_components
    )

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_cols': feature_cols,
        'imputer': imputer, 'scaler': scaler, 'pca': pca,
        'train_counts': Counter(y_train),
        'val_counts': Counter(y_val),
        'test_counts': Counter(y_test),
    }


# =============================================================================
# Output Helpers
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def print_header(title, width=80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title, width=60):
    """Print a formatted subsection header."""
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def check_cache(cache_dir=None):
    """Verify that the feature cache exists and has patient CSVs."""
    cdir = cache_dir or SPECTRAL_CACHE_DIR
    if not os.path.exists(cdir):
        print(f"ERROR: Feature cache not found: {cdir}")
        print(f"")
        print(f"Run the caching script first:")
        print(f"  uv run python -m studies.cache_features")
        print(f"")
        raise SystemExit(1)

    csv_count = len([f for f in os.listdir(cdir) if f.startswith('patient_') and f.endswith('.csv')])
    if csv_count == 0:
        print(f"ERROR: No patient CSV files in cache: {cdir}")
        print(f"Run: uv run python -m studies.cache_features")
        raise SystemExit(1)

    print(f"Feature cache: {cdir} ({csv_count} patients)")
    return cdir

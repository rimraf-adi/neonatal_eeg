"""
Legacy 2.0 - Self-Contained Feature Extraction Module
=====================================================
Extracts features fresh from raw EDF recordings in dataset/ across all 39 annotated patients:
1. Consensus Ground Truth: Strict 3-annotator unanimous agreement ((s1 == s2) & (s2 == s3)).
2. Spectral Slope: 4 frequency bands (delta, theta, alpha, beta) x 3 stats (slope, intercept, arithmetic midband) = 12 features/ch.
3. DWT Wavelets: Level 3 DWT (db4) -> [A3, D3, D2, D1] x 4 stats (power, std, skewness, kurtosis) = 16 features/ch.
4. EMD: 4 IMFs via C++ pyemdcpp -> [IMF1, IMF2, IMF3, IMF4] x 4 stats (power, std, skewness, kurtosis) = 16 features/ch.
5. Feature Fusion: Spectral Slope (12) + Wavelet (16) + EMD (16) = 44 features/ch.

Saves CSVs to legacy2.0/features/{spectral_slope, wavelet, emd, fusion}/patient_{pid:03d}.csv
"""

import os
import sys
import time
import csv
import pywt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, welch
import pyemdcpp

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
OUT_BASE = ROOT_DIR / "legacy2.0" / "features"

SLOPE_DIR = OUT_BASE / "spectral_slope"
WAVELET_DIR = OUT_BASE / "wavelet"
EMD_DIR = OUT_BASE / "emd"
FUSION_DIR = OUT_BASE / "fusion"

# 39 Annotated Patients
EEG_IDX = [
    1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 31, 34,
    36, 38, 39, 40, 41, 44, 47, 50, 51, 52, 62, 63, 66, 67, 69, 73, 75,
    76, 77, 78, 79
]

FS = 256
BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta':  (12.0, 35.0)
}
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta']
SLOPE_METRICS = ['slope', 'intercept', 'midband']
SLOPE_FEATURE_COLS = [f"{b}_{m}" for b in BAND_NAMES for m in SLOPE_METRICS]

SUBBANDS = ['a3', 'd3', 'd2', 'd1']
FOUR_STATS = ['power', 'std', 'skewness', 'kurtosis']
WAVELET_FEATURE_COLS = [f"{sb}_{m}" for sb in SUBBANDS for m in FOUR_STATS]

IMFS = ['imf1', 'imf2', 'imf3', 'imf4']
EMD_FEATURE_COLS = [f"{imf}_{m}" for imf in IMFS for m in FOUR_STATS]

FUSION_FEATURE_COLS = SLOPE_FEATURE_COLS + WAVELET_FEATURE_COLS + EMD_FEATURE_COLS

DESIRED_ORDER = [
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fz-Cz',  'Cz-Pz',
]

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


def _build_channel_maps():
    def normalize(ch): return ch.strip().upper()
    def pair_name(p):
        l = p[0].replace('EEG ', '').replace('-REF', '')
        r = p[1].replace('EEG ', '').replace('-REF', '')
        return f"{l}-{r}"

    clean_pairs = [(normalize(a), normalize(b)) for a, b in BIPOLAR_PAIRS]
    name_map = {pair_name(p): cp for p, cp in zip(BIPOLAR_PAIRS, clean_pairs)}
    reordered = [name_map[name] for name in DESIRED_ORDER if name in name_map]
    anode = [a for a, _ in reordered]
    cathode = [b for _, b in reordered]
    def pretty(ch): return ch.replace('EEG ', '').replace('-REF', '').capitalize()
    ch_names = [f"{pretty(a)}-{pretty(b)}" for a, b in reordered]
    return anode, cathode, ch_names

ANODE, CATHODE, CH_NAMES = _build_channel_maps()


def load_edf_epochs(filename):
    """Read EDF, map to 18 bipolar channels, epoch into 1-sec arrays."""
    try:
        import mne
        mne.set_log_level('WARNING')
        raw = mne.io.read_raw_edf(str(filename), preload=True, verbose=False)
        raw.rename_channels(lambda ch: ch.upper())
        drop = ['ECG EKG', 'RESP EFFORT', 'ECG EKG-REF', 'RESP EFFORT-REF']
        raw.drop_channels([c for c in drop if c in raw.ch_names])
        raw = mne.set_bipolar_reference(raw, anode=ANODE, cathode=CATHODE, ch_name=CH_NAMES, copy=False, verbose=False)
        epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.0, verbose=False)
        return epochs.get_data(copy=False) # (n_epochs, 18, 256)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def load_annotations():
    return [
        pd.read_csv(DATA_DIR / "annotations_2017_A_fixed.csv"),
        pd.read_csv(DATA_DIR / "annotations_2017_B.csv"),
        pd.read_csv(DATA_DIR / "annotations_2017_C.csv"),
    ]


def get_unanimous_labels(anno_dfs, patient_id):
    col = str(patient_id)
    try:
        s1 = anno_dfs[0][col].dropna().values
        s2 = anno_dfs[1][col].dropna().values
        s3 = anno_dfs[2][col].dropna().values
        min_len = min(len(s1), len(s2), len(s3))
        s1, s2, s3 = s1[:min_len], s2[:min_len], s3[:min_len]
        agreement_mask = (s1 == s2) & (s2 == s3)
        labels = s1.astype(int)
        return labels, agreement_mask
    except KeyError:
        return None, None


# Vectorized 4 statistics: [power, std, skewness, kurtosis]
def calc_four_stats_vectorized(data, eps=1e-12):
    """
    Computes power (mean square), std, Fisher-Pearson skewness, and Fisher excess kurtosis.
    data: (..., n_samples)
    returns: (..., 4)
    """
    power = np.mean(data ** 2, axis=-1)
    std = np.std(data, axis=-1)
    
    mean = np.mean(data, axis=-1, keepdims=True)
    diff = data - mean
    var = np.mean(diff ** 2, axis=-1, keepdims=True)
    std_safe = np.sqrt(np.maximum(var, eps))
    
    skew = np.mean(diff ** 3, axis=-1) / (std_safe.squeeze(-1) ** 3)
    kurt = (np.mean(diff ** 4, axis=-1) / (std_safe.squeeze(-1) ** 4)) - 3.0
    
    return np.stack([power, std, skew, kurt], axis=-1)


# 1. Spectral Slope (12 features/ch)
def extract_spectral_slope_batch(eeg_valid):
    """
    eeg_valid: (n_epochs, n_channels, 256)
    returns: (n_epochs, n_channels, 12)
    """
    n_epochs, n_ch, n_samples = eeg_valid.shape
    all_slope_feats = np.full((n_epochs, n_ch, len(BAND_NAMES) * 3), np.nan, dtype=np.float32)

    for b_idx, b_name in enumerate(BAND_NAMES):
        low, high = BANDS[b_name]
        nyq = 0.5 * FS
        sos = butter(4, [low / nyq, high / nyq], btype='band', output='sos')
        
        # Filter all signals along time axis
        filtered = sosfiltfilt(sos, eeg_valid, axis=-1)
        freqs, psds = welch(filtered, fs=FS, nperseg=FS, axis=-1)
        
        idx_band = (freqs >= low) & (freqs <= high)
        f_band = freqs[idx_band]
        p_band = psds[..., idx_band] # (n_epochs, n_ch, n_freq_bins)
        
        log_f = f_band
        log_p = np.log10(p_band + 1e-10)
        
        # Vectorized linear regression (polyfit degree 1)
        # log_p: (N, n_bins) where N = n_epochs * n_ch
        flat_p = log_p.reshape(-1, len(log_f))
        mean_x = np.mean(log_f)
        mean_y = np.mean(flat_p, axis=1, keepdims=True)
        ss_xx = np.sum((log_f - mean_x) ** 2)
        ss_xy = np.sum((log_f - mean_x) * (flat_p - mean_y), axis=1)
        
        slopes = ss_xy / (ss_xx + 1e-12)
        intercepts = mean_y.squeeze(1) - slopes * mean_x
        
        # Arithmetic mean for midband
        mid_freq = (low + high) / 2.0
        midbands = slopes * mid_freq + intercepts
        
        feat_start = b_idx * 3
        all_slope_feats[:, :, feat_start] = slopes.reshape(n_epochs, n_ch)
        all_slope_feats[:, :, feat_start + 1] = intercepts.reshape(n_epochs, n_ch)
        all_slope_feats[:, :, feat_start + 2] = midbands.reshape(n_epochs, n_ch)
        
    return all_slope_feats


# 2. DWT Wavelets (16 features/ch)
def extract_wavelet_batch(eeg_valid):
    """
    eeg_valid: (n_epochs, n_channels, 256)
    returns: (n_epochs, n_channels, 16)
    """
    # coeffs: [a3, d3, d2, d1]
    coeffs = pywt.wavedec(eeg_valid, 'db4', level=3, axis=-1)
    
    subband_feats = []
    for c in coeffs:
        # c shape: (n_epochs, n_channels, n_coeffs)
        f = calc_four_stats_vectorized(c) # (n_epochs, n_channels, 4)
        subband_feats.append(f)
        
    return np.concatenate(subband_feats, axis=-1) # (n_epochs, n_channels, 16)


# 3. EMD (16 features/ch)
def extract_emd_batch(eeg_valid, emd_instance):
    """
    eeg_valid: (n_epochs, n_channels, 256)
    returns: (n_epochs, n_channels, 16)
    """
    n_epochs, n_ch, n_samples = eeg_valid.shape
    emd_feats = np.zeros((n_epochs, n_ch, 16), dtype=np.float32)
    
    for ep in range(n_epochs):
        for ch in range(n_ch):
            sig = eeg_valid[ep, ch]
            res = emd_instance.decompose(sig, 4, 10)
            imfs = res.imfs
            
            ch_imf_feats = []
            for i in range(4):
                if i < len(imfs):
                    imf_sig = imfs[i]
                else:
                    imf_sig = np.zeros(n_samples, dtype=np.float64)
                    
                p = float(np.mean(imf_sig ** 2))
                s = float(np.std(imf_sig))
                diff = imf_sig - np.mean(imf_sig)
                var = np.mean(diff ** 2)
                std_safe = np.sqrt(max(var, 1e-12))
                skew = float(np.mean(diff ** 3) / (std_safe ** 3))
                kurt = float((np.mean(diff ** 4) / (std_safe ** 4)) - 3.0)
                ch_imf_feats.extend([p, s, skew, kurt])
                
            emd_feats[ep, ch] = ch_imf_feats
            
    return emd_feats


def main():
    print("=" * 80)
    print("LEGACY 2.0: FRESH MULTI-PARADIGM FEATURE EXTRACTION")
    print("=" * 80)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUT_BASE}")
    print(f"Paradigms: Spectral Slope (12), Wavelet (16), EMD (16), Fusion (44)")
    print(f"Consensus: Strict 3-annotator unanimous agreement")

    for d in [SLOPE_DIR, WAVELET_DIR, EMD_DIR, FUSION_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    anno_dfs = load_annotations()
    emd_inst = pyemdcpp.EMD()

    total_patients = len(EEG_IDX)
    start_all = time.time()

    for idx_num, pid in enumerate(EEG_IDX, 1):
        f_slope = SLOPE_DIR / f"patient_{pid:03d}.csv"
        f_wavelet = WAVELET_DIR / f"patient_{pid:03d}.csv"
        f_emd = EMD_DIR / f"patient_{pid:03d}.csv"
        f_fusion = FUSION_DIR / f"patient_{pid:03d}.csv"

        if f_slope.exists() and f_wavelet.exists() and f_emd.exists() and f_fusion.exists():
            print(f"[{idx_num:02d}/{total_patients:02d}] Patient {pid:03d}: Already cached. Skipping.")
            continue

        edf_path = DATA_DIR / f"eeg{pid}.edf"
        if not edf_path.exists():
            print(f"[{idx_num:02d}/{total_patients:02d}] Patient {pid:03d}: EDF NOT FOUND ({edf_path})!")
            continue

        labels, valid_mask = get_unanimous_labels(anno_dfs, pid)
        if labels is None or valid_mask is None:
            print(f"[{idx_num:02d}/{total_patients:02d}] Patient {pid:03d}: Missing annotations!")
            continue

        t0 = time.time()
        print(f"[{idx_num:02d}/{total_patients:02d}] Patient {pid:03d}: Loading EDF...", end='', flush=True)
        eeg = load_edf_epochs(edf_path)
        if eeg is None:
            print(" Failed.")
            continue

        n_epochs = min(eeg.shape[0], len(labels), len(valid_mask))
        valid_idx = np.where(valid_mask[:n_epochs])[0]
        if len(valid_idx) == 0:
            print(f" No valid unanimous epochs found.")
            continue

        eeg_valid = eeg[valid_idx] # (n_valid, 18, 256)
        labels_valid = labels[valid_idx]
        n_valid, n_ch, _ = eeg_valid.shape
        t_load = time.time() - t0

        print(f" loaded {n_valid} unanimous epochs ({t_load:.1f}s). Extracting features...", end='', flush=True)

        # 1. Spectral slope
        t1 = time.time()
        slope_feats = extract_spectral_slope_batch(eeg_valid) # (n_valid, n_ch, 12)
        # 2. Wavelet
        wavelet_feats = extract_wavelet_batch(eeg_valid) # (n_valid, n_ch, 16)
        # 3. EMD
        emd_feats = extract_emd_batch(eeg_valid, emd_inst) # (n_valid, n_ch, 16)
        # 4. Fusion
        fusion_feats = np.concatenate([slope_feats, wavelet_feats, emd_feats], axis=-1) # (n_valid, n_ch, 44)
        t_feat = time.time() - t1

        # Flatten epochs x channels into rows for each paradigm
        # Each row has: label, channel, feat_1, feat_2, ...
        channel_names = DESIRED_ORDER[:n_ch]
        
        # Prepare row labels and channels
        rep_labels = np.repeat(labels_valid, n_ch)
        rep_channels = np.tile(channel_names, n_valid)

        # Write each CSV
        targets = [
            (f_slope, ['label', 'channel'] + SLOPE_FEATURE_COLS, slope_feats.reshape(-1, 12)),
            (f_wavelet, ['label', 'channel'] + WAVELET_FEATURE_COLS, wavelet_feats.reshape(-1, 16)),
            (f_emd, ['label', 'channel'] + EMD_FEATURE_COLS, emd_feats.reshape(-1, 16)),
            (f_fusion, ['label', 'channel'] + FUSION_FEATURE_COLS, fusion_feats.reshape(-1, 44)),
        ]

        for out_csv, header, data_mat in targets:
            df = pd.DataFrame(data_mat, columns=header[2:])
            df.insert(0, 'channel', rep_channels)
            df.insert(0, 'label', rep_labels)
            df.to_csv(out_csv, index=False, float_format='%.6g')

        t_total = time.time() - t0
        print(f" Done ({t_feat:.1f}s compute, {t_total:.1f}s total). Seizures: {np.sum(labels_valid==1)}/{n_valid}")

    elapsed = time.time() - start_all
    print("=" * 80)
    print(f"Feature extraction complete in {elapsed/60:.2f} minutes!")
    print("=" * 80)


if __name__ == '__main__':
    main()

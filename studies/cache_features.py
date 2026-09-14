"""
Feature Extraction & Caching Script
====================================

THIS MUST RUN FIRST before any study scripts.

Extracts ALL feature sets from raw EDF recordings in d:\\neonatal\\dataset\\
and caches them as per-patient CSVs for fast loading by study scripts.

Feature sets extracted:
  1. Spectral Slope (slope, intercept, midband per band) — from v2.1.py
  2. Band Power (area under Welch PSD per band) — baseline
  3. Hjorth Parameters (Activity, Mobility, Complexity) — baseline
  4. Time-Domain Stats (variance, RMS, line length, zero crossings) — baseline
  5. Spectral Entropy (Shannon entropy of PSD per band) — baseline

Preprocessing pipeline (exact replica of v2.1.py):
  1. Load EDF -> rename channels UPPER -> drop ECG/RESP
  2. set_bipolar_reference (18-channel bipolar montage)
  3. make_fixed_length_epochs (1s, 256 Hz, no overlap)
  4. 3-annotator unanimous consensus labels
  5. Per-epoch, per-channel feature extraction
  6. Save to CSV per patient

Usage:
  uv run python -m studies.cache_features

Output:
  study_results/feature_cache/spectral_slope/patient_XXX.csv
  study_results/feature_cache/baselines/band_power/patient_XXX.csv
  study_results/feature_cache/baselines/hjorth/patient_XXX.csv
  study_results/feature_cache/baselines/time_domain/patient_XXX.csv
  study_results/feature_cache/baselines/spectral_entropy/patient_XXX.csv
"""

import os
import gc
import csv
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

from studies.utils import (
    EEG_IDX, FS, BANDS, DESIRED_ORDER,
    DATA_DIR, FEATURE_CACHE_DIR, SPECTRAL_CACHE_DIR, BASELINE_CACHE_DIR,
    FEATURE_TYPES, BASELINE_FEATURE_SETS,
    load_edf_epochs, load_annotations, get_unanimous_labels,
    extract_spectral_slope_batch, get_spectral_slope_header,
    ensure_dir, print_header, print_subheader,
)


def extract_patient_spectral_slope(eeg_data, labels, valid_mask, patient_id, output_dir):
    """Extract spectral slope features for one patient and write to CSV.

    Exact replica of v2.1.py preprocess() inner loop.

    Args:
        eeg_data: np.ndarray (n_epochs, n_channels, 256)
        labels: np.ndarray of per-second labels
        valid_mask: np.ndarray of boolean mask (unanimous agreement)
        patient_id: int, 1-indexed
        output_dir: directory to save CSV
    """
    out_csv = os.path.join(output_dir, f'patient_{patient_id:03d}.csv')

    if os.path.exists(out_csv):
        print(f"  [SKIP] Patient {patient_id}: already cached at {out_csv}")
        return True

    header = get_spectral_slope_header(BANDS)

    n_epochs = min(eeg_data.shape[0], len(labels), len(valid_mask))
    rows_written = 0

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep_idx in range(n_epochs):
            if not valid_mask[ep_idx]:
                continue

            label = int(labels[ep_idx])
            n_ch = min(len(DESIRED_ORDER), eeg_data.shape[1])
            epoch_signals = eeg_data[ep_idx, :n_ch]  # (n_channels, 256)

            all_ch_feats = extract_spectral_slope_batch(epoch_signals, FS, BANDS)

            for ch_idx, ch_name in enumerate(DESIRED_ORDER[:n_ch]):
                row = [label, ch_name]
                row.extend(all_ch_feats[ch_idx].tolist())
                writer.writerow(row)
                rows_written += 1

    print(f"  [OK] Patient {patient_id}: {rows_written} rows -> {out_csv}")
    return True


def extract_patient_baseline(eeg_data, labels, valid_mask, patient_id, feat_name, feat_config, output_dir):
    """Extract one baseline feature set for one patient and write to CSV.

    Args:
        eeg_data: np.ndarray (n_epochs, n_channels, 256)
        labels: np.ndarray of per-second labels
        valid_mask: np.ndarray
        patient_id: int
        feat_name: str, e.g. 'band_power'
        feat_config: dict with 'extractor' and 'columns'
        output_dir: directory to save CSV
    """
    out_csv = os.path.join(output_dir, f'patient_{patient_id:03d}.csv')

    if os.path.exists(out_csv):
        print(f"  [SKIP] Patient {patient_id}/{feat_name}: already cached")
        return True

    extractor = feat_config['extractor']
    col_names = feat_config['columns']
    header = ['label', 'channel'] + col_names

    n_epochs = min(eeg_data.shape[0], len(labels), len(valid_mask))
    rows_written = 0

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ep_idx in range(n_epochs):
            if not valid_mask[ep_idx]:
                continue

            label = int(labels[ep_idx])
            n_ch = min(len(DESIRED_ORDER), eeg_data.shape[1])
            epoch_signals = eeg_data[ep_idx, :n_ch]

            feats = extractor(epoch_signals)

            for ch_idx, ch_name in enumerate(DESIRED_ORDER[:n_ch]):
                row = [label, ch_name]
                row.extend(feats[ch_idx].tolist())
                writer.writerow(row)
                rows_written += 1

    print(f"  [OK] Patient {patient_id}/{feat_name}: {rows_written} rows")
    return True


def main():
    t_start = time.time()
    print_header("NEONATAL EEG FEATURE EXTRACTION & CACHING")

    # -- Validate data directory ------------------------------------------
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Dataset directory not found: {DATA_DIR}")
        print(f"Set NEONATAL_DATA_DIR environment variable if it's elsewhere.")
        sys.exit(1)

    edf_count = len([f for f in os.listdir(DATA_DIR) if f.endswith('.edf')])
    print(f"Dataset directory: {DATA_DIR} ({edf_count} EDF files)")

    # -- Create output directories ----------------------------------------
    ensure_dir(SPECTRAL_CACHE_DIR)
    for feat_name in BASELINE_FEATURE_SETS:
        ensure_dir(os.path.join(BASELINE_CACHE_DIR, feat_name))

    print(f"Cache root: {FEATURE_CACHE_DIR}")

    # -- Load annotations ------------------------------------------------
    print_subheader("Loading Annotations (3 annotators)")
    annotation_dfs = load_annotations()
    print(f"  Loaded {len(annotation_dfs)} annotation files")
    for i, adf in enumerate(annotation_dfs):
        print(f"    Annotator {i+1}: {adf.shape[0]} rows x {adf.shape[1]} columns")

    # -- Process each patient ---------------------------------------------
    print_subheader(f"Processing {len(EEG_IDX)} Annotated Patients")

    patients_processed = 0
    patients_skipped = 0
    patients_failed = 0

    for pid in EEG_IDX:
        print(f"\n{'-' * 40}")
        print(f"Patient {pid}")
        print(f"{'-' * 40}")

        # -- Load EDF --
        edf_path = os.path.join(DATA_DIR, f'eeg{pid}.edf')
        if not os.path.exists(edf_path):
            print(f"  [SKIP] EDF not found: {edf_path}")
            patients_skipped += 1
            continue

        eeg_data = load_edf_epochs(edf_path)
        if eeg_data is None:
            print(f"  [FAIL] Could not load EDF: {edf_path}")
            patients_failed += 1
            continue

        print(f"  EDF loaded: {eeg_data.shape[0]} epochs x {eeg_data.shape[1]} channels x {eeg_data.shape[2]} samples")

        # -- Get labels --
        labels, valid_mask = get_unanimous_labels(annotation_dfs, pid)
        if labels is None or valid_mask is None:
            print(f"  [SKIP] No annotation data for patient {pid}")
            patients_skipped += 1
            del eeg_data
            gc.collect()
            continue

        n_total = len(valid_mask)
        n_valid = np.sum(valid_mask)
        n_seizure = np.sum(labels[valid_mask] == 1)
        n_nonseizure = np.sum(labels[valid_mask] == 0)

        if n_seizure == 0:
            print(f"  [SKIP] No seizure epochs (valid={n_valid}/{n_total}, all non-seizure)")
            patients_skipped += 1
            del eeg_data
            gc.collect()
            continue

        print(f"  Annotations: {n_valid}/{n_total} valid (agreement), "
              f"seizure={n_seizure}, non-seizure={n_nonseizure}")

        # -- Extract spectral slope features --
        print(f"  Extracting spectral slope features...")
        extract_patient_spectral_slope(eeg_data, labels, valid_mask, pid, SPECTRAL_CACHE_DIR)

        # -- Extract baseline features --
        for feat_name, feat_config in BASELINE_FEATURE_SETS.items():
            print(f"  Extracting {feat_name} features...")
            feat_output_dir = os.path.join(BASELINE_CACHE_DIR, feat_name)
            extract_patient_baseline(
                eeg_data, labels, valid_mask, pid,
                feat_name, feat_config, feat_output_dir
            )

        patients_processed += 1
        del eeg_data
        gc.collect()

    # -- Summary ----------------------------------------------------------
    elapsed = time.time() - t_start
    print_header("CACHING COMPLETE")
    print(f"  Patients processed: {patients_processed}")
    print(f"  Patients skipped:   {patients_skipped}")
    print(f"  Patients failed:    {patients_failed}")
    print(f"  Time elapsed:       {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"")
    print(f"  Cache locations:")
    print(f"    Spectral slope: {SPECTRAL_CACHE_DIR}")
    for feat_name in BASELINE_FEATURE_SETS:
        print(f"    {feat_name:18s}: {os.path.join(BASELINE_CACHE_DIR, feat_name)}")
    print(f"")
    print(f"  Next steps:")
    print(f"    uv run python -m studies.study_01_classifier_ladder")
    print(f"    uv run python -m studies.study_02_feature_ablation")
    print(f"    uv run --with umap-learn python -m studies.study_03_umap_tsne")
    print(f"    uv run python -m studies.study_04_fisher_ratio")
    print(f"    uv run python -m studies.study_05_baseline_comparison")
    print(f"    uv run python -m studies.study_06_temporal_trajectory")
    print(f"    uv run python -m studies.study_07_cohens_d")
    print(f"    uv run python -m studies.study_08_sampler_ablation")


if __name__ == '__main__':
    main()

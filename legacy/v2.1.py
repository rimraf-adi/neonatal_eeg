import numpy as np
import pandas as pd
import mne
import os
import csv
import warnings
import antropy as ant
import gc
import sys
from pathlib import Path
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
from scipy.signal import stft, welch, butter, sosfiltfilt
# sklearn no longer needed - using np.polyfit instead

mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

class LogColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log(msg, level='info'):
    if level == 'info': color = LogColors.OKBLUE
    elif level == 'success': color = LogColors.OKGREEN
    elif level == 'warn': color = LogColors.WARNING
    elif level == 'error': color = LogColors.FAIL
    else: color = LogColors.ENDC
    print(f"{color}{msg}{LogColors.ENDC}")

class V_One:
    def __init__(self):
        self.annotations = []
        self.eeg_set = []
        self.data_root = './data' 
        self.output_root = './freq_features_updated'
        
        self.bipolar_pairs = [
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
        
        self.desired_order = [
            'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
            'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
            'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
            'Fz-Cz', 'Cz-Pz',
        ]
        
        self._prepare_channel_mappings()

    def _prepare_channel_mappings(self):
        def normalize(ch): return ch.strip().upper()
        def pair_name(p): 
            l = p[0].replace('EEG ', '').replace('-REF', '')
            r = p[1].replace('EEG ', '').replace('-REF', '')
            return f"{l}-{r}"

        clean_pairs = [(normalize(a), normalize(b)) for a, b in self.bipolar_pairs]
        name_map = {pair_name(p): cp for p, cp in zip(self.bipolar_pairs, clean_pairs)}
        
        self.reordered_pairs = [name_map[name] for name in self.desired_order if name in name_map]
        self.anode = [a for a, _ in self.reordered_pairs]
        self.cathode = [b for _, b in self.reordered_pairs]
        
        def pretty(ch): return ch.replace('EEG ', '').replace('-REF', '').capitalize()
        self.ch_names = [f"{pretty(a)}-{pretty(b)}" for a, b in self.reordered_pairs]

    def _get_array(self, filename: str):
        try:
            raw = mne.io.read_raw_edf(filename, preload=True)
            raw.rename_channels(lambda ch: ch.upper())
            
            drop = ['ECG EKG', 'RESP EFFORT', 'ECG EKG-REF', 'RESP EFFORT-REF']
            raw.drop_channels([c for c in drop if c in raw.ch_names])
            raw = mne.set_bipolar_reference(raw, anode=self.anode, cathode=self.cathode, ch_name=self.ch_names, copy=False)
            
            epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.0, verbose=False)
            data = epochs.get_data(copy=False)
            
            return data
            
        except Exception as e:
            log(f"Error reading {filename}: {e}", 'error')
            return None

    def load_data(self):
        log("🔵 Loading EEG data...", 'info')
        self.eeg_set = []
        for i in range(1, 80):
            fp = os.path.join(self.data_root, f'eeg{i}.edf')
            if os.path.exists(fp):
                log(f"Loading {fp}", 'info')
                self.eeg_set.append(self._get_array(fp))
            else:
                self.eeg_set.append(None)
            gc.collect()

    def annotate(self):
        log("🔵 Loading annotations...", 'info')
        try:
            a = pd.read_csv('./annotations_2017_A_fixed.csv')
            b = pd.read_csv('./annotations_2017_B.csv')
            c = pd.read_csv('./annotations_2017_C.csv')
            self.annotation_dfs = [a, b, c]
        except FileNotFoundError:
            log("❌ Annotation files not found.", 'error')
            self.annotation_dfs = []

    def _get_patient_annotation(self, p_idx):
        """Returns (labels, valid_mask) where valid_mask indicates unanimous agreement."""
        if not self.annotation_dfs: return None, None
        col = str(p_idx + 1)
        try:
            s1 = self.annotation_dfs[0][col].dropna().values
            s2 = self.annotation_dfs[1][col].dropna().values
            s3 = self.annotation_dfs[2][col].dropna().values
            
            # Find minimum length across all annotators
            min_len = min(len(s1), len(s2), len(s3))
            s1, s2, s3 = s1[:min_len], s2[:min_len], s3[:min_len]
            
            # Only keep samples where all 3 annotators agree
            agreement_mask = (s1 == s2) & (s2 == s3)
            labels = s1  # All are same where mask is True
            
            return labels.astype(int), agreement_mask
        except KeyError:
            return None, None

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='band', analog=False, output='sos')
        y = sosfiltfilt(sos, data)
        return y

    def _get_spectral_slope_features_batch(self, signals, fs, bands):
        """
        Vectorized spectral slope features for multiple signals.
        signals: shape (n_channels, n_samples)
        Returns: shape (n_channels, n_features)
        """
        n_channels = signals.shape[0]
        n_bands = len(bands)
        n_feats_per_band = 3  # slope, intercept, midband
        all_feats = np.full((n_channels, n_bands * n_feats_per_band), np.nan)
        
        for b_idx, (b_name, (low, high)) in enumerate(bands.items()):
            try:
                # Vectorized bandpass filter across all channels
                filtered_sigs = np.array([self._butter_bandpass_filter(sig, low, high, fs, order=4) for sig in signals])
                
                # Vectorized welch - compute for all channels at once
                freqs, psds = welch(filtered_sigs, fs=fs, nperseg=fs, axis=-1)
                
                idx_band = (freqs >= low) & (freqs <= high)
                f_band = freqs[idx_band]
                p_band = psds[:, idx_band]  # (n_channels, n_freq_bins)
                
                log_f = f_band
                log_p = np.log10(p_band + 1e-10)
                
                # Vectorized linear regression using np.polyfit
                # polyfit returns [slope, intercept] for degree=1
                for ch_idx in range(n_channels):
                    coeffs = np.polyfit(log_f, log_p[ch_idx], 1)
                    slope, intercept = coeffs[0], coeffs[1]
                    mid_freq = (low + high) / 2
                    midband = slope * mid_freq + intercept
                    
                    feat_start = b_idx * n_feats_per_band
                    all_feats[ch_idx, feat_start:feat_start+3] = [slope, intercept, midband]
                    
            except Exception as e:
                pass  # Features remain as NaN
                
        return all_feats

    def preprocess(self):
        if not self.eeg_set: self.load_data()
        log("🚀 Starting Feature Extraction ( Freq Domain)...", 'success')
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 35),
            'gamma': (35, 100)
        }
        feat_subnames = ['slope', 'intercept', 'midband']
        
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
        
        for i, eeg_data in enumerate(self.eeg_set):
            if eeg_data is None: continue
            pid = i + 1
            
            anno, valid_mask = self._get_patient_annotation(i)
            if anno is None or valid_mask is None:
                log(f"⚠️  Patient {pid}: No annotations.", 'warn')
                continue
            
            # Count how many samples were discarded due to disagreement
            n_total = len(valid_mask)
            n_valid = np.sum(valid_mask)
            n_discarded = n_total - n_valid
            if n_discarded > 0:
                log(f"ℹ️  Patient {pid}: Discarded {n_discarded}/{n_total} samples (annotator disagreement)", 'info')
                
            has_seizure = 1 in anno[valid_mask]
            if not has_seizure:
                log(f"⏭️  Patient {pid}: No seizure activity. Skipping.", 'warn')
                continue
            
            out_csv = Path(self.output_root) / f"patient_{pid:03d}.csv"
            
            # Skip if file already exists
            if out_csv.exists():
                log(f"⏭️  Patient {pid}: Output file exists. Skipping.", 'info')
                continue
            
            log(f"⚡ Processing Patient {pid} ({len(anno)} epochs)...", 'info')
            
            with open(out_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['label', 'channel']
                for bn in bands.keys(): 
                    for fsn in feat_subnames:
                        header.append(f"{bn}_{fsn}")
                writer.writerow(header)
                
                n_epochs = min(eeg_data.shape[0], len(anno), len(valid_mask))
                
                for ep_idx in range(n_epochs):
                    # Skip samples where annotators disagreed
                    if not valid_mask[ep_idx]:
                        continue
                        
                    label = int(anno[ep_idx])
                    
                    # Get all channels for this epoch and process in batch
                    n_ch = min(len(self.desired_order), eeg_data.shape[1])
                    epoch_signals = eeg_data[ep_idx, :n_ch]  # (n_channels, n_samples)
                    
                    # Batch compute features for all channels
                    all_ch_feats = self._get_spectral_slope_features_batch(epoch_signals, 256, bands)
                    
                    for ch_idx, ch_name in enumerate(self.desired_order[:n_ch]):
                        row = [label, ch_name]
                        row.extend(all_ch_feats[ch_idx].tolist())
                        writer.writerow(row)
            
            log(f"✅ Patient {pid} features saved.", 'success')
            del anno
            gc.collect()

if __name__ == '__main__':
    v = V_One()
    v.annotate()
    v.preprocess()
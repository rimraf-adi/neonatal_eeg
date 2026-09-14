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
from sklearn.linear_model import LinearRegression

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
        self.output_root = './freq_features'
        
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
        if not self.annotation_dfs: return None
        col = str(p_idx + 1)
        try:
            s1 = self.annotation_dfs[0][col].dropna()
            s2 = self.annotation_dfs[1][col].dropna()
            s3 = self.annotation_dfs[2][col].dropna()
            total = s1 + s2 + s3
            merged = (total >= 2).astype(int)
            return merged.values
        except KeyError:
            return None

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='band', analog=False, output='sos')
        y = sosfiltfilt(sos, data)
        return y

    def _get_spectral_slope_features(self, sig, fs, bands):
        flat_feats = []
        
        for b_name, (low, high) in bands.items():
            try:
                filtered_sig = self._butter_bandpass_filter(sig, low, high, fs, order=4)
                freqs, psd = welch(filtered_sig, fs=fs, nperseg=fs)
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                f_band = freqs[idx_band]
                p_band = psd[idx_band]
                log_f = f_band
                log_p = np.log10(p_band + 1e-10)
                
                X = log_f.reshape(-1, 1)
                y = log_p
                model = LinearRegression()
                model.fit(X, y)
                
                slope = model.coef_[0]
                intercept = model.intercept_
                mid_freq = (low + high) / 2
                midband = slope * mid_freq + intercept
                flat_feats.extend([slope, intercept, midband])
                
            except Exception as e:
                flat_feats.extend([np.nan, np.nan, np.nan])
                
        return flat_feats

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
            
            anno = self._get_patient_annotation(i)
            if anno is None:
                log(f"⚠️  Patient {pid}: No annotations.", 'warn')
                continue
                
            has_seizure = 1 in anno
            if not has_seizure:
                log(f"⏭️  Patient {pid}: No seizure activity. Skipping.", 'warn')
                continue
            
            out_csv = Path(self.output_root) / f"patient_{pid:03d}.csv"
            
            log(f"⚡ Processing Patient {pid} ({len(anno)} epochs)...", 'info')
            
            with open(out_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['label', 'channel']
                for bn in bands.keys(): 
                    for fsn in feat_subnames:
                        header.append(f"{bn}_{fsn}")
                writer.writerow(header)
                
                n_epochs = min(eeg_data.shape[0], len(anno))
                for ep_idx in range(n_epochs):
                    label = int(anno[ep_idx])
                    for ch_idx, ch_name in enumerate(self.desired_order):
                        if ch_idx >= eeg_data.shape[1]: break
                        sig = eeg_data[ep_idx, ch_idx]
                        row = [label, ch_name]
                        
                        feats = self._get_spectral_slope_features(sig, 256, bands)
                        row.extend(feats)
                        
                        writer.writerow(row)
            
            log(f"✅ Patient {pid} features saved.", 'success')
            del anno
            gc.collect()

if __name__ == '__main__':
    v = V_One()
    v.annotate()
    v.preprocess()
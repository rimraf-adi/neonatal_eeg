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
from scipy.signal import stft

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
        self.output_root = './stft_features'
        
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

    def _calc_freq_features(self, mag_spec):
        if len(mag_spec) == 0:
            return {
                'energy': 0.0, 'wiener_entropy': 0.0, 'skewness': 0.0,
                'kurtosis': 0.0, 'psd_rms': 0.0, 'std': 0.0
            }
        
        EPS = 1e-12
        pass_pow = mag_spec ** 2
        energy = np.sum(pass_pow)
        mean_pow = np.mean(pass_pow)
        if mean_pow < EPS:
            wiener = 0.0
        else:
            wiener = np.exp(np.mean(np.log(pass_pow + EPS))) / mean_pow
            
        skew_val = scipy_skew(mag_spec)
        kurt_val = scipy_kurtosis(mag_spec)
        std_val = np.std(mag_spec)
        psd_rms = np.sqrt(mean_pow)
        
        return {
            'energy': float(energy),
            'wiener_entropy': float(wiener),
            'skewness': float(skew_val),
            'kurtosis': float(kurt_val),
            'psd_rms': float(psd_rms),
            'std': float(std_val)
        }

    def preprocess(self):
        if not self.eeg_set: self.load_data()
        log("🚀 Starting Feature Extraction (STFT Freq Domain)...", 'success')
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        band_names = list(bands.keys())
        feats = ['energy', 'wiener_entropy', 'skewness', 'kurtosis', 'psd_rms', 'std']
        
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
                for bn in band_names: header.extend([f"{bn}_{fn}" for fn in feats])
                writer.writerow(header)
                
                n_epochs = min(eeg_data.shape[0], len(anno))
                for ep_idx in range(n_epochs):
                    label = int(anno[ep_idx])
                    for ch_idx, ch_name in enumerate(self.desired_order):
                        if ch_idx >= eeg_data.shape[1]: break
                        sig = eeg_data[ep_idx, ch_idx]
                        row = [label, ch_name]
                        
                        try:
                            f_vals_stft, t_vals, Zxx = stft(sig, fs=256, nperseg=256)
                            
                            for b_name, (low, high) in bands.items():
                                mask = (f_vals_stft >= low) & (f_vals_stft <= high)
                                band_z = Zxx[mask, :]
                                band_mag = np.abs(band_z).flatten()
                                
                                f_res = self._calc_freq_features(band_mag)
                                row.extend([f_res[fn] for fn in feats])
                                
                        except Exception as e:
                            log(f"STFT Error P{pid}Ep{ep_idx}: {e}", 'error')
                            row.extend([np.nan] * (len(band_names) * len(feats)))
                        
                        writer.writerow(row)
            
            log(f"✅ Patient {pid} features saved.", 'success')
            del anno
            gc.collect()

if __name__ == '__main__':
    v = V_One()
    v.annotate()
    v.preprocess()
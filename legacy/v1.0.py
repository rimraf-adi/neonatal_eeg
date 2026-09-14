import numpy as np
import pandas as pd
import mne
import os
import csv
import warnings
import pywt
import antropy as ant
import gc
import sys
from pathlib import Path
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis

# --- Configuration & Logging ---
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
        self.output_root = './features_output'
        
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

    # --- Manual Algorithm Implementations ---

    def _psr(self, x, m, tao):
        """Phase Space Reconstruction (Manual)."""
        N = len(x)
        M = N - (m - 1) * tao
        if M <= 0: return None
        
        # Create Y matrix: M x m
        Y = np.zeros((M, m))
        for i in range(m):
            Y[:, i] = x[i*tao : i*tao + M]
        return Y

    def _lyap_r(self, x, m=5, tao=1, fs=256):
        """Largest Lyapunov Exponent (Rosenstein) - Manual Implementation."""
        meanperiod = 1 
        maxiter = min(50, len(x)//10)
        
        Y = self._psr(x, m, tao)
        if Y is None: return np.nan
        
        M = Y.shape[0]
        if M < 10: return np.nan
        
        # optimized distance matrix with broadcasting
        Y2 = np.sum(Y**2, axis=1)
        D2 = Y2[:, None] + Y2[None, :] - 2 * np.dot(Y, Y.T)
        
        # Mask diagonal band |i-j| <= meanperiod
        idx = np.arange(M)
        mask = np.abs(idx[:, None] - idx[None, :]) <= meanperiod
        D2[mask] = np.inf
        
        nearpos = np.argmin(D2, axis=1)
        
        # Divergence Tracking
        d = np.zeros(maxiter)
        
        for k in range(1, maxiter):
            maxind = M - k
            valid_j = np.where((idx < maxind) & (nearpos < maxind))[0]
            if len(valid_j) == 0: continue
            
            j_k = valid_j + k
            near_k = nearpos[valid_j] + k
            
            delta_Y = Y[j_k] - Y[near_k]
            dist_sq = np.sum(delta_Y**2, axis=1)
            
            valid_dist = dist_sq > 0
            if np.sum(valid_dist) > 0:
                logs = 0.5 * np.log(dist_sq[valid_dist])
                d[k] = np.mean(logs)
        
        # Fit Line
        x_fit = np.arange(1, maxiter)
        y_fit = d[1:]
        
        valid_fit = y_fit != 0
        if np.sum(valid_fit) < 2: return np.nan
        
        slope = np.polyfit(x_fit[valid_fit], y_fit[valid_fit], 1)[0]
        return slope * fs

    def _hurst_rs(self, x):
        """Hurst Exponent (R/S Analysis) - Manual."""
        N = len(x)
        if N < 20: return np.nan
        
        min_chunk, max_chunk = 8, N
        chunk_sizes = []
        c = min_chunk
        while c <= max_chunk:
            chunk_sizes.append(c)
            c *= 2
            
        rs_values = []
        for n in chunk_sizes:
            num_chunks = N // n
            if num_chunks < 1: continue
            
            rs_chunk = []
            for i in range(num_chunks):
                chunk = x[i*n : (i+1)*n]
                mean = np.mean(chunk)
                y = chunk - mean
                z = np.cumsum(y)
                R = np.max(z) - np.min(z)
                S = np.std(chunk)
                if S == 0: continue
                rs_chunk.append(R/S)
            
            if rs_chunk:
                rs_values.append(np.mean(np.array(rs_chunk)))
        
        if len(rs_values) < 2: return np.nan
        
        log_n = np.log10(np.array(chunk_sizes[:len(rs_values)]))
        log_rs = np.log10(np.array(rs_values))
        
        coeffs = np.polyfit(log_n, log_rs, 1)
        return coeffs[0]

    def _calc_features(self, sig, fs=256):
        EPS = 1e-12
        
        energy = np.sum(sig ** 2)
        params = sig ** 2
        mean_params = np.mean(params)
        wiener = np.exp(np.mean(np.log(params + EPS))) / mean_params if mean_params >= EPS else 0.0

        skew_val = scipy_skew(sig)
        kurt_val = scipy_kurtosis(sig)

        try:
            F = np.fft.fft(sig)
            psd = (np.abs(F)**2) / len(sig)
            psd_rms = np.sqrt(np.mean(psd))
        except: psd_rms = 0.0
            
        std_val = np.std(sig)
        
        return {
            'energy': float(energy),
            'wiener_entropy': float(wiener),
            'skewness': float(skew_val),
            'kurtosis': float(kurt_val),
            'psd_rms': float(psd_rms),
            'std': float(std_val)
        }

    def _katz_fd(self, sig):
        """Katz Fractal Dimension."""
        n = sig.shape[-1]
        x_diff = np.abs(np.diff(sig))
        L = np.sum(x_diff)
        d = np.max(np.abs(sig - sig[0]))
        if L == 0: return 0.0
        return np.log10(n) / (np.log10(d / L) + np.log10(n))

    def _extract_raw(self, sig):
        # Features
        fd = float(self._katz_fd(sig))
        hurst = float(self._hurst_rs(sig))
        lyap = float(self._lyap_r(sig))
        
        return {'fractal_dim': fd, 'hurst': hurst, 'lyapunov': lyap}

    def preprocess(self):
        if not self.eeg_set: self.load_data()
        log("🚀 Starting Feature Extraction...", 'success')
        
        bands = ['a4', 'd4', 'd3', 'd2', 'd1']
        feats = ['energy', 'wiener_entropy', 'skewness', 'kurtosis', 'psd_rms', 'std']
        raw_names = ['fractal_dim', 'hurst', 'lyapunov']
        
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
                
            p_dir = Path(self.output_root) / f"patient_{pid:03d}"
            p_dir.mkdir(parents=True, exist_ok=True)
            out_csv = p_dir / "features.csv"
            
            log(f"⚡ Processing Patient {pid} ({len(anno)} epochs)...", 'info')
            
            with open(out_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['label', 'channel']
                for b in bands: header.extend([f"{b}_{fn}" for fn in feats])
                header.extend([f"raw_{rn}" for rn in raw_names])
                writer.writerow(header)
                
                n_epochs = min(eeg_data.shape[0], len(anno))
                for ep_idx in range(n_epochs):
                    label = int(anno[ep_idx])
                    for ch_idx, ch_name in enumerate(self.desired_order):
                        if ch_idx >= eeg_data.shape[1]: break
                        sig = eeg_data[ep_idx, ch_idx]
                        row = [label, ch_name]
                        
                        try:
                            # DWT
                            coeffs = pywt.wavedec(sig, 'db4', level=4)
                            for c in coeffs:
                                f_vals = self._calc_features(c)
                                row.extend([f_vals[fn] for fn in feats])
                        except Exception as e:
                            log(f"DWT Error P{pid}Ep{ep_idx}: {e}", 'error')
                            row.extend([np.nan] * (len(bands) * len(feats)))
                        
                        # Raw
                        r_vals = self._extract_raw(sig)
                        row.extend([r_vals[rn] for rn in raw_names])
                        
                        writer.writerow(row)
            
            log(f"✅ Patient {pid} features saved.", 'success')
            del anno
            gc.collect()

if __name__ == '__main__':
    v = V_One()
    v.annotate()
    v.preprocess()
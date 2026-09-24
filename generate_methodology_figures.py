#!/usr/bin/env python3
"""
================================================================================
Comprehensive Methodology Architectural & Parameter Extraction Diagrams
================================================================================
Generates 3 publication-grade methodological figures for Springer Nature submission:
  1. fig_methodology_spectral_slope.(png/pdf):
     Raw EEG Epoch -> Welch PSD -> Log Transform -> OLS Line Fit -> Parameters Highlighted
     (Slope m_b, Intercept c_b, Midband Power P_mid,b across Delta, Theta, Alpha, Beta)
  2. fig_methodology_dwt.(png/pdf):
     Raw EEG Epoch -> Daubechies 4 Filter Bank Tree -> Multi-Resolution Sub-bands (A3, D3, D2, D1)
     -> Extraction of 6 Descriptors (Energy, Std, Skewness, Kurtosis, Wiener Entropy, PSD RMS)
  3. fig_methodology_emd.(png/pdf):
     Raw EEG Epoch -> Iterative Sifting (Extrema & Cubic Spline Envelopes) -> IMFs (1 to 4) + Residue
     -> Extraction of 6 Descriptors per IMF Mode
================================================================================
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
from scipy.interpolate import CubicSpline
import pywt
import PyEMD
import mne

# ------------------------------------------------------------------------------
# 0. Global Setup & Publication Theme
# ------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = REPO_ROOT / "public"
PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Springer Nature / IEEE Typography
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 10,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11.5,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#D0D0D0',
    'axes.linewidth': 1.0,
    'axes.edgecolor': '#333333',
    'axes.grid': True,
    'grid.color': '#EAEAEA',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

PALETTE = {
    'navy': '#1B365D',
    'crimson': '#D9381E',
    'delta': '#2B5C8F',
    'theta': '#2E8B57',
    'alpha': '#D35400',
    'beta': '#8E44AD',
    'dark': '#2C3E50',
    'light_bg': '#F8F9FA',
    'box_border': '#BDC3C7',
    'gold': '#F39C12',
    'teal': '#16A085',
    'gray': '#7F8C8D',
}

def save_dual(fig, base_name):
    """Saves figure in both public/ and paper/figures/ in PNG and PDF formats."""
    for d in [PUBLIC_DIR, PAPER_FIG_DIR]:
        png_p = d / f"{base_name}.png"
        pdf_p = d / f"{base_name}.pdf"
        fig.savefig(png_p, dpi=300, format='png')
        fig.savefig(pdf_p, dpi=300, format='pdf')
    print(f"  [OK] Saved {base_name}.png and {base_name}.pdf to public/ and paper/figures/")

# ------------------------------------------------------------------------------
# 1. Load Real Patient Data
# ------------------------------------------------------------------------------
def load_real_epochs():
    """Loads representative neonatal seizure & background epochs from eeg1.edf."""
    edf_path = REPO_ROOT / "dataset" / "eeg1.edf"
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.rename_channels(lambda ch: ch.upper())

    # Anterior-posterior bipolar derivation (C3-P3)
    d_c3 = raw.get_data(picks=['EEG C3-REF'])[0]
    d_p3 = raw.get_data(picks=['EEG P3-REF'])[0]
    bipolar_uv = (d_c3 - d_p3) * 1e6  # Convert V to uV
    fs = int(raw.info['sfreq'])

    # Epoch 104 is a unanimous seizure epoch; Epoch 30 is interictal background
    seizure_epoch = bipolar_uv[104 * fs : 105 * fs]
    background_epoch = bipolar_uv[30 * fs : 31 * fs]
    t = np.linspace(0, 1.0, fs, endpoint=False)

    return t, seizure_epoch, background_epoch, fs

# ------------------------------------------------------------------------------
# FIGURE 1: SPECTRAL SLOPE (LOG-PSD) METHODOLOGY
# ------------------------------------------------------------------------------
def generate_spectral_slope_methodology(t, sig, fs):
    """
    Step-by-step visual workflow of the proposed Log-PSD Spectral Slope Biomarker:
      Step 1: Raw EEG 1-sec epoch
      Step 2: Welch PSD estimation (Hann window, fs=256)
      Step 3: Decibel log-transformation y_k = 10 log10(P(f_k))
      Step 4: Multi-band OLS linear fitting (Delta, Theta, Alpha, Beta)
      Step 5: Geometric feature extraction (Slope m_b, Intercept c_b, Midband Power P_mid,b)
    """
    print("Generating Figure 1: Spectral Slope (Log-PSD) Parameter Extraction Methodology...")
    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.2, 1.3], hspace=0.38, wspace=0.28)

    # ---------------- Panel A: Raw 1-second Epoch ----------------
    ax_raw = fig.add_subplot(gs[0, :2])
    ax_raw.plot(t * 1000, sig, color=PALETTE['crimson'], lw=1.6, label='Bipolar EEG (C3–P3)')
    ax_raw.set_title(r"$\mathbf{Step\ 1:\ Raw\ 1-Second\ EEG\ Epoch\ Segment}\ (f_s = 256\ \mathrm{Hz},\ N = 256)$", pad=8)
    ax_raw.set_xlabel("Time (milliseconds)")
    ax_raw.set_ylabel(r"Amplitude ($\mu\mathrm{V}$)")
    ax_raw.set_xlim(0, 1000)
    ax_raw.set_ylim(-11, 6.5)
    ax_raw.legend(loc='upper right', framealpha=0.9)
    ax_raw.annotate("Paroxysmal Rhythmic Seizure Discharge", xy=(550, np.max(sig)*0.65), xytext=(350, 4.3),
                    arrowprops=dict(facecolor=PALETTE['dark'], arrowstyle='->', lw=1.2),
                    fontsize=9, fontweight='bold', color=PALETTE['dark'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD', edgecolor='#FFEEBA', alpha=0.9))

    # ---------------- Panel B: Welch Modified Periodogram ----------------
    freqs, psd = welch(sig, fs=fs, nperseg=fs, window='hann')
    mask_valid = (freqs >= 0.5) & (freqs <= 30.0)
    freqs_valid = freqs[mask_valid]
    psd_valid = psd[mask_valid]

    ax_welch = fig.add_subplot(gs[0, 2:])
    ax_welch.plot(freqs_valid, psd_valid, color=PALETTE['navy'], lw=1.8, marker='o', markersize=3, label="Welch PSD")
    ax_welch.set_title(r"$\mathbf{Step\ 2:\ Welch\ Modified\ Periodogram}\ P(f)\ \mathrm{(Hann\ Window,\ N_{fft}=256)}$", pad=8)
    ax_welch.set_xlabel("Frequency (Hz)")
    ax_welch.set_ylabel(r"PSD ($\mu\mathrm{V}^2/\mathrm{Hz}$)")
    ax_welch.set_xlim(0, 31)

    bands = [
        ('delta', r'$\delta$ (0.5–4 Hz)', 0.5, 4.0, PALETTE['delta']),
        ('theta', r'$\theta$ (4–8 Hz)', 4.0, 8.0, PALETTE['theta']),
        ('alpha', r'$\alpha$ (8–13 Hz)', 8.0, 13.0, PALETTE['alpha']),
        ('beta',  r'$\beta$ (13–30 Hz)', 13.0, 30.0, PALETTE['beta']),
    ]
    for b_key, b_name, flow, fhigh, bcol in bands:
        ax_welch.axvspan(flow, fhigh, color=bcol, alpha=0.15)
        ax_welch.text((flow + fhigh)/2, np.max(psd_valid)*0.88, b_name.split()[0],
                      horizontalalignment='center', color=bcol, fontweight='bold', fontsize=9.5)
    ax_welch.legend(loc='upper right')

    # ---------------- Panel C: Decibel Log-Transformation ----------------
    ax_log = fig.add_subplot(gs[1, :2])
    db_psd = 10 * np.log10(psd_valid + 1e-10)
    ax_log.plot(freqs_valid, db_psd, color='#34495E', lw=1.8, marker='s', markersize=3, label=r"$y_k = 10\log_{10}(P(f_k))$")
    ax_log.set_title(r"$\mathbf{Step\ 3:\ Decibel\ (Logarithmic)\ Transformation}\ y_k = 10\log_{10}(P(f_k))$", pad=8)
    ax_log.set_xlabel("Frequency (Hz, Linear Scale)")
    ax_log.set_ylabel(r"Log-Power Density (dB/Hz)")
    ax_log.set_xlim(0, 31)

    for b_key, b_name, flow, fhigh, bcol in bands:
        ax_log.axvspan(flow, fhigh, color=bcol, alpha=0.12)
        ax_log.text((flow + fhigh)/2, np.min(db_psd) + 2.5, b_name,
                    horizontalalignment='center', color=bcol, fontweight='bold', fontsize=8.5, rotation=0)
    ax_log.legend(loc='upper right')

    # ---------------- Panel D: OLS Linear Regression Fits Across Bands ----------------
    ax_fit = fig.add_subplot(gs[1, 2:])
    ax_fit.plot(freqs_valid, db_psd, color='#95A5A6', lw=1.2, ls=':', label='Decibel Spectrum')
    ax_fit.set_title(r"$\mathbf{Step\ 4:\ Multi-Band\ Ordinary\ Least\ Squares\ (OLS)\ Fitting}\ y_k = m_b f_k + c_b$", pad=8)
    ax_fit.set_xlabel("Frequency (Hz)")
    ax_fit.set_ylabel(r"Log-Power Density (dB/Hz)")
    ax_fit.set_xlim(0, 31)

    fit_results = {}
    for b_key, b_name, flow, fhigh, bcol in bands:
        b_mask = (freqs_valid >= flow) & (freqs_valid <= fhigh)
        bf = freqs_valid[b_mask]
        bp = db_psd[b_mask]

        # OLS fit
        coeffs = np.polyfit(bf, bp, 1)
        slope, intercept = coeffs[0], coeffs[1]
        f_mid = (flow + fhigh) / 2.0
        p_mid = slope * f_mid + intercept
        fit_results[b_key] = {'slope': slope, 'intercept': intercept, 'f_mid': f_mid, 'p_mid': p_mid}

        # Plot regression line
        ax_fit.plot(bf, slope * bf + intercept, color=bcol, lw=2.4, label=f"{b_name} Fit")
        ax_fit.scatter([f_mid], [p_mid], color=bcol, s=45, edgecolors='black', zorder=5)

    ax_fit.legend(loc='lower left', fontsize=8.5, ncol=2)

    # ---------------- Panel E: Deep Dive / Highlighting the 3 Parameters (Theta Band) ----------------
    ax_zoom = fig.add_subplot(gs[2, :3])
    tb_flow, tb_fhigh = 4.0, 8.0
    tb_mask = (freqs_valid >= tb_flow) & (freqs_valid <= tb_fhigh)
    tb_f = freqs_valid[tb_mask]
    tb_p = db_psd[tb_mask]
    tb_res = fit_results['theta']

    # Extend fit for intercept visualization
    ext_f = np.linspace(0.0, 9.0, 200)
    ext_line = tb_res['slope'] * ext_f + tb_res['intercept']

    ax_zoom.plot(ext_f, ext_line, color=PALETTE['theta'], lw=2.5, label=r"Theta OLS Line: $\hat{y} = m_\theta f + c_\theta$")
    ax_zoom.scatter(tb_f, tb_p, color=PALETTE['dark'], s=55, zorder=4, label='Empirical Log-PSD Bins')
    ax_zoom.axvspan(tb_flow, tb_fhigh, color=PALETTE['theta'], alpha=0.10, label=r'Theta Band ($[f_{\mathrm{low}}, f_{\mathrm{high}}] = [4, 8]\ \mathrm{Hz}$)')

    # Parameter 1: Slope Triangle (Roll-off Rate)
    x1, x2 = 5.0, 7.5
    y1 = tb_res['slope'] * x1 + tb_res['intercept']
    y2 = tb_res['slope'] * x2 + tb_res['intercept']
    ax_zoom.plot([x1, x2, x2, x1], [y1, y1, y2, y1], color=PALETTE['crimson'], lw=1.5, ls='--')
    ax_zoom.text((x1 + x2)/2, y1 + 0.6, r"$\Delta f = 2.5\ \mathrm{Hz}$", color=PALETTE['crimson'], fontsize=9, ha='center', fontweight='bold')
    ax_zoom.text(x2 + 0.15, (y1 + y2)/2, r"$\Delta y = m_\theta \cdot \Delta f$", color=PALETTE['crimson'], fontsize=9, va='center', fontweight='bold')
    ax_zoom.annotate(r"$\mathbf{Parameter\ 1:\ Slope\ (m_b)}$" + f"\nRate of spectral roll-off: {tb_res['slope']:.2f} dB/Hz",
                     xy=((x1+x2)/2, (y1+y2)/2), xytext=(5.6, y1 + 4.5),
                     arrowprops=dict(facecolor=PALETTE['crimson'], arrowstyle='->', lw=1.2),
                     fontsize=9, fontweight='bold', color=PALETTE['crimson'],
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', edgecolor='#F5B7B1'))

    # Parameter 2: Intercept at f=0
    c_val = tb_res['intercept']
    ax_zoom.scatter([0.0], [c_val], color=PALETTE['gold'], s=90, edgecolors='black', zorder=6)
    ax_zoom.axvline(0.0, color='gray', ls=':', lw=1.0)
    ax_zoom.annotate(r"$\mathbf{Parameter\ 2:\ Intercept\ (c_b)}$" + f"\nBaseline power: {c_val:.2f} dB",
                     xy=(0.0, c_val), xytext=(0.3, c_val + 2.0),
                     arrowprops=dict(facecolor=PALETTE['gold'], arrowstyle='->', lw=1.2),
                     fontsize=9, fontweight='bold', color='#B7950B',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7', edgecolor='#F9E79F'))

    # Parameter 3: Midband Power
    f_mid = tb_res['f_mid']
    p_mid = tb_res['p_mid']
    ax_zoom.scatter([f_mid], [p_mid], color=PALETTE['teal'], s=100, edgecolors='black', zorder=6)
    ax_zoom.plot([f_mid, f_mid], [-25, p_mid], color=PALETTE['teal'], ls=':', lw=1.2)
    ax_zoom.plot([0, f_mid], [p_mid, p_mid], color=PALETTE['teal'], ls=':', lw=1.2)
    ax_zoom.annotate(r"$\mathbf{Parameter\ 3:\ Midband\ Fit\ (P_{\mathrm{mid}, b})}$" + f"\nCenter power at {f_mid:.1f} Hz: {p_mid:.2f} dB",
                     xy=(f_mid, p_mid), xytext=(4.2, p_mid - 6.5),
                     arrowprops=dict(facecolor=PALETTE['teal'], arrowstyle='->', lw=1.2),
                     fontsize=9, fontweight='bold', color=PALETTE['teal'],
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5', edgecolor='#A3E4D7'))

    ax_zoom.set_title(r"$\mathbf{Step\ 5:\ Geometric\ Extraction\ of\ the\ Three\ Physical\ Parameters\ (\theta-Band\ Exemplar)}$", pad=12)
    ax_zoom.set_xlabel("Frequency (Hz)")
    ax_zoom.set_ylabel(r"Log-Power Density (dB/Hz)")
    ax_zoom.set_xlim(-0.3, 8.8)
    ax_zoom.set_ylim(-24, 16)
    ax_zoom.legend(loc='lower left', fontsize=8.5)
    ax_zoom.set_xlabel("Frequency (Hz)")
    ax_zoom.set_ylabel(r"Log-Power Density (dB/Hz)")
    ax_zoom.set_xlim(-0.3, 8.8)
    ax_zoom.legend(loc='lower left', fontsize=8.5)

    # ---------------- Panel F: Structured Output Vector ----------------
    ax_vec = fig.add_subplot(gs[2, 3])
    ax_vec.axis('off')
    vec_box = FancyBboxPatch((0.05, 0.05), 0.90, 0.90, boxstyle="round,pad=0.06",
                             facecolor='#F4F6F7', edgecolor=PALETTE['navy'], linewidth=1.5)
    ax_vec.add_patch(vec_box)
    ax_vec.text(0.5, 0.90, "Structured Feature Output", ha='center', va='center', fontsize=10.5, fontweight='bold', color=PALETTE['navy'])
    ax_vec.text(0.5, 0.81, "12 Descriptors per Channel", ha='center', va='center', fontsize=9, color='gray')

    vector_items = [
        (r"$\mathbf{\delta\text{-Band}}:$", f"[{fit_results['delta']['slope']:.2f}, {fit_results['delta']['intercept']:.1f}, {fit_results['delta']['p_mid']:.1f}]", PALETTE['delta']),
        (r"$\mathbf{\theta\text{-Band}}:$", f"[{fit_results['theta']['slope']:.2f}, {fit_results['theta']['intercept']:.1f}, {fit_results['theta']['p_mid']:.1f}]", PALETTE['theta']),
        (r"$\mathbf{\alpha\text{-Band}}:$", f"[{fit_results['alpha']['slope']:.2f}, {fit_results['alpha']['intercept']:.1f}, {fit_results['alpha']['p_mid']:.1f}]", PALETTE['alpha']),
        (r"$\mathbf{\beta\text{-Band}}:$", f"[{fit_results['beta']['slope']:.2f}, {fit_results['beta']['intercept']:.1f}, {fit_results['beta']['p_mid']:.1f}]", PALETTE['beta']),
    ]
    y_pos = 0.68
    for label, val, col in vector_items:
        ax_vec.text(0.12, y_pos, label, fontsize=9.5, color=col, va='center')
        ax_vec.text(0.12, y_pos - 0.065, f"  [m, c, P_mid] = {val}", fontsize=8.2, color=PALETTE['dark'], va='center', family='monospace')
        y_pos -= 0.14

    ax_vec.text(0.5, 0.12, r"$\mathbf{Total\ Feature\ Vector:}\ \mathbf{x} \in \mathbb{R}^{12\times 18 = 216}$",
                ha='center', va='center', fontsize=8.8, fontweight='bold', color=PALETTE['crimson'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEDEC', edgecolor='#F5B7B1'))

    plt.suptitle("End-to-End Methodology: Log-PSD Spectral Slope Biomarker Parameterization",
                 fontsize=13.5, fontweight='bold', y=0.995, color=PALETTE['navy'])

    save_dual(fig, "fig_methodology_spectral_slope")


# ------------------------------------------------------------------------------
# FIGURE 2: DISCRETE WAVELET TRANSFORM (DWT) METHODOLOGY
# ------------------------------------------------------------------------------
def generate_dwt_methodology(t, sig, fs):
    """
    Step-by-step visual workflow of the DWT baseline feature extraction pipeline:
      Step 1: Input EEG epoch
      Step 2: Daubechies 4 filter bank decomposition tree (down to level 3: A3, D3, D2, D1)
      Step 3: Multi-resolution sub-band coefficient waveforms
      Step 4: Extraction of 6 descriptors: Energy, Std, Skewness, Kurtosis, Wiener Entropy, PSD RMS
    """
    print("Generating Figure 2: Discrete Wavelet Transform (DWT) Parameter Extraction Methodology...")
    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.3, 1.2], hspace=0.36, wspace=0.28)

    # ---------------- Panel A: Input EEG Epoch ----------------
    ax_raw = fig.add_subplot(gs[0, :2])
    ax_raw.plot(t * 1000, sig, color=PALETTE['crimson'], lw=1.5, label='Epoch Signal x[n]')
    ax_raw.set_title("Step 1: Input Neonatal EEG Epoch Signal x[n] (N=256, fs=256 Hz)", pad=8, fontweight='bold')
    ax_raw.set_xlabel("Time (milliseconds)")
    ax_raw.set_ylabel(r"Amplitude ($\mu\mathrm{V}$)")
    ax_raw.set_xlim(0, 1000)
    ax_raw.legend(loc='upper right')

    # ---------------- Panel B: Daubechies 4 Filter Bank Tree Diagram ----------------
    ax_tree = fig.add_subplot(gs[0, 2:])
    ax_tree.axis('off')
    tree_box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.03",
                              facecolor='#FDFEFE', edgecolor=PALETTE['box_border'], linewidth=1.2)
    ax_tree.add_patch(tree_box)
    ax_tree.set_title("Step 2: Multi-Level Filter Bank Decomposition Tree (Daubechies db4)", pad=8, fontweight='bold')

    # Draw Filter Tree Schematic
    # Root
    ax_tree.text(0.08, 0.50, "x[n]\n(0–128 Hz)", ha='center', va='center',
                 bbox=dict(boxstyle='square,pad=0.3', facecolor='#EAECEE', edgecolor='#2C3E50', lw=1.2), fontsize=8.5, fontweight='bold')
    # Level 1
    ax_tree.annotate('', xy=(0.28, 0.72), xytext=(0.14, 0.52), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax_tree.annotate('', xy=(0.28, 0.28), xytext=(0.14, 0.48), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax_tree.text(0.34, 0.72, "High-Pass\n(h[n]) ↓2", ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F8F5', edgecolor='#16A085'), fontsize=8)
    ax_tree.text(0.34, 0.28, "Low-Pass\n(g[n]) ↓2", ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#EBF5FB', edgecolor='#2980B9'), fontsize=8)

    # D1 Terminal
    ax_tree.annotate('', xy=(0.52, 0.72), xytext=(0.42, 0.72), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax_tree.text(0.68, 0.72, r"$\mathbf{D_1\ (64\text{--}128\ \mathrm{Hz})}$" + "\nDetail Level 1 (N=131)", ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#FDEDEC', edgecolor=PALETTE['crimson'], lw=1.2), fontsize=8.2)

    # Level 2
    ax_tree.annotate('', xy=(0.46, 0.45), xytext=(0.40, 0.32), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax_tree.annotate('', xy=(0.46, 0.15), xytext=(0.40, 0.24), arrowprops=dict(arrowstyle="->", lw=1.1))
    ax_tree.text(0.51, 0.45, "h[n] ↓2", ha='center', va='center', fontsize=7.5)
    ax_tree.text(0.51, 0.15, "g[n] ↓2", ha='center', va='center', fontsize=7.5)

    # D2 Terminal
    ax_tree.text(0.70, 0.45, r"$\mathbf{D_2\ (32\text{--}64\ \mathrm{Hz})}$" + "\nDetail Level 2 (N=69)", ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF9E7', edgecolor=PALETTE['gold'], lw=1.2), fontsize=8.2)

    # Level 3
    ax_tree.annotate('', xy=(0.60, 0.26), xytext=(0.56, 0.17), arrowprops=dict(arrowstyle="->", lw=1.0))
    ax_tree.annotate('', xy=(0.60, 0.06), xytext=(0.56, 0.13), arrowprops=dict(arrowstyle="->", lw=1.0))

    # D3 & A3 Terminals
    ax_tree.text(0.82, 0.26, r"$\mathbf{D_3\ (16\text{--}32\ \mathrm{Hz})}$" + " (N=38)", ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F8F5', edgecolor=PALETTE['teal'], lw=1.2), fontsize=8.0)
    ax_tree.text(0.82, 0.06, r"$\mathbf{A_3\ (0\text{--}16\ \mathrm{Hz})}$" + " (N=38)", ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#EBF5FB', edgecolor=PALETTE['delta'], lw=1.2), fontsize=8.0)

    # ---------------- Panel C: Decomposed Sub-band Waveforms ----------------
    # Perform Daubechies 4 decomposition (level 3)
    coeffs = pywt.wavedec(sig, 'db4', level=3)
    # coeffs order: [A3, D3, D2, D1]
    subband_names = [
        (r'Approximation $A_3$ (0–16 Hz)', coeffs[0], PALETTE['delta']),
        (r'Detail $D_3$ (16–32 Hz)', coeffs[1], PALETTE['teal']),
        (r'Detail $D_2$ (32–64 Hz)', coeffs[2], PALETTE['gold']),
        (r'Detail $D_1$ (64–128 Hz)', coeffs[3], PALETTE['crimson']),
    ]

    for idx, (sb_name, sb_coeff, sb_col) in enumerate(subband_names):
        ax_sb = fig.add_subplot(gs[1, idx])
        n_samples = len(sb_coeff)
        x_idx = np.arange(n_samples)
        ax_sb.plot(x_idx, sb_coeff, color=sb_col, lw=1.4)
        ax_sb.set_title(f"{sb_name}", fontsize=9.2, color=sb_col, pad=6)
        ax_sb.set_xlabel("Coeff Index [n]", fontsize=8.5)
        if idx == 0:
            ax_sb.set_ylabel(r"Amplitude ($\mu\mathrm{V}$)")
        ax_sb.set_xlim(0, n_samples - 1)

    # ---------------- Panel D: The 6 Parameter Extraction Mechanics ----------------
    ax_desc = fig.add_subplot(gs[2, :3])
    ax_desc.axis('off')
    desc_box = FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.04",
                              facecolor='#FBFCFC', edgecolor=PALETTE['navy'], linewidth=1.4)
    ax_desc.add_patch(desc_box)
    ax_desc.set_title("Step 4: Extraction of the Six Statistical & Spectral Descriptors per Sub-Band", pad=6, fontweight='bold')

    # Define equations & physical significance of the 6 descriptors
    descriptors_info = [
        ("1. Total Energy (E):",
         r"$E = \sum_{i=1}^{N} c_i^2$",
         "Captures cumulative spectral power and paroxysmal burst energy in the frequency sub-band."),
        ("2. Standard Deviation (σ):",
         r"$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(c_i - \bar{c})^2}$",
         "Quantifies amplitude dispersion and coefficient spread around the baseline mean."),
        ("3. Skewness (S):",
         r"$S = \frac{\frac{1}{N}\sum (c_i - \bar{c})^3}{\sigma^3}$",
         "Measures asymmetry of the oscillatory coefficient distribution (directional bias)."),
        ("4. Kurtosis (K):",
         r"$K = \frac{\frac{1}{N}\sum (c_i - \bar{c})^4}{\sigma^4}$",
         "Quantifies heavy tails and impulsive spike-and-wave discharges relative to Gaussian noise."),
        ("5. Wiener Entropy (H_W):",
         r"$H_W = \frac{\exp\left(\frac{1}{N}\sum \ln(c_i^2 + \epsilon)\right)}{\frac{1}{N}\sum c_i^2}$",
         "Spectral flatness measure: ratio of geometric mean to arithmetic mean (signal complexity)."),
        ("6. PSD RMS (RMS_PSD):",
         r"$\mathrm{RMS}_{\mathrm{PSD}} = \sqrt{\frac{1}{M}\sum_{m=1}^{M} P(f_m)}$",
         "Root mean square of the power spectral density, evaluating concentrated sub-band power."),
    ]

    y_pos = 0.84
    for idx, (title_desc, formula, interp) in enumerate(descriptors_info):
        col_offset = 0.03 if idx < 3 else 0.52
        curr_y = y_pos if idx % 3 == 0 else (y_pos - 0.28 if idx % 3 == 1 else y_pos - 0.56)

        ax_desc.text(col_offset, curr_y, title_desc, fontsize=9.2, fontweight='bold', color=PALETTE['navy'])
        ax_desc.text(col_offset + 0.22, curr_y, formula, fontsize=9.0, color=PALETTE['crimson'], family='sans-serif')
        ax_desc.text(col_offset, curr_y - 0.12, f"• {interp}", fontsize=8.0, color='#34495E', wrap=True)

    # ---------------- Panel E: Feature Vector Summary ----------------
    ax_out = fig.add_subplot(gs[2, 3])
    ax_out.axis('off')
    out_box = FancyBboxPatch((0.05, 0.02), 0.90, 0.96, boxstyle="round,pad=0.05",
                             facecolor='#F4F6F7', edgecolor=PALETTE['teal'], linewidth=1.5)
    ax_out.add_patch(out_box)
    ax_out.text(0.5, 0.90, "Structured DWT Vector", ha='center', va='center', fontsize=10.5, fontweight='bold', color=PALETTE['teal'])
    ax_out.text(0.5, 0.81, "4 Bands × 6 Descriptors = 24", ha='center', va='center', fontsize=8.8, color='gray')

    # Calculate actual numeric values for A3 subband as example
    a3 = coeffs[0]
    e_val = np.sum(a3**2)
    std_val = np.std(a3)
    skew_val = scipy_skew(a3)
    kurt_val = scipy_kurtosis(a3)
    wiener_val = np.exp(np.mean(np.log(a3**2 + 1e-12))) / (np.mean(a3**2) + 1e-12)
    f_a, p_a = welch(a3, fs=fs//8, nperseg=len(a3))
    psd_rms_val = np.sqrt(np.mean(p_a))

    sample_vals = [
        ("Energy (E)", f"{e_val:.1e}"),
        ("Std Dev (σ)", f"{std_val:.2f} µV"),
        ("Skewness (S)", f"{skew_val:.2f}"),
        ("Kurtosis (K)", f"{kurt_val:.2f}"),
        ("Wiener Ent (H_w)", f"{wiener_val:.3f}"),
        ("PSD RMS", f"{psd_rms_val:.2f}"),
    ]

    ax_out.text(0.12, 0.69, "Exemplar Band Vector (A3):", fontsize=8.5, fontweight='bold', color=PALETTE['delta'])
    y_v = 0.59
    for name, val in sample_vals:
        ax_out.text(0.15, y_v, f"• {name}:", fontsize=8.0, color='#2C3E50')
        ax_out.text(0.72, y_v, val, fontsize=8.0, color=PALETTE['navy'], fontweight='bold', family='monospace')
        y_v -= 0.085

    ax_out.text(0.5, 0.09, r"$\mathbf{Total\ DWT\ Feature\ Vector:}\ \mathbf{x}_{\mathrm{DWT}} \in \mathbb{R}^{24\times 18 = 432}$",
                ha='center', va='center', fontsize=8.2, fontweight='bold', color=PALETTE['crimson'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEDEC', edgecolor='#F5B7B1'))

    plt.suptitle("Comparative Baseline Methodology: Discrete Wavelet Transform (DWT) Feature Extraction",
                 fontsize=13.5, fontweight='bold', y=0.995, color=PALETTE['navy'])

    save_dual(fig, "fig_methodology_dwt")


# ------------------------------------------------------------------------------
# FIGURE 3: EMPIRICAL MODE DECOMPOSITION (EMD) METHODOLOGY
# ------------------------------------------------------------------------------
def generate_emd_methodology(t, sig, fs):
    """
    Step-by-step visual workflow of the EMD baseline feature extraction pipeline:
      Step 1: Input Non-Stationary EEG Epoch
      Step 2: The Sifting Algorithm (Extrema detection, Cubic Spline Upper/Lower Envelopes, Mean extraction)
      Step 3: Decomposed Intrinsic Mode Functions (IMFs 1 to 4) + Residual Trend
      Step 4: Extraction of 6 Descriptors per IMF (Energy, Std, Skewness, Kurtosis, Wiener Entropy, PSD RMS)
    """
    print("Generating Figure 3: Empirical Mode Decomposition (EMD) Parameter Extraction Methodology...")
    fig = plt.figure(figsize=(15.5, 9.8))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.3, 1.2], hspace=0.38, wspace=0.28)

    # ---------------- Panel A: Input EEG Epoch ----------------
    ax_raw = fig.add_subplot(gs[0, :2])
    ax_raw.plot(t * 1000, sig, color=PALETTE['crimson'], lw=1.5, label='Non-Stationary EEG Signal x(t)')
    ax_raw.set_title("Step 1: Input Neonatal EEG Epoch Signal x(t) (fs=256 Hz)", pad=8, fontweight='bold')
    ax_raw.set_xlabel("Time (milliseconds)")
    ax_raw.set_ylabel(r"Amplitude ($\mu\mathrm{V}$)")
    ax_raw.set_xlim(0, 1000)
    ax_raw.legend(loc='upper right')

    # ---------------- Panel B: Sifting Algorithm Mechanics (Envelopes & Mean) ----------------
    ax_sift = fig.add_subplot(gs[0, 2:])

    # Zoom in on a sub-segment (0 to 300 ms) to clearly illustrate envelope interpolation
    sub_n = 77  # ~300 ms
    t_sub = t[:sub_n] * 1000
    s_sub = sig[:sub_n]

    # Find extrema
    max_idx = (np.diff(np.sign(np.diff(s_sub))) < 0).nonzero()[0] + 1
    min_idx = (np.diff(np.sign(np.diff(s_sub))) > 0).nonzero()[0] + 1

    # Add boundary points for spline stability
    max_x = np.concatenate(([t_sub[0]], t_sub[max_idx], [t_sub[-1]]))
    max_y = np.concatenate(([s_sub[0]], s_sub[max_idx], [s_sub[-1]]))
    min_x = np.concatenate(([t_sub[0]], t_sub[min_idx], [t_sub[-1]]))
    min_y = np.concatenate(([s_sub[0]], s_sub[min_idx], [s_sub[-1]]))

    cs_up = CubicSpline(max_x, max_y, bc_type='natural')
    cs_low = CubicSpline(min_x, min_y, bc_type='natural')

    t_fine = np.linspace(t_sub[0], t_sub[-1], 250)
    env_up = cs_up(t_fine)
    env_low = cs_low(t_fine)
    env_mean = (env_up + env_low) / 2.0

    ax_sift.plot(t_sub, s_sub, color='#2C3E50', lw=1.6, label='Signal x(t)')
    ax_sift.scatter(t_sub[max_idx], s_sub[max_idx], color=PALETTE['crimson'], s=40, zorder=5, label='Local Maxima')
    ax_sift.scatter(t_sub[min_idx], s_sub[min_idx], color=PALETTE['delta'], s=40, zorder=5, label='Local Minima')
    ax_sift.plot(t_fine, env_up, color=PALETTE['crimson'], lw=1.2, ls='--', label=r'Upper Envelope $e_{\mathrm{up}}(t)$')
    ax_sift.plot(t_fine, env_low, color=PALETTE['delta'], lw=1.2, ls='--', label=r'Lower Envelope $e_{\mathrm{low}}(t)$')
    ax_sift.plot(t_fine, env_mean, color='#F39C12', lw=1.8, label=r'Mean Envelope $m_1(t)$')

    ax_sift.set_title(r"Step 2: Sifting Mechanics: $e_{\mathrm{up}}(t),\ e_{\mathrm{low}}(t) \rightarrow m_1(t) = \frac{e_{\mathrm{up}}+e_{\mathrm{low}}}{2}$", pad=8, fontweight='bold')
    ax_sift.set_xlabel("Time (milliseconds)")
    ax_sift.set_ylabel(r"Amplitude ($\mu\mathrm{V}$)")
    ax_sift.set_xlim(0, t_sub[-1])
    ax_sift.legend(loc='lower left', fontsize=7.8, ncol=2)

    # ---------------- Panel C: Extracted Intrinsic Mode Functions (IMFs 1 to 4) ----------------
    emd_obj = PyEMD.EMD()
    imfs = emd_obj(sig, max_imf=4)
    # Ensure 4 IMFs
    if imfs.shape[0] < 4:
        imfs = np.pad(imfs, ((0, 4 - imfs.shape[0]), (0, 0)), mode='edge')

    imf_colors = [PALETTE['crimson'], PALETTE['gold'], PALETTE['teal'], PALETTE['delta']]
    imf_titles = [
        ("IMF 1 (High-Freq Spikes)", imfs[0], imf_colors[0]),
        ("IMF 2 (Paroxysmal Waves)", imfs[1], imf_colors[1]),
        ("IMF 3 (Rhythmic Oscillations)", imfs[2], imf_colors[2]),
        ("IMF 4 (Slow Baseline Trend)", imfs[3], imf_colors[3]),
    ]

    for idx, (imf_label, imf_data, imf_col) in enumerate(imf_titles):
        ax_imf = fig.add_subplot(gs[1, idx])
        ax_imf.plot(t * 1000, imf_data, color=imf_col, lw=1.4)
        ax_imf.set_title(imf_label, fontsize=9.2, color=imf_col, pad=6, fontweight='bold')
        ax_imf.set_xlabel("Time (ms)", fontsize=8.5)
        if idx == 0:
            ax_imf.set_ylabel(r"Amplitude ($\mu\mathrm{V}$)")
        ax_imf.set_xlim(0, 1000)

    # ---------------- Panel D: Parameter Extraction Highlighting (The 6 Descriptors) ----------------
    ax_desc = fig.add_subplot(gs[2, :3])
    ax_desc.axis('off')
    desc_box = FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.04",
                              facecolor='#FBFCFC', edgecolor=PALETTE['navy'], linewidth=1.4)
    ax_desc.add_patch(desc_box)
    ax_desc.set_title("Step 4: The Suite of Six Descriptors Computed on Each Intrinsic Mode Function (IMF)", pad=6, fontweight='bold')

    descriptors_info = [
        ("1. Total Energy (E):",
         r"$E_k = \sum_{t=1}^{N} [\mathrm{IMF}_k(t)]^2$",
         "Reflects instantaneous oscillatory energy isolated within the specific IMF mode."),
        ("2. Standard Deviation (σ):",
         r"$\sigma_k = \sqrt{\frac{1}{N}\sum (\mathrm{IMF}_k(t) - \bar{\mu}_k)^2}$",
         "Measures amplitude dispersion and dynamic envelope fluctuation of the IMF mode."),
        ("3. Skewness (S):",
         r"$S_k = \frac{\frac{1}{N}\sum (\mathrm{IMF}_k(t) - \bar{\mu}_k)^3}{\sigma_k^3}$",
         "Evaluates morphological wave asymmetry and unipolar paroxysmal deflections."),
        ("4. Kurtosis (K):",
         r"$K_k = \frac{\frac{1}{N}\sum (\mathrm{IMF}_k(t) - \bar{\mu}_k)^4}{\sigma_k^4}$",
         "Quantifies heavy tails, indicating sharp polyphasic epileptiform discharges."),
        ("5. Wiener Entropy (H_W):",
         r"$H_{W,k} = \frac{\exp\left(\frac{1}{N}\sum \ln([\mathrm{IMF}_k(t)]^2 + \epsilon)\right)}{\frac{1}{N}\sum [\mathrm{IMF}_k(t)]^2}$",
         "Quantifies spectral flatness and structural complexity of the decomposed mode."),
        ("6. PSD RMS (RMS_PSD):",
         r"$\mathrm{RMS}_{\mathrm{PSD}} = \sqrt{\frac{1}{M}\sum P_k(f_m)}$",
         "Root mean square of the IMF's power spectral density, tracking spectral concentration."),
    ]

    y_pos = 0.84
    for idx, (title_desc, formula, interp) in enumerate(descriptors_info):
        col_offset = 0.03 if idx < 3 else 0.52
        curr_y = y_pos if idx % 3 == 0 else (y_pos - 0.28 if idx % 3 == 1 else y_pos - 0.56)

        ax_desc.text(col_offset, curr_y, title_desc, fontsize=9.2, fontweight='bold', color=PALETTE['navy'])
        ax_desc.text(col_offset + 0.22, curr_y, formula, fontsize=9.0, color=PALETTE['crimson'], family='sans-serif')
        ax_desc.text(col_offset, curr_y - 0.12, f"• {interp}", fontsize=8.0, color='#34495E', wrap=True)

    # ---------------- Panel E: Feature Vector Output ----------------
    ax_out = fig.add_subplot(gs[2, 3])
    ax_out.axis('off')
    out_box = FancyBboxPatch((0.05, 0.02), 0.90, 0.96, boxstyle="round,pad=0.05",
                             facecolor='#F4F6F7', edgecolor=PALETTE['gold'], linewidth=1.5)
    ax_out.add_patch(out_box)
    ax_out.text(0.5, 0.90, "Structured EMD Vector", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#B7950B')
    ax_out.text(0.5, 0.81, "4 IMFs × 6 Descriptors = 24", ha='center', va='center', fontsize=8.8, color='gray')

    # Calculate actual numeric values for IMF 1
    imf1 = imfs[0]
    e_val = np.sum(imf1**2)
    std_val = np.std(imf1)
    skew_val = scipy_skew(imf1)
    kurt_val = scipy_kurtosis(imf1)
    wiener_val = np.exp(np.mean(np.log(imf1**2 + 1e-12))) / (np.mean(imf1**2) + 1e-12)
    f_imf, p_imf = welch(imf1, fs=fs, nperseg=len(imf1))
    psd_rms_val = np.sqrt(np.mean(p_imf))

    sample_vals = [
        ("Energy (E)", f"{e_val:.1e}"),
        ("Std Dev (σ)", f"{std_val:.2f} µV"),
        ("Skewness (S)", f"{skew_val:.2f}"),
        ("Kurtosis (K)", f"{kurt_val:.2f}"),
        ("Wiener Ent (H_w)", f"{wiener_val:.3f}"),
        ("PSD RMS", f"{psd_rms_val:.2f}"),
    ]

    ax_out.text(0.12, 0.69, "Exemplar Vector (IMF 1):", fontsize=8.5, fontweight='bold', color=PALETTE['crimson'])
    y_v = 0.59
    for name, val in sample_vals:
        ax_out.text(0.15, y_v, f"• {name}:", fontsize=8.0, color='#2C3E50')
        ax_out.text(0.72, y_v, val, fontsize=8.0, color=PALETTE['navy'], fontweight='bold', family='monospace')
        y_v -= 0.085

    ax_out.text(0.5, 0.09, r"$\mathbf{Total\ EMD\ Feature\ Vector:}\ \mathbf{x}_{\mathrm{EMD}} \in \mathbb{R}^{24\times 18 = 432}$",
                ha='center', va='center', fontsize=8.2, fontweight='bold', color=PALETTE['crimson'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FDEDEC', edgecolor='#F5B7B1'))

    plt.suptitle("Comparative Baseline Methodology: Empirical Mode Decomposition (EMD) Feature Extraction",
                 fontsize=13.5, fontweight='bold', y=0.995, color=PALETTE['navy'])

    save_dual(fig, "fig_methodology_emd")

# ------------------------------------------------------------------------------
# Main Runner
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Generating Methodology Figures (Spectral Slope, DWT, and EMD)...")
    print("=" * 80)

    # 1. Load real data
    t, seizure_sig, bg_sig, fs = load_real_epochs()
    print(f"Loaded real EEG epoch from eeg1.edf (fs={fs} Hz, length={len(seizure_sig)} samples)")

    # 2. Generate Figure 1: Spectral Slope
    generate_spectral_slope_methodology(t, seizure_sig, fs)

    # 3. Generate Figure 2: DWT
    generate_dwt_methodology(t, seizure_sig, fs)

    # 4. Generate Figure 3: EMD
    generate_emd_methodology(t, seizure_sig, fs)

    print("\n" + "=" * 80)
    print("All 3 methodology figures successfully generated and saved to:")
    print(f"  • {PUBLIC_DIR}")
    print(f"  • {PAPER_FIG_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()

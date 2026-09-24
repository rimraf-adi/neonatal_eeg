#!/usr/bin/env python3
"""
Publication-Grade Sequential Methodology Diagrams for Springer Nature LaTeX Paper
==================================================================================
Style: Formal Academic Multi-Panel Layout (IEEE / Nature style)
Color System: Dignified Editorial Scientific Palette (Pastel tints, Crisp Spines)
Layout:
  - Reduced individual box sizes significantly for generous inter-panel spacing.
  - Centered pipeline connector arrows positioned precisely at box vertical center.
  - Y-axis unit titles located above top-left of sequential subpanels to guarantee
    ZERO overlap between arrows, tick labels, and axis titles.
  - 4-spine boxed frames for empirical plots, clean rounded cards for descriptors.
  - Entire figure enclosed in a clean outer border frame.

Generates:
  1. fig_methodology_spectral_slope.(png/pdf)  - 5 panels (a-e)
  2. fig_methodology_dwt.(png/pdf)             - 4 panels (a-d, tree omitted)
  3. fig_methodology_emd.(png/pdf)             - 5 panels (a-e)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from scipy.signal import welch
from scipy.interpolate import CubicSpline
import pywt
import PyEMD
import mne

# Editorial Typography Configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.0,
    'mathtext.fontset': 'dejavusans',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'axes.linewidth': 0.8,
})

REPO_ROOT = Path("d:/neonatal")
PUBLIC_DIR = REPO_ROOT / "public"
PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

THEME = {
    'bg': '#FFFFFF',
    'outer_border': '#94A3B8',
    'box_spine': '#64748B',
    'grid': '#F1F5F9',
    'text_dark': '#0F172A',
    'text_muted': '#64748B',
    'arrow': '#475569',
    
    # Sub-band / Mode Scientific Gradient (Nature editorial palette)
    'c_blue': '#1D4ED8',         # Deep Royal Blue (Low freq / Approx)
    'c_teal': '#0F766E',         # Muted Teal
    'c_amber': '#B45309',        # Muted Amber
    'c_crimson': '#DC2626',      # Deep Crimson (High freq / Detail / Fit)
    
    # Soft badge background fills & borders
    'b_blue_bg': '#EFF6FF',  'b_blue_bd': '#93C5FD',  'b_blue_tx': '#1E40AF',
    'b_teal_bg': '#F0FDF4',  'b_teal_bd': '#86EFAC',  'b_teal_tx': '#166534',
    'b_amber_bg': '#FEFCE8', 'b_amber_bd': '#FDE047', 'b_amber_tx': '#854D0E',
    'b_crimson_bg': '#FEF2F2', 'b_crimson_bd': '#FCA5A5', 'b_crimson_tx': '#991B1B',
    
    'panel_box_bg': '#F8FAFC',
    'panel_box_border': '#CBD5E1',
}

def save_fig_dual(fig, name):
    for d in [PUBLIC_DIR, PAPER_FIG_DIR]:
        fig.savefig(d / f"{name}.png", dpi=300, format='png')
        fig.savefig(d / f"{name}.pdf", dpi=300, format='pdf')
    print(f"  [OK] Saved {name}.png and {name}.pdf to public/ and paper/figures/")
    plt.close(fig)

def load_data():
    edf_p = REPO_ROOT / "dataset" / "eeg1.edf"
    raw = mne.io.read_raw_edf(str(edf_p), preload=True, verbose=False)
    raw.rename_channels(lambda c: c.upper())
    d_c3 = raw.get_data(picks=['EEG C3-REF'])[0]
    d_p3 = raw.get_data(picks=['EEG P3-REF'])[0]
    sig_uv = (d_c3 - d_p3) * 1e6
    fs = int(raw.info['sfreq'])
    epoch = sig_uv[104 * fs : 105 * fs]
    t = np.linspace(0, 1.0, fs, endpoint=False)
    return t, epoch, fs

def setup_boxed_spines(ax):
    """Encloses subplot in a classic 4-spine boxed frame (IEEE / Nature style)."""
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(THEME['box_spine'])
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=THEME['text_dark'], width=0.8, length=3.0, direction='in', labelsize=6.5)

def draw_outer_enclosure_box(ax_bg):
    """Global external border omitted as requested."""
    pass

def draw_center_pipeline_arrow(ax_bg, x1, y1, x2, y2, label=""):

    """Draws a refined scientific pipeline connector centered vertically between subpanels."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                            mutation_scale=9, linewidth=1.1, color=THEME['arrow'], zorder=4)
    ax_bg.add_patch(arrow)
    if label:
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        ax_bg.text(xm, ym + 0.020, label, fontsize=6.2, color=THEME['text_muted'],
                   fontweight='bold', ha='center', va='bottom', zorder=5)

def draw_panel_header(ax_bg, x, y, letter, title):
    """Draws formal academic panel title: (a) Title."""
    full_title = f"({letter})  {title}"
    ax_bg.text(x, y, full_title, fontsize=7.8, fontweight='bold', color=THEME['text_dark'],
               ha='left', va='bottom', zorder=5)

# ==============================================================================
# 1. CLASSIC JOURNAL MULTI-PANEL: SPECTRAL SLOPE (LOG-PSD)
# ==============================================================================
def make_spectral_slope_diagram(t, sig, fs):
    print("Generating Classic Journal Multi-Panel: Spectral Slope (Center Arrows, Reduced Box Sizes)...")
    fig = plt.figure(figsize=(14.4, 3.6), facecolor=THEME['bg'])
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.axis('off')
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)

    draw_outer_enclosure_box(ax_bg)

    # Significantly reduced box dimensions to provide ample breathing room
    pw = 0.115
    ph = 0.54
    y_ax = 0.16
    y_hdr = 0.81
    y_center = y_ax + ph / 2.0  # Exactly at the vertical center of the boxes (0.43)
    xs = [0.040, 0.243, 0.446, 0.649, 0.852]

    # Panel Headers
    draw_panel_header(ax_bg, xs[0], y_hdr, "a", r"Raw EEG Epoch $x(t)$")
    draw_panel_header(ax_bg, xs[1], y_hdr, "b", r"Welch PSD $P(f)$")
    draw_panel_header(ax_bg, xs[2], y_hdr, "c", r"Log-PSD Dynamics $y(f)$")
    draw_panel_header(ax_bg, xs[3], y_hdr, "d", r"Sub-Band OLS Fit $\hat{y}_b(f)$")
    draw_panel_header(ax_bg, xs[4], y_hdr, "e", r"Biomarker Vector $\mathbf{x}_{\mathrm{slope}}$")

    # Pipeline connectors AT VERTICAL CENTER with zero overlap
    draw_center_pipeline_arrow(ax_bg, xs[0] + pw + 0.012, y_center, xs[1] - 0.024, y_center, "Welch")
    draw_center_pipeline_arrow(ax_bg, xs[1] + pw + 0.012, y_center, xs[2] - 0.024, y_center, r"$10\log_{10}$")
    draw_center_pipeline_arrow(ax_bg, xs[2] + pw + 0.012, y_center, xs[3] - 0.024, y_center, "OLS Fit")
    draw_center_pipeline_arrow(ax_bg, xs[3] + pw + 0.012, y_center, xs[4] - 0.012, y_center, "Concatenate")

    # (a) Raw EEG
    ax1 = fig.add_axes([xs[0], y_ax, pw, ph])
    setup_boxed_spines(ax1)
    ax1.plot(t, sig, color='#1E293B', lw=1.1)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Time t (s)", fontsize=6.8, labelpad=2)
    ax1.set_ylabel(r"Amplitude ($\mu\mathrm{V}$)", fontsize=6.8, labelpad=2)
    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_xticklabels(["0", "0.5", "1.0"])
    ax1.grid(True, ls='--', lw=0.4, color=THEME['grid'], alpha=0.8)

    # (b) Welch PSD
    freqs, psd = welch(sig, fs=fs, nperseg=fs, window='hann')
    mask = (freqs >= 0.5) & (freqs <= 30.0)
    f_val, p_val = freqs[mask], psd[mask]

    ax2 = fig.add_axes([xs[1], y_ax, pw, ph])
    setup_boxed_spines(ax2)
    ax2.plot(f_val, p_val, color='#1E293B', lw=1.2)
    ax2.set_xlim(0, 30)
    ax2.set_xlabel("Frequency f (Hz)", fontsize=6.8, labelpad=2)
    ax2.text(-0.06, 1.03, r"PSD ($\mu\mathrm{V}^2/\mathrm{Hz}$)", transform=ax2.transAxes,
             fontsize=6.5, ha='left', va='bottom', color=THEME['text_dark'])
    ax2.grid(True, ls='--', lw=0.4, color=THEME['grid'], alpha=0.8)

    ax2.axvspan(0.5, 4, color='#EFF6FF', alpha=0.7, zorder=0)
    ax2.axvspan(4, 8, color='#F8FAFC', alpha=0.9, zorder=0)
    ax2.axvspan(8, 13, color='#F0FDF4', alpha=0.7, zorder=0)
    ax2.axvspan(13, 30, color='#FAF5FF', alpha=0.7, zorder=0)

    y_top2 = ax2.get_ylim()[1]
    ax2.text(2.25, y_top2 * 0.84, r"$\delta$", color='#1E40AF', fontweight='bold', fontsize=7.2, ha='center')
    ax2.text(6.0, y_top2 * 0.84, r"$\theta$", color='#334155', fontweight='bold', fontsize=7.2, ha='center')
    ax2.text(10.5, y_top2 * 0.84, r"$\alpha$", color='#0F766E', fontweight='bold', fontsize=7.2, ha='center')
    ax2.text(21.5, y_top2 * 0.84, r"$\beta$", color='#6B21A8', fontweight='bold', fontsize=7.2, ha='center')

    # (c) Semi-Log PSD
    db_p = 10 * np.log10(p_val + 1e-10)
    ax3 = fig.add_axes([xs[2], y_ax, pw, ph])
    setup_boxed_spines(ax3)
    ax3.plot(f_val, db_p, color='#1E293B', lw=1.2)
    ax3.set_xlim(0, 30)
    ax3.set_xlabel("Frequency f (Hz)", fontsize=6.8, labelpad=2)
    ax3.text(-0.06, 1.03, r"$10\log_{10}P(f)$ (dB/Hz)", transform=ax3.transAxes,
             fontsize=6.5, ha='left', va='bottom', color=THEME['text_dark'])
    ax3.grid(True, ls='--', lw=0.4, color=THEME['grid'], alpha=0.8)

    ax3.axvspan(0.5, 4, color='#EFF6FF', alpha=0.7, zorder=0)
    ax3.axvspan(4, 8, color='#F8FAFC', alpha=0.9, zorder=0)
    ax3.axvspan(8, 13, color='#F0FDF4', alpha=0.7, zorder=0)
    ax3.axvspan(13, 30, color='#FAF5FF', alpha=0.7, zorder=0)

    # (d) OLS Fit on EXACT empirical spectrum
    ax4 = fig.add_axes([xs[3], y_ax, pw, ph])
    setup_boxed_spines(ax4)

    th_mask = (f_val >= 4.0) & (f_val <= 8.0)
    th_f, th_p = f_val[th_mask], db_p[th_mask]
    coeffs = np.polyfit(th_f, th_p, 1)
    m_val, c_val = coeffs[0], coeffs[1]
    f_mid = 6.0
    p_mid = m_val * f_mid + c_val

    # 1. Exact empirical curve
    ax4.plot(th_f, th_p, color='#1E293B', lw=1.3, marker='o', markersize=3.0,
             label=r'Spectrum $y(f)$', zorder=4)

    # 2. OLS regression line
    f_ext = np.linspace(0, 8.5, 100)
    ax4.plot(f_ext, m_val * f_ext + c_val, color=THEME['c_crimson'], lw=1.5, ls='-',
             label=r'Fit $\hat{y}_b(f)$', zorder=3)

    # 3. Residual projections
    for f_i, p_i in zip(th_f, th_p):
        ax4.plot([f_i, f_i], [p_i, m_val * f_i + c_val], color='#94A3B8', ls=':', lw=0.7, zorder=2)

    # Slope triangle
    ax4.plot([4.2, 7.2, 7.2, 4.2], [m_val*4.2+c_val, m_val*4.2+c_val, m_val*7.2+c_val, m_val*4.2+c_val],
             color=THEME['c_crimson'], lw=0.9, ls='--', zorder=3)
    ax4.text(5.5, m_val*4.2+c_val + 1.4, r"Slope $m_b = \frac{\Delta y}{\Delta f}$",
             fontsize=5.8, color=THEME['c_crimson'], fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.12', facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=0.6), zorder=6)

    # Intercept marker
    ax4.scatter([0], [c_val], color=THEME['c_amber'], marker='D', s=22, zorder=6)
    ax4.annotate(r"$c_b$ (Intercept)", xy=(0, c_val), xytext=(0.4, c_val + 1.4),
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=THEME['c_amber']),
                 fontsize=5.8, color=THEME['c_amber'], fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.12', facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=0.6), zorder=6)

    # Mid-power marker
    ax4.scatter([f_mid], [p_mid], color=THEME['c_teal'], marker='s', s=22, zorder=6)
    ax4.annotate(r"$P_{\mathrm{mid}, b}$", xy=(f_mid, p_mid), xytext=(f_mid - 2.8, p_mid - 2.6),
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=THEME['c_teal']),
                 fontsize=5.8, color=THEME['c_teal'], fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.12', facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=0.6), zorder=6)

    ax4.set_xlim(-0.8, 8.8)
    ax4.set_ylim(-14.0, 16.0)
    ax4.set_xlabel("Frequency f (Hz)", fontsize=6.8, labelpad=2)
    ax4.text(-0.06, 1.03, r"Power (dB/Hz)", transform=ax4.transAxes,
             fontsize=6.5, ha='left', va='bottom', color=THEME['text_dark'])
    ax4.grid(True, ls='--', lw=0.4, color=THEME['grid'], alpha=0.8)
    ax4.legend(loc='lower left', fontsize=5.2, framealpha=0.95, edgecolor='#CBD5E1', handlelength=1.0, borderpad=0.2)

    # (e) Feature Vector Assembly
    ax5 = fig.add_axes([xs[4], y_ax, pw, ph])
    ax5.axis('off')
    
    panel_card = FancyBboxPatch((0.005, 0.005), 0.990, 0.990, boxstyle="round,pad=0.005,rounding_size=0.015",
                                facecolor='#FFFFFF', edgecolor=THEME['box_spine'], lw=0.8, zorder=1)
    ax5.add_patch(panel_card)

    desc_entries = [
        ("1. Sub-Band Vector:", r"$\mathbf{f}_b = [m_b, c_b, P_{\mathrm{mid},b}]^\top$"),
        ("2. Channel Vector:", r"$\mathbf{f}_c = [\mathbf{f}_\delta^\top, \mathbf{f}_\theta^\top, \mathbf{f}_\alpha^\top, \mathbf{f}_\beta^\top]^\top$"),
        ("3. Channel Dim:", r"$\mathrm{dim}(\mathbf{f}_c) = 4 \times 3 = 12\mathrm{D}$"),
        ("4. Multichannel:", r"$\mathbf{x}_{\mathrm{slope}} = [\mathbf{f}_1^\top, \dots, \mathbf{f}_C^\top]^\top$"),
    ]
    y_pos = 0.90
    for d_title, d_eq in desc_entries:
        ax5.text(0.04, y_pos, d_title, fontsize=5.8, fontweight='bold', color=THEME['text_dark'], zorder=3)
        ax5.text(0.08, y_pos - 0.076, d_eq, fontsize=5.8, color=THEME['text_dark'], zorder=3)
        y_pos -= 0.185

    box_total = FancyBboxPatch((0.03, 0.03), 0.94, 0.16, boxstyle="round,pad=0.008,rounding_size=0.012",
                               facecolor=THEME['panel_box_bg'], edgecolor=THEME['panel_box_border'], lw=0.8, zorder=2)
    ax5.add_patch(box_total)
    ax5.text(0.5, 0.11, r"$\mathbf{x}_{\mathrm{slope}} \in \mathbb{R}^{C \times 12}$" + "\n" + r"($C=18 \rightarrow 216\text{D})$",
             ha='center', va='center', fontsize=6.0, fontweight='bold', color=THEME['text_dark'], zorder=3)

    save_fig_dual(fig, "fig_methodology_spectral_slope")

# ==============================================================================
# 2. CLASSIC JOURNAL MULTI-PANEL: DWT (CENTER ARROWS, TREE OMITTED)
# ==============================================================================
def make_dwt_diagram(t, sig, fs):
    print("Generating Classic Journal Multi-Panel: DWT (Center Arrows, Reduced Box Sizes)...")
    fig = plt.figure(figsize=(13.4, 3.6), facecolor=THEME['bg'])
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.axis('off')
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)

    draw_outer_enclosure_box(ax_bg)

    pw = 0.150
    ph = 0.54
    y_ax = 0.16
    y_hdr = 0.81
    y_center = y_ax + ph / 2.0  # Exactly at vertical center (0.43)
    xs = [0.050, 0.301, 0.552, 0.803]

    # 4 Panel Headers
    draw_panel_header(ax_bg, xs[0], y_hdr, "a", r"Raw EEG Sequence $x[n]$")
    draw_panel_header(ax_bg, xs[1], y_hdr, "b", r"Wavelet Sub-Bands ($A_3, D_3, D_2, D_1$)")
    draw_panel_header(ax_bg, xs[2], y_hdr, "c", r"6 Non-Linear Descriptors")
    draw_panel_header(ax_bg, xs[3], y_hdr, "d", r"Wavelet Tensor $\mathbf{x}_{\mathrm{DWT}}$")

    # 3 Pipeline Connectors at VERTICAL CENTER
    draw_center_pipeline_arrow(ax_bg, xs[0] + pw + 0.015, y_center, xs[1] - 0.015, y_center, "db4 DWT")
    draw_center_pipeline_arrow(ax_bg, xs[1] + pw + 0.015, y_center, xs[2] - 0.012, y_center, "Extract")
    draw_center_pipeline_arrow(ax_bg, xs[2] + pw + 0.015, y_center, xs[3] - 0.012, y_center, "Concatenate")

    # (a) Raw EEG Sequence
    ax1 = fig.add_axes([xs[0], y_ax, pw, ph])
    setup_boxed_spines(ax1)
    ax1.plot(t, sig, color='#1E293B', lw=1.1)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Sample Index n", fontsize=6.8, labelpad=2)
    ax1.set_ylabel("Amplitude x[n]", fontsize=6.8, labelpad=2)
    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_xticklabels(["0", "N/2", "N"])
    ax1.grid(True, ls='--', lw=0.4, color=THEME['grid'], alpha=0.8)

    # (b) Decomposed Sub-Bands
    coeffs = pywt.wavedec(sig, 'db4', level=3)  # A3, D3, D2, D1
    ax2 = fig.add_axes([xs[1], y_ax, pw, ph])
    setup_boxed_spines(ax2)

    traces_dwt = [
        ('A3', coeffs[0], THEME['c_blue'], THEME['b_blue_bg'], THEME['b_blue_bd'], THEME['b_blue_tx']),
        ('D3', coeffs[1], THEME['c_teal'], THEME['b_teal_bg'], THEME['b_teal_bd'], THEME['b_teal_tx']),
        ('D2', coeffs[2], THEME['c_amber'], THEME['b_amber_bg'], THEME['b_amber_bd'], THEME['b_amber_tx']),
        ('D1', coeffs[3], THEME['c_crimson'], THEME['b_crimson_bg'], THEME['b_crimson_bd'], THEME['b_crimson_tx']),
    ]
    y_offsets = [9.0, 6.0, 3.0, 0.0]
    for idx, (tname, tdata, tcol, b_bg, b_bd, b_tx) in enumerate(traces_dwt):
        norm_trace = (tdata / (np.max(np.abs(tdata)) + 1e-6)) * 0.70 + y_offsets[idx]
        x_wave = np.linspace(24, 98, len(tdata))
        ax2.plot(x_wave, norm_trace, color=tcol, lw=1.0, zorder=2)
        ax2.axhline(y_offsets[idx], color=THEME['grid'], ls=':', lw=0.6, zorder=1)
        ax2.text(2, y_offsets[idx], tname, color=b_tx, fontsize=6.5, fontweight='bold', ha='left', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=b_bg, edgecolor=b_bd, lw=0.6), zorder=5)

    ax2.set_xlim(0, 100)
    ax2.set_ylim(-1.5, 10.8)
    ax2.set_xlabel("Coefficient Samples", fontsize=6.8, labelpad=2)
    ax2.set_yticks([])
    ax2.set_xticks([])

    # (c) 6 Non-Linear Descriptors
    ax3 = fig.add_axes([xs[2], y_ax, pw, ph])
    ax3.axis('off')
    card_c = FancyBboxPatch((0.005, 0.005), 0.990, 0.990, boxstyle="round,pad=0.005,rounding_size=0.015",
                            facecolor='#FFFFFF', edgecolor=THEME['box_spine'], lw=0.8, zorder=1)
    ax3.add_patch(card_c)

    desc_list = [
        ("1. Energy:", r"$E_j = \sum |c_{j,k}|^2$"),
        ("2. Std Dev:", r"$\sigma_j = \mathrm{std}(c_j)$"),
        ("3. Skewness:", r"$S_j = \mathbb{E}[(c-\mu)^3]/\sigma^3$"),
        ("4. Kurtosis:", r"$K_j = \mathbb{E}[(c-\mu)^4]/\sigma^4$"),
        ("5. Wiener Ent:", r"$H_j = \frac{\exp(\mathbb{E}[\ln c^2])}{\mathbb{E}[c^2]}$"),
        ("6. PSD RMS:", r"$\mathrm{RMS}_j = \sqrt{\frac{1}{M}\sum P_j(f)}$"),
    ]
    y_d = 0.90
    for d_name, d_eq in desc_list:
        ax3.text(0.04, y_d, d_name, fontsize=6.6, fontweight='bold', color=THEME['text_dark'], zorder=3)
        ax3.text(0.46, y_d, d_eq, fontsize=6.4, color=THEME['text_dark'], zorder=3)
        y_d -= 0.150

    # (d) Wavelet Vector Assembly
    ax4 = fig.add_axes([xs[3], y_ax, pw, ph])
    ax4.axis('off')
    card_d = FancyBboxPatch((0.005, 0.005), 0.990, 0.990, boxstyle="round,pad=0.005,rounding_size=0.015",
                            facecolor='#FFFFFF', edgecolor=THEME['box_spine'], lw=0.8, zorder=1)
    ax4.add_patch(card_d)

    desc_entries_dwt = [
        ("1. Sub-Band Vector:", r"$\mathbf{d}_j = [E_j, \sigma_j, S_j, K_j, H_j, \mathrm{RMS}_j]^\top$"),
        ("2. Channel Vector:", r"$\mathbf{d}_c = [\mathbf{d}_{A3}^\top, \mathbf{d}_{D3}^\top, \mathbf{d}_{D2}^\top, \mathbf{d}_{D1}^\top]^\top$"),
        ("3. Channel Dim:", r"$\mathrm{dim}(\mathbf{d}_c) = 4 \times 6 = 24\mathrm{D}$"),
        ("4. Multichannel:", r"$\mathbf{x}_{\mathrm{DWT}} = [\mathbf{d}_1^\top, \dots, \mathbf{d}_C^\top]^\top$"),
    ]
    y_pos = 0.90
    for d_title, d_eq in desc_entries_dwt:
        ax4.text(0.04, y_pos, d_title, fontsize=6.5, fontweight='bold', color=THEME['text_dark'], zorder=3)
        ax4.text(0.08, y_pos - 0.076, d_eq, fontsize=6.3, color=THEME['text_dark'], zorder=3)
        y_pos -= 0.185

    box_dwt = FancyBboxPatch((0.03, 0.03), 0.94, 0.16, boxstyle="round,pad=0.008,rounding_size=0.012",
                             facecolor=THEME['panel_box_bg'], edgecolor=THEME['panel_box_border'], lw=0.8, zorder=2)
    ax4.add_patch(box_dwt)
    ax4.text(0.5, 0.11, r"$\mathbf{x}_{\mathrm{DWT}} \in \mathbb{R}^{C \times 24}$" + "\n" + r"($C=18 \rightarrow 432\text{D})$",
             ha='center', va='center', fontsize=6.6, fontweight='bold', color=THEME['text_dark'], zorder=3)

    save_fig_dual(fig, "fig_methodology_dwt")

# ==============================================================================
# 3. CLASSIC JOURNAL MULTI-PANEL: EMD (CENTER ARROWS)
# ==============================================================================
def make_emd_diagram(t, sig, fs):
    print("Generating Classic Journal Multi-Panel: EMD (Center Arrows, Reduced Box Sizes)...")
    fig = plt.figure(figsize=(14.4, 3.6), facecolor=THEME['bg'])
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.axis('off')
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)

    draw_outer_enclosure_box(ax_bg)

    pw = 0.115
    ph = 0.54
    y_ax = 0.16
    y_hdr = 0.81
    y_center = y_ax + ph / 2.0  # Exactly at vertical center (0.43)
    xs = [0.040, 0.243, 0.446, 0.649, 0.852]

    # Panel Headers
    draw_panel_header(ax_bg, xs[0], y_hdr, "a", r"Raw Non-Stationary EEG $x(t)$")
    draw_panel_header(ax_bg, xs[1], y_hdr, "b", r"Cubic Spline Sifting Loop")
    draw_panel_header(ax_bg, xs[2], y_hdr, "c", r"Intrinsic Modes (IMF 1–4)")
    draw_panel_header(ax_bg, xs[3], y_hdr, "d", r"6 Non-Linear Descriptors")
    draw_panel_header(ax_bg, xs[4], y_hdr, "e", r"EMD Tensor $\mathbf{x}_{\mathrm{EMD}}$")

    # Pipeline Connectors at VERTICAL CENTER
    draw_center_pipeline_arrow(ax_bg, xs[0] + pw + 0.012, y_center, xs[1] - 0.024, y_center, "Extrema")
    draw_center_pipeline_arrow(ax_bg, xs[1] + pw + 0.012, y_center, xs[2] - 0.012, y_center, "Sift Modes")
    draw_center_pipeline_arrow(ax_bg, xs[2] + pw + 0.012, y_center, xs[3] - 0.012, y_center, "Compute")
    draw_center_pipeline_arrow(ax_bg, xs[3] + pw + 0.012, y_center, xs[4] - 0.012, y_center, "Concatenate")

    # (a) Raw Signal
    ax1 = fig.add_axes([xs[0], y_ax, pw, ph])
    setup_boxed_spines(ax1)
    ax1.plot(t, sig, color='#1E293B', lw=1.1)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Time t (s)", fontsize=6.8, labelpad=2)
    ax1.set_ylabel(r"Amplitude ($\mu\mathrm{V}$)", fontsize=6.8, labelpad=2)
    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_xticklabels(["0", "0.5", "1.0"])
    ax1.grid(True, ls='--', lw=0.4, color=THEME['grid'], alpha=0.8)

    # (b) Sifting Loop
    ax2 = fig.add_axes([xs[1], y_ax, pw, ph])
    setup_boxed_spines(ax2)

    sub_n = 70
    t_sub = t[:sub_n] * 1000
    s_sub = sig[:sub_n]
    max_idx = (np.diff(np.sign(np.diff(s_sub))) < 0).nonzero()[0] + 1
    min_idx = (np.diff(np.sign(np.diff(s_sub))) > 0).nonzero()[0] + 1
    max_x = np.concatenate(([t_sub[0]], t_sub[max_idx], [t_sub[-1]]))
    max_y = np.concatenate(([s_sub[0]], s_sub[max_idx], [s_sub[-1]]))
    min_x = np.concatenate(([t_sub[0]], t_sub[min_idx], [t_sub[-1]]))
    min_y = np.concatenate(([s_sub[0]], s_sub[min_idx], [s_sub[-1]]))
    cs_up = CubicSpline(max_x, max_y, bc_type='natural')
    cs_low = CubicSpline(min_x, min_y, bc_type='natural')
    t_f = np.linspace(t_sub[0], t_sub[-1], 150)
    e_up, e_low = cs_up(t_f), cs_low(t_f)
    e_m = (e_up + e_low) / 2.0

    ax2.plot(t_sub, s_sub, color='#0F172A', lw=1.2, label=r'$x(t)$')
    ax2.scatter(t_sub[max_idx], s_sub[max_idx], color=THEME['c_crimson'], s=11, zorder=5)
    ax2.scatter(t_sub[min_idx], s_sub[min_idx], color=THEME['c_blue'], s=11, zorder=5)
    ax2.plot(t_f, e_up, color=THEME['c_crimson'], lw=1.0, ls='--', label=r'$e_{\mathrm{up}}(t)$')
    ax2.plot(t_f, e_low, color=THEME['c_blue'], lw=1.0, ls='--', label=r'$e_{\mathrm{low}}(t)$')
    ax2.plot(t_f, e_m, color=THEME['c_amber'], lw=1.2, ls='-', label=r'$m(t)$')

    ax2.set_xlim(0, t_sub[-1])
    ax2.set_ylim(-5.0, 6.8)
    ax2.set_xlabel("Time (ms)", fontsize=6.8, labelpad=2)
    ax2.text(-0.06, 1.03, r"Amplitude ($\mu\mathrm{V}$)", transform=ax2.transAxes,
             fontsize=6.5, ha='left', va='bottom', color=THEME['text_dark'])
    ax2.grid(True, ls='--', lw=0.4, color=THEME['grid'], alpha=0.8)
    ax2.legend(loc='upper right', fontsize=5.0, framealpha=0.95, edgecolor='#CBD5E1', handlelength=1.0, borderpad=0.2)

    # (c) Decomposed IMFs
    emd_obj = PyEMD.EMD()
    imfs = emd_obj(sig, max_imf=4)
    if imfs.shape[0] < 4:
        imfs = np.pad(imfs, ((0, 4 - imfs.shape[0]), (0, 0)), mode='edge')

    ax3 = fig.add_axes([xs[2], y_ax, pw, ph])
    setup_boxed_spines(ax3)

    traces_emd = [
        ('IMF 1', imfs[0], THEME['c_crimson'], THEME['b_crimson_bg'], THEME['b_crimson_bd'], THEME['b_crimson_tx']),
        ('IMF 2', imfs[1], THEME['c_amber'], THEME['b_amber_bg'], THEME['b_amber_bd'], THEME['b_amber_tx']),
        ('IMF 3', imfs[2], THEME['c_teal'], THEME['b_teal_bg'], THEME['b_teal_bd'], THEME['b_teal_tx']),
        ('IMF 4', imfs[3], THEME['c_blue'], THEME['b_blue_bg'], THEME['b_blue_bd'], THEME['b_blue_tx']),
    ]
    y_imf_offsets = [9.0, 6.0, 3.0, 0.0]
    for idx, (iname, idata, icol, b_bg, b_bd, b_tx) in enumerate(traces_emd):
        norm_imf = (idata / (np.max(np.abs(idata)) + 1e-6)) * 0.70 + y_imf_offsets[idx]
        x_wave = np.linspace(26, 98, len(idata))
        ax3.plot(x_wave, norm_imf, color=icol, lw=1.0, zorder=2)
        ax3.axhline(y_imf_offsets[idx], color=THEME['grid'], ls=':', lw=0.6, zorder=1)
        ax3.text(2, y_imf_offsets[idx], iname, color=b_tx, fontsize=6.5, fontweight='bold', ha='left', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=b_bg, edgecolor=b_bd, lw=0.6), zorder=5)

    ax3.set_xlim(0, 100)
    ax3.set_ylim(-1.5, 10.8)
    ax3.set_xlabel("Time Samples", fontsize=6.8, labelpad=2)
    ax3.set_yticks([])
    ax3.set_xticks([])

    # (d) 6 Non-Linear Descriptors
    ax4 = fig.add_axes([xs[3], y_ax, pw, ph])
    ax4.axis('off')
    card_d = FancyBboxPatch((0.005, 0.005), 0.990, 0.990, boxstyle="round,pad=0.005,rounding_size=0.015",
                            facecolor='#FFFFFF', edgecolor=THEME['box_spine'], lw=0.8, zorder=1)
    ax4.add_patch(card_d)

    desc_list_emd = [
        ("1. Energy:", r"$E_k = \sum |\mathrm{IMF}_k|^2$"),
        ("2. Std Dev:", r"$\sigma_k = \mathrm{std}(\mathrm{IMF}_k)$"),
        ("3. Skewness:", r"$S_k = \mathbb{E}[(\mathrm{IMF}-\mu)^3]/\sigma^3$"),
        ("4. Kurtosis:", r"$K_k = \mathbb{E}[(\mathrm{IMF}-\mu)^4]/\sigma^4$"),
        ("5. Wiener Ent:", r"$H_k = \frac{\exp(\mathbb{E}[\ln \mathrm{IMF}^2])}{\mathbb{E}[\mathrm{IMF}^2]}$"),
        ("6. PSD RMS:", r"$\mathrm{RMS}_k = \sqrt{\frac{1}{M}\sum P_k(f)}$"),
    ]
    y_d = 0.90
    for d_name, d_eq in desc_list_emd:
        ax4.text(0.04, y_d, d_name, fontsize=5.8, fontweight='bold', color=THEME['text_dark'], zorder=3)
        ax4.text(0.42, y_d, d_eq, fontsize=5.6, color=THEME['text_dark'], zorder=3)
        y_d -= 0.150

    # (e) EMD Vector Assembly
    ax5 = fig.add_axes([xs[4], y_ax, pw, ph])
    ax5.axis('off')
    card_e = FancyBboxPatch((0.005, 0.005), 0.990, 0.990, boxstyle="round,pad=0.005,rounding_size=0.015",
                            facecolor='#FFFFFF', edgecolor=THEME['box_spine'], lw=0.8, zorder=1)
    ax5.add_patch(card_e)

    desc_entries_emd = [
        ("1. Mode Vector:", r"$\mathbf{e}_k = [E_k, \sigma_k, S_k, K_k, H_k, \mathrm{RMS}_k]^\top$"),
        ("2. Channel Vector:", r"$\mathbf{e}_c = [\mathbf{e}_1^\top, \mathbf{e}_2^\top, \mathbf{e}_3^\top, \mathbf{e}_4^\top]^\top$"),
        ("3. Channel Dim:", r"$\mathrm{dim}(\mathbf{e}_c) = 4 \times 6 = 24\mathrm{D}$"),
        ("4. Multichannel:", r"$\mathbf{x}_{\mathrm{EMD}} = [\mathbf{e}_1^\top, \dots, \mathbf{e}_C^\top]^\top$"),
    ]
    y_pos = 0.90
    for d_title, d_eq in desc_entries_emd:
        ax5.text(0.04, y_pos, d_title, fontsize=5.8, fontweight='bold', color=THEME['text_dark'], zorder=3)
        ax5.text(0.08, y_pos - 0.076, d_eq, fontsize=5.8, color=THEME['text_dark'], zorder=3)
        y_pos -= 0.185

    box_emd = FancyBboxPatch((0.03, 0.03), 0.94, 0.16, boxstyle="round,pad=0.008,rounding_size=0.012",
                             facecolor=THEME['panel_box_bg'], edgecolor=THEME['panel_box_border'], lw=0.8, zorder=2)
    ax5.add_patch(box_emd)
    ax5.text(0.5, 0.11, r"$\mathbf{x}_{\mathrm{EMD}} \in \mathbb{R}^{C \times 24}$" + "\n" + r"($C=18 \rightarrow 432\text{D})$",
             ha='center', va='center', fontsize=6.0, fontweight='bold', color=THEME['text_dark'], zorder=3)

    save_fig_dual(fig, "fig_methodology_emd")

def main():
    print("=" * 80)
    print("Generating Formal Classic Journal Multi-Panel Methodology Figures (Reduced Boxes, Center Arrows)...")
    print("=" * 80)
    t, sig, fs = load_data()
    make_spectral_slope_diagram(t, sig, fs)
    make_dwt_diagram(t, sig, fs)
    make_emd_diagram(t, sig, fs)
    print("\n" + "=" * 80)
    print("All 3 classic journal figures generated successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()

"""
Legacy 2.0 - 30-Minute Telegram Background Monitor Daemon
=========================================================
Runs in background and sends regular 30-minute status updates to Telegram:
- Current pipeline phase (Feature Extraction / Model Training / Ablation / Figures)
- Extraction progress (X / 39 patients completed)
- Training progress (X / 10 trials completed per paradigm)
- GPU VRAM, utilization, and temperature
- Free disk space on D:
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = ROOT_DIR / "features" / "spectral_slope"
RESULTS_DIR = ROOT_DIR / "results"
ANALYSIS_DIR = ROOT_DIR / "analysis"

from telegram_notifier import send_telegram, get_gpu_info


def get_disk_free():
    try:
        import shutil
        usage = shutil.disk_usage(r"D:\\")
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0


def get_pipeline_status():
    extracted = len(list(FEATURES_DIR.glob("patient_*.csv"))) if FEATURES_DIR.exists() else 0
    results_files = list(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
    
    if extracted < 39:
        phase = "Feature Extraction (Step 1/6)"
        progress = f"{extracted} / 39 patients ({extracted/39*100:.1f}%)"
    elif not results_files:
        phase = "Model Training & 10-Trial CV (Step 2/6)"
        progress = "39/39 extracted. Training initializing..."
    else:
        phase = "Training / Evaluation Active"
        progress = f"{len(results_files)} paradigm/strategy result logs saved"

    return phase, progress, extracted


def build_30min_update():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    phase, progress, extracted = get_pipeline_status()
    gpu = get_gpu_info()
    disk = get_disk_free()

    gpu_str = "N/A"
    if gpu:
        gpu_str = f"RTX A5000 | `{gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.1f} GB` | `{gpu['temp']}°C` ({gpu['util']}% util)"

    msg = (
        f"🤖 *Legacy 2.0: 30-Minute Periodic Update*\n"
        f"🕒 *Time*: `{now_str}`\n\n"
        f"📊 *Current Phase*: `{phase}`\n"
        f"📈 *Progress*: *{progress}*\n"
        f"🔬 *Feature Cache*: `{extracted} patients completed`\n\n"
        f"🖥️ *GPU*: {gpu_str}\n"
        f"💿 *Disk Free*: `{disk:.1f} GB free` on D:\n\n"
        f"⏱️ Next automated update in 30 minutes."
    )
    return msg


def main():
    print("Starting 30-minute Telegram monitor daemon...")
    # Send immediate initial status
    msg = build_30min_update()
    send_telegram(msg)
    print("Initial 30-minute update sent to Telegram.")

    # 30-minute loop (1800 seconds)
    interval = 1800
    while True:
        time.sleep(interval)
        try:
            msg = build_30min_update()
            send_telegram(msg)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 30-min update sent successfully.")
        except Exception as e:
            print(f"Error sending update: {e}")


if __name__ == '__main__':
    main()

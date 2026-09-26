"""
Legacy 2.0 - 30-Minute Telegram Polling Daemon
=============================================
Polls pipeline progress, GPU telemetry, and artifact generation every 30 minutes
and dispatches rich status reports to Telegram until all steps are complete.
"""

import time
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
ANALYSIS_DIR = ROOT_DIR / "analysis"
TABLES_DIR = ROOT_DIR / "tables"
FIGURES_DIR = ROOT_DIR / "figures"
LOG_PATH = Path(r"C:\Users\Machine Learning GPU\.gemini\antigravity\brain\5f2ca0ca-b991-406a-8497-e74554a812fd\.system_generated\tasks\task-1205.log")

from telegram_notifier import send_telegram, get_gpu_info


def inspect_pipeline_status():
    status = {
        'last_log_line': 'No log available',
        'active_step': 'Unknown',
        'is_finished': False,
        'completed_results': 0,
        'figures_count': 0,
        'tables_count': 0
    }

    # Count saved results
    if RESULTS_DIR.exists():
        status['completed_results'] = len(list(RESULTS_DIR.glob("*_cv_results.json")))

    if FIGURES_DIR.exists():
        status['figures_count'] = len(list(FIGURES_DIR.glob("*.png")))

    if TABLES_DIR.exists():
        status['tables_count'] = len(list(TABLES_DIR.glob("*.*")))

    # Read latest log lines
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if lines:
                    status['last_log_line'] = lines[-1]
                    for l in reversed(lines):
                        if "ALL LEGACY 2.0 PIPELINE STEPS COMPLETED" in l:
                            status['is_finished'] = True
                            status['active_step'] = 'ALL STEPS COMPLETE'
                            break
                        elif "STEP 6:" in l:
                            status['active_step'] = 'Step 6: Publication Figures & Tables'
                            break
                        elif "STEP 5:" in l:
                            status['active_step'] = 'Step 5: Explainability (CCA, SHAP)'
                            break
                        elif "STEP 4:" in l:
                            status['active_step'] = 'Step 4: Sensitivity & Feature Ablation'
                            break
                        elif "STEP 3:" in l:
                            status['active_step'] = 'Step 3: Calibration & Bootstrap CIs'
                            break
                        elif "STEP 2:" in l or ">>> Running Paradigm:" in l or "Training 4 models" in l:
                            status['active_step'] = 'Step 2: 10-Trial Cross-Validation'
                            break
        except Exception as e:
            status['last_log_line'] = f"Error reading log: {e}"

    return status


def send_poll_report(poll_num):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st = inspect_pipeline_status()
    gpu = get_gpu_info()

    gpu_str = "N/A"
    if gpu:
        gpu_str = f"RTX A5000 | `{gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.1f} GB` | `{gpu['temp']}°C` ({gpu['util']}% util)"

    if st['is_finished']:
        msg = (
            f"🎉 *LEGACY 2.0 PIPELINE COMPLETE!* 🎉\n"
            f"🕒 *Finished*: `{now_str}`\n\n"
            f"✅ *All 8 Paradigm x Strategy CV Sets Complete* (80/80 trials)\n"
            f"✅ *Step 3*: Bootstrap 95% CIs & Calibration Curves\n"
            f"✅ *Step 4*: Ratios, PCA & Frequency Feature Ablations\n"
            f"✅ *Step 5*: CCA & SHAP Explainability\n"
            f"✅ *Step 6*: {st['figures_count']} Figures & {st['tables_count']} Tables generated!\n\n"
            f"🖥️ *GPU*: {gpu_str}"
        )
    else:
        msg = (
            f"⏱️ *Legacy 2.0 Status Poll #{poll_num} (30-min)*\n"
            f"🕒 *Timestamp*: `{now_str}`\n\n"
            f"📌 *Active Phase*: `{st['active_step']}`\n"
            f"📊 *CV Result Files*: `{st['completed_results']}/8` Completed\n"
            f"📝 *Latest Output*: `{st['last_log_line'][-120:]}`\n\n"
            f"🖥️ *GPU*: {gpu_str}"
        )

    send_telegram(msg)
    return st['is_finished']


def main():
    poll_interval = 1800 # 30 minutes
    poll_count = 2

    while True:
        # Sleep in 60-second increments with heartbeat
        for _ in range(poll_interval // 60):
            time.sleep(60)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Poller active, waiting for next 30-min window...", flush=True)

        poll_count += 1
        is_done = send_poll_report(poll_count)
        if is_done:
            break


if __name__ == '__main__':
    main()

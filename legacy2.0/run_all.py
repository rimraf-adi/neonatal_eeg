"""
Legacy 2.0 - Master Pipeline Orchestrator
=========================================
Runs the entire reproduction and ablation pipeline end-to-end:
  Step 1: Multi-paradigm feature extraction across all 39 patients
  Step 2: 10-trial cross-validation across paradigms & classifiers (CUDA accelerated)
  Step 3: Calibration metrics & bootstrap confidence intervals
  Step 4: Seizure/non-seizure ratio & PCA dimensionality ablation sweeps
  Step 5: Canonical Correlation Analysis (CCA) & SHAP explainability
  Step 6: Master publication figures and supplementary tables generation
"""

import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_step(step_num, title, script_name):
    print("\n" + "#" * 80)
    print(f"STEP {step_num}: {title.upper()}")
    print("#" * 80)
    script_path = BASE_DIR / script_name
    t0 = time.time()
    
    try:
        from telegram_notifier import notify_status
        notify_status(f"Starting Step {step_num}: {title}", {"Script": script_name})
    except Exception:
        pass

    # Execute script
    import subprocess
    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"ERROR: Step {step_num} ({script_name}) failed with return code {result.returncode}!")
        try:
            from telegram_notifier import notify_status
            notify_status(f"❌ FAILED Step {step_num}: {title}", {"Error Code": str(result.returncode)})
        except Exception:
            pass
        sys.exit(result.returncode)
    else:
        print(f"Step {step_num} completed successfully in {dt:.1f}s.")
        try:
            from telegram_notifier import notify_status
            notify_status(f"✅ Completed Step {step_num}: {title}", {"Elapsed": f"{dt/60:.2f} min"})
        except Exception:
            pass

def main():
    total_start = time.time()
    print("=" * 80)
    print("STARTING COMPLETE LEGACY 2.0 REPRODUCTION PIPELINE")
    print("=" * 80)
    
    try:
        from telegram_notifier import notify_status
        notify_status("🚀 Legacy 2.0 Pipeline Started", {
            "Paradigms": "Spectral Slope, Wavelet, EMD, Fusion",
            "Models": "Adaptive NN (CUDA), LogReg, RF, XGBoost"
        })
    except Exception:
        pass

    run_step(1, "Multi-Paradigm Feature Extraction", "extract_features.py")
    run_step(2, "10-Trial Cross-Validation & Model Training", "train_models.py")
    run_step(3, "Calibration Metrics & Bootstrap CIs", "calibration_and_ci.py")
    run_step(4, "Ratio & PCA Sensitivity Ablation", "sensitivity_ablation.py")
    run_step(5, "Explainability (CCA, PCA Loadings, SHAP)", "explainability_cca.py")
    run_step(6, "Publication Figures & Supplementary Tables", "generate_figures_and_tables.py")
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"ALL LEGACY 2.0 PIPELINE STEPS COMPLETED IN {total_elapsed/60:.2f} MINUTES!")
    print("=" * 80)

    try:
        from telegram_notifier import notify_status
        notify_status("🎉 All Legacy 2.0 Experiments Completed!", {
            "Total Time": f"{total_elapsed/60:.2f} min",
            "Figures": "Saved to legacy2.0/figures/",
            "Tables": "Saved to legacy2.0/tables/"
        })
    except Exception:
        pass

if __name__ == '__main__':
    main()

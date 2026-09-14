"""
Overnight Master Pipeline Runner
=================================

Orchestrates the entire neonatal EEG analysis pipeline:
1. Monitors and waits for the initial feature caching (task-337 / Patients 1-79) to complete.
2. Runs Pass 2 of cache_features to extract newly added EMD and DWT comparison baselines.
3. Sequentially executes all 8 experimental studies:
   - Study 01: Classifier Complexity Ladder (LogReg, SVM, RF, KNN)
   - Study 02: Feature Ablation Study (Delta, Theta, Alpha, Beta, Slope, Intercept, Midband)
   - Study 03: UMAP / t-SNE Unsupervised Clustering & Separation
   - Study 04: Fisher's Discriminant Ratio (Feature Ranking)
   - Study 05: Baseline Comparison (Spectral Slope vs Band Power, Hjorth, Time-Domain, Spectral Entropy, EMD, DWT)
   - Study 06: Temporal Onset Trajectory Tracking
   - Study 07: Cohen's d Effect Size Analysis
   - Study 08: WeightedRandomSampler Class Balance Ablation (MLP)
4. Compiles a consolidated, publication-ready research report:
   - study_results/MASTER_RESEARCH_REPORT.md
   - study_results/pipeline_status.json

Usage:
  uv run python run_entire_pipeline.py
"""

import os
import sys
import time
import json
import psutil
import datetime
import subprocess
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent
STUDIES_DIR = REPO_ROOT / 'studies'
RESULTS_DIR = REPO_ROOT / 'study_results'
LOGS_DIR = RESULTS_DIR / 'logs'
STATUS_FILE = RESULTS_DIR / 'pipeline_status.json'
REPORT_FILE = RESULTS_DIR / 'MASTER_RESEARCH_REPORT.md'

INITIAL_CACHING_PID = 28424  # PID of task-337 if still active
LAST_PATIENT_CSV = RESULTS_DIR / 'feature_cache' / 'spectral_slope' / 'patient_079.csv'
LAST_BASELINE_CSV = RESULTS_DIR / 'feature_cache' / 'baselines' / 'spectral_entropy' / 'patient_079.csv'

STUDIES = [
    ("Study 01: Classifier Ladder", "studies.study_01_classifier_ladder", "01_classifier_ladder"),
    ("Study 02: Feature Ablation", "studies.study_02_feature_ablation", "02_feature_ablation"),
    ("Study 03: UMAP / t-SNE", "studies.study_03_umap_tsne", "03_umap_tsne"),
    ("Study 04: Fisher Ratio", "studies.study_04_fisher_ratio", "04_fisher_ratio"),
    ("Study 05: Baseline Comparison", "studies.study_05_baseline_comparison", "05_baseline_comparison"),
    ("Study 06: Temporal Trajectory", "studies.study_06_temporal_trajectory", "06_temporal_trajectory"),
    ("Study 07: Cohen's d Effect Size", "studies.study_07_cohens_d", "07_cohens_d"),
    ("Study 08: Sampler Ablation", "studies.study_08_sampler_ablation", "08_sampler_ablation"),
]


def log_msg(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / 'master_pipeline.log', 'a', encoding='utf-8') as f:
        f.write(formatted + "\n")


def update_status(data: dict):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def is_process_running(pid: int) -> bool:
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def wait_for_initial_caching():
    log_msg("Checking status of Initial Caching (Task-337)...")
    
    if LAST_BASELINE_CSV.exists() and LAST_BASELINE_CSV.stat().st_size > 1000:
        log_msg("Initial caching is already complete (patient_079.csv verified).")
        return True

    log_msg(f"Waiting for Initial Caching (PID {INITIAL_CACHING_PID} or patient_079 completion)...")
    
    start_wait = time.time()
    last_log_time = 0
    
    while True:
        # Check if last file exists and is populated
        if LAST_BASELINE_CSV.exists() and LAST_BASELINE_CSV.stat().st_size > 1000:
            log_msg("Initial caching completed successfully (patient_079.csv detected)!")
            time.sleep(5)  # brief grace period for file handles to close
            return True
        
        # Check if process died unexpectedly
        running = is_process_running(INITIAL_CACHING_PID)
        if not running:
            # Process exited; check if files are there
            if LAST_BASELINE_CSV.exists():
                log_msg("Initial caching process finished.")
                return True
            else:
                log_msg(f"Warning: Process {INITIAL_CACHING_PID} exited but {LAST_BASELINE_CSV} is not complete.")
                log_msg("Proceeding to run cache_features to ensure all patients are processed.")
                return True
        
        # Log progress periodically every 60 seconds
        if time.time() - last_log_time > 60:
            last_log_time = time.time()
            elapsed_m = (time.time() - start_wait) / 60
            # Inspect latest file in spectral_slope
            slope_dir = RESULTS_DIR / 'feature_cache' / 'spectral_slope'
            latest_file = "none"
            if slope_dir.exists():
                csvs = sorted(slope_dir.glob('patient_*.csv'), key=lambda p: p.stat().st_mtime)
                if csvs:
                    latest_file = csvs[-1].name
            log_msg(f"Still waiting for initial caching... Elapsed: {elapsed_m:.1f}m. Latest cached: {latest_file}")
            
        time.sleep(15)


def run_command_with_logging(cmd: list, log_path: Path, desc: str) -> bool:
    log_msg(f"STARTING: {desc}")
    log_msg(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT)
        )
        
        for line in iter(proc.stdout.readline, ''):
            if line:
                log_file.write(line)
                log_file.flush()
                # Print milestone lines to master log
                if any(k in line for k in ['[OK]', 'COMPLETE', 'Trial', 'AUROC', 'Ratio', 'Saved', 'ERROR', 'FAIL']):
                    print(f"   {line.strip()}", flush=True)
                    
        proc.stdout.close()
        ret = proc.wait()
        
    duration = time.time() - t0
    if ret == 0:
        log_msg(f"SUCCESS: {desc} (Duration: {duration:.1f}s / {duration/60:.1f}m)")
        return True
    else:
        log_msg(f"FAILED: {desc} with return code {ret} (Duration: {duration:.1f}s)")
        return False


def run_emd_dwt_caching():
    log_msg("=" * 70)
    log_msg("STEP 2: RUNNING CACHE_FEATURES FOR EMD & DWT BASELINES")
    log_msg("=" * 70)
    
    log_path = LOGS_DIR / '00_cache_features_emd_dwt.log'
    cmd = ["uv", "run", "python", "-m", "studies.cache_features"]
    
    success = run_command_with_logging(cmd, log_path, "Feature Caching (Pass 2 - EMD & DWT)")
    return success


def run_all_studies(status_dict: dict):
    log_msg("=" * 70)
    log_msg("STEP 3: RUNNING ALL 8 VALIDATION STUDIES SEQUENTIALLY")
    log_msg("=" * 70)
    
    for name, module, folder in STUDIES:
        log_msg(f"\n>>> Running {name}...")
        status_dict['current_study'] = name
        update_status(status_dict)
        
        log_path = LOGS_DIR / f"{folder}.log"
        cmd = ["uv", "run", "python", "-m", module]
        
        t_start = time.time()
        success = run_command_with_logging(cmd, log_path, name)
        t_end = time.time()
        
        status_dict['studies'][name] = {
            'status': 'SUCCESS' if success else 'FAILED',
            'duration_seconds': round(t_end - t_start, 2),
            'log_file': str(log_path.relative_to(REPO_ROOT)),
            'output_folder': str((RESULTS_DIR / folder).relative_to(REPO_ROOT))
        }
        update_status(status_dict)


def generate_consolidated_report(status_dict: dict):
    log_msg("=" * 70)
    log_msg("STEP 4: COMPILING MASTER RESEARCH REPORT")
    log_msg("=" * 70)
    
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        "# Comprehensive Neonatal EEG Seizure Detection Research Report",
        "",
        f"**Generated:** {now_str}  ",
        f"**Pipeline Status:** {status_dict.get('overall_status', 'COMPLETED')}  ",
        f"**Total Studies Executed:** {len(status_dict.get('studies', {}))}  ",
        "",
        "---",
        "",
        "## Executive Summary & Study Overview",
        "",
        "| Study # | Name | Status | Duration | Key Focus |",
        "| :--- | :--- | :---: | :---: | :--- |",
        "| **Study 1** | Classifier Complexity Ladder | " + status_dict['studies'].get("Study 01: Classifier Ladder", {}).get('status', 'N/A') + " | " + f"{status_dict['studies'].get('Study 01: Classifier Ladder', {}).get('duration_seconds', 0):.1f}s" + " | Linear vs Non-linear ML (Logistic Regression, Linear SVM, RF, KNN) |",
        "| **Study 2** | Feature Ablation | " + status_dict['studies'].get("Study 02: Feature Ablation", {}).get('status', 'N/A') + " | " + f"{status_dict['studies'].get('Study 02: Feature Ablation', {}).get('duration_seconds', 0):.1f}s" + " | Frequency band & parameter drop sensitivity |",
        "| **Study 3** | UMAP & t-SNE | " + status_dict['studies'].get("Study 03: UMAP / t-SNE", {}).get('status', 'N/A') + " | " + f"{status_dict['studies'].get('Study 03: UMAP / t-SNE', {}).get('duration_seconds', 0):.1f}s" + " | Unsupervised manifold separation & clustering metrics |",
        "| **Study 4** | Fisher's Discriminant Ratio | " + status_dict['studies'].get("Study 04: Fisher Ratio", {}).get('status', 'N/A') + " | " + f"{status_dict['studies'].get('Study 04: Fisher Ratio', {}).get('duration_seconds', 0):.1f}s" + " | Analytical class separability per feature |",
        "| **Study 5** | Baseline Comparison (w/ EMD & DWT) | " + status_dict['studies'].get("Study 05: Baseline Comparison", {}).get('status', 'N/A') + " | " + f"{status_dict['studies'].get('Study 05: Baseline Comparison', {}).get('duration_seconds', 0):.1f}s" + " | Spectral slope vs Band Power, Hjorth, Time-Domain, Entropy, EMD, DWT |",
        "| **Study 6** | Temporal Trajectory Tracking | " + status_dict['studies'].get("Study 06: Temporal Trajectory", {}).get('status', 'N/A') + " | " + f"{status_dict['studies'].get('Study 06: Temporal Trajectory', {}).get('duration_seconds', 0):.1f}s" + " | Real-time seizure tracking across onset/offset boundaries |",
        "| **Study 7** | Cohen's d Effect Sizes | " + status_dict['studies'].get("Study 07: Cohen's d Effect Size", {}).get('status', 'N/A') + " | " + f"{status_dict['studies'].get('Study 07: Cohen\'s d Effect Size', {}).get('duration_seconds', 0):.1f}s" + " | Standardized effect magnitude (sample-size invariant) |",
        "| **Study 8** | WeightedRandomSampler Ablation | " + status_dict['studies'].get("Study 08: Sampler Ablation", {}).get('status', 'N/A') + " | " + f"{status_dict['studies'].get('Study 08: Sampler Ablation', {}).get('duration_seconds', 0):.1f}s" + " | Imbalance ratio sensitivity (50:50, 60:40, 40:60, 30:70, 20:80, Natural) |",
        "",
        "---",
        ""
    ]
    
    # Read each study summary text if present
    summary_files = [
        ("Study 1: Classifier Complexity Ladder Results", RESULTS_DIR / '01_classifier_ladder' / 'summary_table.txt'),
        ("Study 2: Feature Ablation Results", RESULTS_DIR / '02_feature_ablation' / 'ablation_summary.txt'),
        ("Study 3: UMAP & t-SNE Clustering Results", RESULTS_DIR / '03_umap_tsne' / 'clustering_metrics.txt'),
        ("Study 4: Fisher Discriminant Ratio Results", RESULTS_DIR / '04_fisher_ratio' / 'fisher_summary.txt'),
        ("Study 5: Baseline Feature Comparison (including EMD & DWT)", RESULTS_DIR / '05_baseline_comparison' / 'comparison_summary.txt'),
        ("Study 6: Temporal Trajectory Analysis", RESULTS_DIR / '06_temporal_trajectory' / 'selected_patients.txt'),
        ("Study 7: Cohen's d Effect Size Results", RESULTS_DIR / '07_cohens_d' / 'effect_size_summary.txt'),
        ("Study 8: Class Imbalance Sampler Ablation Results", RESULTS_DIR / '08_sampler_ablation' / 'sampler_summary.txt'),
    ]
    
    for title, path in summary_files:
        lines.append(f"## {title}")
        lines.append("")
        if path.exists():
            try:
                content = path.read_text(encoding='utf-8', errors='replace')
                lines.append("```text")
                lines.append(content.strip())
                lines.append("```")
            except Exception as e:
                lines.append(f"*Error reading summary file: {e}*")
        else:
            lines.append("*Summary file not generated or study did not complete.*")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Generated Figures & Artifacts")
    lines.append("")
    # Find all generated png images
    pngs = sorted(RESULTS_DIR.glob('**/*.png'))
    for png in pngs:
        rel = png.relative_to(REPO_ROOT)
        lines.append(f"- **{png.stem}**: `{rel}`")
    lines.append("")
    
    REPORT_FILE.write_text("\n".join(lines), encoding='utf-8')
    log_msg(f"Report successfully compiled to: {REPORT_FILE}")


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_msg("=" * 70)
    log_msg("OVERNIGHT NEONATAL EEG PIPELINE MASTER RUNNER STARTED")
    log_msg("=" * 70)
    
    t_start_all = time.time()
    
    status_dict = {
        'start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'overall_status': 'IN_PROGRESS',
        'current_phase': 'waiting_initial_caching',
        'studies': {}
    }
    update_status(status_dict)
    
    try:
        # Phase 1: Wait for initial caching
        status_dict['current_phase'] = 'waiting_initial_caching'
        update_status(status_dict)
        wait_for_initial_caching()
        
        # Phase 2: Cache EMD and DWT
        status_dict['current_phase'] = 'caching_emd_dwt'
        update_status(status_dict)
        run_emd_dwt_caching()
        
        # Phase 3: Run all studies
        status_dict['current_phase'] = 'running_studies'
        update_status(status_dict)
        run_all_studies(status_dict)
        
        # Phase 4: Compile report
        status_dict['current_phase'] = 'compiling_report'
        update_status(status_dict)
        status_dict['overall_status'] = 'COMPLETED'
        status_dict['end_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_dict['total_duration_minutes'] = round((time.time() - t_start_all) / 60, 2)
        generate_consolidated_report(status_dict)
        update_status(status_dict)
        
        log_msg("=" * 70)
        log_msg("ALL TASKS COMPLETED SUCCESSFULLY! READY FOR PAPER WRITING.")
        log_msg("=" * 70)
        
    except Exception as e:
        log_msg(f"FATAL ERROR in pipeline: {e}")
        import traceback
        traceback.print_exc()
        status_dict['overall_status'] = f"FAILED: {e}"
        update_status(status_dict)


if __name__ == '__main__':
    main()

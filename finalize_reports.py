"""
Finalize Master Research Report and Pipeline Status across all 8 Studies.
"""
import os
import json
import datetime
from pathlib import Path

REPO_ROOT = Path("D:/neonatal")
RESULTS_DIR = REPO_ROOT / "study_results"
REPORT_FILE = RESULTS_DIR / "MASTER_RESEARCH_REPORT.md"
STATUS_FILE = RESULTS_DIR / "pipeline_status.json"

STUDY_INFO = [
    ("Study 1: Classifier Complexity Ladder", "01_classifier_ladder", "summary_table.txt", "Linear vs Non-linear ML (Logistic Regression, Linear SVM, RF, KNN)"),
    ("Study 2: Feature Ablation", "02_feature_ablation", "ablation_summary.txt", "Frequency band & parameter drop sensitivity"),
    ("Study 3: UMAP & t-SNE Clustering", "03_umap_tsne", "clustering_metrics.txt", "Unsupervised manifold separation & clustering metrics"),
    ("Study 4: Fisher Discriminant Ratio", "04_fisher_ratio", "fisher_summary.txt", "Analytical class separability per feature"),
    ("Study 5: Baseline Feature Comparison (w/ EMD & DWT)", "05_baseline_comparison", "comparison_summary.txt", "Spectral slope vs Band Power, Hjorth, Time-Domain, Entropy, EMD, DWT"),
    ("Study 6: Temporal Trajectory Analysis", "06_temporal_trajectory", "selected_patients.txt", "Real-time seizure tracking across onset/offset boundaries"),
    ("Study 7: Cohen's d Effect Sizes", "07_cohens_d", "effect_size_summary.txt", "Standardized effect magnitude (sample-size invariant)"),
    ("Study 8: Class Imbalance Sampler Ablation", "08_sampler_ablation", "sampler_summary.txt", "Imbalance ratio sensitivity (50:50, 60:40, 40:60, 30:70, 20:80, Natural)"),
]

def compile_master_report():
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check status of each study
    studies_status = {}
    for name, folder, summary_filename, description in STUDY_INFO:
        summary_path = RESULTS_DIR / folder / summary_filename
        success = summary_path.exists() and summary_path.stat().st_size > 50
        studies_status[name] = {
            'status': 'SUCCESS' if success else 'PENDING',
            'summary_path': summary_path,
            'folder': folder,
            'description': description
        }
        
    all_success = all(s['status'] == 'SUCCESS' for s in studies_status.values())
    
    lines = [
        "# Comprehensive Neonatal EEG Seizure Detection Research Report",
        "",
        f"**Generated:** {now_str}  ",
        f"**Pipeline Status:** {'COMPLETED' if all_success else 'IN_PROGRESS'}  ",
        f"**Total Studies Executed:** {len(STUDY_INFO)}  ",
        "",
        "---",
        "",
        "## Executive Summary & Study Overview",
        "",
        "| Study # | Name | Status | Key Focus |",
        "| :--- | :--- | :---: | :--- |",
    ]
    
    for idx, (name, folder, _, desc) in enumerate(STUDY_INFO, 1):
        status = studies_status[name]['status']
        lines.append(f"| **Study {idx}** | {name.split(': ')[-1]} | {status} | {desc} |")
        
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Append content of each study summary
    for name, folder, summary_filename, _ in STUDY_INFO:
        lines.append(f"## {name}")
        lines.append("")
        path = RESULTS_DIR / folder / summary_filename
        if path.exists() and path.stat().st_size > 0:
            try:
                content = path.read_text(encoding='utf-8', errors='replace')
                lines.append("```text")
                lines.append(content.strip())
                lines.append("```")
            except Exception as e:
                lines.append(f"*Error reading summary file: {e}*")
        else:
            lines.append("*Summary file pending generation.*")
        lines.append("")
        lines.append("---")
        lines.append("")
        
    lines.append("## Generated Figures & Artifacts")
    lines.append("")
    pngs = sorted(RESULTS_DIR.glob('**/*.png'))
    for png in pngs:
        try:
            rel = png.relative_to(REPO_ROOT)
            lines.append(f"- **{png.stem}**: `{rel}`")
        except ValueError:
            lines.append(f"- **{png.stem}**: `{png}`")
    lines.append("")
    
    REPORT_FILE.write_text("\n".join(lines), encoding='utf-8')
    print(f"Master report saved to: {REPORT_FILE}")
    
    # Update pipeline_status.json
    status_dict = {
        'timestamp': now_str,
        'overall_status': 'COMPLETED' if all_success else 'IN_PROGRESS',
        'studies': {name: {'status': studies_status[name]['status'], 'folder': studies_status[name]['folder']} for name in studies_status}
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(status_dict, f, indent=2)
    print(f"Pipeline status saved to: {STATUS_FILE}")

if __name__ == '__main__':
    compile_master_report()

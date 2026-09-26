from telegram_notifier import send_telegram, get_gpu_info
from datetime import datetime

gpu = get_gpu_info()
gpu_str = f"RTX A5000 | {gpu['temp']}°C | {gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.1f} GB" if gpu else "N/A"
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

msg = (
    f"🎉 *LEGACY 2.0: ALL PIPELINE EXPERIMENTS COMPLETE!* 🎉\n"
    f"🕒 *Finished*: `{now_str}`\n\n"
    f"🏆 *Key Findings Summary*:\n"
    f"• *Feature Fusion (44 feats)*: Highest Overall AUROC = `0.695 +/- 0.075` (MLP, Downsample 2:1)\n"
    f"• *Spectral Slope (12 feats)*: Highest Recall = `0.874 +/- 0.169`, F1 = `0.668 +/- 0.117` (XGBoost)\n"
    f"• *Wavelet DWT (16 feats)*: Consistent AUROC = `0.683 +/- 0.058` (MLP, Full Dataset)\n"
    f"• *EMD (16 feats)*: AUROC = `0.584 +/- 0.125` (LogReg)\n\n"
    f"🔬 *Feature Ablation Insights*:\n"
    f"• *Theta band (2-4 Hz)* is the single strongest isolated band (AUROC `0.748`)\n"
    f"• *Midband parameter* alone captures `0.761` AUROC (~99% of full 12-feature slope model!)\n"
    f"• *Top SHAP Driver*: `a3_power` (0.196) and `delta_midband` (0.177)\n\n"
    f"📁 *Deliverables Generated*:\n"
    f"• `tables/master_supplementary_table.{{md,tex,csv}}`\n"
    f"• `tables/frequency_feature_ablation_table.md`\n"
    f"• `tables/pca_ablation_table.md`\n"
    f"• `figures/` (7 publication charts: SHAP, PCA, Ablation, Comparison)\n\n"
    f"🖥️ *GPU*: `{gpu_str}`"
)

success = send_telegram(msg)
print(f"Telegram status: {success}")

"""
Legacy 2.0 - Rich Telegram Notification Module
==============================================
Sends detailed, publication-grade formatted updates, visual ASCII tables,
model comparisons, confusion matrices, and GPU telemetry via Telegram Bot API.
"""

import json
import urllib.request
import urllib.parse
import subprocess
from pathlib import Path
from datetime import datetime

CONFIG_PATH = Path(r"D:\marathi-asr\telegram_config.json")


def load_config():
    if not CONFIG_PATH.exists():
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "vram_used_gb": float(parts[0]) / 1024.0,
            "vram_total_gb": float(parts[1]) / 1024.0,
            "util": parts[2],
            "temp": parts[3]
        }
    except Exception:
        return None


def send_telegram(text):
    config = load_config()
    if not config:
        return False
    bot_token = config.get("bot_token")
    chat_id = config.get("chat_id")
    if not bot_token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }

    try:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def make_progress_bar(current, total, length=10):
    frac = min(max(current / max(total, 1), 0.0), 1.0)
    filled = int(round(length * frac))
    return "█" * filled + "░" * (length - filled)


def notify_status(title, details=None):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu = get_gpu_info()
    gpu_str = "N/A"
    if gpu:
        gpu_str = f"RTX A5000 | `{gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.1f} GB` | `{gpu['temp']}°C` ({gpu['util']}% util)"

    msg = (
        f"🧠 *Neonatal EEG Analysis - Legacy 2.0*\n"
        f"📌 *{title}*\n"
        f"🕒 *Time*: `{now_str}`\n\n"
    )
    if details:
        for k, v in details.items():
            msg += f"• *{k}*: `{v}`\n"
        msg += "\n"

    msg += f"🖥️ *GPU*: {gpu_str}"
    return send_telegram(msg)


def notify_rich_benchmark(paradigm, strategy, trial_idx, total_trials, models_dict, elapsed_sec):
    """
    Sends a rich, multi-model comparison table and diagnostics to Telegram.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = make_progress_bar(trial_idx + 1, total_trials, length=10)
    pct = ((trial_idx + 1) / total_trials) * 100
    strat_clean = "Downsampling (2:1)" if strategy == 'downsample_2x' else "WeightedRandomSampler"

    gpu = get_gpu_info()
    gpu_str = "N/A"
    if gpu:
        gpu_str = f"RTX A5000 (`{gpu['vram_used_gb']:.1f}/{gpu['vram_total_gb']:.1f} GB` | `{gpu['temp']}°C` | `{gpu['util']}%` util)"

    table_header = (
        "```\n"
        "Model         |  F1   | AUROC | Sens  | Spec  | Brier\n"
        "--------------+-------+-------+-------+-------+------\n"
    )
    table_rows = []
    short_names = {
        'AdaptiveNN': 'Adaptive NN  ',
        'XGBoost': 'XGBoost (GPU)',
        'RandomForest': 'Random Forest',
        'LogisticRegression': 'Logistic Reg '
    }

    diagnostics = []
    for m_key, m_data in models_dict.items():
        m_name = short_names.get(m_key, m_key[:13].ljust(13))
        tm = m_data['test_metrics']
        f1 = tm['f1']
        auc = tm['auroc']
        rec = tm['recall']
        spec = tm['specificity']
        brier = tm['brier']
        w = m_data.get('best_ma_window', 1)
        t = m_data.get('best_threshold', 0.5)

        table_rows.append(f"{m_name} | {f1:.3f} | {auc:.3f} | {rec:.3f} | {spec:.3f} | {brier:.3f}")
        diagnostics.append(f"• *{m_key}*: Opt $W={w}$, $T={t:.2f}$ | Acc: `{tm['accuracy']*100:.1f}%`")

    table_str = table_header + "\n".join(table_rows) + "\n```\n"
    diag_str = "\n".join(diagnostics)

    msg = (
        f"📊 *Legacy 2.0 Benchmark: Trial {trial_idx+1:02d} / {total_trials:02d}*\n"
        f"🔬 *Paradigm*: `{paradigm.upper()}`\n"
        f"⚙️ *Strategy*: `{strat_clean}`\n"
        f"📈 *Progress*: `[{bar}]` *{pct:.0f}%*\n"
        f"🕒 *Timestamp*: `{now_str}` (took `{elapsed_sec:.1f}s`)\n\n"
        f"{table_str}\n"
        f"🎯 *Calibration & Diagnostics*:\n"
        f"{diag_str}\n\n"
        f"🖥️ *GPU*: {gpu_str}"
    )
    return send_telegram(msg)


if __name__ == '__main__':
    # Test sample rich benchmark
    dummy_models = {
        'AdaptiveNN': {
            'best_ma_window': 19, 'best_threshold': 0.51,
            'test_metrics': {'f1': 0.7454, 'auroc': 0.8516, 'recall': 0.6913, 'specificity': 0.8649, 'accuracy': 0.7871, 'brier': 0.142}
        },
        'XGBoost': {
            'best_ma_window': 15, 'best_threshold': 0.48,
            'test_metrics': {'f1': 0.7621, 'auroc': 0.8679, 'recall': 0.7241, 'specificity': 0.8812, 'accuracy': 0.8095, 'brier': 0.131}
        },
        'RandomForest': {
            'best_ma_window': 12, 'best_threshold': 0.45,
            'test_metrics': {'f1': 0.7380, 'auroc': 0.8412, 'recall': 0.6804, 'specificity': 0.8541, 'accuracy': 0.7762, 'brier': 0.149}
        },
        'LogisticRegression': {
            'best_ma_window': 1, 'best_threshold': 0.50,
            'test_metrics': {'f1': 0.6512, 'auroc': 0.7481, 'recall': 0.6120, 'specificity': 0.7601, 'accuracy': 0.6940, 'brier': 0.198}
        }
    }
    notify_rich_benchmark('fusion', 'downsample_2x', 0, 10, dummy_models, 42.5)

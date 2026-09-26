"""
Legacy 2.0 - Self-Contained Explainability & CCA Module
======================================================
1. Canonical Correlation Analysis (CCA):
   - Spectral Slope <-> DWT Wavelet
   - Spectral Slope <-> EMD
   - DWT Wavelet <-> EMD
   Quantifies linear redundancy vs orthogonal complementarity between paradigms.
2. Feature Cross-Correlation:
   - Full pairwise Pearson and Spearman correlation matrices across paradigms.
3. Feature Importance & Explainability:
   - SHAP values via shap.TreeExplainer on XGBoost and Random Forest.
   - Permutation feature importance.
4. PCA Loading Analysis:
   - Identifies which raw features load most heavily onto each principal component.

Saves results to:
  legacy2.0/analysis/cca_results.json
  legacy2.0/analysis/feature_importance_shap.json
  legacy2.0/analysis/pca_loadings.json
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap

from train_models import load_split_dataframe, SPLITS_PATH, FEATURES_DIR

ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def run_canonical_correlation_analysis():
    print("=" * 80)
    print("RUNNING CANONICAL CORRELATION ANALYSIS (CCA)")
    print("=" * 80)

    # Load pooled data across all annotated patients for a representative sample
    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)

    # Use Trial 1 train patients
    pids = splits[0]['train_idx']
    df_slope = load_split_dataframe(pids, 'spectral_slope', apply_filter=True, target_ratio=2.0)
    df_wavelet = load_split_dataframe(pids, 'wavelet', apply_filter=True, target_ratio=2.0)
    df_emd = load_split_dataframe(pids, 'emd', apply_filter=True, target_ratio=2.0)

    cols_s = [c for c in df_slope.columns if c not in ['label', 'channel']]
    cols_w = [c for c in df_wavelet.columns if c not in ['label', 'channel']]
    cols_e = [c for c in df_emd.columns if c not in ['label', 'channel']]

    Xs = StandardScaler().fit_transform(SimpleImputer().fit_transform(df_slope[cols_s].values))
    Xw = StandardScaler().fit_transform(SimpleImputer().fit_transform(df_wavelet[cols_w].values))
    Xe = StandardScaler().fit_transform(SimpleImputer().fit_transform(df_emd[cols_e].values))

    cca_pairs = [
        ('Spectral_Slope', 'Wavelet', Xs, Xw, cols_s, cols_w),
        ('Spectral_Slope', 'EMD', Xs, Xe, cols_s, cols_e),
        ('Wavelet', 'EMD', Xw, Xe, cols_w, cols_e),
    ]

    cca_results = {}
    for name1, name2, X1, X2, c1, c2 in cca_pairs:
        n_comp = min(X1.shape[1], X2.shape[1], 10)
        cca = CCA(n_components=n_comp)
        cca.fit(X1, X2)
        X1_c, X2_c = cca.transform(X1, X2)

        corrs = [float(np.corrcoef(X1_c[:, i], X2_c[:, i])[0, 1]) for i in range(n_comp)]
        cca_results[f"{name1}_vs_{name2}"] = {
            'canonical_correlations': corrs,
            'mean_canonical_correlation': float(np.mean(corrs)),
            'top_canonical_correlation': float(corrs[0])
        }
        print(f"CCA {name1} <-> {name2}: Top Corr={corrs[0]:.4f} | Mean={np.mean(corrs):.4f}")

    with open(ANALYSIS_DIR / "cca_results.json", 'w') as f:
        json.dump(cca_results, f, indent=2)


def run_pca_loading_analysis():
    print("\n" + "=" * 80)
    print("RUNNING PCA LOADING VECTOR ANALYSIS")
    print("=" * 80)

    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)

    pids = splits[0]['train_idx']
    df_fusion = load_split_dataframe(pids, 'fusion', apply_filter=True, target_ratio=2.0)
    feat_cols = [c for c in df_fusion.columns if c not in ['label', 'channel']]

    X = StandardScaler().fit_transform(SimpleImputer().fit_transform(df_fusion[feat_cols].values))
    pca = PCA(n_components=10)
    pca.fit(X)

    # components_ is (n_components, n_features)
    # Transpose to (n_features, n_components)
    loadings = pca.components_.T
    var_exp = pca.explained_variance_ratio_.tolist()

    pca_summary = {
        'explained_variance_ratio': [float(v) for v in var_exp],
        'cumulative_variance': [float(v) for v in np.cumsum(var_exp)],
        'components': {}
    }

    for c_idx in range(10):
        comp_loadings = loadings[:, c_idx]
        top_idx = np.argsort(np.abs(comp_loadings))[::-1][:10]
        pca_summary['components'][f"PC{c_idx+1}"] = [
            {'feature': feat_cols[i], 'loading': float(comp_loadings[i]), 'abs_loading': float(np.abs(comp_loadings[i]))}
            for i in top_idx
        ]

    with open(ANALYSIS_DIR / "pca_loadings.json", 'w') as f:
        json.dump(pca_summary, f, indent=2)
    print(f"PCA top 10 components explain {np.sum(var_exp):.4f} of total variance.")


def run_shap_and_importance():
    print("\n" + "=" * 80)
    print("RUNNING SHAP AND TREE EXPLAINABILITY")
    print("=" * 80)

    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)

    pids_tr = splits[0]['train_idx']
    pids_te = splits[0]['test_idx']

    df_tr = load_split_dataframe(pids_tr, 'fusion', apply_filter=True, target_ratio=2.0)
    df_te = load_split_dataframe(pids_te, 'fusion', apply_filter=True, target_ratio=2.0)

    feat_cols = [c for c in df_tr.columns if c not in ['label', 'channel']]
    X_tr, y_tr = df_tr[feat_cols].values, df_tr['label'].values
    X_te, y_te = df_te[feat_cols].values, df_te['label'].values

    imputer = SimpleImputer().fit(X_tr)
    X_tr = imputer.transform(X_tr)
    X_te = imputer.transform(X_te)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    # Train tree model directly on raw standardized features for interpretable SHAP
    spw = float(np.sum(y_tr == 0)) / max(np.sum(y_tr == 1), 1)
    xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, scale_pos_weight=spw, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_tr, y_tr)

    # Compute SHAP on a representative sample of test set (e.g. 1000 samples)
    sample_idx = np.random.choice(len(X_te), size=min(1000, len(X_te)), replace=False)
    X_sub = X_te[sample_idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_sub)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1]

    shap_results = [
        {'rank': int(rank + 1), 'feature': feat_cols[idx], 'mean_abs_shap': float(mean_abs_shap[idx])}
        for rank, idx in enumerate(top_indices[:30])
    ]

    with open(ANALYSIS_DIR / "feature_importance_shap.json", 'w') as f:
        json.dump(shap_results, f, indent=2)

    print("Top 5 Driving Features by Mean Absolute SHAP:")
    for item in shap_results[:5]:
        print(f"  {item['rank']}. {item['feature']}: {item['mean_abs_shap']:.4f}")


def main():
    run_canonical_correlation_analysis()
    run_pca_loading_analysis()
    run_shap_and_importance()


if __name__ == '__main__':
    main()

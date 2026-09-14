# Comprehensive Neonatal EEG Seizure Detection Research Report

**Generated:** 2026-09-14 10:00:55  
**Pipeline Status:** COMPLETED  
**Total Studies Executed:** 8  

---

## Executive Summary & Study Overview

| Study # | Name | Status | Key Focus |
| :--- | :--- | :---: | :--- |
| **Study 1** | Classifier Complexity Ladder | SUCCESS | Linear vs Non-linear ML (Logistic Regression, Linear SVM, RF, KNN) |
| **Study 2** | Feature Ablation | SUCCESS | Frequency band & parameter drop sensitivity |
| **Study 3** | UMAP & t-SNE Clustering | SUCCESS | Unsupervised manifold separation & clustering metrics |
| **Study 4** | Fisher Discriminant Ratio | SUCCESS | Analytical class separability per feature |
| **Study 5** | Baseline Feature Comparison (w/ EMD & DWT) | SUCCESS | Spectral slope vs Band Power, Hjorth, Time-Domain, Entropy, EMD, DWT |
| **Study 6** | Temporal Trajectory Analysis | SUCCESS | Real-time seizure tracking across onset/offset boundaries |
| **Study 7** | Cohen's d Effect Sizes | SUCCESS | Standardized effect magnitude (sample-size invariant) |
| **Study 8** | Class Imbalance Sampler Ablation | SUCCESS | Imbalance ratio sensitivity (50:50, 60:40, 40:60, 30:70, 20:80, Natural) |

---

## Study 1: Classifier Complexity Ladder

```text
====================================================================================================
STUDY 1: CLASSIFIER COMPLEXITY LADDER � SUMMARY
====================================================================================================

Goal: If Logistic Regression achieves AUROC > 0.80, the spectral slope
features are linearly separable � the features do the work, not the model.


--------------------------------------------------------------------------------
  VAL SET RESULTS (mean � std across 10 trials)
--------------------------------------------------------------------------------

Model                      ACCURACY            PRECISION           RECALL              F1                  AUROC             
-------------------------  ------------------  ------------------  ------------------  ------------------  ------------------
Logistic Regression        0.6698 � 0.1093     0.3335 � 0.1500     0.3732 � 0.1747     0.3096 � 0.1078     0.6401 � 0.1027   
Linear SVM                 0.6732 � 0.1114     0.3328 � 0.1499     0.3585 � 0.1804     0.3009 � 0.1089     0.6395 � 0.1056   
Random Forest              0.5901 � 0.1111     0.2993 � 0.1435     0.4566 � 0.1484     0.3196 � 0.1061     0.6116 � 0.0860   
KNN (k=5)                  0.5439 � 0.0359     0.5474 � 0.0435     0.5134 � 0.0850     0.5270 � 0.0543     0.5665 � 0.0483   

--------------------------------------------------------------------------------
  TEST SET RESULTS (mean � std across 10 trials)
--------------------------------------------------------------------------------

Model                      ACCURACY            PRECISION           RECALL              F1                  AUROC             
-------------------------  ------------------  ------------------  ------------------  ------------------  ------------------
Logistic Regression        0.6693 � 0.1079     0.3567 � 0.1993     0.3674 � 0.2011     0.3254 � 0.1438     0.6153 � 0.1631   
Linear SVM                 0.6746 � 0.1079     0.3586 � 0.2019     0.3530 � 0.2030     0.3184 � 0.1439     0.6165 � 0.1660   
Random Forest              0.5792 � 0.0728     0.3048 � 0.1709     0.4857 � 0.1375     0.3417 � 0.1049     0.5966 � 0.0814   
KNN (k=5)                  0.5369 � 0.0366     0.5359 � 0.0366     0.5310 � 0.0803     0.5321 � 0.0537     0.5538 � 0.0509   


================================================================================
INTERPRETATION
================================================================================

[WARN]  Logistic Regression test AUROC = 0.6153 (< 0.80)
   -> Features may require non-linear modelling for full potential.
```

---

## Study 2: Feature Ablation

```text
====================================================================================================
STUDY 2: FEATURE ABLATION STUDY � SUMMARY
====================================================================================================

Goal: Prove each band/parameter contributes non-redundant information.
If dropping a band causes a large AUROC drop, it is essential.


------------------------------------------------------------------------------------------
  VAL SET (mean across 10 trials)
------------------------------------------------------------------------------------------

Ablation                  #Feats    AUROC        DeltaAUROC  F1           DeltaF1   
--------------------------------------------------------------------------------
Full (baseline)           12        0.6401         �         0.3096         �       
Drop Delta                9         0.6357       -0.0044     0.3043       -0.0053   
Drop Theta                9         0.6410       +0.0009     0.3096       -0.0000   
Drop Alpha                9         0.6389       -0.0012     0.3058       -0.0038   
Drop Beta                 9         0.6240       -0.0161     0.2937       -0.0160   
Slope only                4         0.6071       -0.0330     0.2844       -0.0253   
Intercept only            4         0.6158       -0.0243     0.2883       -0.0213   
Midband only              4         0.6386       -0.0015     0.3091       -0.0005   

------------------------------------------------------------------------------------------
  TEST SET (mean across 10 trials)
------------------------------------------------------------------------------------------

Ablation                  #Feats    AUROC        DeltaAUROC  F1           DeltaF1   
--------------------------------------------------------------------------------
Full (baseline)           12        0.6153         �         0.3254         �       
Drop Delta                9         0.6090       -0.0064     0.3206       -0.0048   
Drop Theta                9         0.6160       +0.0007     0.3243       -0.0011   
Drop Alpha                9         0.6147       -0.0007     0.3227       -0.0028   
Drop Beta                 9         0.6026       -0.0127     0.3109       -0.0146   
Slope only                4         0.6004       -0.0149     0.3044       -0.0211   
Intercept only            4         0.6028       -0.0126     0.3074       -0.0181   
Midband only              4         0.6189       +0.0036     0.3267       +0.0012   


================================================================================
INTERPRETATION
================================================================================

Largest AUROC drop: 'Slope only' (Delta = -0.0149)
  -> This component contributes the most discriminative information.

Possibly redundant: ['Drop Delta', 'Drop Theta', 'Drop Alpha', 'Midband only']
```

---

## Study 3: UMAP & t-SNE Clustering

```text
================================================================================
STUDY 3: UMAP / t-SNE — CLUSTERING METRICS SUMMARY
================================================================================

Metrics computed on raw (non-PCA) spectral slope features.
Silhouette Score: [-1, 1], higher = better cluster separation.
Davies-Bouldin Index: [0, inf), lower = better cluster separation.

Trial    Silhouette     Davies-Bouldin   N Samples   
--------------------------------------------------
1        0.0089         8.8121           10000       
2        0.0123         6.3373           10000       
3        0.0045         9.7740           10000       
4        0.0205         5.2009           10000       
5        0.0258         3.8564           10000       
6        0.0494         3.4618           10000       
7        0.0628         3.0959           10000       
8        0.0355         3.9887           10000       
9        0.0029         11.3361          10000       
10       0.0268         4.5017           10000       

--------------------------------------------------
Mean     0.0249         6.0365          
Std      0.0197         2.9274          

================================================================================
INTERPRETATION
================================================================================

[WARN]  Mean Silhouette = 0.0249 (0 < x < 0.25)
   -> Weak but present cluster structure.
```

---

## Study 4: Fisher Discriminant Ratio

```text
================================================================================
STUDY 4: FISHER'S DISCRIMINANT RATIO � SUMMARY
================================================================================

Fisher Ratio J = (mu_1 - mu_0)^2 / (sigma^2_1 + sigma^2_0)
Higher J = better class separation for that feature.
Unlike t-tests, Fisher ratio does NOT inflate with sample size.

Total features analyzed: 12

TOP 30 FEATURES BY FISHER RATIO
------------------------------------------------------------
Rank   Feature                        Band     Type         J (mean�std)      
------------------------------------------------------------
1      delta_midband                  delta    midband      0.0543 � 0.0414
2      delta_intercept                delta    intercept    0.0503 � 0.0408
3      delta_slope                    delta    slope        0.0412 � 0.0358
4      theta_midband                  theta    midband      0.0234 � 0.0176
5      theta_intercept                theta    intercept    0.0181 � 0.0135
6      alpha_intercept                alpha    intercept    0.0085 � 0.0059
7      theta_slope                    theta    slope        0.0081 � 0.0060
8      alpha_midband                  alpha    midband      0.0068 � 0.0061
9      alpha_slope                    alpha    slope        0.0061 � 0.0037
10     beta_slope                     beta     slope        0.0028 � 0.0016
11     beta_intercept                 beta     intercept    0.0006 � 0.0006
12     beta_midband                   beta     midband      0.0004 � 0.0004


FISHER RATIO BY BAND (mean of means)
----------------------------------------
  delta     : J = 0.0486
  theta     : J = 0.0165
  alpha     : J = 0.0071
  beta      : J = 0.0013

FISHER RATIO BY TYPE (mean of means)
----------------------------------------
  midband     : J = 0.0212
  intercept   : J = 0.0194
  slope       : J = 0.0146


================================================================================
INTERPRETATION
================================================================================

Most discriminative band: delta (J = 0.0486)
Most discriminative type: midband (J = 0.0212)
```

---

## Study 5: Baseline Feature Comparison (w/ EMD & DWT)

```text
====================================================================================================
STUDY 5: BASELINE FEATURE COMPARISON � SUMMARY
====================================================================================================

All feature sets evaluated with Logistic Regression (class_weight='balanced')
across the same 10-trial patient-level CV splits.


------------------------------------------------------------------------------------------
  VAL SET (mean � std across 10 trials)
------------------------------------------------------------------------------------------

Feature Set               #Feats    AUROC               F1                  RECALL              PRECISION         
----------------------------------------------------------------------------------------------------
Spectral Slope            12        0.6401 � 0.1027     0.3096 � 0.1078     0.3732 � 0.1747     0.3335 � 0.1500   
Band Power                4         0.5603 � 0.0768     0.2852 � 0.1261     0.5652 � 0.3266     0.2528 � 0.1095   
Hjorth                    3         0.5956 � 0.0833     0.3927 � 0.1637     0.7383 � 0.1019     0.2876 � 0.1508   
Time Domain               4         0.4728 � 0.0965     0.3386 � 0.1434     0.6621 � 0.1135     0.2435 � 0.1278   
Spectral Entropy          4         0.6068 � 0.1027     0.3345 � 0.1130     0.4526 � 0.1491     0.3123 � 0.1470   
Emd                       6         0.5744 � 0.1346     0.2653 � 0.1715     0.5823 � 0.4216     0.2296 � 0.1196   
Dwt                       5         0.5680 � 0.1093     0.3657 � 0.1735     0.9013 � 0.0915     0.2411 � 0.1415   

------------------------------------------------------------------------------------------
  TEST SET (mean � std across 10 trials)
------------------------------------------------------------------------------------------

Feature Set               #Feats    AUROC               F1                  RECALL              PRECISION         
----------------------------------------------------------------------------------------------------
Spectral Slope            12        0.6153 � 0.1631     0.3254 � 0.1438     0.3674 � 0.2011     0.3567 � 0.1993   
Band Power                4         0.5058 � 0.0984     0.2899 � 0.1481     0.5351 � 0.3128     0.2735 � 0.1988   
Hjorth                    3         0.5599 � 0.0939     0.3848 � 0.1781     0.7212 � 0.0939     0.2804 � 0.1721   
Time Domain               4         0.4582 � 0.0683     0.3439 � 0.1717     0.6636 � 0.1196     0.2500 � 0.1646   
Spectral Entropy          4         0.5945 � 0.1455     0.3504 � 0.1404     0.4738 � 0.1925     0.3281 � 0.1919   
Emd                       6         0.5259 � 0.1589     0.2649 � 0.2080     0.5720 � 0.4370     0.2388 � 0.1544   
Dwt                       5         0.4739 � 0.0889     0.3756 � 0.1772     0.8974 � 0.0855     0.2516 � 0.1515   


================================================================================
INTERPRETATION
================================================================================

[PASS] Spectral Slope (AUROC = 0.6153) outperforms ALL baselines.
   -> Log-PSD linear fitting extracts information simpler features miss.
```

---

## Study 6: Temporal Trajectory Analysis

```text
============================================================
SELECTED PATIENTS (top by seizure duration)
============================================================

Patient  41:  7856s seizure (56 regions) / 8062s total (97.4%)
Patient   5:  3136s seizure (7 regions) / 3628s total (86.4%)
Patient  38:  2681s seizure (21 regions) / 4257s total (63.0%)
Patient  69:  2250s seizure (14 regions) / 2757s total (81.6%)
```

---

## Study 7: Cohen's d Effect Sizes

```text
================================================================================
STUDY 7: COHEN'S d EFFECT SIZE � SUMMARY
================================================================================

d = (mu_1 - mu_0) / sigma_pooled
sigma_pooled = sqrt((sigma_1^2 + sigma_0^2) / 2)

Unlike p-values, Cohen's d does NOT inflate with sample size.
With thousands of epochs, p < 0.001 for nearly everything.
Effect size tells you how MEANINGFULLY different distributions are.

Total features analyzed: 12

EFFECT SIZE DISTRIBUTION
----------------------------------------
  Large       :    0 features (  0.0%)
  Medium      :    0 features (  0.0%)
  Small       :    3 features ( 25.0%)
  Negligible  :    9 features ( 75.0%)


TOP 30 FEATURES BY |COHEN'S d|
----------------------------------------------------------------------
Rank   Feature                        Band     Type         d (mean�std)         Category    
----------------------------------------------------------------------
1      delta_midband                  delta    midband      +0.3060 � 0.1292   Small       
2      delta_intercept                delta    intercept    +0.2915 � 0.1313   Small       
3      delta_slope                    delta    slope        -0.2607 � 0.1268   Small       
4      theta_midband                  theta    midband      +0.1984 � 0.0910   Negligible  
5      theta_intercept                theta    intercept    +0.1753 � 0.0783   Negligible  
6      alpha_intercept                alpha    intercept    +0.1215 � 0.0500   Negligible  
7      theta_slope                    theta    slope        -0.1166 � 0.0527   Negligible  
8      alpha_slope                    alpha    slope        -0.1054 � 0.0358   Negligible  
9      alpha_midband                  alpha    midband      +0.1028 � 0.0581   Negligible  
10     beta_slope                     beta     slope        -0.0721 � 0.0230   Negligible  
11     beta_intercept                 beta     intercept    +0.0251 � 0.0247   Negligible  
12     beta_midband                   beta     midband      -0.0209 � 0.0172   Negligible  


EFFECT SIZE BY BAND (mean |d|)
----------------------------------------
  delta     : |d| = 0.2861
  theta     : |d| = 0.1634
  alpha     : |d| = 0.1099
  beta      : |d| = 0.0408

EFFECT SIZE BY TYPE (mean |d|)
----------------------------------------
  midband     : |d| = 0.1572
  intercept   : |d| = 0.1542
  slope       : |d| = 0.1387


================================================================================
INTERPRETATION
================================================================================

[WARN]  Only 0/12 features have medium-to-large effects.
   -> Most features have weak individual effect sizes, but may
     combine to produce strong classification performance.
```

---

## Study 8: Class Imbalance Sampler Ablation

```text
==============================================================================================================
STUDY 8: WEIGHTED RANDOM SAMPLER RATIO ABLATION � SUMMARY
==============================================================================================================

MLP architecture: Input->128->BN->ReLU->Drop(0.3)->64->BN->ReLU->Drop(0.3)->32->BN->ReLU->Drop(0.3)->2
Uses WeightedRandomSampler (not dataset filtering). Full dataset preserved.
50:50 = dominant config (prevents classifier collapse).


----------------------------------------------------------------------------------------------------
  VAL SET (mean � std across 10 trials)
----------------------------------------------------------------------------------------------------

Ratio         ACCURACY            PRECISION           RECALL              F1                  AUROC             
---------------------------------------------------------------------------------------------------------
50:50         0.2363 � 0.1261     0.2359 � 0.1265     0.9998 � 0.0004     0.3672 � 0.1588     0.6087 � 0.1066    <
60:40         0.2361 � 0.1264     0.2359 � 0.1266     0.9999 � 0.0003     0.3672 � 0.1589     0.5982 � 0.0977   
40:60         0.2399 � 0.1223     0.2361 � 0.1263     0.9976 � 0.0051     0.3675 � 0.1585     0.6086 � 0.0974   
30:70         0.2991 � 0.1173     0.2395 � 0.1272     0.9331 � 0.1291     0.3690 � 0.1609     0.6100 � 0.0956   
20:80         0.6475 � 0.1566     0.3487 � 0.1565     0.4942 � 0.2252     0.3692 � 0.1120     0.6135 � 0.0983   
Natural       0.5960 � 0.1729     0.3336 � 0.1755     0.4307 � 0.2230     0.3040 � 0.0964     0.6244 � 0.0983   

----------------------------------------------------------------------------------------------------
  TEST SET (mean � std across 10 trials)
----------------------------------------------------------------------------------------------------

Ratio         ACCURACY            PRECISION           RECALL              F1                  AUROC             
---------------------------------------------------------------------------------------------------------
50:50         0.2561 � 0.1519     0.2558 � 0.1519     0.9999 � 0.0003     0.3881 � 0.1786     0.5685 � 0.1195    <
60:40         0.2559 � 0.1519     0.2558 � 0.1519     1.0000 � 0.0001     0.3880 � 0.1786     0.5758 � 0.1161   
40:60         0.2595 � 0.1517     0.2561 � 0.1519     0.9955 � 0.0124     0.3881 � 0.1785     0.5677 � 0.1125   
30:70         0.2942 � 0.1380     0.2533 � 0.1549     0.9218 � 0.1426     0.3800 � 0.1853     0.5787 � 0.1156   
20:80         0.6458 � 0.1840     0.3553 � 0.1630     0.4779 � 0.2968     0.3798 � 0.1752     0.5725 � 0.1189   
Natural       0.6132 � 0.1824     0.3366 � 0.1692     0.4265 � 0.2318     0.3407 � 0.1400     0.5798 � 0.1205   


================================================================================
INTERPRETATION
================================================================================

Best test F1: 40:60 (F1 = 0.3881)

50:50 recall = 0.9999
Natural recall = 0.4265
Recall improvement from balanced sampling: +0.5733
```

---

## Generated Figures & Artifacts

- **test_auroc_comparison**: `study_results\01_classifier_ladder\test_auroc_comparison.png`
- **test_f1_comparison**: `study_results\01_classifier_ladder\test_f1_comparison.png`
- **val_auroc_comparison**: `study_results\01_classifier_ladder\val_auroc_comparison.png`
- **val_f1_comparison**: `study_results\01_classifier_ladder\val_f1_comparison.png`
- **ablation_chart**: `study_results\02_feature_ablation\ablation_chart.png`
- **tsne_scatter**: `study_results\03_umap_tsne\tsne_scatter.png`
- **umap_scatter**: `study_results\03_umap_tsne\umap_scatter.png`
- **fisher_by_band**: `study_results\04_fisher_ratio\fisher_by_band.png`
- **fisher_by_type**: `study_results\04_fisher_ratio\fisher_by_type.png`
- **top30_fisher**: `study_results\04_fisher_ratio\top30_fisher.png`
- **comparison_chart**: `study_results\05_baseline_comparison\comparison_chart.png`
- **patient_005_all_features**: `study_results\06_temporal_trajectory\patient_005_all_features.png`
- **patient_005_delta_slope**: `study_results\06_temporal_trajectory\patient_005_delta_slope.png`
- **patient_038_all_features**: `study_results\06_temporal_trajectory\patient_038_all_features.png`
- **patient_038_delta_slope**: `study_results\06_temporal_trajectory\patient_038_delta_slope.png`
- **patient_041_all_features**: `study_results\06_temporal_trajectory\patient_041_all_features.png`
- **patient_041_delta_slope**: `study_results\06_temporal_trajectory\patient_041_delta_slope.png`
- **patient_069_all_features**: `study_results\06_temporal_trajectory\patient_069_all_features.png`
- **patient_069_delta_slope**: `study_results\06_temporal_trajectory\patient_069_delta_slope.png`
- **cohens_d_by_band**: `study_results\07_cohens_d\cohens_d_by_band.png`
- **cohens_d_distribution**: `study_results\07_cohens_d\cohens_d_distribution.png`
- **cohens_d_top30**: `study_results\07_cohens_d\cohens_d_top30.png`
- **auroc_vs_ratio**: `study_results\08_sampler_ablation\auroc_vs_ratio.png`
- **f1_vs_ratio**: `study_results\08_sampler_ablation\f1_vs_ratio.png`
- **recall_vs_ratio**: `study_results\08_sampler_ablation\recall_vs_ratio.png`

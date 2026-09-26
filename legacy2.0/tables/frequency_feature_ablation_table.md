# Frequency Feature Ablation Study (All Classifiers)

Systematic evaluation of individual sub-bands and parameter types versus the full 12-feature baseline:

| Configuration            |   Features | Model              | AUROC           | F1              | Recall          | Specificity     |
|:-------------------------|-----------:|:-------------------|:----------------|:----------------|:----------------|:----------------|
| Full (12 feats)          |         12 | AdaptiveNN         | 0.767 +/- 0.087 | 0.666 +/- 0.108 | 0.862 +/- 0.051 | 0.428 +/- 0.173 |
| Full (12 feats)          |         12 | XGBoost            | 0.735 +/- 0.093 | 0.639 +/- 0.120 | 0.793 +/- 0.193 | 0.485 +/- 0.149 |
| Full (12 feats)          |         12 | LogisticRegression | 0.765 +/- 0.082 | 0.664 +/- 0.094 | 0.841 +/- 0.073 | 0.450 +/- 0.241 |
| Full (12 feats)          |         12 | RandomForest       | 0.734 +/- 0.091 | 0.652 +/- 0.122 | 0.839 +/- 0.191 | 0.436 +/- 0.192 |
| Delta only (3 feats)     |          3 | AdaptiveNN         | 0.732 +/- 0.061 | 0.634 +/- 0.081 | 0.922 +/- 0.087 | 0.196 +/- 0.239 |
| Delta only (3 feats)     |          3 | XGBoost            | 0.723 +/- 0.062 | 0.644 +/- 0.092 | 0.925 +/- 0.092 | 0.237 +/- 0.253 |
| Delta only (3 feats)     |          3 | LogisticRegression | 0.743 +/- 0.061 | 0.651 +/- 0.076 | 0.887 +/- 0.103 | 0.329 +/- 0.277 |
| Delta only (3 feats)     |          3 | RandomForest       | 0.721 +/- 0.061 | 0.645 +/- 0.091 | 0.937 +/- 0.080 | 0.218 +/- 0.229 |
| Theta only (3 feats)     |          3 | AdaptiveNN         | 0.745 +/- 0.046 | 0.651 +/- 0.093 | 0.883 +/- 0.080 | 0.321 +/- 0.208 |
| Theta only (3 feats)     |          3 | XGBoost            | 0.743 +/- 0.052 | 0.651 +/- 0.097 | 0.933 +/- 0.069 | 0.242 +/- 0.219 |
| Theta only (3 feats)     |          3 | LogisticRegression | 0.748 +/- 0.064 | 0.650 +/- 0.096 | 0.894 +/- 0.059 | 0.304 +/- 0.193 |
| Theta only (3 feats)     |          3 | RandomForest       | 0.742 +/- 0.050 | 0.652 +/- 0.097 | 0.933 +/- 0.070 | 0.246 +/- 0.218 |
| Alpha only (3 feats)     |          3 | AdaptiveNN         | 0.735 +/- 0.063 | 0.648 +/- 0.084 | 0.873 +/- 0.085 | 0.337 +/- 0.260 |
| Alpha only (3 feats)     |          3 | XGBoost            | 0.731 +/- 0.059 | 0.643 +/- 0.093 | 0.926 +/- 0.078 | 0.216 +/- 0.241 |
| Alpha only (3 feats)     |          3 | LogisticRegression | 0.727 +/- 0.060 | 0.640 +/- 0.096 | 0.920 +/- 0.096 | 0.242 +/- 0.296 |
| Alpha only (3 feats)     |          3 | RandomForest       | 0.733 +/- 0.058 | 0.643 +/- 0.092 | 0.915 +/- 0.076 | 0.240 +/- 0.244 |
| Beta only (3 feats)      |          3 | AdaptiveNN         | 0.721 +/- 0.068 | 0.630 +/- 0.085 | 0.846 +/- 0.084 | 0.335 +/- 0.125 |
| Beta only (3 feats)      |          3 | XGBoost            | 0.644 +/- 0.060 | 0.625 +/- 0.094 | 0.918 +/- 0.061 | 0.180 +/- 0.106 |
| Beta only (3 feats)      |          3 | LogisticRegression | 0.694 +/- 0.092 | 0.641 +/- 0.099 | 0.945 +/- 0.070 | 0.164 +/- 0.275 |
| Beta only (3 feats)      |          3 | RandomForest       | 0.647 +/- 0.060 | 0.625 +/- 0.093 | 0.915 +/- 0.053 | 0.186 +/- 0.113 |
| Slope only (4 feats)     |          4 | AdaptiveNN         | 0.716 +/- 0.073 | 0.640 +/- 0.094 | 0.884 +/- 0.094 | 0.277 +/- 0.265 |
| Slope only (4 feats)     |          4 | XGBoost            | 0.702 +/- 0.081 | 0.642 +/- 0.113 | 0.862 +/- 0.176 | 0.337 +/- 0.253 |
| Slope only (4 feats)     |          4 | LogisticRegression | 0.739 +/- 0.056 | 0.649 +/- 0.082 | 0.890 +/- 0.107 | 0.317 +/- 0.290 |
| Slope only (4 feats)     |          4 | RandomForest       | 0.692 +/- 0.084 | 0.639 +/- 0.112 | 0.859 +/- 0.176 | 0.330 +/- 0.271 |
| Intercept only (4 feats) |          4 | AdaptiveNN         | 0.748 +/- 0.078 | 0.663 +/- 0.096 | 0.872 +/- 0.068 | 0.396 +/- 0.186 |
| Intercept only (4 feats) |          4 | XGBoost            | 0.706 +/- 0.079 | 0.641 +/- 0.110 | 0.870 +/- 0.167 | 0.325 +/- 0.227 |
| Intercept only (4 feats) |          4 | LogisticRegression | 0.741 +/- 0.066 | 0.660 +/- 0.083 | 0.898 +/- 0.086 | 0.339 +/- 0.241 |
| Intercept only (4 feats) |          4 | RandomForest       | 0.700 +/- 0.075 | 0.638 +/- 0.106 | 0.871 +/- 0.170 | 0.308 +/- 0.250 |
| Midband only (4 feats)   |          4 | AdaptiveNN         | 0.757 +/- 0.076 | 0.658 +/- 0.100 | 0.875 +/- 0.055 | 0.381 +/- 0.193 |
| Midband only (4 feats)   |          4 | XGBoost            | 0.729 +/- 0.092 | 0.653 +/- 0.120 | 0.850 +/- 0.185 | 0.419 +/- 0.181 |
| Midband only (4 feats)   |          4 | LogisticRegression | 0.761 +/- 0.083 | 0.660 +/- 0.094 | 0.840 +/- 0.071 | 0.440 +/- 0.244 |
| Midband only (4 feats)   |          4 | RandomForest       | 0.732 +/- 0.091 | 0.653 +/- 0.121 | 0.852 +/- 0.187 | 0.414 +/- 0.186 |
# Frequency Feature Ablation Study (All Classifiers)

Systematic evaluation of individual sub-bands and parameter types versus the full 12-feature baseline:

| Configuration            |   Features | Model              | AUROC           | F1              | Recall          | Specificity     |
|:-------------------------|-----------:|:-------------------|:----------------|:----------------|:----------------|:----------------|
| Full (12 feats)          |         12 | AdaptiveNN         | 0.755 +/- 0.087 | 0.656 +/- 0.097 | 0.777 +/- 0.098 | 0.540 +/- 0.147 |
| Full (12 feats)          |         12 | XGBoost            | 0.735 +/- 0.093 | 0.639 +/- 0.120 | 0.793 +/- 0.193 | 0.485 +/- 0.149 |
| Full (12 feats)          |         12 | LogisticRegression | 0.765 +/- 0.082 | 0.664 +/- 0.094 | 0.841 +/- 0.073 | 0.450 +/- 0.241 |
| Full (12 feats)          |         12 | RandomForest       | 0.734 +/- 0.091 | 0.652 +/- 0.122 | 0.839 +/- 0.191 | 0.436 +/- 0.192 |
| Delta only (3 feats)     |          3 | AdaptiveNN         | 0.715 +/- 0.074 | 0.625 +/- 0.074 | 0.879 +/- 0.103 | 0.244 +/- 0.246 |
| Delta only (3 feats)     |          3 | XGBoost            | 0.723 +/- 0.062 | 0.644 +/- 0.092 | 0.925 +/- 0.092 | 0.237 +/- 0.253 |
| Delta only (3 feats)     |          3 | LogisticRegression | 0.743 +/- 0.061 | 0.651 +/- 0.076 | 0.887 +/- 0.103 | 0.329 +/- 0.277 |
| Delta only (3 feats)     |          3 | RandomForest       | 0.721 +/- 0.061 | 0.645 +/- 0.091 | 0.937 +/- 0.080 | 0.218 +/- 0.229 |
| Theta only (3 feats)     |          3 | AdaptiveNN         | 0.729 +/- 0.047 | 0.647 +/- 0.092 | 0.911 +/- 0.089 | 0.268 +/- 0.229 |
| Theta only (3 feats)     |          3 | XGBoost            | 0.743 +/- 0.052 | 0.651 +/- 0.097 | 0.933 +/- 0.069 | 0.242 +/- 0.219 |
| Theta only (3 feats)     |          3 | LogisticRegression | 0.748 +/- 0.064 | 0.650 +/- 0.096 | 0.894 +/- 0.059 | 0.304 +/- 0.193 |
| Theta only (3 feats)     |          3 | RandomForest       | 0.742 +/- 0.050 | 0.652 +/- 0.097 | 0.933 +/- 0.070 | 0.246 +/- 0.218 |
| Alpha only (3 feats)     |          3 | AdaptiveNN         | 0.731 +/- 0.062 | 0.637 +/- 0.089 | 0.922 +/- 0.085 | 0.196 +/- 0.250 |
| Alpha only (3 feats)     |          3 | XGBoost            | 0.731 +/- 0.059 | 0.643 +/- 0.093 | 0.926 +/- 0.078 | 0.216 +/- 0.241 |
| Alpha only (3 feats)     |          3 | LogisticRegression | 0.727 +/- 0.060 | 0.640 +/- 0.096 | 0.920 +/- 0.096 | 0.242 +/- 0.296 |
| Alpha only (3 feats)     |          3 | RandomForest       | 0.733 +/- 0.058 | 0.643 +/- 0.092 | 0.915 +/- 0.076 | 0.240 +/- 0.244 |
| Beta only (3 feats)      |          3 | AdaptiveNN         | 0.708 +/- 0.074 | 0.637 +/- 0.092 | 0.883 +/- 0.079 | 0.301 +/- 0.167 |
| Beta only (3 feats)      |          3 | XGBoost            | 0.644 +/- 0.060 | 0.625 +/- 0.094 | 0.918 +/- 0.061 | 0.180 +/- 0.106 |
| Beta only (3 feats)      |          3 | LogisticRegression | 0.694 +/- 0.092 | 0.641 +/- 0.099 | 0.945 +/- 0.070 | 0.164 +/- 0.275 |
| Beta only (3 feats)      |          3 | RandomForest       | 0.647 +/- 0.060 | 0.625 +/- 0.093 | 0.915 +/- 0.053 | 0.186 +/- 0.113 |
| Slope only (4 feats)     |          4 | AdaptiveNN         | 0.713 +/- 0.075 | 0.636 +/- 0.091 | 0.912 +/- 0.093 | 0.213 +/- 0.269 |
| Slope only (4 feats)     |          4 | XGBoost            | 0.702 +/- 0.081 | 0.642 +/- 0.113 | 0.862 +/- 0.176 | 0.337 +/- 0.253 |
| Slope only (4 feats)     |          4 | LogisticRegression | 0.739 +/- 0.056 | 0.649 +/- 0.082 | 0.890 +/- 0.107 | 0.317 +/- 0.290 |
| Slope only (4 feats)     |          4 | RandomForest       | 0.692 +/- 0.084 | 0.639 +/- 0.112 | 0.859 +/- 0.176 | 0.330 +/- 0.271 |
| Intercept only (4 feats) |          4 | AdaptiveNN         | 0.738 +/- 0.078 | 0.645 +/- 0.089 | 0.883 +/- 0.093 | 0.306 +/- 0.212 |
| Intercept only (4 feats) |          4 | XGBoost            | 0.706 +/- 0.079 | 0.641 +/- 0.110 | 0.870 +/- 0.167 | 0.325 +/- 0.227 |
| Intercept only (4 feats) |          4 | LogisticRegression | 0.741 +/- 0.066 | 0.660 +/- 0.083 | 0.898 +/- 0.086 | 0.339 +/- 0.241 |
| Intercept only (4 feats) |          4 | RandomForest       | 0.700 +/- 0.075 | 0.638 +/- 0.106 | 0.871 +/- 0.170 | 0.308 +/- 0.250 |
| Midband only (4 feats)   |          4 | AdaptiveNN         | 0.758 +/- 0.083 | 0.662 +/- 0.111 | 0.863 +/- 0.077 | 0.417 +/- 0.209 |
| Midband only (4 feats)   |          4 | XGBoost            | 0.729 +/- 0.092 | 0.653 +/- 0.120 | 0.850 +/- 0.185 | 0.419 +/- 0.181 |
| Midband only (4 feats)   |          4 | LogisticRegression | 0.761 +/- 0.083 | 0.660 +/- 0.094 | 0.840 +/- 0.071 | 0.440 +/- 0.244 |
| Midband only (4 feats)   |          4 | RandomForest       | 0.732 +/- 0.091 | 0.653 +/- 0.121 | 0.852 +/- 0.187 | 0.414 +/- 0.186 |
**In plain words:**

- On **641 mammograms the model never saw during development**, it ranks a randomly chosen malignant image above a randomly chosen benign one **82% of the time** (test AUC 0.817, 95% CI 0.774–0.858).
- At the decision threshold fixed before testing, it flags **65% of malignant images** and correctly clears **82% of benign ones**.
- During development (5-fold patient-grouped cross-validation) it scored AUC **0.850 ± 0.025**; the stretch target of 0.90 was **not** reached.

| Metric | Split | Value | 95% CI |
|---|---|---|---|
| `cv_auc_mean` | 5-fold patient-grouped CV in official train (out-of-fold) | 0.8500 ± 0.0246 | — |
| `cv_auc_mean_excluding_selection_fold` | 5-fold patient-grouped CV in official train (out-of-fold), folds ≠ 0 | 0.8527 | — |
| `cv_auc_pooled_oof` | 5-fold patient-grouped CV in official train (out-of-fold) | 0.8434 | [0.8255, 0.8615] |
| `cv_pr_auc_pooled_oof` | 5-fold patient-grouped CV in official train (out-of-fold) | 0.8328 | — |
| `cv_sens_at_spec80` | 5-fold patient-grouped CV in official train (out-of-fold) | 0.7286 | — |
| `cv_sens_at_spec90` | 5-fold patient-grouped CV in official train (out-of-fold) | 0.5991 | — |
| `cv_spec_at_sens90` | 5-fold patient-grouped CV in official train (out-of-fold) | 0.5174 | — |
| `cv_brier_calibrated` | 5-fold patient-grouped CV in official train (out-of-fold) | 0.1586 | — |
| `test_auc_ensemble` | official test (641 images, 5-model ensemble) | 0.8170 | [0.7745, 0.8585] |
| `test_auc_single_model_mean` | official test, each fold model alone | 0.8003 ± 0.0100 | — |
| `test_pr_auc_ensemble` | official test (641 images, 5-model ensemble) | 0.7927 | — |
| `test_sens_at_oof_threshold` | official test (641 images, 5-model ensemble) | 0.6450 | — |
| `test_spec_at_oof_threshold` | official test (641 images, 5-model ensemble) | 0.8206 | — |
| `test_sens_at_spec90` | official test (641 images, 5-model ensemble) | 0.5725 | — |
| `test_brier` | official test (641 images, 5-model ensemble) | 0.1672 | — |

Configuration: `s4_cv_s3_tta_hflip` (train hash `1e9764457296`); 11 configurations evaluated on validation (`experiments.csv`). Fold 0 was the selection fold.

*Generated from `results/metrics.json` (2026-11-12T17:45:48Z) by notebook milestone 17.*

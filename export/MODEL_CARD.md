# Model card — s4_cv_s3_tta_hflip

**Intended use:** portfolio and research only. **Not for clinical use**; no clinical claims.

**Task:** single-view whole-mammogram benign vs malignant classification on CBIS-DDSM (all images contain a biopsied lesion; there are no normal images).

**Architecture:** efficientnet_b3, input 1536×1184, pooling avg, EMA False, TTA hflip True; 5 fold models averaged; temperature scaling and threshold fitted on out-of-fold validation only.

**Data limits:** 1990s US scanned film digitised from analogue mammograms (not FFDM); imprecise DDSM lesion outlines; label noise; small validation folds.

## Results

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

## Failure modes (from milestone 16, counts by code)

- oof: 219 false positives and 314 false negatives of 2356 images at the frozen threshold; per-subgroup FN/FP rates in `results/error_rates_*.csv`.
- test: 68 false positives and 93 false negatives of 641 images at the frozen threshold; per-subgroup FN/FP rates in `results/error_rates_*.csv`.

**Citation:** CBIS-DDSM (CC BY 3.0), DOI 10.7937/K9/TCIA.2016.7O02S9CY; Lee et al., Sci Data 4, 170177 (2017); Clark et al., J Digit Imaging 26 (2013).

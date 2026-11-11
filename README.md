# CBIS-DDSM Whole-Mammogram Malignancy Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GITHUB_USER/cbis-ddsm-mammo-classifier/blob/main/notebooks/cbis_ddsm_whole_image_classifier.ipynb)

Single-view, whole-mammogram **benign vs malignant** classifier trained on CBIS-DDSM on one Colab A100 GPU,
with patient-grouped validation and a single, locked evaluation on the official test split.

> ⚠️ **Not for clinical use.** This is a portfolio and research project. It makes no clinical claims.

**Status:** milestones 0–18 implemented — the data pipeline, the fold-0 experiments (Stages 0–3), 5-fold
cross-validation, the review-gated one-time test evaluation, Grad-CAM, error analysis and export. Review gates 1 and 2
are approved. Fold-0 experiments are running on Colab; cross-validation and the test evaluation have not run yet, so
there are no headline results.

## Results

<!-- RESULTS:START -->
*Generated from `results/metrics.json` by notebook milestone 17 — not yet available.*
<!-- RESULTS:END -->

### Comparison with published work

Numbers from different splits are **not comparable**; the split column is there to stop that comparison.

| Study | Split | Model | Whole-image AUC |
|---|---|---|---|
| Shen et al. 2019 | Custom patient-level 85/15 (376 test images) | ResNet-50, patch pre-trained, 1152×896 | 0.87 (0.88 TTA) |
| Shen et al. 2019 | Same | 4-model average | 0.91 |
| Petrini et al. 2025 | Official (645 test images) | EfficientNet-B3, single view | 0.8325 ± 0.0171 |
| Petrini et al. 2025 | Official | EfficientNet-B3, two views | 0.8643 ± 0.0241 |
| Wei et al. (via Petrini) | Official / random | 4-model ensemble + TTA | 0.8313 / 0.9427 |
| **This project** | 5-fold CV in official train / official test | — | *pending* |

**Expectation.** Based on the results above, a single-view model on the official test split will most likely land in the
low-to-mid 0.80s. The 0.90 target applies to patient-grouped validation and is a stretch goal. Anything well above the
published numbers gets checked for leakage before it is reported.

## Task

- **Input:** one full-field mammogram (one breast, one view). **Output:** malignancy probability.
- **Label:** 1 if any biopsy-labelled abnormality on the image is `MALIGNANT`; 0 for `BENIGN` / `BENIGN_WITHOUT_CALLBACK`.
- CBIS-DDSM has **no normal images**, so this is benign-vs-malignant among lesion-bearing images — harder than, and not
  comparable to, cancer-vs-normal screening AUCs.
- **Validation:** `StratifiedGroupKFold(5)` grouped by base patient ID inside the official training split.
  **Test:** official split, evaluated once from a frozen, hashed model (`FINAL_EVAL` flag, off by default).

## Data notes (from the metadata run, IDC data v25)

Produced by notebook milestones 4–5 (`results/counts_check.csv`); these are dataset counts, not model results.

- The IDC collection ID is `cbis_ddsm`. All 3,568 CSV abnormality rows match an IDC series by SeriesInstanceUID.
- **IDC metadata defect:** 246 Calc-Test full-mammogram series have DICOM PatientID `P_xxxxx_SIDE_VIEW.dcm` (instead of
  `Calc-Test_P_xxxxx_SIDE_VIEW`) and an empty SeriesDescription. The notebook matches by UID, accepts only that exact
  pattern, and takes split and patient from the CSVs.
- **Image counts vs Petrini Table 1:** the CSVs reference 3,103 distinct full-image series (2,458 train / 645 test),
  matching Petrini. 42 images (38 train, 4 test) are listed in both the mass and calcification CSVs as separate
  series; de-duplicated per image as the PRD specifies, that gives **3,061 images (2,420 train / 641 test)**.
- **Cross-split patients:** 31 base patient IDs appear in both the official train and test splits. They are dropped
  from train/validation (never from test).
- **Orientation:** DDSM films are not consistently oriented relative to the breast-side label, so the preprocessing
  flips each image from its breast mask (chest wall always on the left), not from the side label.
- **Crop limitation:** Otsu + largest-component cropping occasionally keeps a corner of a film label.

## Reproduce

1. Open the notebook in Colab with the badge above and pick an **A100** runtime.
2. Run all. The notebook mounts Drive, downloads the metadata CSVs from TCIA and the full-mammogram DICOMs (~95 GB)
   from NCI Imaging Data Commons with `idc-index`, and caches everything to Drive.
3. It stops at each review gate; set the gate flag in the Config cell after reviewing and re-run.

Data is **not** committed to this repository. A metadata-only check (milestones 0–5 without image download) runs on any
machine: set `cfg.download_dicom = False`.

## Repository layout

```
├── README.md
├── CLAUDE.md                 # rules for Claude Code
├── docs/PRD.md               # specification
├── notebooks/cbis_ddsm_whole_image_classifier.ipynb
├── results/
│   ├── metrics.json          # every reported number, written by code
│   ├── experiments.csv       # every configuration tried
│   └── figures/
├── requirements.txt
└── LICENSE
```

## Limitations

1990s US scanned film (not digital mammography), no normal images, imprecise DDSM lesion annotations, label noise, and a
small validation set per fold. A full model card follows the final evaluation.

## Citations

CBIS-DDSM is licensed under **CC BY 3.0**; please cite:

- Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). *Curated Breast Imaging Subset of Digital Database for
  Screening Mammography (CBIS-DDSM)* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
- Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., & Rubin, D. L. (2017). A curated mammography data set
  for use in computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177.
- Clark, K., et al. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information
  Repository. *Journal of Digital Imaging*, 26(6), 1045–1057.
- Fedorov, A., et al. (2023). National Cancer Institute Imaging Data Commons: Toward Transparency, Reproducibility, and
  Scalability in Imaging Artificial Intelligence. *RadioGraphics*, 43(12).

Related work: Shen et al. 2019 (*Scientific Reports*); Petrini et al. 2025 (arXiv:2503.19945).

## License

Code: MIT (see `LICENSE`). Data: CC BY 3.0, not redistributed here.

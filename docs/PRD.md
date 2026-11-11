# PRD — CBIS-DDSM Whole-Mammogram Malignancy Classifier (Colab A100)

Nov 10, 2026 · Abhishek Bhadkamkar

> Markdown transcription of `PRD_CBIS-DDSM_Whole-Mammogram_Malignancy_Classifier_(Colab_A100).pdf`. The PDF is the original; if the two ever disagree, the PDF wins and this file is fixed.

## 1. Overview

Build one Google Colab notebook that trains a single-view, whole-mammogram benign-vs-malignant classifier on CBIS-DDSM on one A100 GPU. The target is a patient-grouped validation AUC of 0.90 or higher. The frozen model is then scored once on the official CBIS-DDSM test split, and that number is reported as it comes out. The result is a portfolio project hosted on GitHub.

- **Primary metric:** image-level ROC AUC on validation data carved from the official training split (see section 6). Target: 0.90 or higher.
- **Required secondary metric:** official test-split AUC with a 95% bootstrap confidence interval, computed exactly once.
- **Honesty rule:** if 0.90 is not reached, the repo reports the best score reached and the gap analysis. Nothing may be tuned on the test split, and no leakage shortcuts are allowed.
- **Audience:** recruiters and ML engineers reading the GitHub repo. Rigor and transparency count for more than the headline number.

**How to use this PRD with Claude Code:** save it as `docs/PRD.md` in the repo and point `CLAUDE.md` at it. Claude Code builds the notebook milestone by milestone (section 10). It stops at the review gates in sections 12 and 13 and waits for your approval.

## 2. Verified dataset facts

CBIS-DDSM is a curated subset of DDSM: scanned film mammograms from the 1990s, stored as DICOM. Every row in its metadata is a biopsy-labelled abnormality, either a mass or a calcification. The table below lists only facts taken from the sources given in it. Claude Code must re-check each count in code and report any mismatch; it must not patch around one.

| Fact | Value | Source |
|---|---|---|
| Host, DOI, license | TCIA; DOI 10.7937/K9/TCIA.2016.7O02S9CY; CC BY 3.0, citation required | TCIA collection page |
| Full collection | 10,239 images, 6,775 series, about 163.6 GB of DICOM. Includes full mammograms, ROI masks and cropped lesions | TCIA |
| Patients | 1,566 real participants. DICOM metadata shows 6,671 "subjects" because each participant has several patient IDs, such as `Calc-Test_P_00038_LEFT_CC` | TCIA |
| Official split | About 20% of cases go to test, stratified by BI-RADS. Masses and calcifications were split separately. A "case" is one abnormality seen on both the CC and MLO views | TCIA |
| Metadata CSVs | `mass_case_description_train_set.csv`, `calc_case_description_train_set.csv`, `mass_case_description_test_set.csv`, `calc_case_description_test_set.csv` | TCIA |
| Full-mammogram count | 3,103 images: 2,212 train, 246 validation (taken from train), 645 test | Petrini et al., arXiv 2503.19945, Table 1 |
| Pathology classes | Benign, benign without callback, malignant | CBIS-DDSM classification thesis |
| Pixel processing | Converted losslessly from LJPEG to 16-bit TIFF, then to DICOM. Optical density is clipped to 0.05–3.0 | TCIA wiki |
| Curation | A mammographer removed 254 images in which the annotated mass was not clearly visible | TCIA |

**Access routes, in order of preference:**

1. **NCI Imaging Data Commons (IDC):** TCIA states the images are searchable there. The `idc-index` Python package downloads IDC data from public cloud buckets with `s5cmd`, which works well inside Colab (idc-index docs). Claude Code must look up the exact collection ID with the client (for example `get_collections()`) rather than assume it.
2. **TCIA directly:** a `.tcia` manifest opened with the NBIA Data Retriever. This is a heavier install on Colab.
3. **Kaggle JPEG mirror** (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`): a community conversion. I could not confirm its bit depth, size or file layout, so the notebook must inspect them before use. Use it only as a fallback.

**Consequence for the task:** CBIS-DDSM has no normal (lesion-free) images. The task is therefore benign vs malignant among images that all contain a lesion. That is harder than cancer vs normal, and AUCs from the two tasks must never be compared.

## 3. Target realism: what 0.90 means

A single-view AUC of 0.90 is above the best published single-view result I found on the official CBIS-DDSM test split, which is about 0.83. It roughly matches the 4-model ensemble from Shen et al., which was scored on a custom split. Treat 0.90 on validation as a stretch goal, and expect the official test AUC to come in lower.

| Study | Test split | Model | Whole-image AUC |
|---|---|---|---|
| Shen et al. 2019 | Custom patient-level 85/15 split (376 test images) | Best single model: ResNet-50 pre-trained on patches, input 1152×896 | 0.87 (0.88 with test-time augmentation) |
| Shen et al. 2019 | Same | Average of the 4 best models | 0.91 |
| Petrini et al. 2025 | Official split (645 test images) | Best single view, EfficientNet-B3 | 0.8325 ± 0.0171 (0.8343 with TTA) |
| Petrini et al. 2025 | Official split | Two views (CC + MLO), EfficientNet-B3 | 0.8643 ± 0.0241 |
| Wei et al., as reported by Petrini et al. | Official vs random split | 4-model ensemble with TTA | 0.8313 on the official split, 0.9427 on a random split |

What the literature implies for this project:

- **The split moves the number more than the model does.** One method scored 0.83 on the official split and 0.94 on a random one. A random split is not patient-grouped, so the same patient's images can sit on both sides. The README must state exactly which split each number comes from.
- **Small validation sets are noisy.** A fold of about 250 images gives a wider AUC error bar than Petrini's ±0.017 on 645 test images. Picking the best of many configurations on one such fold inflates the score. The final validation figure is therefore the mean 5-fold out-of-fold AUC (section 6), not the best single-fold score.
- **Resolution matters.** Petrini found that halving 1152×896 inputs "led to a significant decrease in AUC". Shen used 1152×896 only because of an 8 GB GPU. An A100 lets us test higher resolutions.
- **Patch pre-training is unproven here.** Shen relied on it. Petrini found no significant gain from it on CBIS-DDSM (p = 0.47). Run it as an ablation, not as an assumption.
- **Two-view fusion** helped Petrini by about 0.03 AUC. Two-view results are exam-level, not single-image, so they are a stretch extension reported separately (section 7).

**Expectation to write into the README:** based on the published results above, a single-view model on the official test split will most likely land in the low-to-mid 0.80s. Anything above that should be checked for leakage before it is celebrated.

## 4. Task definition

The model takes one full mammogram (one breast, one view) and outputs a malignancy probability. The image is labelled malignant if any abnormality on it is malignant.

- **Unit of prediction:** one full-field image, identified by patient, breast side and view (CC or MLO). Abnormality rows from the mass and calcification CSVs are merged on this key. Claude Code reads the real column names from the CSVs before writing the join.
- **Label:** 1 if any abnormality row for the image has pathology `MALIGNANT`. Otherwise 0, which covers both benign classes.
- **De-duplication:** an image can carry several abnormalities, and may appear in both the mass and the calcification CSVs. The notebook collapses rows to one per image and logs how many images were merged.
- **Grouping key:** the base patient ID (for example `P_00038`), with the `Mass-` / `Calc-` and `Training` / `Test` prefixes and the side/view suffixes removed. All splits group on this key.
- **Subgroup tags kept for reporting:** abnormality type (mass only, calcification only, both), view, and the breast-density and BI-RADS assessment columns, if present in the CSVs.
- **Out of scope:** lesion localisation as an output (heatmaps are for explanation only), BI-RADS prediction, and normal-vs-abnormal screening.

## 5. Data acquisition and preprocessing

Download only the full-mammogram DICOM series (plus ROI masks if the patch stage runs). Convert them once to cropped 16-bit PNGs, cache them on Google Drive, and train from a copy on the Colab local disk.

1. **Check resources first.** Print `df -h /content` and the free RAM, and stop with a clear message if space is short. Do not assume Colab disk capacity. The full 163.6 GB collection includes crops and masks the task does not need.
2. **Get the metadata.** Download the four CSVs from the TCIA collection page. Print row counts per file.
3. **Get the images through IDC.** Install `idc-index`, find the CBIS-DDSM collection ID, and list its series. Inspect the actual `SeriesDescription` values; do not hard-code a guessed one. Select full-mammogram series, run a dry run to print the download size, then download to `/content/data/dicom` with progress shown and resume enabled.
4. **Match CSV rows to files.** Map every CSV image path to a downloaded DICOM through the study or series UIDs. Assert that 100% of rows match. Any unmatched row stops the notebook with the offending IDs listed.
5. **Decode.** Read with `pydicom`. Honour `PhotometricInterpretation` and invert `MONOCHROME1`. Keep 16-bit depth and log the min/max per image.
6. **Crop and orient.** Find the breast with Otsu thresholding on a downsampled copy and keep the largest connected component. Crop to its bounding box with a small margin, and flip so every breast faces the same side. Shen et al. did no cropping, so keep an uncropped variant available for one ablation.
7. **Resize and cache.** Resize preserving aspect ratio to a stored long side of 2048 px, so training can downsample to any target. Save as 16-bit PNG, and write a manifest CSV (image key, label, groups, path, original size, crop box).
8. **Cache to Drive.** Tar the PNG cache and manifest to Google Drive. Later sessions copy and untar it to local disk, which is faster than reading many small files from Drive.
9. **Kaggle fallback.** If IDC fails, use `kagglehub` with credentials from Colab Secrets. Print the bit depth and size of a sample and record the route in the results metadata.

**Normalisation:** use per-image normalisation (z-score over breast pixels), so no statistics leak from validation or test into training.

## 6. Splitting and leakage prevention

The official test split is locked away until the end. All development happens in patient-grouped folds inside the official training split.

- **Test lock:** a config flag `FINAL_EVAL = False` by default. Test images are never loaded unless it is `True`. The test cell asserts that the flag is set and that a frozen checkpoint hash is recorded. Test data is never used for training, early stopping, threshold choice, calibration or model selection.
- **Validation scheme:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on the official training images, grouped by base patient ID and stratified by label. Fold 0 is the development hold-out for ablations. The chosen configuration is then trained on all 5 folds.
- **Headline validation metric:** mean (± SD) of the 5 per-fold AUCs, plus the pooled out-of-fold AUC. This is the number compared with the 0.90 target.
- **Cross-split patient check:** count base patient IDs that appear in both the official train and test splits. If any do, drop those patients from train/validation (never from test) and report the count. Leakage-free beats exact comparability.
- **Automatic asserts:** no patient appears in more than one of train, validation and test. Both views of a breast always land in the same fold. Label prevalence per fold is printed.
- **Patch-stage rule:** if patch pre-training runs, its ROI patches come only from that fold's training images.
- **Selection transparency:** every configuration evaluated on validation is logged in `experiments.csv`, and the README states how many were tried.

## 7. Modelling approach

Work in stages. Each stage is a controlled experiment on fold 0, and only changes that raise validation AUC are kept. Backbones come from `timm` with ImageNet weights. Grayscale input is either repeated to 3 channels or fed to a 1-channel stem by summing the pretrained RGB filters.

1. **Stage 0 — pipeline baseline.** ResNet-50 at a short side of 512 px for a few epochs. The goal is to prove that data, labels and metrics work end to end. Expect a weak score; a suspiciously high one points to leakage.
2. **Stage 1 — direct whole-image classifier (DC).** EfficientNet-B3 (the strongest single-view backbone in Petrini et al.) at 1152×896, then larger inputs such as 1536×1152 if memory allows. Head: global average or GeM pooling, dropout, one logit.
3. **Stage 2 — patch pre-training (Shen-style ablation).** Train the same backbone on 224 px patches from training-fold ROIs, with 5 classes: background, benign or malignant mass, benign or malignant calcification. Sample around each ROI with high overlap, plus an equal number of background patches. Then initialise the whole-image model from it and compare with Stage 1.
4. **Stage 3 — one change at a time**, each logged as an ablation row:
   - resolution (896, 1152, 1536 px long side);
   - backbone (EfficientNet-B3/B4, ConvNeXt-Tiny/Small, ResNet-50);
   - cropped vs uncropped input;
   - pooling (average vs GeM vs attention-based MIL pooling over the feature map);
   - loss (BCE with `pos_weight` vs focal loss);
   - EMA of weights;
   - horizontal-flip TTA.
5. **Stage 4 — final model.** Train the best configuration on all 5 folds. Report per-fold and out-of-fold validation AUC. The test prediction is the average of the 5 fold models (an ensemble), reported separately from single-model numbers.
6. **Stage 5 (optional stretch) — two-view exam model.** Fuse the CC and MLO features of the same breast. Report it as exam-level AUC, clearly separated from the single-image task.

## 8. Training configuration on Colab A100

Use PyTorch with bf16 mixed precision, detect the GPU at runtime, and checkpoint to Drive every epoch so a disconnect costs at most one epoch.

| Setting | Requirement |
|---|---|
| Libraries | PyTorch, `timm`, `albumentations`, `scikit-learn`, `pydicom`, `idc-index`. Use whatever torch the Colab image ships, and pin the exact versions you used in `requirements.txt` |
| GPU check | Print `nvidia-smi`. Read `torch.cuda.get_device_properties(0)`, since Colab offers both 40 GB and 80 GB A100s (third-party summary). Set batch size from free memory. Warn, but do not crash, on L4 or T4 |
| Precision and speed | `torch.autocast(dtype=torch.bfloat16)`, `channels_last`, TF32 on. Try `torch.compile` and keep it only if it measurably speeds up an epoch |
| Batch | Largest that fits, with gradient accumulation to an effective batch of 16–32 |
| Optimiser | AdamW with linear warm-up (1 epoch), then cosine decay. Lower learning rate for the backbone than for the head. Weight decay searched in the 1e-4 to 1e-2 range |
| Epochs | 20–40, early stopping on validation AUC with patience of about 8 |
| Augmentation (train only) | Horizontal and vertical flips, rotation within ±25°, scale 0.8–1.2, brightness/contrast/gamma jitter. These follow Shen et al.'s ranges. No heavy elastic warps |
| Class imbalance | Print prevalence per fold. Use BCE `pos_weight` or a balanced sampler, chosen by ablation |
| Data loading | Read PNGs from local disk with `num_workers = os.cpu_count()`. Cache decoded tensors in RAM only if the free-RAM check allows |
| Checkpoints | `best.pt` and `last.pt` per fold on Drive, holding model, optimiser, scheduler, scaler, epoch and RNG state. Resume automatically |
| Reproducibility | Seed Python, NumPy and torch. Log the config, git commit and data-manifest hash with every run |
| Secrets | W&B or Kaggle keys come from Colab Secrets (`google.colab.userdata`), never from notebook text |
| Compute budget | Log GPU-minutes per run. One third-party measurement puts an A100 40 GB at about 5.4 compute units per hour (source); check your own Colab usage panel |

## 9. Evaluation, reporting and explainability

Every reported number comes from code into `results/metrics.json`, with a confidence interval and the name of its split.

- **Discrimination:** ROC AUC (primary) with a 95% CI from 2,000 patient-level bootstrap resamples. Also report PR-AUC, sensitivity at 80% and 90% specificity, and specificity at 90% sensitivity.
- **Threshold:** chosen on out-of-fold validation predictions only, then applied unchanged to test.
- **Calibration:** reliability diagram and Brier score. Optional temperature scaling, fitted on validation only.
- **Subgroups:** AUC by abnormality type (mass, calcification, both), by view (CC vs MLO), and by density and BI-RADS if available. Each shows its n and CI, and small groups are flagged.
- **Figures:** ROC with CI band, PR curve, confusion matrix at the validation-chosen threshold, training curves per fold, and Grad-CAM overlays for true positives, false positives and false negatives. Overlays show the ground-truth ROI outline for comparison. Use validation images, plus test images only after the final evaluation.
- **Comparison table:** our numbers beside the section 3 results, with a split column, so no one compares numbers from different splits.
- **Model card:** intended use (portfolio and research, not clinical), data limits (1990s US scanned film, no normal images, imprecise DDSM annotations), and failure modes found in error analysis.

## 10. Notebook structure

One notebook, `notebooks/cbis_ddsm_whole_image_classifier.ipynb`, that runs top to bottom on a fresh A100 runtime. Expensive stages sit behind flags and skip themselves when their cached outputs already exist. Each numbered item is a markdown header plus code cells, and each is also a Claude Code milestone.

0. **Title and summary:** goal, results table (filled from `metrics.json` at the end), "not for clinical use" notice, dataset citation.
1. **Config:** one dataclass with paths, seed, image size, backbone, fold, and the flags `RUN_DOWNLOAD`, `RUN_PREPROCESS`, `RUN_PATCH_PRETRAIN`, `RUN_CV`, `FINAL_EVAL` (default `False`).
2. **Environment check:** GPU name and memory, CUDA and torch versions, disk and RAM.
3. **Drive mount and folder layout.**
4. **Download:** CSVs and DICOM through IDC. Idempotent.
5. **Metadata build:** merge, label, de-duplicate, patient IDs, and a counts table checked against section 2. **Review gate 1.**
6. **EDA:** class balance per split, image-size histogram, sample images per class and abnormality type.
7. **Preprocessing and caching** (section 5).
8. **Splits and leakage asserts** (section 6).
9. **Datasets, dataloaders and augmentation preview grid.**
10. **Models:** backbone factory, heads, patch model.
11. **Training utilities:** AMP, EMA, checkpoint/resume, metric logging, bootstrap AUC.
12. **Experiments:** Stage 0 baseline (**review gate 2**), then Stages 1–3 as config overrides, appending rows to `experiments.csv`, with an ablation table rendered inline.
13. **5-fold cross-validation** of the chosen configuration; out-of-fold AUC.
14. **Final test evaluation:** gated by `FINAL_EVAL`, run once. **Review gate 3** comes before it.
15. **Explainability:** Grad-CAM panels.
16. **Error analysis:** worst false positives and false negatives, and subgroup tables.
17. **Export:** `metrics.json`, figures, `experiments.csv`, and weights to Drive (optionally the Hugging Face Hub).
18. **Conclusions and limitations.**

**Notebook rules:** helper code lives in the notebook (no hidden local modules), so it opens and runs straight from GitHub in Colab. No cell depends on state from a later cell. Long-running cells print progress. The outputs saved to GitHub stay small: no huge logs or embedded full-size images.

## 11. GitHub repo deliverables

The repo must let a reader see the result in 30 seconds and reproduce it with one click.

```
cbis-ddsm-mammo-classifier/
├── README.md            # results table, figures, Open-in-Colab badge, limitations, citations
├── CLAUDE.md            # rules for Claude Code (section 12)
├── docs/PRD.md          # this document
├── notebooks/cbis_ddsm_whole_image_classifier.ipynb
├── results/
│   ├── metrics.json     # every reported number, written by code
│   ├── experiments.csv  # every configuration tried, with validation AUC
│   └── figures/         # ROC, PR, calibration, Grad-CAM, training curves
├── requirements.txt     # versions pinned from the Colab run
├── LICENSE              # MIT for code
└── .gitignore           # data/, checkpoints/, *.dcm, *.png caches
```

- README numbers are generated from `metrics.json` by a notebook cell, never typed by hand.
- Data is not committed. The README explains how to get it and cites the TCIA dataset, the Lee et al. 2017 Scientific Data paper and the TCIA paper, as the CC BY 3.0 license requires.
- Weights go to a GitHub release or the Hugging Face Hub, with the model card from section 9.
- Commit history follows the milestones in section 10, with one commit per milestone or experiment batch.

## 12. Claude Code + Remote Control workflow

Claude Code writes and edits the notebook in your local repo, but it does not execute it on the Colab GPU. You run the notebook on Colab, and the results come back to the repo for Claude Code to read. Remote Control only lets you steer that local Claude Code session from your phone or a browser.

**Remote Control facts (official docs):**

- It connects claude.ai/code or the Claude mobile app to a Claude Code session running on your machine. Execution and file access stay on that machine, so it must stay on with the `claude` process running.
- It needs a Pro, Max, Team or Enterprise plan and a claude.ai login. API keys are not supported.
- Start it with `claude remote-control` (server mode), `claude --remote-control` (an interactive session), or `/remote-control` inside an existing session.
- If Claude Code runs on a remote machine, start it inside `tmux` or `screen` so it survives SSH disconnects.

**Two ways to run the notebook on Colab:**

- **Colab in the browser:** open the notebook from GitHub ("Open in Colab" badge), choose the A100 runtime, and run.
- **VS Code Colab extension:** Google's official extension connects a local `.ipynb` to a Colab runtime, including Pro-tier GPUs (Google announcement). The Colab runtime cannot read your local files (open issue), so the notebook must fetch all data itself, which it does anyway.

**Iteration loop:**

1. Claude Code implements the next milestone on a branch, commits and pushes.
2. You run it on Colab A100.
3. The notebook writes `summary.json` and `experiments.csv` to Drive. You commit the executed notebook back, through Colab's "Save a copy in GitHub" or by copying it into the repo.
4. Claude Code reads the outputs, explains them, and proposes one next change with its hypothesis. You approve from your phone via Remote Control.

**Rules to put in CLAUDE.md:**

- Never invent, estimate or hand-edit a metric. Report only what is in notebook outputs or `metrics.json`.
- Never set `FINAL_EVAL = True` or load test images without explicit approval in chat.
- Check every dataset assumption with an assert (columns, counts, file matches). On a mismatch, stop and report it; never silently patch.
- Make one change per experiment, log it with its hypothesis, and keep the notebook runnable top to bottom.
- Stop at review gates 1 (counts), 2 (baseline) and 3 (before the test run) and wait for approval.
- No secrets in code or outputs.
- If 0.90 looks unreachable, say so with evidence; do not reach for leakage-prone shortcuts such as random image-level splits or tuning on test.

**Starter prompt:**

> Read docs/PRD.md and CLAUDE.md. Create the repo skeleton from section 11, then build notebook milestones 0–5 from section 10. Stop at review gate 1 and show me what the metadata-counts cell will check against section 2. Do not write training code yet.

## 13. Acceptance criteria, risks and non-goals

The project is done when every box below is ticked. That includes the case where 0.90 is missed but the miss is honestly documented.

- [ ] Run-all on a fresh A100 runtime completes with default flags (using cached data).
- [ ] The counts table matches the section 2 facts, or every mismatch is explained.
- [ ] All leakage asserts pass, and the cross-split patient overlap is reported.
- [ ] 5-fold patient-grouped validation AUC is reported as mean ± SD plus pooled out-of-fold AUC, against the 0.90 target.
- [ ] Official test AUC with a 95% CI is reported once, from a frozen, hashed model.
- [ ] An ablation table has at least 4 rows, covering resolution, patch pre-training vs direct training, backbone, and one training trick.
- [ ] The README shows figures, the comparison table with a split column, limitations, citations and the clinical-use disclaimer.

| Risk | Likelihood | Mitigation |
|---|---|---|
| 0.90 not reached | High (section 3) | Report the best result honestly; try 5-fold ensembling and the two-view stretch; document the gap |
| Validation AUC inflated by small folds and many configs | Medium | 5-fold CV for the headline; log every configuration tried |
| A100 not available | Medium | GPU-aware batch sizing; same code runs on L4 or T4, just slower |
| Colab disconnects | High | Checkpoint every epoch to Drive with auto-resume; cache data |
| Download or path-matching failure | Medium | 100%-match assert; Kaggle fallback route |
| Disk fills up | Medium | Download only the needed series; delete DICOM after the PNG cache is verified |
| Label and annotation noise in DDSM | Certain | State it in the model card; don't over-interpret Grad-CAM |

**Non-goals:** clinical use or claims, generalisation claims to digital (FFDM) mammography, detection or segmentation outputs, and hyperparameter sweeps beyond the compute budget.

## 14. Sources

- TCIA CBIS-DDSM collection page — license, counts, split, CSVs
- TCIA wiki: CBIS-DDSM detailed description — conversion and clipping
- Lee et al. 2017, *A curated mammography data set*, Scientific Data
- Shen et al. 2019, *Deep Learning to Improve Breast Cancer Detection on Screening Mammography*, Scientific Reports
- Petrini et al. 2025, *Optimizing Breast Cancer Detection in Mammograms*, arXiv 2503.19945
- idc-index documentation
- Kaggle CBIS-DDSM JPEG mirror
- Claude Code Remote Control docs
- Google: Colab is coming to VS Code
- Colab GPU tiers (third-party) and A100 compute-unit rate (third-party)

**Not verified**, so the notebook checks these at runtime instead of assuming them: the exact IDC collection ID and series descriptions, the CSV column names, the Kaggle mirror's format, Colab disk size, and current Colab compute-unit prices.

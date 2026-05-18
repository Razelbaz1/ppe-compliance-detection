# PPE Compliance Detection on Construction Sites

> **Two-stage computer-vision pipeline for detecting Personal Protective Equipment (PPE) compliance in construction-site imagery.**
> Final project for the *Deep Learning Workshop* course at Ariel University (M.Sc. Knowledge & Data Engineering, Dr. Anat Goldstein).

| Authors | ID |
|---|---|
| Raz Albaz | 315837658 |
| Mika Melamed | 207488453 |

---

## 1. Project Overview

On construction sites, workers are required to wear personal protective equipment — most importantly a **hard-hat** and a **high-visibility vest**. Manual monitoring is impractical on large sites.

This project trains a model that receives an image of a construction site, detects the workers in it, and decides — per worker — which PPE items they are actually wearing.

The system is built as a **two-stage pipeline** followed by a small compliance-logic layer:

```
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│ Stage 1                  │    │ Stage 2                  │    │ Stage 3                  │
│ Person Detection         │ →  │ PPE Classification       │ →  │ Compliance Logic         │
│ (YOLOv8 fine-tuned)      │    │ (CNN + Transfer Learning)│    │ (helmet ∧ vest = OK)     │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
        person boxes                 multi-label tags                   per-worker verdict
```

- **Stage 1** locates every worker in the input image (a single-class person detector).
- **Stage 2** crops each detected worker and runs a multi-label classifier (helmet: 0/1, vest: 0/1).
- **Stage 3** applies a simple rule: a worker is **compliant** if they wear both a helmet and a vest.

---

## 2. Data

The pipeline is trained on a merged corpus of two public datasets, both already in YOLOv8 annotation format.

| Dataset | Images | Source |
|---|---|---|
| Ultralytics Construction-PPE | 1,414 | [docs.ultralytics.com/datasets/detect/construction-ppe](https://docs.ultralytics.com/datasets/detect/construction-ppe/) |
| Kaggle Construction Site Safety | 2,799 | [kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) |
| **Merged (after deduplication)** | **4,213** | — |

**Why merge them?** The two datasets are complementary:
- *Ultralytics* mostly shows compliant workers (~61 % wearing both items).
- *Kaggle CSS* mostly shows non-compliant workers (~54 % missing both items).
- Combined, the merged corpus is more balanced and reduces dataset bias.

**Class taxonomy (after Intersection — Strategy B):** five unified classes used in the pipeline.

| Unified class | # samples |
|---|---|
| Person | 12,134 |
| Helmet (worn) | 5,080 |
| Vest (worn) | 4,763 |
| No-Vest | 4,158 |
| No-Helmet | 2,912 |

**Pre-defined train / val / test splits** (we respect the splits set by each dataset's authors):

| Split | Ultralytics | Kaggle | Total |
|---|---|---|---|
| Train | 1,132 | 2,605 | **3,737** |
| Val | 143 | 114 | **257** |
| Test | 139 | 80 | **219** |

**Data hygiene — duplicate & leakage check.** We applied perceptual hashing (pHash) across all images in both datasets, in every train/val/test combination, to detect images that — despite different filenames or compression — represent the same scene. We removed **4 overlapping images** that would have caused information leakage between training and evaluation. See [src/utils/deduplicate.py](src/utils/deduplicate.py) for the verification script.

---

## 3. Repository Layout

```
ppe-compliance-detection/
├── README.md                       This file
├── LICENSE
├── requirements.txt                Python dependencies
├── .gitignore                      Excludes datasets, model weights, results
├── data/
│   ├── README.md                   How to download both datasets
│   └── download_datasets.py        One-shot script that fetches both
├── src/
│   ├── utils/
│   │   ├── deduplicate.py          pHash deduplication & leakage check
│   │   └── analyze_data.py         Class-distribution & compliance breakdown
│   ├── stage1_detection/
│   │   └── train.py                YOLOv8 fine-tuning (Stage 1)
│   ├── stage2_classification/
│   │   ├── build_crop_dataset.py   Person-crop extraction with multi-label tags
│   │   └── train.py                CNN training (Stage 2)
│   └── pipeline/
│       └── inference.py            End-to-end pipeline (Stages 1 → 2 → 3)
├── notebooks/                      Jupyter notebooks for exploration & ablations
└── results/                        Trained-model artefacts, figures, metrics
```

---

## 4. Getting Started

### 4.1 Environment

```bash
git clone https://github.com/razelbaz1/ppe-compliance-detection.git
cd ppe-compliance-detection
python -m venv .venv
.venv\Scripts\activate          # PowerShell:  .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2 Datasets

The Ultralytics dataset is downloaded automatically by the `ultralytics` package the first time training starts.

The Kaggle dataset requires a Kaggle API token. Once you have `~/.kaggle/kaggle.json` (or `~/.kaggle/access_token` for the newer token format), run:

```bash
python data/download_datasets.py
```

This fetches both datasets and verifies their structure. See [data/README.md](data/README.md) for detailed instructions.

### 4.3 Verify No Duplicates (optional but recommended)

```bash
python -m src.utils.deduplicate
```

This re-runs the pHash leakage check and reports any cross-split duplicates that should be removed.

### 4.4 Train Stage 1 — Person Detection

```bash
python -m src.stage1_detection.train --model yolov8n.pt --epochs 50
```

### 4.5 Train Stage 2 — PPE Classifier

```bash
# First build the per-person crop dataset from the detection annotations
python -m src.stage2_classification.build_crop_dataset

# Then train the multi-label classifier
python -m src.stage2_classification.train --backbone resnet50 --epochs 30
```

### 4.6 End-to-End Inference

```bash
python -m src.pipeline.inference --image path/to/image.jpg
```

---

## 5. Results

| Metric | Target | Achieved |
|---|---|---|
| Stage 1 — mAP@0.5 (person detection) | 70 – 85 % | _TBD_ |
| Stage 2 — Helmet accuracy | 85 – 92 % | _TBD_ |
| Stage 2 — Vest accuracy | 80 – 88 % | _TBD_ |
| End-to-end compliance accuracy | 70 – 80 % | _TBD_ |

Results will be populated in [`results/`](results/) as each stage is trained.

---

## 6. Acknowledgements

- **Ultralytics** for the open YOLOv8 framework and the Construction-PPE dataset.
- **Roboflow / Snehil Sanyal** for the Kaggle Construction Site Safety dataset.
- **Dr. Anat Goldstein** (course instructor, Ariel University) for the project framing.

---

## 7. License

MIT — see [LICENSE](LICENSE).

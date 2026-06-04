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

The project uses two public datasets, both already annotated in YOLOv8 format. Rather than merging them, we treat them as two distinct domains for a **Cross-Dataset Evaluation** — see notebook Chapter 4 for the full discussion and rationale.

| Dataset | Images | Role | Source |
|---|---|---|---|
| Kaggle Construction Site Safety | 2,799 | Training / validation domain | [kaggle.com/.../construction-site-safety-image-dataset-roboflow](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) |
| Ultralytics Construction-PPE | 1,414 | Held-out out-of-domain test set | [docs.ultralytics.com/datasets/detect/construction-ppe](https://docs.ultralytics.com/datasets/detect/construction-ppe/) |
| **Total (after deduplication)** | **4,213** | — | — |

**Why two datasets in different roles?** The two have complementary biases:
- *Kaggle CSS* mostly shows non-compliant workers (~54% missing both items).
- *Ultralytics* mostly shows compliant workers (~61% wearing both items).

By training on one and testing on the other, the project measures how well the model **generalizes across data sources** — a stricter test than a random in-distribution split. The narrative of the project is *"the model succeeds in both worlds"* — both on its training domain and on a previously-unseen one.

### 2.1 Data splits

The model is trained and evaluated on **four** distinct splits:

| Split | Source | Size | Role |
|---|---|---|---|
| Train | Kaggle CSS — train | 2,605 | Model learns here |
| Validation | Kaggle CSS — val | 114 | Early stopping, hyperparameter selection |
| Test In-Domain | Kaggle CSS — test | 80 | Performance on the trained domain |
| Test Out-of-Domain | Ultralytics — entire dataset | 1,414 | Generalization to a new domain |
| **Total** | | **4,213** | |

The test sets are **never seen during EDA, training, or hyperparameter tuning**. They are evaluated once at the end of each stage.

### 2.2 Class taxonomy

After unifying class labels across the two datasets, the pipeline operates on these classes:

| Class | Total samples |
|---|---|
| Person | 12,134 |
| Helmet (worn) | 5,080 |
| Vest (worn) | 4,763 |
| No-Vest | 4,158 |
| No-Helmet | 2,912 |

### 2.3 Data hygiene — duplicate & leakage check

We applied perceptual hashing (pHash) across all images in both datasets, in every train/val/test combination, to detect images that — despite different filenames or compression — represent the same scene. We removed **4 overlapping images** that would have caused information leakage between training and evaluation. See [src/utils/deduplicate.py](src/utils/deduplicate.py) for the verification script.

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

The **primary deliverable** of this project is the Jupyter notebook **[notebooks/01_main_pipeline.ipynb](notebooks/01_main_pipeline.ipynb)**. It walks through the full pipeline from data exploration to end-to-end evaluation, including theoretical background and detailed explanations of every step.

### 4.1 Environment setup

```powershell
git clone https://github.com/Razelbaz1/ppe-compliance-detection.git
cd ppe-compliance-detection

# Create and activate a virtual environment (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install "tensorflow>=2.15.0"   # Required from Chapter 10 (Stage 2 classifier)
```

### 4.2 Get the datasets

The Ultralytics dataset is fetched automatically by the `ultralytics` package on first use.

The Kaggle dataset requires a Kaggle API token. Place `~/.kaggle/kaggle.json` (or `~/.kaggle/access_token`) and run:

```powershell
python data\download_datasets.py
```

See [data/README.md](data/README.md) for detailed instructions.

### 4.3 Run the notebook

```powershell
jupyter lab notebooks\01_main_pipeline.ipynb
```

The notebook **auto-detects its runtime environment** — local machine, remote workstation (via SSH), or Google Colab — and resolves data paths accordingly. See Chapter 3 inside the notebook for the full environment-setup logic, and for instructions on how to set `PPE_DATA_ROOT` if the datasets live in a non-standard location.

### 4.4 Verify no duplicates (optional)

```powershell
python -m src.utils.deduplicate
```

This re-runs the pHash leakage check and reports any cross-split duplicates.

### 4.5 CLI scripts (planned)

The modules under `src/stage1_detection/`, `src/stage2_classification/`, and `src/pipeline/` are scaffolds for an eventual CLI extraction of the notebook logic. They currently raise `NotImplementedError` — all working code lives in the notebook.

---

## 5. Runtime Environments & Workflow

### 5.1 Auto-detected runtimes

The notebook auto-detects which environment it is running in by checking whether `google.colab` can be imported and whether `SSH_CONNECTION` is set.

| Environment | Detected by | Typical use |
|---|---|---|
| Google Colab | `import google.colab` succeeds + Drive mounted | Backup training when no other GPU is available |
| Remote workstation | `SSH_CONNECTION` env var set | **Primary training environment** — better GPU; code synced via `git pull` |
| Laptop | Neither of the above | EDA, debugging, development, post-training analysis |

### 5.2 Where the notebook looks for the data

The notebook searches for the `datasets/` folder in this order, and stops at the first one that exists:

1. The `PPE_DATA_ROOT` environment variable, if set (explicit override — recommended for non-standard setups).
2. `../datasets/` or `./datasets/` relative to the repository.
3. `~/datasets/` or `~/ppe-data/datasets/` (typical on Linux/SSH).
4. `/data/` or `/mnt/data/` (typical shared mounts on Linux).
5. Colab Drive: `/content/drive/MyDrive/Deep_Learning_course/datasets/`.
6. Windows Google Drive: `G:/My Drive/Deep_Learning_course/datasets/`.
7. Windows OneDrive: `~/OneDrive*/Desktop/.../Deep_Learning_course/datasets/`.

If none of these is found, a clear error message tells you how to set `PPE_DATA_ROOT`.

### 5.3 Cross-machine workflow (laptop ↔ workstation)

The team workflow is git-based:

1. **Develop on the laptop** — edit code/notebook, run EDA, debug.
2. **Commit + push from the laptop** — `git add && git commit && git push`.
3. **On the workstation: `git pull` + train** — run the long training jobs (Stage 1 / Stage 2).
4. **Commit + push results from the workstation** — logs, model artifacts, evaluation plots under `results/`.
5. **Pull back to the laptop and analyse** — open the new artifacts locally.

#### First-time setup on the workstation (Windows + PowerShell)

```powershell
# 1. Clone the repo
git clone https://github.com/Razelbaz1/ppe-compliance-detection.git
cd ppe-compliance-detection

# 2. Get the datasets — pick ONE option:
#    Option A: same Microsoft account on the workstation → OneDrive auto-syncs
#              datasets/ under C:\Users\<you>\OneDrive...\Deep_Learning_course\
#    Option B: robocopy from the laptop over LAN (datasets is ~430 MB)
robocopy `
  "\\<LAPTOP_NAME>\C$\Users\<you>\OneDrive...\Deep_Learning_course\datasets" `
  "C:\Users\<you>\datasets" `
  /E /R:2
#    Option C: re-download from scratch on the workstation
python data\download_datasets.py

# 3. Set PPE_DATA_ROOT (persistent across sessions, if datasets/ is non-standard)
[Environment]::SetEnvironmentVariable('PPE_DATA_ROOT', 'C:\Users\<you>\datasets', 'User')

# 4. Install packages
pip install -r requirements.txt
pip install "tensorflow>=2.15.0"   # required from notebook Chapter 10 onwards

# 5. (Optional) Run Jupyter with port-forwarding to the laptop
jupyter lab --no-browser --port=8888
# On the laptop, in a separate PowerShell window:
ssh -L 8888:localhost:8888 <you>@<workstation>
# Then open http://localhost:8888 in the browser.
```

#### Syncing results back to the laptop

```powershell
# On the workstation, after a training run:
cd C:\path\to\ppe-compliance-detection
git add results\
git commit -m "Stage 1: YOLOv8n 50 epochs"
git push

# On the laptop:
git pull
# results\ now contains the new artifacts.
```

> **⚠ Heads up:** YOLOv8 and CNN model weights can reach 50–200 MB. If `git push` rejects a file >100 MB, add it to `.gitignore` or use Git LFS. Plots, logs, and metric tables are always safe to commit directly.

### 5.4 Reproducibility (SEED = 42)

The notebook seeds `random`, `numpy`, `tensorflow`, and `torch` with `SEED = 42` so runs are reproducible.

**Caveat:** on a GPU, cuDNN's non-deterministic operations may produce slightly different results between runs even with the same seed. This is not significant for relative comparisons across models — it only affects exact bit-equivalence of outputs.

---

## 6. Results

Numeric success thresholds are deliberately **not set in advance**. They will be established by comparing the trained models against the baselines listed below — see chapters 8 and 11 of the notebook for the formal evaluation.

| Stage | Metrics tracked | Status |
|---|---|---|
| Stage 1 — Person Detection | `mAP@0.5`, with Precision and Recall breakdown | _TBD_ |
| Stage 2 — Helmet | `F1`, with Precision and Recall breakdown | _TBD_ |
| Stage 2 — Vest | `F1`, with Precision and Recall breakdown | _TBD_ |
| End-to-End | Compliance accuracy per worker | _TBD_ |
| Cross-Dataset Gap | (In-Domain F1) − (Out-of-Domain F1) | _TBD_ |

**Baselines for comparison** (computed inside the notebook):

1. **Majority-class** predictor — always predicts the most common class.
2. **Random per-class prior** — samples class according to its prevalence in train.
3. **Pretrained YOLOv8n without fine-tuning** — measures how much fine-tuning on construction data actually contributed.

A model that fails to clearly beat these three is, by definition, not useful regardless of its absolute number.

Results will be populated in [`results/`](results/) as each stage is trained.

---

## 7. Acknowledgements

- **Ultralytics** for the open YOLOv8 framework and the Construction-PPE dataset.
- **Roboflow / Snehil Sanyal** for the Kaggle Construction Site Safety dataset.
- **Dr. Anat Goldstein** (course instructor, Ariel University) for the project framing.

---

## 8. License

MIT — see [LICENSE](LICENSE).

# Project Handoff — PPE Compliance Detection Pipeline

**Last session:** 2026-06-07
**Status:** Chapters 0–12 complete. 13–17 pending.
**Primary deliverable:** `notebooks/01_main_pipeline.ipynb` (49 cells).

---

## 1. Where we are

### Chapter status

| # | Chapter | Status |
|---|---------|--------|
| 0 | Cover | ✅ |
| 1 | Introduction | ✅ |
| 2 | Theoretical background | ✅ |
| 3 | Environment setup | ✅ |
| 4 | Data split (Cross-Dataset strategy) | ✅ |
| 5 | EDA | ✅ |
| 6 | Deduplication | ✅ |
| 7 | Stage 1 data preparation | ✅ |
| 8 | Stage 1 — YOLOv8 training | ✅ |
| 9 | Stage 1 evaluation | ✅ |
| 10 | Stage 2 crop dataset construction | ✅ |
| 11 | Stage 2 — CNN training (refs Colab notebook) | ✅ |
| 12 | Stage 2 evaluation | ✅ |
| 13 | Stage 3 — Compliance Logic | ⏳ |
| 14 | End-to-End evaluation | ⏳ |
| 15 | Discussion | ⏳ |
| 16 | Summary | ⏳ |
| 17 | References | ⏳ |

### Headline numbers

**Stage 1 (YOLOv8n — Person detection)**

| Split | mAP@0.5 | Precision | Recall |
|---|---:|---:|---:|
| Val (Kaggle val) | 0.829 | 0.85 | 0.753 |
| Test In-Domain (Kaggle test) | 0.816 | 0.808 | 0.756 |
| Test Out-of-Domain (Ultralytics) | 0.617 | 0.693 | 0.604 |
| **Cross-Dataset Gap** | **+0.199 (~20 pp)** | +0.114 | +0.149 |

**Stage 2 (ResNet50 fine-tuned — Helmet/Vest classification)**

| Split | AUC | Precision | Recall |
|---|---:|---:|---:|
| Val (Kaggle val) | 0.974 | 0.939 | 0.838 |
| Test In-Domain (Kaggle test) | 0.979 | 0.954 | 0.906 |
| Test Out-of-Domain (Ultralytics) | 0.912 | 0.938 | 0.731 |
| **Cross-Dataset Gap** | **+0.067 (~7 pp)** | +0.015 | +0.174 |

**Key academic finding:** The Stage 2 Cross-Dataset Gap (7 pp AUC) is about **one third** of the Stage 1 gap (20 pp mAP). The two-stage architecture is validated: the classifier sees focused person crops without a shifting background, so ImageNet features (helmet colour, vest texture) transfer between Kaggle and Ultralytics; the detector has the harder problem of localising people against an unseen scene.

---

## 2. Repository structure

### In git
```
ppe-compliance-detection/
├── README.md                          synced with notebook
├── HANDOFF.md                         this file
├── .gitignore
├── requirements.txt
├── data/
│   ├── README.md
│   └── download_datasets.py
├── src/                               scaffolds, NotImplementedError
│   ├── utils/{deduplicate.py, analyze_data.py}
│   ├── stage1_detection/train.py
│   ├── stage2_classification/{build_crop_dataset.py, train.py}
│   └── pipeline/inference.py
├── notebooks/
│   ├── 01_main_pipeline.ipynb         the deliverable
│   └── stage2_colab.ipynb             standalone Colab notebook for TF/Keras training
└── results/
    ├── stage1/
    │   ├── yolov8n_50e/
    │   │   ├── weights/{best.pt, last.pt}   ← 6 MB each
    │   │   ├── results.csv, results.png
    │   │   └── confusion_matrix.png, BoxPR_curve.png, etc.
    │   └── evaluation/
    │       ├── summary.csv
    │       ├── comparison.png
    │       └── test_in_domain/, test_out_of_domain/   per-split PR curves
    └── stage2/
        ├── vgg16/{best.keras (57 MB), history.csv}    ablation baseline
        ├── resnet50/{best.keras (94 MB), history.csv}  ablation winner (FE)
        ├── resnet50_finetuned/history.csv             ← best.keras NOT in git (204 MB)
        ├── comparison.png                              ablation plot
        ├── finetune_trajectory.png                     FE→FT plot
        └── evaluation/{summary.csv, comparison.png}
```

### NOT in git
- `notebooks/_build_chapter*.py` — private build scripts (gitignored via `notebooks/_*.py`)
- `notebooks/_list_cells.py`, `_test_chapter5.py` — private utilities
- `datasets/` — never in git
- `results/stage2/resnet50_finetuned/best.keras` — too large (204 MB), lives in Google Drive

### Google Drive layout (Raz's personal account)
```
MyDrive/PPE_Project/
├── stage2_crops.zip                                   210 MB, source for Colab
└── stage2_results/
    ├── vgg16/{best.keras, history.csv}
    ├── resnet50/{best.keras, history.csv}
    ├── resnet50_finetuned/{best.keras, history.csv}   ← only place best.keras lives
    ├── comparison.png
    ├── finetune_trajectory.png
    └── evaluation/{summary.csv, comparison.png}
```

---

## 3. Environments

### Laptop (development, analysis)
- Windows 11, **Python 3.14** ← incompatible with TensorFlow (no wheels exist)
- pytorch 2.11.0+cpu, ultralytics 8.4.48
- numpy 2.3, pandas 2.3, matplotlib 3.10, opencv-python, PIL
- Can run: chapters 1–9, 10, 11 (display only), 12 (display only)
- Cannot run: any TF training/inference code
- `DATA_ROOT`: `G:\My Drive\Deep_Learning_course\datasets\` (via Google Drive Desktop)

### Workstation (training)
- HP Z6 G5 A, Windows, **Python 3.12.10**
- NVIDIA RTX PRO 4000 Blackwell (24 GB VRAM), CUDA driver 13.0
- pytorch **2.11.0+cu128** ← required for Blackwell sm_120, NOT cu126
- tensorflow 2.21.0 (CPU only on Windows — Google dropped TF GPU on Windows from 2.11)
- ultralytics 8.4.60
- `DATA_ROOT`: `C:\Users\LAB\RAZ\Deep_Learning_course\ppe-compliance-detection\datasets\`
- venv: `.\.venv\` inside the repo
- Communication: Claude-on-SSH (executor only, see §6)

### Colab (Stage 2 Keras training)
- T4 or L4 GPU
- All TensorFlow / Keras training happens here
- Mount path: `/content/drive/MyDrive/PPE_Project/`
- Extract path: `/content/stage2_crops/`

---

## 4. Workflow

### Standard development cycle
```
Laptop:   edit code/notebooks, EDA, analysis      → git push
Workstation: git pull → run training (PyTorch)    → git push (results)
Colab:    open stage2_colab.ipynb from GitHub     → run → results saved to Drive
Laptop:   git pull (or Drive sync)                → analyze, update main notebook
```

### Build-script pattern
Each chapter in `01_main_pipeline.ipynb` is generated by a private Python script
`notebooks/_build_chapterN.py`. To edit a chapter:
1. Edit the script
2. Run `python notebooks/_build_chapterN.py` (idempotent: replaces only that chapter's cells)
3. Run `python notebooks/_list_cells.py` to verify structure
4. `git add notebooks/01_main_pipeline.ipynb && commit && push`

The build scripts are gitignored (private to the laptop).

### Useful one-liners
```bash
# List all cells with IDs and line counts
python notebooks/_list_cells.py

# Run a specific cell headlessly (for local verification)
python -c "
import matplotlib; matplotlib.use('Agg')
import json
nb = json.load(open('notebooks/01_main_pipeline.ipynb', encoding='utf-8'))
cells = {c.get('id'): c for c in nb['cells']}
ns = {}
for cid in ['03-env-paths', '03-env-imports', '12-stage2-eval-load']:
    exec(''.join(cells[cid]['source']), ns)
"
```

---

## 5. Next chapters — design notes

### Chapter 13 — Stage 3 (Compliance Logic)
**Short and simple.** A pure function:
```python
def compliance(helmet: bool, vest: bool) -> dict:
    return {
        "compliant":     helmet and vest,
        "missing_items": [item for item, present in [("helmet", helmet), ("vest", vest)] if not present],
    }
```
Plus a brief markdown explaining the logic and showing a few sample applications. ~3 cells.

### Chapter 14 — End-to-End Evaluation
The integration test. For each test image:
1. Load `results/stage1/yolov8n_50e/weights/best.pt` (already in git)
2. Run YOLO → list of Person bboxes
3. For each bbox, crop and run through ResNet50 fine-tuned
4. Apply Stage 3 logic
5. Output: original image with worker boxes coloured by compliance status

**Decision pending:** where to run this. Options:
- **Workstation** has both PyTorch GPU AND TensorFlow CPU. Best place to run it. Requires uploading the 204 MB FT model to the workstation (one-time, e.g. via WeTransfer or Drive web).
- **Colab** is also viable — it has everything already; would write a new `end_to_end_colab.ipynb`.
- **Laptop** cannot run TF at all (Python 3.14).

**Metric to compute:** per-image compliance accuracy = matches between predicted and true compliance per worker. Plus the existing Stage 1 + Stage 2 metrics layered.

### Chapter 15 — Discussion
Markdown only. Cover:
- Why the Cross-Dataset Gap exists (class-prior shift, visual domain shift)
- Why Stage 2 generalizes better than Stage 1 (focused crops, smaller distribution shift in foreground)
- Limitations: small val set (150 crops), Strategy A label noise, single random seed (no `mean±std`)
- Future work: domain adaptation, more PPE classes (gloves, boots), multi-seed reporting, more aggressive augmentation

### Chapter 16 — Summary
One paragraph + final-numbers table + recommendations. Markdown only.

### Chapter 17 — References
- **Datasets**: Ultralytics Construction-PPE, Kaggle Construction Site Safety (Roboflow)
- **Models**: YOLOv8 (Ultralytics 2023), VGG16 (Simonyan & Zisserman 2014), ResNet50 (He et al. 2015)
- **Frameworks**: PyTorch, TensorFlow/Keras, Ultralytics
- **Course material**: Unit 8 (Transfer Learning, Cats vs Dogs)

---

## 6. Operational gotchas to remember

1. **PowerShell `Compress-Archive` writes ZIP entries with backslashes.** Linux/Colab cannot extract them as folders without normalisation. `stage2_colab.ipynb` cell 03 handles this manually.

2. **TensorFlow has no GPU on native Windows from 2.11+.** Use Colab or WSL2. Workstation TF runs CPU-only.

3. **Python 3.14 on the laptop has no TF wheels.** Only pandas/matplotlib code is runnable locally.

4. **PyTorch cu126 lacks Blackwell sm_120 kernels.** Workstation needs **cu128**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

5. **GitHub rejects files > 100 MB.** The fine-tuned ResNet50 (`best.keras`, ~204 MB) lives only in Drive. VGG16 (57 MB) and ResNet50 FE (94 MB) are in git, just below the warning threshold.

6. **PowerShell DataLoader multiprocessing crashes** when running Python via `exec(...)`. Set `workers=0` in YOLO training. Already in `08-train-run`.

7. **Git auth on workstation:** Windows Credential Manager doesn't always cooperate. The reliable workaround is the inline-PAT trick:
   ```powershell
   $origUrl = git remote get-url origin
   git remote set-url origin "https://Razelbaz1:<PAT>@github.com/Razelbaz1/ppe-compliance-detection.git"
   git push
   git remote set-url origin $origUrl
   ```

8. **`stage2_crops/processed_stage1/` location.** Currently lives inside `datasets/`, which is fine on the workstation but lands inside Google Drive on the laptop. That broke `shutil.rmtree` in chapter 7 once. Workaround: run heavy data-prep on the workstation, not the laptop.

9. **The val set is small (150 crops for Stage 2).** Numbers there are noisy. Test In-Domain (155) slightly beating val is just statistical variation, not overfitting.

---

## 7. Multi-Claude pattern (continued from prior session)

There are TWO Claude instances in this project:

- **Claude on the laptop (the "brain")** — plans architecture, writes code, makes decisions, updates the main notebook, handles git on the laptop side. Edits to `01_main_pipeline.ipynb` happen only here.
- **Claude on the workstation via SSH (the "hammer")** — execution only. Runs the exact commands it is given, returns stdout/stderr verbatim, never improvises, never edits the notebook, stops on error and reports the message.

Raz passes prompts between them by copy-paste. Every workstation instruction is a self-contained `INSTRUCTION N` block with explicit "stop on error, verbatim" framing.

---

## 8. User preferences (carried over from CLAUDE.md and memory)

- **Language:** Hebrew interaction, technical terms in English.
- **Explanations:** 3-layer format (intuition → mathematical formulation → why it matters).
- **Course material is authoritative.** Don't invent material not taught; explicitly flag anything ahead of the syllabus.
- **Keras-first** for any classifier work (course is Keras-first); PyTorch only for YOLO.
- **Notebooks/code in English** even though chat is in Hebrew.
- **Don't put "Raz" in public artifacts** (HTML, SVGs, summaries). Memory and CLAUDE.md are fine.
- **No drawings/diagrams without approval.** Pre-approval workflow described in `feedback_visual_workflow.md`.
- **Be terse.** No long summaries unless asked. No filler.

---

## 9. Quick-resume checklist for next session

Open this file, then:

1. `cd C:\Users\razel\github\ppe-compliance-detection && git pull`
2. `python notebooks/_list_cells.py` — confirm 49 cells, chapters 0–12 present
3. Inspect `results/stage2/evaluation/summary.csv` to recall the headline numbers
4. Read §5 above to choose the next chapter
5. Suggest Chapter 13 to the user (Stage 3 — Compliance Logic), it is small and quick

The user (Raz) will tell you what to do next. Default to building Chapter 13.

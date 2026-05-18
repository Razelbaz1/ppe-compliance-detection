# Data Folder

This directory is the **download target** for both datasets used by the pipeline.
None of the actual image files are committed to the repository — only this README
and the download script.

## Expected layout after running the download script

```
data/
├── construction-ppe/                   ← Ultralytics dataset
│   ├── data.yaml
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── kaggle-css/                         ← Kaggle dataset
    └── css-data/
        ├── train/{images,labels}/
        ├── valid/{images,labels}/
        └── test/{images,labels}/
```

## How to download

### Option A — `download_datasets.py` (recommended)

```bash
python data/download_datasets.py
```

This single script:
1. Downloads the **Ultralytics Construction-PPE** dataset (~178 MB) via the `ultralytics` library.
2. Downloads the **Kaggle Construction Site Safety** dataset (~206 MB) via the Kaggle API.
3. Verifies both directory structures.

### Prerequisite — Kaggle credentials

The Kaggle dataset requires an API token.

1. Sign in at <https://www.kaggle.com>.
2. Open *Account → Settings → API → Create New API Token*.
3. Save the token at one of:
   - `~/.kaggle/kaggle.json` (legacy JSON format)
   - `~/.kaggle/access_token` (new `KGAT_*` token format)

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

### Option B — manual

If the script fails for any reason:

- **Ultralytics:** triggered automatically the first time `ultralytics.YOLO(...).train(data='construction-ppe.yaml', ...)` runs.
- **Kaggle:** download the ZIP from <https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow> and extract it into `data/kaggle-css/`.

## After downloading — run the leakage check

```bash
python -m src.utils.deduplicate
```

This re-runs the pHash deduplication check and confirms that no image leaks between any train/val/test split combination across the two datasets. See [src/utils/deduplicate.py](../src/utils/deduplicate.py) for the actual logic.

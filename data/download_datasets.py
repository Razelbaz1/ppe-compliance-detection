"""
Downloads both datasets used by the PPE compliance pipeline.

  1. Ultralytics Construction-PPE  (~178 MB)
  2. Kaggle Construction Site Safety  (~206 MB)

Both end up under ./data/ next to this script.

Run from the repository root:
    python data/download_datasets.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent          # .../ppe-compliance-detection/data
REPO_ROOT = HERE.parent                          # .../ppe-compliance-detection

ULTRALYTICS_TARGET = HERE / "construction-ppe"
KAGGLE_TARGET = HERE / "kaggle-css"
KAGGLE_SLUG = "snehilsanyal/construction-site-safety-image-dataset-roboflow"


# ─────────────────────────────────────────────────────────────────────────────
# Ultralytics Construction-PPE
# ─────────────────────────────────────────────────────────────────────────────

def download_ultralytics() -> None:
    """Trigger the Ultralytics dataset download via the official YAML config."""
    print("\n[1/2] Ultralytics Construction-PPE")
    print(f"      target → {ULTRALYTICS_TARGET}")

    if (ULTRALYTICS_TARGET / "data.yaml").exists():
        print("      ✓ already present, skipping")
        return

    # Point Ultralytics' datasets_dir at this folder, then trigger the download
    from ultralytics import settings
    settings.update({"datasets_dir": str(HERE)})

    from ultralytics.data.utils import check_det_dataset
    check_det_dataset("construction-ppe.yaml")  # ← downloads + extracts
    print("      ✓ done")


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle Construction Site Safety
# ─────────────────────────────────────────────────────────────────────────────

def download_kaggle() -> None:
    """Download the Kaggle dataset via the official Kaggle API."""
    print("\n[2/2] Kaggle Construction Site Safety")
    print(f"      target → {KAGGLE_TARGET}")

    if (KAGGLE_TARGET / "css-data").exists():
        print("      ✓ already present, skipping")
        return

    KAGGLE_TARGET.mkdir(parents=True, exist_ok=True)

    # The Kaggle API reads credentials from ~/.kaggle/kaggle.json
    # or, for the newer access-token format, from ~/.kaggle/access_token.
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: WPS433

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        KAGGLE_SLUG,
        path=str(KAGGLE_TARGET),
        unzip=True,
        quiet=False,
    )
    print("      ✓ done")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def verify() -> None:
    print("\n[verify]")
    checks: list[tuple[str, Path, int]] = [
        ("Ultralytics train images", ULTRALYTICS_TARGET / "images" / "train", 1100),
        ("Ultralytics val images", ULTRALYTICS_TARGET / "images" / "val", 100),
        ("Ultralytics test images", ULTRALYTICS_TARGET / "images" / "test", 100),
        ("Kaggle train images", KAGGLE_TARGET / "css-data" / "train" / "images", 2500),
        ("Kaggle valid images", KAGGLE_TARGET / "css-data" / "valid" / "images", 100),
        ("Kaggle test images", KAGGLE_TARGET / "css-data" / "test" / "images", 70),
    ]
    ok = True
    for label, path, expected_min in checks:
        if not path.exists():
            print(f"      ✗ missing: {label} ({path})")
            ok = False
            continue
        n = sum(1 for f in path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        marker = "✓" if n >= expected_min else "⚠"
        print(f"      {marker} {label}: {n} images")
    if not ok:
        sys.exit(1)


def main() -> None:
    os.chdir(REPO_ROOT)  # so Ultralytics resolves the relative `datasets/` dir under HERE
    download_ultralytics()
    download_kaggle()
    verify()
    print("\nAll datasets ready. Next:  python -m src.utils.deduplicate")


if __name__ == "__main__":
    main()

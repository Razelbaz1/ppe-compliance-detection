"""
Stage 2 — build the per-person crop dataset.

For every Person bounding box in the merged detection corpus:
  1. Crop the box from the original image, resize to 224×224.
  2. Compute multi-label tags:
       helmet = 1 ⇔ at least one helmet/Hardhat box overlaps ≥ 30 % of its area with this person
       vest   = 1 ⇔ same, for vest/Safety-Vest boxes
  3. Save crop to data/merged/crops/{split}/<id>.jpg
     Save labels to data/merged/crops/{split}.csv  (cols: filename, helmet, vest)

Run:
    python -m src.stage2_classification.build_crop_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "data" / "merged" / "crops"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=224, help="Output crop side length")
    p.add_argument("--overlap", type=float, default=0.3,
                   help="Min fraction of a PPE box that must fall inside the person box "
                        "to count as worn. Default 0.30.")
    return p.parse_args()


def main() -> None:
    args = parse_args()    # noqa: F841

    # TODO — implement once Stage 1 is in place. Planned flow:
    #   for each (image, label) pair across both datasets:
    #       parse the YOLO label file
    #       for each Person bbox:
    #           crop & resize → OUT_ROOT/{split}/<id>.jpg
    #           determine helmet/vest tags via overlap test
    #           append row to OUT_ROOT/{split}.csv
    raise NotImplementedError(
        "Stage-2 crop extraction is scaffolded but not yet implemented."
    )


if __name__ == "__main__":
    main()

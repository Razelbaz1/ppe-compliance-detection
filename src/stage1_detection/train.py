"""
Stage 1 — Train a YOLOv8 person detector on the merged Construction-PPE corpus.

The merged corpus is currently kept as two side-by-side directories
(data/construction-ppe and data/kaggle-css) because their original
train/val/test splits are respected. This script builds a YOLO-format
data.yaml on the fly that points at both and collapses every "Person"
variant to a single class.

Run:
    python -m src.stage1_detection.train --model yolov8n.pt --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolov8n.pt",
                   help="Pre-trained YOLOv8 weight (n/s/m/l/x). Default: yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default=str(REPO_ROOT / "results" / "stage1"))
    p.add_argument("--name", default="run")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # TODO — implement once data is in place:
    #   1. Build a temporary data.yaml that points at:
    #      data/construction-ppe/images/{train,val,test} (re-mapped to "Person" only)
    #      data/kaggle-css/css-data/{train,valid,test}/images (re-mapped likewise)
    #   2. Re-write the YOLO labels into a side directory where every
    #      person-related class is collapsed to class 0 ("Person").
    #   3. Call `ultralytics.YOLO(args.model).train(data=tmp_yaml, ...)`
    #
    # For now, scaffold only:
    raise NotImplementedError(
        "Stage-1 training is scaffolded but not yet implemented. "
        "See module docstring for the planned flow."
    )


if __name__ == "__main__":
    main()

"""
Stage 2 — train the multi-label PPE classifier.

Three model strategies, compared during training (per Lecture 8 of the course):
  1. From-scratch CNN baseline
  2. Feature extraction — frozen ImageNet backbone, train classifier head only
  3. Fine-tuning — backbone partially unfrozen + lower learning rate

Loss: Binary Cross-Entropy with logits, one head per PPE class (helmet, vest).

Run:
    python -m src.stage2_classification.train --backbone resnet50 --epochs 30 --strategy fine-tune
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    p.add_argument("--strategy", choices=["scratch", "feature-extract", "fine-tune"],
                   default="fine-tune")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "stage2"))
    return p.parse_args()


def main() -> None:
    args = parse_args()    # noqa: F841

    # TODO — implement once Stage-2 crops exist (build_crop_dataset.py). Planned flow:
    #   1. Load CSVs at data/merged/crops/{train,val,test}.csv
    #   2. Build a tf.data / torch DataLoader that yields (img, [helmet, vest]) pairs
    #   3. Instantiate the chosen backbone + 2-unit sigmoid head
    #   4. Compile with BCE loss, accuracy & per-label F1 metrics
    #   5. Train, save best checkpoint to results/stage2/<run-id>/
    raise NotImplementedError(
        "Stage-2 training is scaffolded but not yet implemented."
    )


if __name__ == "__main__":
    main()

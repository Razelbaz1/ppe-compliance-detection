"""
End-to-end pipeline — run Stage 1 → Stage 2 → Compliance Logic on a single image.

For each worker detected by Stage 1, crops the bounding box, runs Stage 2,
and applies the rule

    compliant  ⇔  helmet = 1 ∧ vest = 1

Draws a green box around compliant workers and a red box around non-compliant ones,
then writes the annotated image to disk and prints a JSON summary.

Run:
    python -m src.pipeline.inference --image path/to/site.jpg
    python -m src.pipeline.inference --image path/to/site.jpg --out annotated.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_S1_WEIGHTS = REPO_ROOT / "results" / "stage1" / "run" / "weights" / "best.pt"
DEFAULT_S2_WEIGHTS = REPO_ROOT / "results" / "stage2" / "best.keras"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="Path to the input image.")
    p.add_argument("--out", default=None, help="Where to save the annotated image.")
    p.add_argument("--s1-weights", default=str(DEFAULT_S1_WEIGHTS),
                   help="Stage-1 YOLOv8 checkpoint.")
    p.add_argument("--s2-weights", default=str(DEFAULT_S2_WEIGHTS),
                   help="Stage-2 classifier checkpoint.")
    return p.parse_args()


def main() -> None:
    args = parse_args()    # noqa: F841

    # TODO — implement once both stages are trained. Planned flow:
    #   1. Load Stage-1 YOLO model (ultralytics.YOLO(args.s1_weights))
    #   2. Run inference → list of person boxes
    #   3. For each box: crop → resize → Stage-2 forward pass → (helmet_pred, vest_pred)
    #   4. Apply compliance rule (helmet ∧ vest)
    #   5. Draw boxes (green/red) on a copy of the image
    #   6. Save image and emit JSON: list of {bbox, helmet, vest, compliant}
    raise NotImplementedError(
        "End-to-end pipeline is scaffolded but not yet implemented."
    )


if __name__ == "__main__":
    main()

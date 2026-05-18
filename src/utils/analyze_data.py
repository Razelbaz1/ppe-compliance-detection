"""
Compute the unified class distribution and Stage-2 compliance breakdown
across the merged dataset (Ultralytics + Kaggle CSS).

Run from the repository root:
    python -m src.utils.analyze_data
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
ULT_LABELS = REPO_ROOT / "data" / "construction-ppe" / "labels"
KAG_LABELS = REPO_ROOT / "data" / "kaggle-css" / "css-data"


@dataclass(frozen=True)
class DatasetSpec:
    label_root: Path
    splits: tuple[str, ...]
    person_cls: int
    helmet_cls: int
    vest_cls: int
    no_helmet_cls: int | None
    no_vest_cls: int | None
    label_subdir: str | None      # "labels" for Kaggle layout, None for Ultralytics


ULTRALYTICS = DatasetSpec(
    label_root=ULT_LABELS,
    splits=("train", "val", "test"),
    person_cls=6, helmet_cls=0, vest_cls=2,
    no_helmet_cls=7, no_vest_cls=None,           # Ultralytics has no explicit no_vest
    label_subdir=None,
)

KAGGLE = DatasetSpec(
    label_root=KAG_LABELS,
    splits=("train", "valid", "test"),
    person_cls=5, helmet_cls=0, vest_cls=7,
    no_helmet_cls=2, no_vest_cls=4,
    label_subdir="labels",
)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO label parsing
# ─────────────────────────────────────────────────────────────────────────────

Box = tuple[float, float, float, float]   # x1, y1, x2, y2 (normalised)


def _parse_label_file(path: Path) -> list[tuple[int, Box]]:
    rows: list[tuple[int, Box]] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        rows.append((cls, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return rows


def _overlap_ratio_inside_person(person: Box, item: Box) -> float:
    """Fraction of *item* that lies inside the *person* box."""
    xa, ya = max(person[0], item[0]), max(person[1], item[1])
    xb, yb = min(person[2], item[2]), min(person[3], item[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    item_area = (item[2] - item[0]) * (item[3] - item[1])
    return inter / (item_area + 1e-9)


def _iter_label_files(spec: DatasetSpec) -> Iterable[Path]:
    for split in spec.splits:
        if spec.label_subdir is None:
            yield from (spec.label_root / split).glob("*.txt")
        else:
            yield from (spec.label_root / split / spec.label_subdir).glob("*.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def class_distribution() -> Counter[str]:
    counts: Counter[str] = Counter()
    for spec in (ULTRALYTICS, KAGGLE):
        for label_file in _iter_label_files(spec):
            for cls, _ in _parse_label_file(label_file):
                if cls == spec.person_cls:    counts["Person"] += 1
                if cls == spec.helmet_cls:    counts["Helmet"] += 1
                if cls == spec.vest_cls:      counts["Vest"] += 1
                if cls == spec.no_helmet_cls: counts["No-Helmet"] += 1
                if spec.no_vest_cls is not None and cls == spec.no_vest_cls:
                    counts["No-Vest"] += 1
    return counts


def stage2_compliance_breakdown(overlap_threshold: float = 0.3) -> Counter[tuple[int, int]]:
    """Counts of (helmet, vest) ∈ {0,1}² per person box."""
    combo: Counter[tuple[int, int]] = Counter()
    for spec in (ULTRALYTICS, KAGGLE):
        for label_file in _iter_label_files(spec):
            rows = _parse_label_file(label_file)
            persons = [b for c, b in rows if c == spec.person_cls]
            helmets = [b for c, b in rows if c == spec.helmet_cls]
            vests = [b for c, b in rows if c == spec.vest_cls]
            for person in persons:
                has_h = any(_overlap_ratio_inside_person(person, h) > overlap_threshold for h in helmets)
                has_v = any(_overlap_ratio_inside_person(person, v) > overlap_threshold for v in vests)
                combo[(int(has_h), int(has_v))] += 1
    return combo


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not ULT_LABELS.exists() or not KAG_LABELS.exists():
        sys.exit("Datasets not found. Run `python data/download_datasets.py` first.")

    counts = class_distribution()
    combo = stage2_compliance_breakdown()
    total = sum(combo.values())

    print("Class distribution (5 unified classes):")
    for cls in ("Person", "Helmet", "Vest", "No-Vest", "No-Helmet"):
        print(f"  {cls:<10} {counts.get(cls, 0):>6}")

    print(f"\nStage-2 compliance breakdown ({total} person crops):")
    labels = {
        (1, 1): "compliant      (helmet + vest)",
        (1, 0): "missing vest",
        (0, 1): "missing helmet",
        (0, 0): "missing both",
    }
    for key, label in labels.items():
        n = combo.get(key, 0)
        pct = 100 * n / total if total else 0
        print(f"  {label:<30} {n:>6}  ({pct:5.1f}%)")

    helmet_1 = combo.get((1, 1), 0) + combo.get((1, 0), 0)
    vest_1 = combo.get((1, 1), 0) + combo.get((0, 1), 0)
    print(f"\nMarginals:")
    print(f"  helmet = 1        {helmet_1:>6}  ({100*helmet_1/total:5.1f}%)")
    print(f"  vest   = 1        {vest_1:>6}  ({100*vest_1/total:5.1f}%)")
    print(f"  compliant         {combo.get((1,1), 0):>6}  ({100*combo.get((1,1), 0)/total:5.1f}%)")


if __name__ == "__main__":
    main()

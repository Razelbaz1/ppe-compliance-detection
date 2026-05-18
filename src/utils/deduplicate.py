"""
Perceptual-hash deduplication & cross-split leakage check.

Scans every image in both datasets (Ultralytics Construction-PPE and Kaggle CSS),
computes a 64-bit pHash, and reports any pair of images whose Hamming distance is
below a threshold. Matches are broken down by (Ultralytics-split, Kaggle-split)
to expose train→test leakage.

Run from the repository root:
    python -m src.utils.deduplicate
    python -m src.utils.deduplicate --threshold 8 --delete

Without --delete this script is read-only.

Outputs:
    - Matrix of cross-split match counts (printed to stdout)
    - Full list of matching pairs (printed to stdout, sorted by distance)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imagehash
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Dataset locations & configuration
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
ULT_ROOT = REPO_ROOT / "data" / "construction-ppe" / "images"
KAG_ROOT = REPO_ROOT / "data" / "kaggle-css" / "css-data"

ULT_SPLITS = ("train", "val", "test")
KAG_SPLITS = ("train", "valid", "test")


@dataclass
class HashedImage:
    source: str        # "U" or "K"
    split: str
    path: Path
    hash: imagehash.ImageHash


# ─────────────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────────────

def _iter_images(root: Path) -> Iterable[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        yield from root.glob(ext)


def _hash_split(root: Path, source: str, split: str) -> list[HashedImage]:
    hashed: list[HashedImage] = []
    for img_path in _iter_images(root):
        try:
            with Image.open(img_path) as im:
                h = imagehash.phash(im)
            hashed.append(HashedImage(source, split, img_path, h))
        except Exception as exc:                # noqa: BLE001
            print(f"      skipped {img_path.name}: {exc}", file=sys.stderr)
    return hashed


def hash_all() -> tuple[list[HashedImage], list[HashedImage]]:
    """Compute pHash for every image in both datasets."""
    if not ULT_ROOT.exists() or not KAG_ROOT.exists():
        sys.exit(
            "Datasets not found. Run `python data/download_datasets.py` first."
        )

    print("Computing pHash for Ultralytics …")
    ult: list[HashedImage] = []
    for s in ULT_SPLITS:
        ult += _hash_split(ULT_ROOT / s, "U", s)

    print("Computing pHash for Kaggle CSS …")
    kag: list[HashedImage] = []
    for s in KAG_SPLITS:
        kag += _hash_split(KAG_ROOT / s / "images", "K", s)

    print(f"  Ultralytics: {len(ult):>5} hashed")
    print(f"  Kaggle CSS:  {len(kag):>5} hashed")
    return ult, kag


# ─────────────────────────────────────────────────────────────────────────────
# Cross-split comparison
# ─────────────────────────────────────────────────────────────────────────────

def cross_dataset_matches(
    ult: list[HashedImage],
    kag: list[HashedImage],
    threshold: int,
) -> list[tuple[int, HashedImage, HashedImage]]:
    """Return all (distance, ult_image, kaggle_image) tuples within threshold."""
    matches: list[tuple[int, HashedImage, HashedImage]] = []
    for u in ult:
        for k in kag:
            d = u.hash - k.hash
            if d <= threshold:
                matches.append((d, u, k))
    matches.sort(key=lambda x: x[0])
    return matches


def within_dataset_matches(
    images: list[HashedImage],
    threshold: int,
) -> list[tuple[int, HashedImage, HashedImage]]:
    """Return pairs of images within the same dataset that share an identical hash."""
    by_hash: dict[imagehash.ImageHash, list[HashedImage]] = defaultdict(list)
    for img in images:
        by_hash[img.hash].append(img)
    matches: list[tuple[int, HashedImage, HashedImage]] = []
    for items in by_hash.values():
        if len(items) > 1:
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    matches.append((0, items[i], items[j]))
    # also catch near-matches above 0 — but for 'within' we usually inspect exact only
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_matrix(matches: list[tuple[int, HashedImage, HashedImage]], threshold: int) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for _, u, k in matches:
        counts[(u.split, k.split)] += 1

    print(f"\nCross-dataset matrix (Hamming distance ≤ {threshold}):")
    print(f"{'':>10}  {'K-train':>10}  {'K-valid':>10}  {'K-test':>10}")
    for usp in ULT_SPLITS:
        row = f"{'U-' + usp:>10}  "
        for ksp in KAG_SPLITS:
            n = counts[(usp, ksp)]
            row += f"{'✓' if n == 0 else '⚠ ' + str(n):>10}  "
        print(row)


def print_matches(matches: list[tuple[int, HashedImage, HashedImage]]) -> None:
    if not matches:
        print("  ✓ no matches")
        return
    print(f"\nMatching pairs (sorted by distance, {len(matches)} total):")
    for d, u, k in matches:
        print(f"  dist={d}  [U/{u.split}] {u.path.name}  ↔  [K/{k.split}] {k.path.name[:50]}")


def remove_pairs(matches: list[tuple[int, HashedImage, HashedImage]]) -> None:
    """Delete the Kaggle side of every cross-dataset match (and its label)."""
    print(f"\nDeleting {len(matches)} Kaggle-side images …")
    deleted = 0
    for _, _u, k in matches:
        if not k.path.exists():
            continue
        k.path.unlink()
        label = k.path.parent.parent / "labels" / (k.path.stem + ".txt")
        if label.exists():
            label.unlink()
        deleted += 1
    print(f"  ✓ deleted {deleted} images (and labels)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=5,
                        help="Hamming-distance threshold (default: 5)")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete the Kaggle side of every match")
    args = parser.parse_args()

    t0 = time.time()
    ult, kag = hash_all()
    print(f"  ({time.time() - t0:.0f}s)")

    cross = cross_dataset_matches(ult, kag, args.threshold)
    print_matrix(cross, args.threshold)
    print_matches(cross)

    within_u = within_dataset_matches(ult, 0)
    within_k = within_dataset_matches(kag, 0)
    print(f"\nWithin-dataset exact duplicates:")
    print(f"  Ultralytics: {len(within_u)} pair(s)")
    print(f"  Kaggle CSS:  {len(within_k)} pair(s)")

    if args.delete and cross:
        remove_pairs(cross)


if __name__ == "__main__":
    main()

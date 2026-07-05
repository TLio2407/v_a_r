"""
Validate and clean raw drone imagery before COLMAP.

- Drops corrupt/unreadable files
- Optionally drops very blurry frames (motion blur is common in drone footage
  and hurts SfM matching + splat quality)
- Resizes to a max dimension for faster SfM + training while preserving
  aspect ratio (full-res originals are kept untouched)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from utils import get_logger, list_images, ensure_dir, variance_of_laplacian

logger = get_logger("dataset_prep")


def resize_longest_edge(image, max_dim: int):
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def prepare(images_dir: Path, out_dir: Path, max_dim: int | None, blur_threshold: float | None,
            masks_dir: Path | None = None):
    out_dir = ensure_dir(out_dir)
    out_masks_dir = ensure_dir(out_dir.parent / (out_dir.name + "_masks")) if masks_dir else None

    files = list_images(images_dir)
    kept, dropped_corrupt, dropped_blur = 0, 0, 0

    for f in tqdm(files, desc="Processing frames"):
        img = cv2.imread(str(f))
        if img is None:
            logger.warning(f"Skipping unreadable file: {f}")
            dropped_corrupt += 1
            continue

        if blur_threshold is not None:
            score = variance_of_laplacian(img)
            if score < blur_threshold:
                dropped_blur += 1
                continue

        if max_dim is not None:
            img = resize_longest_edge(img, max_dim)

        cv2.imwrite(str(out_dir / f.name), img)

        if masks_dir is not None:
            mask_path = Path(masks_dir) / f.name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if max_dim is not None:
                    mask = resize_longest_edge(mask, max_dim)
                cv2.imwrite(str(out_masks_dir / f.name), mask)

        kept += 1

    logger.info(f"Kept {kept}/{len(files)} frames "
                f"(dropped {dropped_corrupt} corrupt, {dropped_blur} too blurry)")
    if kept < 20:
        logger.warning("Fewer than 20 usable frames — SfM and 3DGS quality will likely suffer. "
                        "Consider loosening the blur threshold or capturing more views.")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, type=Path, help="Raw drone image directory")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for processed images")
    ap.add_argument("--masks", type=Path, default=None, help="Optional directory of transient-object masks")
    ap.add_argument("--max-image-dim", type=int, default=1600)
    ap.add_argument("--blur-threshold", type=float, default=None,
                     help="Drop frames with variance-of-Laplacian below this. Omit to disable.")
    args = ap.parse_args()

    prepare(args.images, args.out, args.max_image_dim, args.blur_threshold, args.masks)


if __name__ == "__main__":
    main()

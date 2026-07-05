"""
Runs COLMAP sparse reconstruction (SfM) on processed drone images.

Pipeline: feature_extractor -> {sequential,exhaustive}_matcher -> mapper -> undistorter

Sequential matching is strongly recommended for drone footage: frames are
captured in flight order, so sequential + loop-closure matching is far
cheaper than exhaustive matching (O(n) vs O(n^2)) while still recovering
loop closures around the tower.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils import get_logger, ensure_dir, run_cmd

logger = get_logger("colmap_runner")


def run_colmap(images_dir: Path, workdir: Path, camera_model: str = "OPENCV",
                matcher: str = "sequential", single_camera: bool = True, gpu: bool = True):
    workdir = ensure_dir(workdir)
    db_path = workdir / "database.db"
    sparse_dir = ensure_dir(workdir / "sparse")
    dense_dir = ensure_dir(workdir / "dense")
    gpu_flag = "1" if gpu else "0"

    # 1. Feature extraction
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--SiftExtraction.use_gpu", gpu_flag,
    ], logger)

    # 2. Matching
    if matcher == "sequential":
        run_cmd([
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", gpu_flag,
            "--SequentialMatching.loop_detection", "1",
        ], logger)
    elif matcher == "exhaustive":
        run_cmd([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", gpu_flag,
        ], logger)
    else:
        raise ValueError(f"Unknown matcher '{matcher}' (use 'sequential' or 'exhaustive')")

    # 3. Sparse reconstruction (mapper)
    run_cmd([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ], logger)

    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        raise RuntimeError(
            f"COLMAP mapper did not produce a reconstruction at {model_dir}. "
            "Check overlap between images / try exhaustive matching / verify camera_model."
        )

    # 4. Undistort images so downstream NVS training uses pinhole-consistent pixels
    run_cmd([
        "colmap", "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(model_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ], logger)

    logger.info(f"COLMAP reconstruction complete. Sparse model: {model_dir}, "
                f"undistorted images/model: {dense_dir}")
    return dense_dir  # contains images/ and sparse/ (undistorted, pinhole-consistent)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--camera-model", default="OPENCV")
    ap.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive"])
    ap.add_argument("--single-camera", action="store_true", default=True)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    run_colmap(args.images, args.workdir, args.camera_model, args.matcher,
               args.single_camera, gpu=not args.no_gpu)


if __name__ == "__main__":
    main()

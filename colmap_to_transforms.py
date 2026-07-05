"""
Converts an undistorted COLMAP sparse model (cameras.bin/images.bin/points3D.bin,
as produced by `colmap image_undistorter`) into:

  1. transforms.json         — nerfstudio's camera-pose format, so you can
                                train with the "nerfstudio" dataparser and,
                                more importantly, reuse the exact same
                                convention to describe *requested novel
                                viewpoints* at submission time.
  2. <out>/images/            — copy/symlink of the undistorted images
  3. <out>/colmap/sparse/0/   — symlink to the original model, so `ns-train
                                splatfacto --data <out> colmap` also works
                                out of the box if you prefer nerfstudio's
                                native COLMAP dataparser instead.

COLMAP camera convention: +X right, +Y down, +Z forward (looking direction).
Nerfstudio/OpenGL convention: +X right, +Y up, +Z backward.
We apply the standard flip of the Y and Z axes of the camera-to-world matrix
when converting between the two.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from utils import get_logger, ensure_dir

logger = get_logger("colmap_to_transforms")


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


COLMAP_TO_NERFSTUDIO = np.diag([1, -1, -1, 1])  # flip Y and Z of camera-to-world


def convert(colmap_model_dir: Path, images_dir: Path, out_dir: Path):
    import pycolmap

    out_dir = ensure_dir(out_dir)
    out_images_dir = ensure_dir(out_dir / "images")

    reconstruction = pycolmap.Reconstruction(str(colmap_model_dir))
    logger.info(f"Loaded COLMAP model: {len(reconstruction.images)} images, "
                f"{len(reconstruction.cameras)} camera(s), "
                f"{len(reconstruction.points3D)} 3D points")

    if len(reconstruction.cameras) != 1:
        logger.warning(f"{len(reconstruction.cameras)} distinct cameras found — "
                        "per-frame intrinsics will be written instead of a single global set.")

    frames = []
    shared_intrinsics = None

    for image_id, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]
        params = camera.params  # OPENCV model: fx, fy, cx, cy, k1, k2, p1, p2 (post-undistortion: k*=0)

        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]

        # COLMAP stores world-to-camera as (R, t); invert to get camera-to-world.
        # NOTE: in some pycolmap versions cam_from_world is a property (Rigid3d),
        # in others it's a callable method — handle both transparently.
        cam_from_world = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        R_wc = qvec2rotmat(np.array(cam_from_world.rotation.quat))
        t_wc = np.array(cam_from_world.translation)
        c2w = np.eye(4)
        c2w[:3, :3] = R_wc.T
        c2w[:3, 3] = -R_wc.T @ t_wc
        c2w = c2w @ COLMAP_TO_NERFSTUDIO  # convert axis convention

        src_img_path = Path(images_dir) / image.name
        dst_img_path = out_images_dir / image.name
        if not dst_img_path.exists() and src_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)

        frame = {
            "file_path": f"images/{image.name}",
            "transform_matrix": c2w.tolist(),
            "w": int(camera.width),
            "h": int(camera.height),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }
        frames.append(frame)
        shared_intrinsics = (camera.width, camera.height, fx, fy, cx, cy)

    if not frames:
        raise RuntimeError("No registered images found in the COLMAP model — reconstruction likely failed.")

    transforms = {
        "camera_model": "OPENCV",
        "frames": sorted(frames, key=lambda f: f["file_path"]),
    }
    # If all cameras share intrinsics, hoist them to top level (nerfstudio convention)
    if len(reconstruction.cameras) == 1:
        w, h, fx, fy, cx, cy = shared_intrinsics
        transforms.update({"w": w, "h": h, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy})

    out_json = out_dir / "transforms.json"
    with open(out_json, "w") as f:
        json.dump(transforms, f, indent=2)
    logger.info(f"Wrote {out_json} with {len(frames)} registered frames "
                f"(dropped {len(list(Path(images_dir).glob('*')))} - {len(frames)} unregistered/failed images)")

    # Also expose the raw COLMAP model for the native "colmap" dataparser as a fallback path.
    colmap_out = ensure_dir(out_dir / "colmap" / "sparse" / "0")
    for fname in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]:
        src = Path(colmap_model_dir) / fname
        if src.exists():
            shutil.copy2(src, colmap_out / fname)

    return out_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--colmap", required=True, type=Path,
                     help="Path to undistorted COLMAP sparse model dir (contains cameras.bin, images.bin, points3D.bin) "
                          "— i.e. <colmap_workdir>/dense/sparse")
    ap.add_argument("--images", required=True, type=Path,
                     help="Path to undistorted images dir — i.e. <colmap_workdir>/dense/images")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    convert(args.colmap, args.images, args.out)


if __name__ == "__main__":
    main()
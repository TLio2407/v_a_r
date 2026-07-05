"""
Packages rendered novel-view images into the contest submission format.

The exact required layout (filenames, resolution, zip vs folder, manifest
file) depends on VAR-2026's submission spec, which isn't public without
logging in. This script centralizes that logic in one place so you only
need to edit `build_manifest()` / `SUBMISSION_LAYOUT` once you have the
sample-submission spec, rather than hunting through the pipeline.

Current default behavior: copies all renders into
  <out>/<scene_id>/<original_filename>
and writes a manifest.json listing scene_id, image count, and checksums,
then zips the result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from utils import get_logger, ensure_dir, list_images

logger = get_logger("submission")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(scene_id: str, files: list[Path]) -> dict:
    return {
        "scene_id": scene_id,
        "num_images": len(files),
        "images": [
            {"file_name": f.name, "md5": md5sum(f)}
            for f in files
        ],
    }


def package(renders_dir: Path, scene_id: str, out_dir: Path) -> Path:
    out_dir = ensure_dir(out_dir)
    staging = ensure_dir(out_dir / scene_id)

    files = list_images(renders_dir)
    for f in files:
        shutil.copy2(f, staging / f.name)

    manifest = build_manifest(scene_id, files)
    with open(staging / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    zip_path = out_dir / f"{scene_id}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in staging.rglob("*"):
            zf.write(p, arcname=p.relative_to(staging.parent))

    logger.info(f"Packaged {len(files)} images for scene '{scene_id}' -> {zip_path}")
    return zip_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renders", required=True, type=Path)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    package(args.renders, args.scene_id, args.out)


if __name__ == "__main__":
    main()

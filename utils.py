"""Shared helpers used across the pipeline stages."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
                               datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def list_images(directory: Path, exts: Iterable[str] = (".jpg", ".jpeg", ".png")) -> list[Path]:
    directory = Path(directory)
    files = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images with extensions {exts} found in {directory}")
    return files


def variance_of_laplacian(image: np.ndarray) -> float:
    """Blur metric: higher = sharper. Used to drop near-duplicate/blurry drone frames."""
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_cmd(cmd: list[str], logger: logging.Logger | None = None) -> None:
    """Run a subprocess command, streaming output, raising on non-zero exit."""
    import subprocess
    logger = logger or get_logger("run_cmd")
    logger.info("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}")

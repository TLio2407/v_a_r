"""
COLMAP binary format parser for reading camera poses, intrinsics, and 3D point clouds.
Reference: https://colmap.github.io/format.html
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def read_next_bytes(fid, num_bytes: int) -> bytes:
    """Read the next bytes from a binary file."""
    return fid.read(num_bytes)


def read_cameras(path: Path) -> Dict[int, Dict]:
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = struct.unpack("Q", read_next_bytes(fid, 8))[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24)
            camera_id, model_id, width, height = struct.unpack("IIQ", camera_properties[:16])
            params_size = struct.unpack("Q", camera_properties[16:24])[0]
            params = np.fromfile(fid, np.float64, params_size)
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images(path: Path) -> Dict[int, Dict]:
    """Read COLMAP images.bin file."""
    images = {}
    with open(path, "rb") as fid:
        num_images = struct.unpack("Q", read_next_bytes(fid, 8))[0]
        for _ in range(num_images):
            binary_image_properties = read_next_bytes(fid, 64)
            image_id, qvec_0, qvec_1, qvec_2, qvec_3, tvec_0, tvec_1, tvec_2 = struct.unpack(
                "Qddddddd", binary_image_properties
            )
            qvec = np.array([qvec_0, qvec_1, qvec_2, qvec_3])
            tvec = np.array([tvec_0, tvec_1, tvec_2])

            camera_id = struct.unpack("I", read_next_bytes(fid, 4))[0]
            image_name_len = struct.unpack("Q", read_next_bytes(fid, 8))[0]
            image_name = read_next_bytes(fid, image_name_len).decode("utf-8")

            num_points2d = struct.unpack("Q", read_next_bytes(fid, 8))[0]
            xys = np.fromfile(fid, np.float64, 2 * num_points2d).reshape(-1, 2) if num_points2d > 0 else np.array([])
            point3d_ids = np.fromfile(fid, np.int64, num_points2d) if num_points2d > 0 else np.array([])

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
                "xys": xys,
                "point3d_ids": point3d_ids,
            }
    return images


def read_points3d(path: Path) -> Dict[int, Dict]:
    """Read COLMAP points3D.bin file."""
    points3d = {}
    with open(path, "rb") as fid:
        num_points = struct.unpack("Q", read_next_bytes(fid, 8))[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43)
            point3d_id, x, y, z, r, g, b, error, track_length = struct.unpack(
                "Qfffcccfd", binary_point_line_properties
            )
            track_length = int(track_length)
            
            # Read track elements (image_id, point2d_idx pairs)
            image_ids = np.fromfile(fid, np.uint32, track_length)
            point2d_idxs = np.fromfile(fid, np.uint32, track_length)

            xyz = np.array([x, y, z], dtype=np.float32)
            rgb = np.array([
                int.from_bytes(r, "little") if isinstance(r, bytes) else ord(r),
                int.from_bytes(g, "little") if isinstance(g, bytes) else ord(g),
                int.from_bytes(b, "little") if isinstance(b, bytes) else ord(b),
            ])
            
            points3d[point3d_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "image_ids": image_ids,
                "point2d_idxs": point2d_idxs,
            }
    return points3d


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )
    return R


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

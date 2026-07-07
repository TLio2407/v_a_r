"""
Camera utilities for handling poses, intrinsics, and transformations.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path


class CameraIntrinsics:
    """Camera intrinsic parameters."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, width: int, height: int):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    def to_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix K."""
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32
        )

    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }


class CameraPose:
    """Camera extrinsic parameters (rotation and translation)."""

    def __init__(self, quat: np.ndarray, tvec: np.ndarray):
        """
        Args:
            quat: quaternion [w, x, y, z] or [x, y, z, w] depending on convention
            tvec: translation vector [x, y, z]
        """
        self.quat = quat.astype(np.float32)
        self.tvec = tvec.astype(np.float32)

    def to_matrix(self) -> np.ndarray:
        """Return 4x4 camera extrinsic matrix [R | t]."""
        from colmap_parser import qvec2rotmat

        R = qvec2rotmat(self.quat)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = self.tvec
        return T

    def get_rotation_matrix(self) -> np.ndarray:
        """Return 3x3 rotation matrix."""
        from colmap_parser import qvec2rotmat

        return qvec2rotmat(self.quat)

    def get_translation(self) -> np.ndarray:
        """Return translation vector."""
        return self.tvec.copy()

    def to_dict(self) -> dict:
        return {"quat": self.quat.tolist(), "tvec": self.tvec.tolist()}


class Camera:
    """Complete camera with intrinsics and extrinsics."""

    def __init__(self, intrinsics: CameraIntrinsics, pose: CameraPose):
        self.intrinsics = intrinsics
        self.pose = pose

    def get_projection_matrix(self) -> np.ndarray:
        """Return 3x4 projection matrix P = K [R | t]."""
        K = self.intrinsics.to_matrix()
        T = self.pose.to_matrix()
        return K @ T[:3, :]

    def to_dict(self) -> dict:
        return {"intrinsics": self.intrinsics.to_dict(), "pose": self.pose.to_dict()}


def parse_test_poses_csv(csv_path: Path) -> Tuple[List[Dict], List[Camera]]:
    """
    Parse test_poses.csv format:
    image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
    """
    poses_data = []
    cameras = []

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            if not line.strip():
                continue
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 14:
                continue

            image_name = parts[0]
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            fx, fy = float(parts[8]), float(parts[9])
            cx, cy = float(parts[10]), float(parts[11])
            width, height = int(parts[12]), int(parts[13])

            pose_dict = {
                "image_name": image_name,
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
            }
            poses_data.append(pose_dict)

            quat = np.array([qw, qx, qy, qz], dtype=np.float32)
            tvec = np.array([tx, ty, tz], dtype=np.float32)
            intrinsics = CameraIntrinsics(fx, fy, cx, cy, width, height)
            pose = CameraPose(quat, tvec)
            camera = Camera(intrinsics, pose)
            cameras.append(camera)

    return poses_data, cameras


def estimate_scene_scale(cameras: List[Camera]) -> float:
    """Estimate scene scale from camera translation vectors."""
    tvecs = np.array([cam.pose.get_translation() for cam in cameras])
    distances = np.linalg.norm(tvecs, axis=1)
    return float(np.mean(distances[distances > 0]) if np.any(distances > 0) else 1.0)


def estimate_scene_bounds(cameras: List[Camera], point_cloud: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate scene bounding box from cameras and optionally 3D points."""
    tvecs = np.array([cam.pose.get_translation() for cam in cameras])
    
    if point_cloud is not None:
        points = point_cloud
    else:
        points = tvecs

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    # Add some margin
    margin = np.linalg.norm(max_bound - min_bound) * 0.1
    min_bound -= margin
    max_bound += margin

    return min_bound, max_bound

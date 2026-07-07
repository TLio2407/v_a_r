"""
Dataset loader for VAI NVS competition data.
Handles loading training images, poses, and camera parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

from camera_utils import parse_test_poses_csv, Camera, CameraIntrinsics, CameraPose
from colmap_parser import read_cameras, read_images, read_points3d, qvec2rotmat


class SceneDataset:
    """Load and manage a single scene's data."""

    def __init__(self, scene_dir: Path):
        self.scene_dir = Path(scene_dir)
        self.train_dir = self.scene_dir / "train"
        self.test_dir = self.scene_dir / "test"
        
        self.train_images: List[Path] = []
        self.test_poses: List[Dict] = []
        self.test_cameras: List[Camera] = []
        
        self.colmap_cameras: Dict = {}
        self.colmap_images: Dict = {}
        self.colmap_points3d: Dict = {}
        
        self._load_data()

    def _load_data(self):
        """Load all scene data."""
        # Load training images
        images_dir = self.train_dir / "images"
        if images_dir.exists():
            self.train_images = sorted(images_dir.glob("*"))

        # Load test poses
        test_poses_path = self.test_dir / "test_poses.csv"
        if test_poses_path.exists():
            self.test_poses, self.test_cameras = parse_test_poses_csv(test_poses_path)

        # Load COLMAP data
        sparse_dir = self.train_dir / "sparse" / "0"
        if sparse_dir.exists():
            try:
                cameras_file = sparse_dir / "cameras.bin"
                images_file = sparse_dir / "images.bin"
                points3d_file = sparse_dir / "points3D.bin"
                
                if cameras_file.exists():
                    self.colmap_cameras = read_cameras(cameras_file)
                if images_file.exists():
                    self.colmap_images = read_images(images_file)
                if points3d_file.exists():
                    self.colmap_points3d = read_points3d(points3d_file)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load COLMAP data: {e}. Continuing without COLMAP pose information.")

    def get_training_image(self, idx: int) -> Tuple[np.ndarray, Path]:
        """Load training image as numpy array."""
        if idx >= len(self.train_images):
            raise IndexError(f"Training image index {idx} out of range")
        img_path = self.train_images[idx]
        img = Image.open(img_path).convert("RGB")
        return np.array(img), img_path

    def get_all_training_images(self) -> List[np.ndarray]:
        """Load all training images."""
        images = []
        for i in range(len(self.train_images)):
            img, _ = self.get_training_image(i)
            images.append(img)
        return images

    def get_training_camera_poses(self) -> List[Camera]:
        """Extract camera poses from COLMAP data for training images."""
        cameras = []
        for img_id in sorted(self.colmap_images.keys()):
            img_data = self.colmap_images[img_id]
            camera_id = img_data["camera_id"]
            if camera_id not in self.colmap_cameras:
                continue
            
            cam_data = self.colmap_cameras[camera_id]
            qvec = img_data["qvec"]
            tvec = img_data["tvec"]
            
            # Extract intrinsics from COLMAP camera params
            params = cam_data["params"]
            fx = fy = params[0]  # Assuming pinhole model with equal focal lengths
            cx, cy = params[1], params[2]
            width, height = cam_data["width"], cam_data["height"]
            
            intrinsics = CameraIntrinsics(fx, fy, cx, cy, width, height)
            pose = CameraPose(qvec, tvec)
            camera = Camera(intrinsics, pose)
            cameras.append(camera)
        
        return cameras

    def get_test_cameras(self) -> List[Camera]:
        """Get camera parameters for test views."""
        return self.test_cameras

    def get_test_poses_dict(self) -> List[Dict]:
        """Get test poses as dictionaries."""
        return self.test_poses

    def get_point_cloud(self) -> np.ndarray:
        """Extract 3D point cloud from COLMAP."""
        points = []
        for pt_id in sorted(self.colmap_points3d.keys()):
            pt_data = self.colmap_points3d[pt_id]
            points.append(pt_data["xyz"])
        
        if not points:
            return np.array([], dtype=np.float32).reshape(0, 3)
        return np.array(points, dtype=np.float32)

    def get_scene_scale(self) -> float:
        """Estimate scene scale from camera poses and points."""
        cameras = self.get_training_camera_poses()
        if not cameras:
            return 1.0
        
        tvecs = np.array([cam.pose.get_translation() for cam in cameras])
        distances = np.linalg.norm(tvecs, axis=1)
        scale = float(np.mean(distances[distances > 1e-6]) if np.any(distances > 1e-6) else 1.0)
        
        # Also consider 3D points
        points = self.get_point_cloud()
        if len(points) > 0:
            pt_distances = np.linalg.norm(points, axis=1)
            pt_scale = float(np.mean(pt_distances[pt_distances > 1e-6]) if np.any(pt_distances > 1e-6) else 1.0)
            scale = max(scale, pt_scale)
        
        return scale

    def to_nerfstudio_format(self, output_dir: Path) -> Path:
        """
        Convert to Nerfstudio format (transforms.json) with Point Cloud.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        
        # Use COLMAP poses if available, otherwise use test poses as reference
        cameras = self.get_training_camera_poses()
        if not cameras:
            cameras = self.get_test_cameras()

        for idx, camera in enumerate(cameras):
            # Get the corresponding image path
            if idx < len(self.train_images):
                img_rel_path = str(self.train_images[idx].relative_to(self.scene_dir))
            else:
                img_rel_path = f"frame_{idx:04d}.png"

            # Camera to world matrix (inverse of world to camera)
            w2c = camera.pose.to_matrix()
            c2w = np.linalg.inv(w2c)

            # Nerfstudio expects c2w matrix
            frame = {
                "file_path": img_rel_path,
                "transform_matrix": c2w.tolist(),
            }
            frames.append(frame)

        # Create transform metadata
        transforms = {
            "camera_model": "OPENCV",
            "frames": frames,
        }

        # Add camera intrinsics from first camera
        if cameras:
            intr = cameras[0].intrinsics
            transforms["fl_x"] = intr.fx
            transforms["fl_y"] = intr.fy
            transforms["cx"] = intr.cx
            transforms["cy"] = intr.cy
            transforms["w"] = intr.width
            transforms["h"] = intr.height

        # --- ADDED: Extract and save point cloud for Splatfacto initialization ---
        points = self.get_point_cloud()
        if len(points) > 0:
            ply_path = output_dir / "sparse_pc.ply"
            with open(ply_path, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("end_header\n")
                for pt in points:
                    f.write(f"{float(pt[0])} {float(pt[1])} {float(pt[2])}\n")
            
            # Tell Nerfstudio where to find the initialization points
            transforms["ply_file_path"] = "sparse_pc.ply"

        # Save transforms.json
        transforms_path = output_dir / "transforms.json"
        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=2)

        return transforms_path

    def __repr__(self) -> str:
        return (
            f"SceneDataset({self.scene_dir.name}) "
            f"[{len(self.train_images)} training images, "
            f"{len(self.test_poses)} test poses, "
            f"{len(self.colmap_points3d)} 3D points]"
        )


def discover_all_scenes(base_dir: Path) -> List[SceneDataset]:
    """Discover all scenes in the dataset."""
    scenes = []
    for collection_dir in sorted(base_dir.iterdir()):
        if not collection_dir.is_dir():
            continue
        for scene_dir in sorted(collection_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            train_images_dir = scene_dir / "train" / "images"
            test_poses_path = scene_dir / "test" / "test_poses.csv"
            if train_images_dir.exists() and test_poses_path.exists():
                try:
                    dataset = SceneDataset(scene_dir)
                    scenes.append(dataset)
                except Exception as e:
                    print(f"Failed to load scene {scene_dir}: {e}")
    
    return scenes
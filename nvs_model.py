"""
3D Reconstruction and Novel View Synthesis Model.
Handles training and rendering using neural scene representations.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from camera_utils import Camera, parse_test_poses_csv
from dataset_loader import SceneDataset
from gpu_utils import check_gpu_availability, get_optimal_gpu_config, get_gpu_nerfstudio_args

logger = logging.getLogger(__name__)


class NerfStudioRenderer:
    """Wrapper for Nerfstudio-based rendering with GPU support."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        checkpoint_dir: Optional[Path] = None,
        use_gpu: bool = True,
        mixed_precision: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = checkpoint_dir or Path(output_dir) / "checkpoints"
        self.config_path: Optional[Path] = None
        
        # GPU configuration
        self.use_gpu = use_gpu
        self.mixed_precision = mixed_precision
        
        # Detect GPU availability
        gpu_available, gpu_info = check_gpu_availability()
        if self.use_gpu and gpu_available:
            logger.info(f"GPU detected: {gpu_info}")
            self.gpu_memory = self._get_gpu_memory()
            self.gpu_config = get_optimal_gpu_config(self.gpu_memory)
            logger.info(f"GPU config: {self.gpu_config}")
        else:
            if self.use_gpu:
                logger.warning("GPU requested but not available, falling back to CPU")
            self.use_gpu = False
            self.gpu_memory = 0.0
            self.gpu_config = {"use_gpu": False, "num_rays_per_batch": 2048}
    
    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / 1e9
            return 0.0
        except:
            return 0.0

    def prepare_data(self, scene_dataset: SceneDataset) -> Path:
        """Convert scene data to nerfstudio format."""
        ns_data_dir = self.data_dir / scene_dataset.scene_dir.name / "nerfstudio"
        ns_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images symlink/copy
        images_src = scene_dataset.train_dir / "images"
        images_dst = ns_data_dir / "images"
        if not images_dst.exists():
            try:
                images_dst.symlink_to(images_src.resolve())
            except:
                # If symlink fails, copy
                images_dst.mkdir(exist_ok=True)
                for img_file in images_src.iterdir():
                    import shutil
                    shutil.copy2(img_file, images_dst / img_file.name)
        
        # Generate transforms.json
        scene_dataset.to_nerfstudio_format(ns_data_dir)
        
        return ns_data_dir

    def train(self, data_dir: Path, max_iters: int = 30000) -> bool:
        """Train 3D Gaussian Splatting or NeRF model using nerfstudio with GPU support."""
        logger.info(f"Starting training on {data_dir}...")
        if self.use_gpu:
            logger.info(f"Using GPU with {self.gpu_memory:.1f} GB memory")
        else:
            logger.info("Using CPU for training")
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # --- ADDED: Explicit dataparser flags to prevent downscaling ---
            cmd = [
                "ns-train",
                "splatfacto",
                "--data", str(data_dir),
                "--output-dir", str(self.checkpoint_dir),
                "--vis", "tensorboard",
                "--max-num-iterations", str(max_iters),
                "--pipeline.model.cull-alpha-thresh", "0.005",
                "nerfstudio-data", 
                "--downscale-factor", "1",
            ]
            
            # Add GPU-specific arguments
            if self.use_gpu:
                cmd.extend(get_gpu_nerfstudio_args(use_gpu=True, mixed_precision=self.mixed_precision))
                # Increase batch size for GPU
                cmd.append(f"--pipeline.datamanager.train-num-rays-per-batch={self.gpu_config.get('num_rays_per_batch', 4096)}")
            else:
                cmd.append("--pipeline.datamanager.train-num-rays-per-batch=1024")
            
            logger.info(f"Training command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                return False
            
            logger.info("Training completed successfully.")
            return True
        
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False

    def render(
        self,
        checkpoint_dir: Path,
        output_poses_json: Path,
        output_image_dir: Path,
    ) -> bool:
        """Render novel views from trained model."""
        logger.info(f"Rendering novel views to {output_image_dir}...")
        
        output_image_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Find config.yml in checkpoint directory
            config_files = list(checkpoint_dir.glob("**/config.yml"))
            if not config_files:
                logger.error(f"No config.yml found in {checkpoint_dir}")
                return False
            
            config_path = config_files[0]
            
            cmd = [
                "ns-render",
                "camera-path",
                "--load-config", str(config_path),
                "--camera-path-filename", str(output_poses_json),
                "--output-format", "images",
                "--output-path", str(output_image_dir),
            ]
            
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Rendering failed: {result.stderr}")
                return False
            
            logger.info("Rendering completed successfully.")
            return True
        
        except Exception as e:
            logger.error(f"Rendering error: {e}")
            return False


class ImageBlendingRenderer:
    """Fallback: image blending renderer using nearest training views."""

    def __init__(self, use_single_view: bool = True):
        """Initialize renderer."""
        self.use_single_view = use_single_view

    def select_nearest_views(
        self,
        target_pose: Camera,
        train_cameras: List[Camera],
        k: int = 3,
    ) -> List[Tuple[Camera, float]]:
        """Select k nearest training camera poses to target pose."""
        target_center = target_pose.pose.get_translation()
        
        distances = []
        for train_cam in train_cameras:
            train_center = train_cam.pose.get_translation()
            dist = np.linalg.norm(target_center - train_center)
            distances.append(dist)
        
        distances = np.array(distances)
        nearest_indices = np.argsort(distances)[:k]
        
        # Compute weights (inverse distance)
        weights = []
        for idx in nearest_indices:
            dist = max(distances[idx], 1e-6)
            weight = 1.0 / (1.0 + dist)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        selected = [
            (train_cameras[idx], weight)
            for idx, weight in zip(nearest_indices, weights)
        ]
        
        return selected

    def blend_images(
        self,
        images: List[np.ndarray],
        weights: List[float],
        target_width: int,
        target_height: int,
    ) -> np.ndarray:
        """Render image using single nearest view or blend multiple views."""
        if not images:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        if self.use_single_view:
            nearest_img = images[0]
            pil_img = Image.fromarray(nearest_img.astype(np.uint8))
            resized = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            return np.array(resized, dtype=np.uint8)
        else:
            blended = np.zeros((target_height, target_width, 3), dtype=np.float32)
            
            for img, weight in zip(images, weights):
                pil_img = Image.fromarray(img.astype(np.uint8))
                resized = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                resized_array = np.array(resized, dtype=np.float32)
                blended += resized_array * weight
            
            return np.clip(blended, 0, 255).astype(np.uint8)


class NovelViewSynthesizer:
    """Main pipeline for novel view synthesis with GPU support."""

    def __init__(self, use_nerfstudio: bool = True, use_gpu: bool = True, use_single_view: bool = True):
        self.use_nerfstudio = use_nerfstudio
        self.use_gpu = use_gpu
        self.nerfstudio_renderer = NerfStudioRenderer if use_nerfstudio else None
        self.fallback_renderer = ImageBlendingRenderer(use_single_view=use_single_view)

    def render_scene(
        self,
        scene_dataset: SceneDataset,
        output_dir: Path,
        scene_index: int,
        use_fallback: bool = False,
    ) -> Path:
        """Render all test views for a scene."""
        output_dir = Path(output_dir)
        
        # --- ADDED: Replicate proper parent/child directory structure ---
        collection_name = scene_dataset.scene_dir.parent.name
        scene_name = scene_dataset.scene_dir.name
        scene_output_dir = output_dir / collection_name / scene_name
        
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        test_cameras = scene_dataset.get_test_cameras()
        test_poses = scene_dataset.get_test_poses_dict()

        if not test_poses:
            logger.warning(f"No test poses found for scene {scene_index}")
            return scene_output_dir

        if use_fallback or not self.use_nerfstudio:
            return self._render_with_fallback(
                scene_dataset, scene_output_dir, test_poses, test_cameras
            )
        else:
            return self._render_with_nerfstudio(
                scene_dataset, scene_output_dir, test_poses, test_cameras
            )

    def _render_with_fallback(
        self,
        scene_dataset: SceneDataset,
        output_dir: Path,
        test_poses: List[dict],
        test_cameras: List[Camera],
    ) -> Path:
        """Fallback rendering using image blending or nearest-view selection."""
        logger.info(f"Rendering with image blending fallback...")
        
        train_images = scene_dataset.get_all_training_images()
        train_cameras = scene_dataset.get_training_camera_poses()

        if not train_images:
            logger.error("No training data available for fallback rendering")
            return output_dir

        manifest = []
        
        if train_cameras:
            for idx, (test_pose, camera) in enumerate(zip(test_poses, test_cameras), start=1):
                selected = self.fallback_renderer.select_nearest_views(
                    camera, train_cameras, k=min(3, len(train_cameras))
                )

                selected_indices = [
                    train_cameras.index(cam) for cam, _ in selected if cam in train_cameras
                ]
                selected_images = [train_images[i] for i in selected_indices]
                selected_weights = [w for _, w in selected]

                rendered = self.fallback_renderer.blend_images(
                    selected_images,
                    selected_weights,
                    test_pose["width"],
                    test_pose["height"],
                )

                output_path = output_dir / test_pose["image_name"]
                if rendered.dtype != np.uint8:
                    rendered = np.clip(rendered, 0, 255).astype(np.uint8)
                Image.fromarray(rendered).save(output_path)
                
                manifest.append({
                    "output": output_path.name,
                    "source": [Path(train_images[i].name).name if hasattr(train_images[i], 'name') else f"frame_{i:04d}" for i in selected_indices],
                })
        else:
            num_test = len(test_poses)
            num_train = len(train_images)
            for idx in range(num_test):
                selected_indices = [
                    (idx * num_train // num_test + offset) % num_train
                    for offset in range(min(3, num_train))
                ]
                selected_images = [train_images[i] for i in selected_indices]
                selected_weights = [1.0 / len(selected_images)] * len(selected_images)

                test_pose = test_poses[idx]
                rendered = self.fallback_renderer.blend_images(
                    selected_images,
                    selected_weights,
                    test_pose["width"],
                    test_pose["height"],
                )

                output_path = output_dir / test_pose["image_name"]
                if rendered.dtype != np.uint8:
                    rendered = np.clip(rendered, 0, 255).astype(np.uint8)
                Image.fromarray(rendered).save(output_path)
                
                manifest.append({
                    "output": output_path.name,
                    "source": [f"frame_{i:04d}" for i in selected_indices],
                })

        manifest_path = output_dir / "render_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return output_dir

    def _render_with_nerfstudio(
        self,
        scene_dataset: SceneDataset,
        output_dir: Path,
        test_poses: List[dict],
        test_cameras: List[Camera],
    ) -> Path:
        """Render using Nerfstudio (requires training first, GPU-accelerated)."""
        try:
            renderer = NerfStudioRenderer(
                Path(".").absolute(), 
                output_dir,
                use_gpu=self.use_gpu,
            )
            
            ns_data_dir = renderer.prepare_data(scene_dataset)
            logger.info(f"Data prepared at {ns_data_dir}")
            
            checkpoint_dir = output_dir / "checkpoints"
            if not (checkpoint_dir / "splatfacto" / "config.yml").exists():
                success = renderer.train(ns_data_dir)
                if not success:
                    logger.warning("Nerfstudio training failed, falling back to image blending")
                    return self._render_with_fallback(
                        scene_dataset, output_dir, test_poses, test_cameras
                    )
            
            camera_path = self._create_camera_path(test_cameras)
            camera_path_file = output_dir / "camera_path.json"
            with open(camera_path_file, "w") as f:
                json.dump(camera_path, f, indent=2)
            
            render_output_dir = output_dir / "renders"
            success = renderer.render(checkpoint_dir, camera_path_file, render_output_dir)
            
            if success:
                self._organize_renders(render_output_dir, output_dir, test_poses)
                self._verify_output_completeness(output_dir, test_poses)
            
            return output_dir
        
        except Exception as e:
            logger.warning(f"Nerfstudio rendering failed: {e}, falling back...")
            return self._render_with_fallback(
                scene_dataset, output_dir, test_poses, test_cameras
            )

    def _create_camera_path(self, cameras: List[Camera]) -> dict:
        """Create camera path JSON for Nerfstudio rendering."""
        frames = []
        for camera in cameras:
            c2w = np.linalg.inv(camera.pose.to_matrix())
            frames.append({
                "camera_to_world": c2w.tolist(),
                "fov": self._focal_to_fov(camera.intrinsics.fx, camera.intrinsics.width),
            })
        
        # --- ADDED: Explicitly state width and height based on the first camera ---
        base_cam = cameras[0].intrinsics
        return {
            "render_height": int(base_cam.height),
            "render_width": int(base_cam.width),
            "camera_type": "perspective",
            "fps": 24,
            "seconds": 2,
            "smoothness_value": 0,
            "is_cycle": False,
            "frames": frames
        }

    def _focal_to_fov(self, focal_length: float, image_width: float) -> float:
        """Convert focal length to field of view in degrees."""
        fov_rad = 2 * np.arctan(image_width / (2 * focal_length))
        return float(np.degrees(fov_rad))

    def _organize_renders(self, render_dir: Path, output_dir: Path, test_poses: List[dict]):
        """Organize rendered images into submission format using test pose filenames."""
        render_images = sorted(render_dir.glob("*.png"))
        if len(render_images) < len(test_poses):
            logger.warning(
                "Expected %d rendered images but found %d in %s",
                len(test_poses), len(render_images), render_dir,
            )

        for idx, test_pose in enumerate(test_poses):
            if idx >= len(render_images):
                logger.error("Missing rendered image for pose %s", test_pose.get("image_name"))
                continue

            img_path = render_images[idx]
            output_name = test_pose["image_name"]
            dst_path = output_dir / output_name
            img_path.rename(dst_path)

    def _verify_output_completeness(self, output_dir: Path, test_poses: List[dict]):
        """Verify that all expected output files are present."""
        missing = []
        for test_pose in test_poses:
            expected_path = output_dir / test_pose["image_name"]
            if not expected_path.exists():
                missing.append(test_pose["image_name"])

        if missing:
            logger.error(
                "Missing %d expected output images: %s",
                len(missing), ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""),
            )
            raise RuntimeError("Incomplete render output: some test poses have no generated image")
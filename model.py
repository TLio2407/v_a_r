"""
Novel View Synthesis Model - Main Pipeline
Integrates 3D reconstruction and view synthesis using neural representations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from dataset_loader import SceneDataset, discover_all_scenes
from nvs_model import NovelViewSynthesizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def render_scene(
    scene_dataset: SceneDataset,
    output_dir: Path,
    scene_index: int,
    use_nerfstudio: bool = False,
    use_fallback: bool = False,
    use_gpu: bool = True,
    use_single_view: bool = True,
) -> Path:
    """
    Render all novel views for a single scene.
    
    Args:
        scene_dataset: Loaded scene data
        output_dir: Output directory for renders
        scene_index: Scene index for naming
        use_nerfstudio: If True, use nerfstudio for training/rendering
        use_fallback: If True, use image blending fallback
        use_gpu: If True, use GPU acceleration for NeRF training
        use_single_view: If True, use single nearest view (clean). If False, blend views.
    
    Returns:
        Path to scene output directory
    """
    logger.info(f"Rendering scene {scene_index}: {scene_dataset.scene_dir.name}")
    logger.info(f"  Training images: {len(scene_dataset.train_images)}")
    logger.info(f"  Test poses: {len(scene_dataset.test_poses)}")
    
    synthesizer = NovelViewSynthesizer(use_nerfstudio=use_nerfstudio, use_gpu=use_gpu, use_single_view=use_single_view)
    scene_output = synthesizer.render_scene(
        scene_dataset,
        output_dir,
        scene_index,
        use_fallback=use_fallback,
    )
    
    logger.info(f"Scene {scene_index} rendering complete: {scene_output}")
    return scene_output


def render_submission(
    base_dir: Path,
    submission_dir: Path,
    use_nerfstudio: bool = False,
    use_fallback: bool = False,
    use_gpu: bool = True,
    use_single_view: bool = True,
) -> List[Path]:
    """
    Generate submission by rendering all scenes.
    
    Args:
        base_dir: Root directory containing scenes (e.g., phase1/)
        submission_dir: Output submission directory
        use_nerfstudio: If True, use nerfstudio for training/rendering
        use_fallback: If True, use image blending fallback
        use_gpu: If True, use GPU acceleration for NeRF training
        use_single_view: If True, use single nearest view (clean). If False, blend views.
    
    Returns:
        List of output scene directories
    """
    submission_dir = Path(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Discovering scenes in {base_dir}...")
    scenes = discover_all_scenes(Path(base_dir))
    logger.info(f"Found {len(scenes)} scenes")
    
    output_dirs: List[Path] = []
    for idx, scene_dataset in enumerate(scenes, start=1):
        scene_output = render_scene(
            scene_dataset,
            submission_dir,
            idx,
            use_nerfstudio=use_nerfstudio,
            use_fallback=use_fallback,
            use_gpu=use_gpu,
            use_single_view=use_single_view,
        )
        output_dirs.append(scene_output)
    
    logger.info(f"Submission generation complete: {len(output_dirs)} scenes rendered")
    return output_dirs

import argparse
import logging
from pathlib import Path

from model import render_submission
from gpu_utils import check_gpu_availability

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VAR2026_Inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novel View Synthesis Inference with GPU Support")
    parser.add_argument("--data_dir", type=str, default="D:\Cuong\Viettel\VAI_NVS_DATA\data\phase1", help="Root directory containing the contest scenes")
    parser.add_argument("--output_dir", type=str, default="submission", help="Output directory for rendered RGB images")
    parser.add_argument("--use_nerfstudio", action="store_true", default=False, help="Use Nerfstudio for training/rendering")
    parser.add_argument("--use_fallback", action="store_true", default=False, help="Use image blending fallback")
    parser.add_argument("--blend", action="store_true", default=False, help="Blend multiple views (default: single nearest view for clean output)")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU acceleration (default: auto-detect)")
    parser.add_argument("--no_gpu", action="store_true", default=False, help="Disable GPU acceleration and use CPU only")

    args = parser.parse_args()

    # Handle GPU settings
    use_gpu = not args.no_gpu
    gpu_available, gpu_info = check_gpu_availability()
    
    if use_gpu and gpu_available:
        logger.info(f"✓ GPU available: {gpu_info}")
    elif use_gpu and not gpu_available:
        logger.warning(f"⚠ GPU requested but not available: {gpu_info}. Falling back to CPU.")
        use_gpu = False
    else:
        logger.info("Running on CPU")

    output_dir = Path(args.output_dir)
    base_dir = Path(args.data_dir)
    
    logger.info(f"Starting inference:")
    logger.info(f"  Data: {base_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Nerfstudio: {args.use_nerfstudio}")
    logger.info(f"  View mode: {'Blend' if args.blend else 'Single (clean)'}")
    logger.info(f"  GPU: {use_gpu}")
    
    render_submission(
        base_dir,
        output_dir,
        use_nerfstudio=args.use_nerfstudio,
        use_fallback=args.use_fallback,
        use_gpu=use_gpu,
        use_single_view=not args.blend,
    )
    logger.info(f"✓ Submission generated at {output_dir.resolve()}")

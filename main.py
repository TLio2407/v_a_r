"""
Main entry point for VAI Novel View Synthesis Pipeline.
Generates novel view images from multi-view input data.
"""

from pathlib import Path

from model import render_submission


if __name__ == "__main__":
    # Default configuration
    base_dir = Path("D:/Cuong/Viettel/VAI_NVS_DATA/data/phase1")
    submission_dir = Path("submission")
    
    # Use fallback image blending by default (fastest, no dependencies)
    # Set use_nerfstudio=True to use neural reconstruction (requires nerfstudio)
    render_submission(
        base_dir,
        submission_dir,
        use_nerfstudio=False,  # Set to True for NeRF-based rendering
        use_fallback=True,     # Use image blending as default
    )
    print(f"✓ Submission generated at {submission_dir.resolve()}")

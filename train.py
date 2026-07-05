"""
Trains a 3D Gaussian Splatting model (nerfstudio "splatfacto") on the
prepared scene data, with hyperparameters tuned for outdoor, unbounded,
static BTS-tower scenes captured from a drone.

Just a thin, explicit wrapper around `ns-train` so every flag that matters
for this problem is visible and versioned instead of hidden in defaults.

IMPORTANT: splatfacto's exact CLI flag names have drifted across nerfstudio
releases (e.g. appearance handling was `--pipeline.model.use-appearance-
embedding` in some versions, `--pipeline.model.appearance-embed-dim` in
others, and is absent/handled differently in newer ones). Rather than
hardcode names that may not exist in your installed version and crash the
whole run, this script introspects `ns-train splatfacto --help` first and
only appends flags that are actually supported, logging anything it skips
so you know your installed defaults are being used instead.

SPEED: the two biggest levers by far are (1) training resolution and
(2) iteration count. Full-resolution drone photos (often 4000x3000+) make
every rasterization step far more expensive than it needs to be for an SfM
point cloud that's already fairly coarse. Defaults here downscale 4x during
training and cap eval/checkpoint overhead — use --downscale-factor 1 and/or
--max-iterations 30000 only once you've confirmed a fast run looks
reasonable and you're chasing the last bit of quality.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from utils import get_logger, run_cmd

logger = get_logger("train")


def check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning(
                "No CUDA GPU detected by PyTorch! Training splatfacto on CPU is "
                "roughly 50-100x slower and is almost certainly why this is taking "
                "too long. In Colab: Runtime -> Change runtime type -> Hardware "
                "accelerator -> GPU (T4 or better), then restart and rerun."
            )
        else:
            name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {name}")
    except ImportError:
        logger.warning("Could not import torch to check GPU availability.")


def get_supported_flags() -> str:
    """Returns the raw --help text for `ns-train splatfacto` so we can check
    which flags exist before using them."""
    try:
        result = subprocess.run(
            ["ns-train", "splatfacto", "--help"],
            capture_output=True, text=True, timeout=60,
        )
        return result.stdout + result.stderr
    except Exception as e:
        logger.warning(f"Could not introspect `ns-train splatfacto --help` ({e}); "
                        "will attempt all flags and let ns-train validate them.")
        return ""


def add_if_supported(cmd: list[str], help_text: str, flag: str, value: str, label: str):
    if not help_text or flag in help_text:
        cmd += [flag, value]
    else:
        logger.warning(f"Flag '{flag}' not found in this nerfstudio version's splatfacto "
                        f"help output — skipping '{label}' and using its installed default. "
                        f"Run `ns-train splatfacto --help | grep -i <keyword>` to find the "
                        f"equivalent flag in your version if you need this behavior.")


def build_command(
    data_dir: Path,
    scene_id: str,
    output_dir: Path,
    max_iterations: int = 15000,
    camera_optimizer_mode: str = "SO3xR3",
    use_appearance_embedding: bool = True,
    background_color: str = "random",
    downscale_factor: int = 4,
    eval_every: int = 5000,
    save_every: int = 5000,
) -> list[str]:
    help_text = get_supported_flags()
    has_transforms = (Path(data_dir) / "transforms.json").exists()
    if has_transforms:
        if "nerfstudio-data" in help_text:
            dataparser = "nerfstudio-data"
        elif "nerfstudio" in help_text:
            dataparser = "nerfstudio"
        else:
            logger.warning("Could not confirm the nerfstudio-format dataparser subcommand name "
                            "from --help output; defaulting to 'nerfstudio-data'. If this fails, "
                            "run `ns-train splatfacto --help` and check the 'Available subcommands' "
                            "list for the correct name.")
            dataparser = "nerfstudio-data"
    else:
        dataparser = "colmap"

    cmd = [
        "ns-train", "splatfacto",
        "--data", str(data_dir),
        "--output-dir", str(output_dir),
        "--experiment-name", scene_id,
        "--max-num-iterations", str(max_iterations),
        "--vis", "tensorboard",
    ]

    add_if_supported(cmd, help_text, "--pipeline.model.camera-optimizer.mode",
                      camera_optimizer_mode, "camera pose optimization")
    add_if_supported(cmd, help_text, "--pipeline.model.background-color",
                      background_color, "background color")
    add_if_supported(cmd, help_text, "--pipeline.datamanager.camera-res-scale-factor",
                      str(1.0 / downscale_factor), "downscale factor")

    # These control how often training pauses to render eval images / write
    # checkpoints — doing this too often on large images is a common silent
    # source of "training feels stuck" complaints. Push them out.
    add_if_supported(cmd, help_text, "--steps-per-eval-image", str(eval_every), "eval frequency")
    add_if_supported(cmd, help_text, "--steps-per-eval-all-images", str(max_iterations), "full-eval frequency")
    add_if_supported(cmd, help_text, "--steps-per-save", str(save_every), "checkpoint frequency")

    if use_appearance_embedding:
        if "--pipeline.model.use-appearance-embedding" in help_text:
            cmd += ["--pipeline.model.use-appearance-embedding", "True"]
        elif "--pipeline.model.appearance-embed-dim" in help_text:
            cmd += ["--pipeline.model.appearance-embed-dim", "32"]
        else:
            logger.warning("No appearance-embedding flag found for splatfacto in this "
                            "nerfstudio version — skipping (using installed default).")

    # Positional dataparser subcommand comes last for ns-train.
    cmd.append(dataparser)
    if dataparser == "colmap":
        cmd += ["--colmap-path", "colmap/sparse/0"]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="Nerfstudio-formatted data dir")
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--max-iterations", type=int, default=15000,
                     help="30000 is nerfstudio's full default; 15000 trains ~2x faster "
                          "and is usually enough for a single static structure. "
                          "Bump back up once you've confirmed the pipeline works end to end.")
    ap.add_argument("--camera-optimizer-mode", default="SO3xR3",
                     choices=["off", "SO3xR3", "SE3"])
    ap.add_argument("--no-appearance-embedding", action="store_true")
    ap.add_argument("--background-color", default="random")
    ap.add_argument("--downscale-factor", type=int, default=4,
                     help="Train on 1/N resolution images. 4 is a good fast default for "
                          "full-res drone photos; use 1 only for a final high-quality run.")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--skip-gpu-check", action="store_true")
    args = ap.parse_args()

    if not args.skip_gpu_check:
        check_gpu()

    cmd = build_command(
        args.data, args.scene_id, args.output_dir, args.max_iterations,
        args.camera_optimizer_mode, not args.no_appearance_embedding,
        args.background_color, args.downscale_factor,
        args.eval_every, args.save_every,
    )
    run_cmd(cmd, logger)
    logger.info(f"Training complete. Checkpoints/configs under: {args.output_dir}/{args.scene_id}/splatfacto/")


if __name__ == "__main__":
    main()
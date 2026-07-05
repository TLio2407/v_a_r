"""
Three things, selectable via subcommand:

  eval     Runs nerfstudio's built-in `ns-eval` on the held-out validation
           split (whatever split the dataparser reserved) and writes a
           metrics JSON (PSNR/SSIM/LPIPS as computed by nerfstudio itself).

  render   Renders RGB images at a set of *requested* novel camera poses
           (e.g. the ones the contest gives you) via `ns-render camera-path`,
           using a camera-path JSON in nerfstudio's format.

  compare  Standalone PSNR/SSIM/LPIPS between two image directories with
           matching filenames — useful for sanity-checking renders against
           any ground truth you do have, independent of nerfstudio's eval.

Usage:
  python render_eval.py eval    --load-config outputs/scene/splatfacto/*/config.yml --out eval_metrics.json
  python render_eval.py render  --load-config outputs/scene/splatfacto/*/config.yml --camera-path novel_poses.json --out renders/
  python render_eval.py compare --pred renders/ --gt gt_images/ --out compare_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import get_logger, ensure_dir, run_cmd, list_images

logger = get_logger("render_eval")


def cmd_eval(args):
    out = Path(args.out)
    run_cmd([
        "ns-eval",
        "--load-config", str(args.load_config),
        "--output-path", str(out),
    ], logger)
    logger.info(f"Eval metrics written to {out}")


def cmd_render(args):
    out_dir = ensure_dir(args.out)
    run_cmd([
        "ns-render", "camera-path",
        "--load-config", str(args.load_config),
        "--camera-path-filename", str(args.camera_path),
        "--output-path", str(out_dir),
        "--output-format", "images",
    ], logger)
    logger.info(f"Rendered novel views written to {out_dir}")


def _load_rgb(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cmd_compare(args):
    import lpips
    import torch
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    pred_files = {p.name: p for p in list_images(Path(args.pred))}
    gt_files = {p.name: p for p in list_images(Path(args.gt))}
    common = sorted(set(pred_files) & set(gt_files))
    if not common:
        raise RuntimeError("No matching filenames between --pred and --gt directories.")
    missing_pred = set(gt_files) - set(pred_files)
    if missing_pred:
        logger.warning(f"{len(missing_pred)} ground-truth images have no matching prediction "
                        f"(e.g. {sorted(missing_pred)[:3]})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = lpips.LPIPS(net="alex").to(device)

    results = {}
    for name in tqdm(common, desc="Scoring"):
        pred = _load_rgb(pred_files[name]).astype(np.float32) / 255.0
        gt = _load_rgb(gt_files[name]).astype(np.float32) / 255.0
        if pred.shape != gt.shape:
            import cv2
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        p = psnr_fn(gt, pred, data_range=1.0)
        s = ssim_fn(gt, pred, data_range=1.0, channel_axis=2)

        t_pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        t_gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            l = lpips_model(t_pred, t_gt).item()

        results[name] = {"psnr": float(p), "ssim": float(s), "lpips": float(l)}

    agg = {
        metric: float(np.mean([r[metric] for r in results.values()]))
        for metric in ("psnr", "ssim", "lpips")
    }
    out = {"per_image": results, "average": agg}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Average PSNR: {agg['psnr']:.2f} dB | SSIM: {agg['ssim']:.4f} | LPIPS: {agg['lpips']:.4f}")
    logger.info(f"Full results written to {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--load-config", required=True, type=Path)
    p_eval.add_argument("--out", type=Path, default=Path("eval_metrics.json"))
    p_eval.set_defaults(func=cmd_eval)

    p_render = sub.add_parser("render")
    p_render.add_argument("--load-config", required=True, type=Path)
    p_render.add_argument("--camera-path", required=True, type=Path)
    p_render.add_argument("--out", type=Path, default=Path("renders"))
    p_render.set_defaults(func=cmd_render)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--pred", required=True, type=Path)
    p_compare.add_argument("--gt", required=True, type=Path)
    p_compare.add_argument("--out", type=Path, default=Path("compare_metrics.json"))
    p_compare.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

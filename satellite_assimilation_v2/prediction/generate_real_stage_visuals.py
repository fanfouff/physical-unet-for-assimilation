#!/usr/bin/env python3
"""
Generate real sample visual panels for PASNet stage storytelling.

Outputs (per sample):
  1) raw observation swath map (scatter)
  2) background map (level-wise)
  3) stage-1 feature mean map
  4) model output analysis map (level-wise)
  5) label/target map (level-wise)
  6) 5-panel storyboard

Notes:
  - Background/output/label use the same color range for fair visual comparison.
  - Raw samples come from *_X.npy/*_Y.npy/*_lat.npy/*_lon.npy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate real stage visual panels from raw npy samples")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--raw_dir", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--num_samples", default=6, type=int)
    p.add_argument("--level_idx", default=22, type=int)
    p.add_argument("--grid_h", default=64, type=int)
    p.add_argument("--grid_w", default=64, type=int)
    p.add_argument("--resolution", default=0.25, type=float)
    p.add_argument("--stats_file", default="", type=str)
    p.add_argument("--increment_stats", default="", type=str)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def resolve_stats_paths(args: argparse.Namespace, ckpt_args: SimpleNamespace) -> Tuple[Path, Optional[Path], bool]:
    cand_stats: List[Path] = []
    if args.stats_file:
        cand_stats.append(Path(args.stats_file).expanduser())
    if getattr(ckpt_args, "stats_file", ""):
        cand_stats.append(Path(str(ckpt_args.stats_file)).expanduser())
    cand_stats.append(Path("/data/lrx_true/era_obs/npz/stats.npz"))
    cand_stats.append(Path("/data1/lrx/npz_64_real/stats.npz"))
    cand_stats.append(Path("/data2/lrx/npz_64_real/stats.npz"))

    stats_file = None
    for p in cand_stats:
        if p.exists():
            stats_file = p
            break
    if stats_file is None:
        raise FileNotFoundError("Cannot find stats.npz. Please pass --stats_file explicitly.")

    use_increment = _as_bool(getattr(ckpt_args, "use_increment", False))
    cand_inc: List[Path] = []
    if args.increment_stats:
        cand_inc.append(Path(args.increment_stats).expanduser())
    if getattr(ckpt_args, "increment_stats", ""):
        cand_inc.append(Path(str(ckpt_args.increment_stats)).expanduser())
    cand_inc.append(stats_file.parent / "increment_stats.npz")
    cand_inc.append(Path("/data1/lrx/npz_64_real/increment_stats.npz"))
    cand_inc.append(Path("/data2/lrx/npz_64_real/increment_stats.npz"))

    inc_file: Optional[Path] = None
    for p in cand_inc:
        if p.exists():
            inc_file = p
            break

    if use_increment and inc_file is None:
        # Fallback: still run, but interpret output as target-normalized prediction.
        use_increment = False

    return stats_file, inc_file, use_increment


def load_model(checkpoint_path: Path, device: torch.device):
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from models.backbone import create_model, UNetConfig

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    ckpt_args = SimpleNamespace(**ckpt["args"])

    model_name = str(getattr(ckpt_args, "model", "physics_unet"))
    if model_name in ("physics_unet", "pasnet", "physics_unet_lite", "physics_unet_large"):
        config = UNetConfig(
            fusion_mode=getattr(ckpt_args, "fusion_mode", "gated"),
            use_aux=_as_bool(getattr(ckpt_args, "use_aux", True)),
            mask_aware=_as_bool(getattr(ckpt_args, "mask_aware", True)),
            use_spectral_stem=_as_bool(getattr(ckpt_args, "use_spectral_stem", True)),
            deep_supervision=_as_bool(getattr(ckpt_args, "deep_supervision", False)),
        )
        model = create_model(model_name, config=config)
    elif model_name in ("fuxi_da", "fengwu", "background_only", "obs_only"):
        use_aux = _as_bool(getattr(ckpt_args, "use_aux", True))
        model = create_model(model_name, aux_channels=4 if use_aux else 0)
    elif model_name == "mamba":
        model = create_model(
            "mamba",
            fusion_mode=getattr(ckpt_args, "fusion_mode", "gated"),
            use_aux=_as_bool(getattr(ckpt_args, "use_aux", True)),
            mask_aware=_as_bool(getattr(ckpt_args, "mask_aware", True)),
        )
    else:
        model = create_model(model_name)

    sd = ckpt["model_state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model, ckpt_args


def pick_sample_ids(raw_dir: Path, n: int) -> List[str]:
    ids: List[str] = []
    for x in sorted(raw_dir.glob("*_X.npy")):
        sid = x.name.replace("_X.npy", "")
        need = [
            raw_dir / f"{sid}_Y.npy",
            raw_dir / f"{sid}_lat.npy",
            raw_dir / f"{sid}_lon.npy",
        ]
        if all(p.exists() for p in need):
            ids.append(sid)
    if len(ids) < n:
        return ids
    # Uniformly spread across the month for visual diversity
    idx = np.linspace(0, len(ids) - 1, n, dtype=int)
    return [ids[i] for i in idx]


def norm_field(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[:, None, None]) / std[:, None, None]


def denorm_field(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std[:, None, None] + mean[:, None, None]


def _add_globe_inset(fig, parent_ax, center_lon: float, center_lat: float) -> None:
    bbox = parent_ax.get_position()
    w = bbox.width * 0.24
    h = bbox.height * 0.24
    x0 = bbox.x0 + bbox.width * 0.70
    y0 = bbox.y0 + bbox.height * 0.04
    ax_in = fig.add_axes([x0, y0, w, h], projection=ccrs.Orthographic(center_lon, center_lat))
    ax_in.set_global()
    ax_in.stock_img()
    ax_in.coastlines(linewidth=0.4)
    ax_in.set_title("Earth", fontsize=7, pad=1)


def save_scatter_map(path: Path, lon: np.ndarray, lat: np.ndarray, val: np.ndarray,
                     title: str, cmap: str = "turbo", vmin=None, vmax=None) -> None:
    fig = plt.figure(figsize=(6.4, 6.0), dpi=220)
    ax = plt.axes(projection=ccrs.PlateCarree())
    extent = [float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))]
    pad_lon = (extent[1] - extent[0]) * 0.05 + 1e-6
    pad_lat = (extent[3] - extent[2]) * 0.05 + 1e-6
    ax.set_extent([extent[0] - pad_lon, extent[1] + pad_lon, extent[2] - pad_lat, extent[3] + pad_lat], ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
    im = ax.scatter(lon, lat, c=val, s=3, cmap=cmap, vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    _add_globe_inset(fig, ax, center_lon=float(np.nanmedian(lon)), center_lat=float(np.nanmedian(lat)))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_grid_map(path: Path, lon2d: np.ndarray, lat2d: np.ndarray, field: np.ndarray,
                  title: str, cmap: str = "RdYlBu_r", vmin=None, vmax=None) -> None:
    fig = plt.figure(figsize=(6.4, 6.0), dpi=220)
    ax = plt.axes(projection=ccrs.PlateCarree())
    extent = [float(np.nanmin(lon2d)), float(np.nanmax(lon2d)), float(np.nanmin(lat2d)), float(np.nanmax(lat2d))]
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
    im = ax.pcolormesh(lon2d, lat2d, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
                       transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    _add_globe_inset(fig, ax, center_lon=float(np.nanmedian(lon2d)), center_lat=float(np.nanmedian(lat2d)))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_storyboard(path: Path, image_paths: List[Path], title: str) -> None:
    imgs = [plt.imread(str(p)) for p in image_paths]
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.8), dpi=180)
    if n == 1:
        axes = [axes]
    for ax, im in zip(axes, imgs):
        ax.imshow(im)
        ax.axis("off")
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    panels_dir = out_dir / "panels"
    story_dir = out_dir / "storyboards"
    trip_dir = out_dir / "triptychs"
    for d in (out_dir, panels_dir, story_dir, trip_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_model(Path(args.checkpoint), device)
    use_aux = _as_bool(getattr(ckpt_args, "use_aux", True))

    stats_file, inc_file, use_increment = resolve_stats_paths(args, ckpt_args)
    stats = np.load(stats_file)
    inc_stats = np.load(inc_file) if (use_increment and inc_file is not None) else None

    obs_mean, obs_std = stats["obs_mean"], stats["obs_std"]
    bkg_mean, bkg_std = stats["bkg_mean"], stats["bkg_std"]
    tgt_mean, tgt_std = stats["target_mean"], stats["target_std"]

    sample_ids = pick_sample_ids(raw_dir, args.num_samples)
    if not sample_ids:
        raise RuntimeError(f"No valid samples found in {raw_dir}")

    # Import here to avoid heavy startup for --help mode.
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from data_process.prepare_v3_data import convert_single_sample

    manifest: Dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "stats_file": str(stats_file.resolve()),
        "increment_stats": str(inc_file.resolve()) if inc_file is not None else None,
        "level_idx": args.level_idx,
        "samples": [],
    }

    for sid in sample_ids:
        x_path = raw_dir / f"{sid}_X.npy"
        y_path = raw_dir / f"{sid}_Y.npy"
        lat_path = raw_dir / f"{sid}_lat.npy"
        lon_path = raw_dir / f"{sid}_lon.npy"

        obs_raw = np.load(x_path).astype(np.float64)
        tgt_raw = np.load(y_path).astype(np.float64)
        lat = np.load(lat_path).astype(np.float64).ravel()
        lon = np.load(lon_path).astype(np.float64).ravel()

        sample = convert_single_sample(
            x_path=x_path,
            y_path=y_path,
            lat=lat,
            lon=lon,
            obs_data=obs_raw,
            tgt_data=tgt_raw,
            h=args.grid_h,
            w=args.grid_w,
            resolution=args.resolution,
            interp_method="linear",
            mask_radius=args.resolution * 2.0,
        )

        obs_phys = sample["obs"]
        bkg_phys = sample["bkg"]
        tgt_phys = sample["target"]
        mask = sample["mask"]
        aux = sample["aux"]
        lat2d = sample["lat2d"]
        lon2d = sample["lon2d"]

        obs_n = norm_field(obs_phys, obs_mean, obs_std)
        bkg_n = norm_field(bkg_phys, bkg_mean, bkg_std)

        obs_t = torch.from_numpy(obs_n[None]).float().to(device)
        bkg_t = torch.from_numpy(bkg_n[None]).float().to(device)
        mask_t = torch.from_numpy(mask[None]).float().to(device)
        aux_t = torch.from_numpy(aux[None]).float().to(device) if use_aux else None

        with torch.no_grad():
            if hasattr(model, "stem"):
                stem = model.stem(obs_t, bkg_t, mask_t, aux_t)
                stage1_map = stem.mean(dim=1)[0].detach().cpu().numpy()
            else:
                stage1_map = np.zeros((args.grid_h, args.grid_w), dtype=np.float32)

            out = model(obs_t, bkg_t, mask_t, aux_t)
            pred_n = out[0] if isinstance(out, tuple) else out
            pred_n = pred_n[0].detach().cpu().numpy()

        if use_increment and inc_stats is not None:
            inc_mean = inc_stats["inc_mean"]
            inc_std = inc_stats["inc_std"]
            pred_inc = denorm_field(pred_n, inc_mean, inc_std)
            ana_phys = bkg_phys + pred_inc
        else:
            ana_phys = denorm_field(pred_n, tgt_mean, tgt_std)

        li = int(np.clip(args.level_idx, 0, ana_phys.shape[0] - 1))
        bkg_lv = bkg_phys[li]
        out_lv = ana_phys[li]
        lbl_lv = tgt_phys[li]

        temp_vmin = float(np.nanmin([bkg_lv.min(), out_lv.min(), lbl_lv.min()]))
        temp_vmax = float(np.nanmax([bkg_lv.max(), out_lv.max(), lbl_lv.max()]))

        p_obs = panels_dir / f"{sid}_input_raw_obs_ch1_map.png"
        p_bkg = panels_dir / f"{sid}_input_bkg_level{li}.png"
        p_st1 = panels_dir / f"{sid}_stage1_feature_mean.png"
        p_out = panels_dir / f"{sid}_output_map_level{li}.png"
        p_lbl = panels_dir / f"{sid}_label_map_level{li}.png"

        save_scatter_map(
            p_obs,
            lon=lon,
            lat=lat,
            val=obs_raw[:, 0],
            title=f"{sid} | Input raw obs ch1",
            cmap="turbo",
        )
        save_grid_map(
            p_bkg,
            lon2d=lon2d,
            lat2d=lat2d,
            field=bkg_lv,
            title=f"{sid} | Background level {li}",
            cmap="RdYlBu_r",
            vmin=temp_vmin,
            vmax=temp_vmax,
        )
        save_grid_map(
            p_st1,
            lon2d=lon2d,
            lat2d=lat2d,
            field=stage1_map,
            title=f"{sid} | Stage-1 feature mean",
            cmap="magma",
        )
        save_grid_map(
            p_out,
            lon2d=lon2d,
            lat2d=lat2d,
            field=out_lv,
            title=f"{sid} | Output analysis level {li}",
            cmap="RdYlBu_r",
            vmin=temp_vmin,
            vmax=temp_vmax,
        )
        save_grid_map(
            p_lbl,
            lon2d=lon2d,
            lat2d=lat2d,
            field=lbl_lv,
            title=f"{sid} | Label target level {li}",
            cmap="RdYlBu_r",
            vmin=temp_vmin,
            vmax=temp_vmax,
        )

        p_tri = trip_dir / f"{sid}_triptych.png"
        p_story = story_dir / f"{sid}_five_panel.png"

        save_storyboard(
            p_tri,
            [p_obs, p_st1, p_out],
            title=f"{sid} | Input vs Stage-1 vs Output",
        )
        save_storyboard(
            p_story,
            [p_obs, p_bkg, p_st1, p_out, p_lbl],
            title=f"{sid} | Obs / Bkg / Stage-1 / Output / Label",
        )

        manifest["samples"].append({
            "sample_id": sid,
            "input_obs": str(p_obs.resolve()),
            "input_bkg": str(p_bkg.resolve()),
            "stage1": str(p_st1.resolve()),
            "output": str(p_out.resolve()),
            "label": str(p_lbl.resolve()),
            "triptych": str(p_tri.resolve()),
            "five_panel": str(p_story.resolve()),
            "colorbar_shared_temp_range": [temp_vmin, temp_vmax],
        })

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Save a reusable Gemini prompt template next to visuals.
    prompt_text = f"""# Gemini Image Prompt (202402, real-image anchored)\n\nUse the provided real panels as hard references (do not invent textures):\n- Input observation map\n- Background map\n- Stage-1 feature map\n- Output analysis map\n- Label target map\n\nVisual style should combine scientific clarity seen in ClimaX / FengWu / FuXi-DA / Pangu figures:\nclean white canvas, minimal grid, publication typography, consistent spacing.\n\nRequirements:\n1. Build a horizontal 5-stage story: Input Obs -> Input Bkg -> Stage-1 Feature -> Output -> Label.\n2. Preserve geospatial mapping (lat/lon-consistent appearance) from the real reference images.\n3. Keep the same colormap and same value range for Background and Output (Label can share the same scale).\n4. Stage-1 should use a distinct feature colormap (e.g., magma) and remain visually different from temperature maps.\n5. Keep inset globe in each panel to indicate Earth location.\n6. Add concise panel titles and no extra decorative elements.\n7. Final aspect ratio around 5:1, high-resolution, journal-ready.\n\nOutput folder: {str(out_dir.resolve())}\nManifest: {str((out_dir / 'manifest.json').resolve())}\n"""
    (out_dir / "gemini_prompt_202402.md").write_text(prompt_text, encoding="utf-8")

    print(f"Done. Output dir: {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

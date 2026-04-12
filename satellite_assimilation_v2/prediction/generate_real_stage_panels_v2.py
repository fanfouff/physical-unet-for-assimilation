#!/usr/bin/env python3
"""
Generate real sample stage panels for PASNet storytelling.

Per sample outputs:
1) raw observation swath map (scatter)
2) background map (level-wise)
3) stage-1 feature mean map
4) output analysis map (level-wise)
5) target/label map (level-wise)
6) 5-panel storyboard and 3-panel triptych

Key points:
- increment mode: analysis = bkg + denorm(pred_delta)
- bkg/output/target share one common color range
- raw obs swath is plotted from original scattered points
- each map includes a lower-right globe inset
- optional real ERA5 background replacement to enforce time semantics
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


ERA5_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825,
    850, 875, 900, 925, 950, 975, 1000,
]
_ERA5_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate real stage visual panels (v2)")
    p.add_argument("--checkpoint", required=True, type=str, help="Model checkpoint")
    p.add_argument("--raw_dir", required=True, type=str, help="Raw npy folder")
    p.add_argument("--output_dir", required=True, type=str, help="Output folder")
    p.add_argument("--num_samples", default=6, type=int, help="Number of samples")
    p.add_argument("--level_idx", default=22, type=int, help="Pressure level index")
    p.add_argument("--grid_h", default=64, type=int)
    p.add_argument("--grid_w", default=64, type=int)
    p.add_argument("--resolution", default=0.25, type=float)
    p.add_argument("--stats_file", default="", type=str, help="stats.npz path")
    p.add_argument("--increment_stats", default="", type=str, help="increment_stats.npz path")
    p.add_argument("--force_increment", action="store_true", help="Force increment mode")
    p.add_argument("--obs_channel", default=0, type=int, help="Raw obs channel for scatter map")
    p.add_argument("--obs_point_scale", default=1.35, type=float, help="Raw obs scatter point size scale")
    p.add_argument("--temp_smooth_sigma", default=0.9, type=float, help="Gaussian sigma for visual smoothing of temp maps")
    p.add_argument("--temp_upsample", default=4, type=int, help="Display upsample factor for temp maps")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--dpi", default=220, type=int)

    # Optional ERA5 background replacement (to enforce bkg=t, target=t+3 semantics)
    p.add_argument("--use_real_bkg", action="store_true", help="Replace simulated bkg with ERA5 bkg")
    p.add_argument("--era5_root", default="", type=str, help="ERA5 GRIB root, e.g. /data2/lrx/split_data")
    p.add_argument("--target_lead_hours", default=3, type=int, help="Assumed target lead wrt rounded obs time")
    p.add_argument("--bkg_lag_hours", default=3, type=int, help="bkg time lag wrt target time")
    p.add_argument("--diagnose_target_time", action="store_true", help="Compare target against nearby ERA5 times")
    return p.parse_args()


def resolve_stats_paths(args: argparse.Namespace, ckpt_args: SimpleNamespace) -> Tuple[Path, Optional[Path], bool]:
    cand_stats: List[Path] = []
    if args.stats_file:
        cand_stats.append(Path(args.stats_file).expanduser())
    if getattr(ckpt_args, "stats_file", ""):
        cand_stats.append(Path(str(ckpt_args.stats_file)).expanduser())
    for p in [
        "/data/lrx_true/era_obs/npz/stats.npz",
        "/data1/lrx/npz_64_real/stats.npz",
        "/data2/lrx/npz_64_real/stats.npz",
        str(Path(args.raw_dir).parent / "stats.npz"),
        str(Path(args.raw_dir).parent.parent / "npz" / "stats.npz"),
    ]:
        cand_stats.append(Path(p))

    stats_file = None
    for p in cand_stats:
        if p.exists():
            stats_file = p
            break
    if stats_file is None:
        raise FileNotFoundError(
            f"Cannot find stats.npz. Searched: {[str(c) for c in cand_stats]}"
        )

    use_increment = args.force_increment or _as_bool(getattr(ckpt_args, "use_increment", False))

    cand_inc: List[Path] = []
    if args.increment_stats:
        cand_inc.append(Path(args.increment_stats).expanduser())
    if getattr(ckpt_args, "increment_stats", ""):
        cand_inc.append(Path(str(ckpt_args.increment_stats)).expanduser())
    cand_inc.append(stats_file.parent / "increment_stats.npz")
    for p in [
        "/data/lrx_true/era_obs/npz/increment_stats.npz",
        "/data1/lrx/npz_64_real/increment_stats.npz",
        "/data2/lrx/npz_64_real/increment_stats.npz",
    ]:
        cand_inc.append(Path(p))

    inc_file: Optional[Path] = None
    for p in cand_inc:
        if p.exists():
            inc_file = p
            break

    if use_increment and inc_file is None:
        print("[WARN] Increment mode requested but increment_stats.npz not found. Fallback to absolute mode.")
        use_increment = False

    print(f"[INFO] stats: {stats_file}")
    print(f"[INFO] increment_stats: {inc_file}")
    print(f"[INFO] increment mode: {'ON' if use_increment else 'OFF'}")
    return stats_file, inc_file, use_increment


def load_model(checkpoint_path: Path, device: torch.device):
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    for candidate in [
        repo_root,
        Path("/home/lrx/Unet/satellite_assimilation_v2"),
        Path("/home/lrx/Unet"),
    ]:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    from models.backbone import create_model, UNetConfig

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
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

    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing keys: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    print("[INFO] checkpoint summary")
    print(f"       model={model_name}, use_increment={getattr(ckpt_args, 'use_increment', 'N/A')}, use_aux={getattr(ckpt_args, 'use_aux', 'N/A')}")

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
    if len(ids) <= n:
        return ids
    idx = np.linspace(0, len(ids) - 1, n, dtype=int)
    return [ids[i] for i in idx]


def norm_field(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[:, None, None]) / (std[:, None, None] + 1e-8)


def denorm_field(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std[:, None, None] + mean[:, None, None]


def _sid_to_datetime(sid: str) -> Optional[datetime]:
    m = re.match(r"collocation_(\d{8})_(\d{4})$", sid)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")


def _round_to_3h(dt: datetime) -> datetime:
    total_min = dt.hour * 60 + dt.minute
    rounded_h = round(total_min / 180.0) * 3
    base = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if rounded_h >= 24:
        return base + timedelta(days=1)
    return base + timedelta(hours=rounded_h)


def _resolve_era5_root(args: argparse.Namespace) -> Optional[Path]:
    candidates = []
    if args.era5_root:
        candidates.append(Path(args.era5_root).expanduser())
    for p in [
        "/data2/lrx/split_data",
        "/data/lrx_true/split_data",
        "/data1/lrx/split_data",
    ]:
        candidates.append(Path(p))

    for c in candidates:
        if c.exists():
            return c
    return None


def _grib_path(era5_root: Path, dt: datetime) -> Path:
    return era5_root / dt.strftime("%Y/%m") / f"era5_{dt.strftime('%Y%m%d_%H')}.grib"


def _load_era5_grib(grib_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pygrib

    grb = pygrib.open(str(grib_file))
    msgs = list(grb)
    grb.close()

    # De-duplicate levels if needed.
    seen = set()
    uniq = []
    for m in sorted(msgs, key=lambda it: it.level):
        if m.level in seen:
            continue
        seen.add(m.level)
        uniq.append(m)

    if len(uniq) != len(ERA5_LEVELS):
        raise RuntimeError(f"Expected {len(ERA5_LEVELS)} levels, got {len(uniq)} in {grib_file}")

    _, lats2d, lons2d = uniq[0].latlons()
    lats_1d = lats2d[:, 0]
    lons_1d = lons2d[0, :]

    stack = np.stack([m.values for m in uniq], axis=0).astype(np.float32)
    # Flip to 1000 hPa -> 1 hPa channel order, consistent with target files.
    stack = stack[::-1, :, :].copy()
    return stack, lats_1d, lons_1d


def _get_era5(era5_root: Path, dt: datetime) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    key = dt.strftime("%Y%m%d_%H")
    if key in _ERA5_CACHE:
        return _ERA5_CACHE[key]

    gp = _grib_path(era5_root, dt)
    if not gp.exists():
        return None

    result = _load_era5_grib(gp)
    if len(_ERA5_CACHE) > 6:
        _ERA5_CACHE.pop(next(iter(_ERA5_CACHE)))
    _ERA5_CACHE[key] = result
    return result


def _interp_era5_to_tile(
    era5_data: np.ndarray,
    lats_1d: np.ndarray,
    lons_1d: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    if lats_1d[0] > lats_1d[-1]:
        lats_1d = lats_1d[::-1].copy()
        era5_data = era5_data[:, ::-1, :].copy()

    h, w = lat2d.shape
    tile_points = np.column_stack([lat2d.ravel(), lon2d.ravel()])

    out = np.zeros((era5_data.shape[0], h, w), dtype=np.float32)
    for lev in range(era5_data.shape[0]):
        interp = RegularGridInterpolator(
            (lats_1d, lons_1d),
            era5_data[lev],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        out[lev] = interp(tile_points).reshape(h, w).astype(np.float32)
    return out


def build_real_bkg(
    sid: str,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    args: argparse.Namespace,
    tgt_phys: np.ndarray,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    info: Dict[str, object] = {
        "sid": sid,
        "enabled": True,
        "reason": "",
    }

    era5_root = _resolve_era5_root(args)
    if era5_root is None:
        info["reason"] = "era5_root_not_found"
        return None, info

    t_obs = _sid_to_datetime(sid)
    if t_obs is None:
        info["reason"] = "sid_time_parse_failed"
        return None, info

    t_anchor = _round_to_3h(t_obs)
    t_target = t_anchor + timedelta(hours=int(args.target_lead_hours))
    t_bkg = t_target - timedelta(hours=int(args.bkg_lag_hours))

    # Prefer requested bkg time, then +-3h fallbacks.
    cand = [
        t_bkg,
        t_bkg - timedelta(hours=3),
        t_bkg + timedelta(hours=3),
        t_anchor,
    ]

    selected = None
    era5_data = None
    lats_1d = None
    lons_1d = None
    for dt in cand:
        got = _get_era5(era5_root, dt)
        if got is None:
            continue
        era5_data, lats_1d, lons_1d = got
        selected = dt
        break

    info.update(
        {
            "era5_root": str(era5_root),
            "t_obs": t_obs.strftime("%Y-%m-%d %H:%M"),
            "t_anchor": t_anchor.strftime("%Y-%m-%d %H:%M"),
            "t_target_assumed": t_target.strftime("%Y-%m-%d %H:%M"),
            "t_bkg_requested": t_bkg.strftime("%Y-%m-%d %H:%M"),
            "t_bkg_selected": selected.strftime("%Y-%m-%d %H:%M") if selected else None,
        }
    )

    if selected is None:
        info["reason"] = "no_grib_found_for_candidates"
        return None, info

    try:
        bkg = _interp_era5_to_tile(era5_data, lats_1d, lons_1d, lat2d, lon2d)
    except Exception as e:
        info["reason"] = f"interp_failed: {e}"
        return None, info

    # Optional: diagnose which nearby ERA5 time matches target best.
    if args.diagnose_target_time:
        diag = {}
        for name, dt in [
            ("anchor", t_anchor),
            ("target", t_target),
            ("target_minus_3h", t_target - timedelta(hours=3)),
            ("target_plus_3h", t_target + timedelta(hours=3)),
        ]:
            got = _get_era5(era5_root, dt)
            if got is None:
                continue
            era5_d, lat_d, lon_d = got
            try:
                x = _interp_era5_to_tile(era5_d, lat_d, lon_d, lat2d, lon2d)
                rmse = float(np.sqrt(np.nanmean((x - tgt_phys) ** 2)))
                diag[name] = {
                    "dt": dt.strftime("%Y-%m-%d %H:%M"),
                    "rmse_vs_target_K": rmse,
                }
            except Exception:
                continue
        info["target_time_diagnosis"] = diag

    return bkg.astype(np.float32), info


def _add_globe_inset(fig, parent_ax, center_lon: float, center_lat: float, position: str = "lower_right") -> None:
    bbox = parent_ax.get_position()
    w = bbox.width * 0.22
    h = bbox.height * 0.22
    if position == "lower_right":
        x0 = bbox.x0 + bbox.width * 0.76
        y0 = bbox.y0 + bbox.height * 0.04
    else:
        x0 = bbox.x0 + bbox.width * 0.02
        y0 = bbox.y0 + bbox.height * 0.04

    ax_in = fig.add_axes([x0, y0, w, h], projection=ccrs.Orthographic(center_lon, center_lat))
    ax_in.set_global()
    ax_in.stock_img()
    ax_in.coastlines(linewidth=0.3, color="gray")
    ax_in.plot(
        center_lon,
        center_lat,
        marker="o",
        markersize=4,
        color="red",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )


def save_scatter_map(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    val: np.ndarray,
    title: str,
    cmap: str = "turbo",
    vmin=None,
    vmax=None,
    point_scale: float = 1.35,
    dpi: int = 220,
) -> None:
    fig = plt.figure(figsize=(6.4, 5.6), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())

    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    pad_lon = max((lon_max - lon_min) * 0.08, 1.0)
    pad_lat = max((lat_max - lat_min) * 0.08, 1.0)
    ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], ccrs.PlateCarree())

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)

    n_pts = len(val)
    pt_size = max(1.6, min(10.5, (3000.0 / max(n_pts, 1)) * max(0.5, point_scale)))
    im = ax.scatter(
        lon,
        lat,
        c=val,
        s=pt_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        edgecolors="none",
        alpha=0.85,
    )

    ax.set_title(title, fontsize=10, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cb.ax.tick_params(labelsize=7)

    _add_globe_inset(fig, ax, float(np.nanmedian(lon)), float(np.nanmedian(lat)))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_grid_map(
    path: Path,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: str = "RdYlBu_r",
    vmin=None,
    vmax=None,
    smooth_sigma: float = 0.0,
    upsample: int = 1,
    unit_label: str = "K",
    dpi: int = 220,
) -> None:
    fig = plt.figure(figsize=(6.4, 5.6), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())

    lon_min, lon_max = float(np.nanmin(lon2d)), float(np.nanmax(lon2d))
    lat_min, lat_max = float(np.nanmin(lat2d)), float(np.nanmax(lat2d))
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)

    lon_plot = lon2d
    lat_plot = lat2d
    field_plot = field
    if smooth_sigma > 0.0 or upsample > 1:
        try:
            from scipy.ndimage import gaussian_filter, zoom

            if smooth_sigma > 0.0:
                field_plot = gaussian_filter(field_plot, sigma=float(smooth_sigma), mode="nearest")

            if upsample > 1:
                zf = int(max(1, upsample))
                field_plot = zoom(field_plot, zf, order=3)
                lon_plot = zoom(lon2d, zf, order=1)
                lat_plot = zoom(lat2d, zf, order=1)
        except Exception as e:
            print(f"[WARN] temp smoothing skipped: {e}")

    im = ax.pcolormesh(
        lon_plot,
        lat_plot,
        field_plot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    ax.set_title(title, fontsize=10, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(unit_label, fontsize=8)

    _add_globe_inset(fig, ax, float(np.nanmedian(lon2d)), float(np.nanmedian(lat2d)))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_five_panel_storyboard(path: Path, image_paths: List[Path], panel_labels: List[str], suptitle: str, dpi: int = 220) -> None:
    assert len(image_paths) == 5
    assert len(panel_labels) == 5

    imgs = [plt.imread(str(p)) for p in image_paths]
    fig = plt.figure(figsize=(26, 5.5), dpi=dpi)
    gs = gridspec.GridSpec(1, 5, wspace=0.05)

    for idx, (img, label) in enumerate(zip(imgs, panel_labels)):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=11, fontweight="bold", pad=4)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def save_triptych(path: Path, image_paths: List[Path], panel_labels: List[str], suptitle: str, dpi: int = 220) -> None:
    imgs = [plt.imread(str(p)) for p in image_paths]
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2), dpi=dpi)
    if n == 1:
        axes = [axes]
    for ax, im, label in zip(axes, imgs, panel_labels):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(label, fontsize=11, fontweight="bold")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _build_gemini_prompt(out_dir: Path, use_increment: bool) -> str:
    return f"""# Gemini Image Prompt (real-image anchored, v3)

## Goal
Create one publication-ready horizontal 5-stage storyboard for PASNet data assimilation.
Use the generated real panel images as hard references. No hallucinated textures.

## Mandatory hard constraints
1. Panel (a) Input raw observation swath must be the real scatter map from manifest.
2. Panel (b) Background map must be the real mapped background field from manifest.
3. Panel (c) Stage-1 feature map must be the real Stage-1 image from manifest.
4. Panel (d) Output analysis map must be the real output image from manifest.
5. Panel (e) Target map must be the real label image from manifest.
6. Keep geospatial extent exactly aligned with source maps.
7. Keep lower-right globe inset on every panel.
8. Keep one shared colorbar scale for (b)(d)(e). Do not re-normalize them independently.
9. Stage-1 keeps a distinct feature colormap (magma-like), separate from temperature maps.
10. Final figure ratio about 5:1, high resolution (>= 300 dpi).

## Scientific meaning to show
- Input Obs + Background -> PASNet Stage-1 fusion -> refined analysis output.
- Increment mode: analysis = background + delta.
- Current increment mode flag: {'ON' if use_increment else 'OFF'}.

## Visual language (inspired by paper figure style)
Reference style from these local docs:
- /home/lrx/Unet/imgs/climax.pdf
- /home/lrx/Unet/imgs/fengwu.pdf
- /home/lrx/Unet/imgs/fuxida.pdf
- /home/lrx/Unet/imgs/pangu.pdf

Design rules:
- clean white canvas
- compact typography
- thin arrows between stages
- minimal decoration
- journal-ready spacing and alignment

## Suggested top annotation
Satellite Obs ----+ 
                  +--> PASNet Stage-1 --> PASNet Stage-2 --> Analysis
ERA5 Background --+

## Inputs
- Manifest: {str((out_dir / 'manifest.json').resolve())}
- Panel directory: {str((out_dir / 'panels').resolve())}
- Storyboards directory: {str((out_dir / 'storyboards').resolve())}

## Safety checks before final render
- Check that (b)(d)(e) visually correspond to the same numeric color range in manifest.
- Check panel (a) is scatter swath (not gridded texture).
- Check inset globe exists in each panel.
- Do not add fabricated meteorological structures.
"""


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
    print(f"[INFO] device: {device}")

    model, ckpt_args = load_model(Path(args.checkpoint), device)
    use_aux = _as_bool(getattr(ckpt_args, "use_aux", True))

    stats_file, inc_file, use_increment = resolve_stats_paths(args, ckpt_args)
    stats = np.load(stats_file)
    inc_stats = np.load(inc_file) if (use_increment and inc_file is not None) else None

    obs_mean, obs_std = stats["obs_mean"], stats["obs_std"]
    bkg_mean, bkg_std = stats["bkg_mean"], stats["bkg_std"]
    tgt_mean, tgt_std = stats["target_mean"], stats["target_std"]

    if inc_stats is not None:
        print(
            f"[INFO] increment stats range: mean[{inc_stats['inc_mean'].min():.3f}, {inc_stats['inc_mean'].max():.3f}] "
            f"std[{inc_stats['inc_std'].min():.3f}, {inc_stats['inc_std'].max():.3f}]"
        )

    sample_ids = pick_sample_ids(raw_dir, args.num_samples)
    if not sample_ids:
        raise RuntimeError(f"No valid samples found in {raw_dir}")

    import sys

    repo_root = Path(__file__).resolve().parent.parent
    for candidate in [repo_root, Path("/home/lrx/Unet/satellite_assimilation_v2"), Path("/home/lrx/Unet")]:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    from data_process.prepare_v3_data import convert_single_sample

    manifest: Dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "stats_file": str(stats_file.resolve()),
        "increment_stats": str(inc_file.resolve()) if inc_file else None,
        "use_increment": use_increment,
        "use_real_bkg": bool(args.use_real_bkg),
        "target_lead_hours": int(args.target_lead_hours),
        "bkg_lag_hours": int(args.bkg_lag_hours),
        "level_idx": int(args.level_idx),
        "paper_refs": {
            "climax": "/home/lrx/Unet/imgs/climax.pdf",
            "fengwu": "/home/lrx/Unet/imgs/fengwu.pdf",
            "fuxida": "/home/lrx/Unet/imgs/fuxida.pdf",
            "pangu": "/home/lrx/Unet/imgs/pangu.pdf",
        },
        "samples": [],
    }

    for i, sid in enumerate(sample_ids, start=1):
        print("-" * 60)
        print(f"[INFO] sample {i}/{len(sample_ids)}: {sid}")

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

        bkg_source = "prepare_v3_simulated"
        bkg_time_info: Dict[str, object] = {}
        if args.use_real_bkg:
            real_bkg, bkg_time_info = build_real_bkg(sid, lat2d, lon2d, args, tgt_phys)
            if real_bkg is not None:
                bkg_phys = real_bkg
                bkg_source = "era5_interpolated"
            else:
                bkg_source = "prepare_v3_simulated_fallback"

        obs_n = norm_field(obs_phys, obs_mean, obs_std)
        bkg_n = norm_field(bkg_phys, bkg_mean, bkg_std)

        obs_t = torch.from_numpy(obs_n[None]).float().to(device)
        bkg_t = torch.from_numpy(bkg_n[None]).float().to(device)
        mask_t = torch.from_numpy(mask[None]).float().to(device)
        aux_t = torch.from_numpy(aux[None]).float().to(device) if use_aux and aux is not None else None

        with torch.no_grad():
            if hasattr(model, "stem"):
                try:
                    stem_out = model.stem(obs_t, bkg_t, mask_t, aux_t)
                    if isinstance(stem_out, tuple):
                        stem_out = stem_out[0]
                    stage1_map = stem_out.mean(dim=1)[0].detach().cpu().numpy()
                except Exception as e:
                    print(f"[WARN] stem forward failed: {e}")
                    stage1_map = np.zeros((args.grid_h, args.grid_w), dtype=np.float32)
            else:
                stage1_map = np.zeros((args.grid_h, args.grid_w), dtype=np.float32)

            out = model(obs_t, bkg_t, mask_t, aux_t)
            pred_n = out[0] if isinstance(out, tuple) else out
            pred_n = pred_n[0].detach().cpu().numpy()

        if use_increment and inc_stats is not None:
            pred_inc_phys = denorm_field(pred_n, inc_stats["inc_mean"], inc_stats["inc_std"])
            ana_phys = bkg_phys + pred_inc_phys
            mode_tag = "increment"
        else:
            ana_phys = denorm_field(pred_n, tgt_mean, tgt_std)
            mode_tag = "absolute"

        li = int(np.clip(args.level_idx, 0, ana_phys.shape[0] - 1))
        bkg_lv = bkg_phys[li]
        out_lv = ana_phys[li]
        lbl_lv = tgt_phys[li]

        temp_vmin = float(np.nanmin([bkg_lv.min(), out_lv.min(), lbl_lv.min()]))
        temp_vmax = float(np.nanmax([bkg_lv.max(), out_lv.max(), lbl_lv.max()]))

        rmse_bkg_vs_lbl = float(np.sqrt(np.nanmean((bkg_lv - lbl_lv) ** 2)))
        rmse_out_vs_lbl = float(np.sqrt(np.nanmean((out_lv - lbl_lv) ** 2)))

        print(
            f"[INFO] level={li} source={bkg_source} mode={mode_tag} "
            f"range_shared=[{temp_vmin:.2f}, {temp_vmax:.2f}] "
            f"rmse(bkg,target)={rmse_bkg_vs_lbl:.3f} rmse(out,target)={rmse_out_vs_lbl:.3f}"
        )

        p_obs = panels_dir / f"{sid}_1_raw_obs_swath.png"
        p_bkg = panels_dir / f"{sid}_2_bkg_level{li}.png"
        p_st1 = panels_dir / f"{sid}_3_stage1_feature.png"
        p_out = panels_dir / f"{sid}_4_output_level{li}.png"
        p_lbl = panels_dir / f"{sid}_5_target_level{li}.png"

        obs_ch = min(max(args.obs_channel, 0), obs_raw.shape[1] - 1) if obs_raw.ndim == 2 else 0
        obs_vals = obs_raw[:, obs_ch] if obs_raw.ndim == 2 else obs_raw.ravel()

        # Keep raw scatter on the same geographic tile as grid maps.
        lat_min, lat_max = float(np.nanmin(lat2d)), float(np.nanmax(lat2d))
        lon_min, lon_max = float(np.nanmin(lon2d)), float(np.nanmax(lon2d))
        pad = float(args.resolution * 3.0)
        obs_keep = (
            np.isfinite(lat)
            & np.isfinite(lon)
            & (lat >= lat_min - pad)
            & (lat <= lat_max + pad)
            & (lon >= lon_min - pad)
            & (lon <= lon_max + pad)
        )
        if int(obs_keep.sum()) < 32:
            # Fallback to full swath only when local points are unexpectedly too few.
            obs_keep = np.isfinite(lat) & np.isfinite(lon)

        save_scatter_map(
            p_obs,
            lon=lon[obs_keep],
            lat=lat[obs_keep],
            val=obs_vals[obs_keep],
            title=f"(a) Raw Obs Swath [ch={obs_ch}]",
            cmap="turbo",
            point_scale=args.obs_point_scale,
            dpi=args.dpi,
        )
        save_grid_map(
            p_bkg,
            lon2d=lon2d,
            lat2d=lat2d,
            field=bkg_lv,
            title=f"(b) Background (t) [level {li}]",
            cmap="RdYlBu_r",
            vmin=temp_vmin,
            vmax=temp_vmax,
            smooth_sigma=args.temp_smooth_sigma,
            upsample=args.temp_upsample,
            dpi=args.dpi,
        )
        save_grid_map(
            p_st1,
            lon2d=lon2d,
            lat2d=lat2d,
            field=stage1_map,
            title="(c) Stage-1 Feature Mean",
            cmap="magma",
            unit_label="feature",
            dpi=args.dpi,
        )
        save_grid_map(
            p_out,
            lon2d=lon2d,
            lat2d=lat2d,
            field=out_lv,
            title=f"(d) Output {'(bkg+delta)' if mode_tag == 'increment' else ''} [level {li}]",
            cmap="RdYlBu_r",
            vmin=temp_vmin,
            vmax=temp_vmax,
            smooth_sigma=args.temp_smooth_sigma,
            upsample=args.temp_upsample,
            dpi=args.dpi,
        )
        save_grid_map(
            p_lbl,
            lon2d=lon2d,
            lat2d=lat2d,
            field=lbl_lv,
            title=f"(e) ERA5 Target (t+dt) [level {li}]",
            cmap="RdYlBu_r",
            vmin=temp_vmin,
            vmax=temp_vmax,
            dpi=args.dpi,
        )

        p_story = story_dir / f"{sid}_five_panel.png"
        save_five_panel_storyboard(
            p_story,
            image_paths=[p_obs, p_bkg, p_st1, p_out, p_lbl],
            panel_labels=["(a) Raw Obs", "(b) Background", "(c) Stage-1", "(d) Output", "(e) Target"],
            suptitle=(
                f"Sample: {sid} | level {li} | mode={mode_tag} | bkg={bkg_source}"
            ),
            dpi=args.dpi,
        )

        p_tri = trip_dir / f"{sid}_triptych.png"
        save_triptych(
            p_tri,
            image_paths=[p_obs, p_st1, p_out],
            panel_labels=["Raw Obs", "Stage-1 Feature", "Output Analysis"],
            suptitle=f"{sid} | Obs -> Stage-1 -> Output",
            dpi=args.dpi,
        )

        manifest["samples"].append(
            {
                "sample_id": sid,
                "panels": {
                    "raw_obs": str(p_obs.resolve()),
                    "background": str(p_bkg.resolve()),
                    "stage1": str(p_st1.resolve()),
                    "output": str(p_out.resolve()),
                    "target": str(p_lbl.resolve()),
                },
                "composites": {
                    "five_panel": str(p_story.resolve()),
                    "triptych": str(p_tri.resolve()),
                },
                "bkg_source": bkg_source,
                "time_info": bkg_time_info,
                "colorbar_shared_temp_range_K": [temp_vmin, temp_vmax],
                "level_idx": li,
                "rmse_level": {
                    "bkg_vs_target": rmse_bkg_vs_lbl,
                    "output_vs_target": rmse_out_vs_lbl,
                },
            }
        )

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    prompt_path = out_dir / "gemini_prompt_v3.md"
    prompt_path.write_text(_build_gemini_prompt(out_dir, use_increment), encoding="utf-8")

    print("=" * 70)
    print(f"[DONE] output_dir: {out_dir}")
    print(f"[DONE] manifest:   {manifest_path}")
    print(f"[DONE] prompt:     {prompt_path}")
    print(f"[DONE] panels:     {panels_dir}")
    print(f"[DONE] storyboard: {story_dir}")
    print(f"[DONE] triptych:   {trip_dir}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        raise

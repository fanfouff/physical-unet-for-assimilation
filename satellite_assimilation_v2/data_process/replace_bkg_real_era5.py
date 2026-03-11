"""
replace_bkg_real_era5.py
========================
将现有 npz 数据集中的伪造背景场（target + 0.3K噪声）替换为
真实 ERA5 6h 预报场（T_bkg = T_analysis - 6h），同时重算 stats.npz。

用法:
  python replace_bkg_real_era5.py \
      --src_dir /data2/lrx/npz_64 \
      --dst_dir /data2/lrx/npz_64_real \
      --era5_root /data2/lrx/split_data \
      --workers 8
"""

import argparse
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.interpolate import RegularGridInterpolator


ERA5_37_LEVELS = [1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,
                  300,350,400,450,500,550,600,650,700,750,775,800,825,
                  850,875,900,925,950,975,1000]


def parse_obs_time(stem):
    m = re.match(r'collocation_(\d{8})_(\d{4})', stem)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M')


def round_to_3h(dt):
    total_min = dt.hour * 60 + dt.minute
    rounded_h = round(total_min / 180) * 3
    base = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if rounded_h >= 24:
        return base + timedelta(days=1)
    return base + timedelta(hours=rounded_h)


def grib_path(era5_root, dt):
    return era5_root / dt.strftime('%Y/%m') / f'era5_{dt.strftime("%Y%m%d_%H")}.grib'


def load_era5_grib(grib_file):
    import pygrib
    grb = pygrib.open(str(grib_file))
    msgs = list(grb)
    grb.close()
    # 去重：部分月份（如12月）GRIB每层存了两份完全相同的消息
    seen_levels = set()
    msgs_sorted = []
    for m in sorted(msgs, key=lambda m: m.level):
        if m.level not in seen_levels:
            seen_levels.add(m.level)
            msgs_sorted.append(m)
    assert len(msgs_sorted) == 37, f"Expected 37 levels, got {len(msgs_sorted)} in {grib_file}"
    data0, lats2d, lons2d = msgs_sorted[0].latlons()
    lats_1d = lats2d[:, 0]
    lons_1d = lons2d[0, :]
    # m.values 比 m.data()[0] 快 4x (不重算 latlons meshgrid)
    stack = np.stack([m.values for m in msgs_sorted], axis=0).astype(np.float32)
    # Flip level axis: sorted ascending (1hPa first) -> flip to 1000hPa first to match target order
    stack = stack[::-1, :, :].copy()
    return stack, lats_1d, lons_1d


def interp_era5_to_tile(era5_data, lats_1d, lons_1d, lat2d, lon2d):
    if lats_1d[0] > lats_1d[-1]:
        lats_1d = lats_1d[::-1].copy()
        era5_data = era5_data[:, ::-1, :].copy()

    H, W = lat2d.shape
    tile_points = np.column_stack([lat2d.ravel(), lon2d.ravel()])

    result = np.zeros((37, H, W), dtype=np.float32)
    for lev in range(37):
        interp = RegularGridInterpolator(
            (lats_1d, lons_1d),
            era5_data[lev],
            method='linear',
            bounds_error=False,
            fill_value=None,
        )
        result[lev] = interp(tile_points).reshape(H, W).astype(np.float32)
    return result


_grib_cache = {}
_MAX_CACHE = 4


def get_era5(era5_root, dt):
    key = dt.strftime('%Y%m%d_%H')
    if key not in _grib_cache:
        if len(_grib_cache) >= _MAX_CACHE:
            oldest = next(iter(_grib_cache))
            del _grib_cache[oldest]
        gp = grib_path(era5_root, dt)
        if not gp.exists():
            return None
        _grib_cache[key] = load_era5_grib(gp)
    return _grib_cache[key]


def process_one(args):
    src_path, dst_path, era5_root_str = args
    era5_root = Path(era5_root_str)
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    try:
        stem = src_path.stem
        t_obs = parse_obs_time(stem)
        if t_obs is None:
            return stem, 'skip', ''

        t_an = round_to_3h(t_obs)
        t_bkg = t_an - timedelta(hours=6)

        bkg_dt = None
        for candidate in [t_bkg, t_an - timedelta(hours=3), t_an]:
            if grib_path(era5_root, candidate).exists():
                bkg_dt = candidate
                break
        if bkg_dt is None:
            return stem, 'fail', f'no grib found for {t_an}'

        era5_result = get_era5(era5_root, bkg_dt)
        if era5_result is None:
            return stem, 'fail', f'grib load failed'

        era5_data, lats_1d, lons_1d = era5_result
        src_data = dict(np.load(src_path, allow_pickle=False))
        lat2d = src_data['lat2d']
        lon2d = src_data['lon2d']

        bkg_real = interp_era5_to_tile(era5_data, lats_1d, lons_1d, lat2d, lon2d)
        src_data['bkg'] = bkg_real

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(dst_path), **src_data)

        lag_h = (t_an - bkg_dt).total_seconds() / 3600
        return stem, 'ok', f'lag={lag_h:.0f}h'

    except Exception as e:
        return src_path.stem, 'fail', str(e)


def recompute_stats(dst_dir, split='train'):
    split_dir = dst_dir / split
    files = sorted(f for f in split_dir.glob('*.npz') if f.name != 'stats.npz')
    if not files:
        print(f'[stats] No files in {split_dir}')
        return

    print(f'[stats] Computing from {len(files)} training files ...')
    sums = {}
    sumsq = {}
    counts = {}
    keys = ['obs', 'bkg', 'target', 'aux']

    for f in files:
        d = np.load(f, allow_pickle=False)
        for k in keys:
            if k not in d:
                continue
            arr = d[k].astype(np.float64)
            C = arr.shape[0]
            vals = arr.reshape(C, -1)
            if k not in sums:
                sums[k] = np.zeros(C)
                sumsq[k] = np.zeros(C)
                counts[k] = 0
            sums[k] += vals.mean(axis=1)
            sumsq[k] += (vals ** 2).mean(axis=1)
            counts[k] += 1

    stats = {}
    for k in keys:
        if k not in counts:
            continue
        n = counts[k]
        mean = sums[k] / n
        std = np.sqrt(np.maximum(sumsq[k] / n - mean ** 2, 1e-12))
        stats[f'{k}_mean'] = mean.astype(np.float32)
        stats[f'{k}_std'] = std.astype(np.float32)
        print(f'  {k}: mean=[{mean.min():.2f},{mean.max():.2f}]  std=[{std.min():.4f},{std.max():.4f}]')

    out = dst_dir / 'stats.npz'
    np.savez(str(out), **stats)
    print(f'[stats] Saved -> {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True)
    parser.add_argument('--dst_dir', required=True)
    parser.add_argument('--era5_root', required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    era5_root = Path(args.era5_root)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(f for f in src_dir.glob('**/*.npz')
                       if f.name not in ('stats.npz', 'dataset_split.json'))
    print(f'Found {len(src_files)} npz files to process')

    task_args = []
    for sf in src_files:
        rel = sf.relative_to(src_dir)
        df = dst_dir / rel
        task_args.append((str(sf), str(df), str(era5_root)))

    ok_count = skip_count = fail_count = 0
    fail_list = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futs = {executor.submit(process_one, a): a[0] for a in task_args}
        for i, fut in enumerate(as_completed(futs), 1):
            stem, status, msg = fut.result()
            if status == 'ok':
                ok_count += 1
            elif status == 'skip':
                skip_count += 1
            else:
                fail_count += 1
                fail_list.append(f'{stem}: {msg}')
            if i % 200 == 0 or i == len(task_args):
                print(f'  [{i}/{len(task_args)}] ok={ok_count} skip={skip_count} fail={fail_count}')

    print(f'\n=== Done: ok={ok_count}, skip={skip_count}, fail={fail_count} ===')
    if fail_list:
        print('Failures:')
        for s in fail_list[:20]:
            print(' ', s)

    recompute_stats(dst_dir, split='train')
    print(f'\nDataset saved to: {dst_dir}')


if __name__ == '__main__':
    main()

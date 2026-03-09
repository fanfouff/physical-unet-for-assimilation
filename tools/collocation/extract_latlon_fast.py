#!/usr/bin/env python3
"""
快速从原始FY-3F HDF文件中提取经纬度，生成 _lat.npy / _lon.npy
每个文件约 0.15s，3957 个文件用 8 进程约 1-2 分钟即可完成。
"""
import os, re, glob, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import h5py


def load_hdf_bt_latlon(hdf_path):
    with h5py.File(hdf_path, 'r') as f:
        if 'Data/Earth_Obs_BT' in f:
            ds = f['Data/Earth_Obs_BT']
            bt_raw = ds[:].astype(np.float64)
            slope     = float(np.asarray(ds.attrs.get('Slope',     1.0)).ravel()[0])
            intercept = float(np.asarray(ds.attrs.get('Intercept', 0.0)).ravel()[0])
            bt = bt_raw * slope + intercept
        elif 'Brightness_Temperature' in f:
            bt = f['Brightness_Temperature'][:].astype(np.float64)
        else:
            raise ValueError('No BT data')
        if 'Geolocation/Latitude' in f:
            lat = f['Geolocation/Latitude'][:]
            lon = f['Geolocation/Longitude'][:]
        elif 'Latitude' in f:
            lat = f['Latitude'][:]
            lon = f['Longitude'][:]
        else:
            raise ValueError('No lat/lon')
    # Transpose (Ch,Scan,Pos) → (Scan,Pos,Ch)
    if bt.ndim == 3 and lat.ndim == 2 and bt.shape[1:] == lat.shape:
        bt = np.transpose(bt, (1, 2, 0))
    n_scan, n_pos, _ = bt.shape
    bt_flat  = bt.reshape(-1, bt.shape[2])
    lat_flat = lat.flatten() if lat.ndim == 2 else np.repeat(lat, n_pos)
    lon_flat = lon.flatten() if lon.ndim == 2 else np.repeat(lon, n_pos)
    return bt_flat, lat_flat, lon_flat


def apply_auto_qc(bt_flat):
    bt_qc = bt_flat.copy()
    valid_vals = bt_flat[(bt_flat > 0) & ~np.isnan(bt_flat)]
    if len(valid_vals):
        p1, p99 = np.percentile(valid_vals, [1, 99])
        bt_min = max(50., p1 - 50); bt_max = min(400., p99 + 50)
    else:
        bt_min, bt_max = 100., 400.
    invalid = (bt_qc < bt_min) | (bt_qc > bt_max) | (bt_qc <= 0) | np.isnan(bt_qc)
    bt_qc[invalid] = np.nan
    return bt_qc


def sequential_match(bt_clean, X):
    """O(K) forward scan matching; returns bool mask over bt_clean."""
    K, N = len(bt_clean), len(X)
    valid_mask = np.zeros(K, dtype=bool)
    i = 0
    for k in range(K):
        if i >= N:
            break
        bk = bt_clean[k]
        xi = X[i]
        if not np.isnan(bk[0]) and not np.isnan(bk[1]):
            if abs(bk[0] - xi[0]) < 0.01 and abs(bk[1] - xi[1]) < 0.01:
                valid_mask[k] = True
                i += 1
    return valid_mask, i


def find_hdf(fy3f_root, date_str, time_str):
    year, month = date_str[:4], date_str[4:6]
    for pat in [
        f"{fy3f_root}/{year}/{month}/FY3F*{date_str}*{time_str}*.HDF",
        f"{fy3f_root}/{year}/*/FY3F*{date_str}*{time_str}*.HDF",
    ]:
        files = glob.glob(pat)
        if files:
            return files[0]
    return None


def process_one(args_tuple):
    x_path, fy3f_root, overwrite = args_tuple
    x_path = Path(x_path)
    pfx = str(x_path).replace('_X.npy', '')
    lat_p = Path(pfx + '_lat.npy')
    lon_p = Path(pfx + '_lon.npy')
    if not overwrite and lat_p.exists() and lon_p.exists():
        return 'skip', x_path.name
    m = re.search(r'collocation_(\d{8})_(\d{4})', x_path.name)
    if not m:
        return 'fail', f'{x_path.name}: parse error'
    hdf = find_hdf(fy3f_root, m.group(1), m.group(2))
    if not hdf:
        return 'fail', f'{x_path.name}: HDF not found'
    try:
        bt_flat, lat_flat, lon_flat = load_hdf_bt_latlon(hdf)
        bt_qc = apply_auto_qc(bt_flat)
        vi = ~np.all(np.isnan(bt_qc), axis=1)
        bt_c = bt_qc[vi]; lat_c = lat_flat[vi]; lon_c = lon_flat[vi]
        X = np.load(str(x_path))
        vmask, matched = sequential_match(bt_c, X)
        if matched != len(X):
            return 'fail', f'{x_path.name}: matched {matched}/{len(X)}'
        np.save(str(lat_p), lat_c[vmask].astype(np.float32))
        np.save(str(lon_p), lon_c[vmask].astype(np.float32))
        return 'ok', x_path.name
    except Exception as e:
        return 'fail', f'{x_path.name}: {e}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  default='/data/lrx_true/era_obs')
    p.add_argument('--fy3f_dir',  default='/data2/lrx/fy3f_organized')
    p.add_argument('--workers',   type=int, default=8)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--year',  default=None)
    p.add_argument('--month', default=None)
    args = p.parse_args()

    all_x = sorted(glob.glob(f'{args.data_dir}/**/*_X.npy', recursive=True))
    if args.year:  all_x = [f for f in all_x if f'/{args.year}/' in f]
    if args.month: all_x = [f for f in all_x if f'/{args.month}/' in f]
    existing = sum(1 for f in all_x if Path(f.replace('_X.npy','_lat.npy')).exists())
    print(f'Files: {len(all_x)}  already_done: {existing}  to_process: {len(all_x)-existing}')

    tasks = [(f, args.fy3f_dir, args.overwrite) for f in all_x]
    n_ok = n_skip = n_fail = 0; failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(process_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            st, msg = fut.result()
            if   st == 'ok':   n_ok += 1
            elif st == 'skip': n_skip += 1
            else:              n_fail += 1; failures.append(msg)
            if done % 200 == 0 or done == len(tasks):
                print(f'  [{done}/{len(tasks)}] ok={n_ok} skip={n_skip} fail={n_fail}')
    print(f'\nDone: ok={n_ok}  skip={n_skip}  fail={n_fail}')
    for f in failures[:10]: print(f'  FAIL: {f[:200]}')

if __name__ == '__main__':
    main()

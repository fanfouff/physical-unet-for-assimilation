#!/usr/bin/env python3
"""
诊断FY-3F亮温数据的实际值和范围
"""

import h5py
import numpy as np
import sys

def diagnose_brightness_temperature(filepath):
    """诊断亮温数据"""
    print("="*70)
    print("亮温数据诊断")
    print("="*70)
    print(f"文件: {filepath}\n")
    
    with h5py.File(filepath, 'r') as f:
        # 读取亮温数据
        if 'Data/Earth_Obs_BT' in f:
            bt = f['Data/Earth_Obs_BT'][:]
            print("✓ 找到: Data/Earth_Obs_BT\n")
        else:
            print("✗ 未找到亮温数据")
            return
        
        # 基本信息
        print("📊 数据基本信息:")
        print(f"   形状: {bt.shape}")
        print(f"   数据类型: {bt.dtype}")
        print(f"   总元素数: {bt.size:,}")
        print()
        
        # 统计信息
        print("📈 统计信息:")
        print(f"   最小值: {np.nanmin(bt):.4f}")
        print(f"   最大值: {np.nanmax(bt):.4f}")
        print(f"   平均值: {np.nanmean(bt):.4f}")
        print(f"   中位数: {np.nanmedian(bt):.4f}")
        print(f"   标准差: {np.nanstd(bt):.4f}")
        print()
        
        # NaN统计
        n_nan = np.sum(np.isnan(bt))
        print(f"   NaN数量: {n_nan:,} ({n_nan/bt.size*100:.2f}%)")
        
        # 检查特殊值
        n_zero = np.sum(bt == 0)
        n_neg = np.sum(bt < 0)
        print(f"   零值数量: {n_zero:,} ({n_zero/bt.size*100:.2f}%)")
        print(f"   负值数量: {n_neg:,} ({n_neg/bt.size*100:.2f}%)")
        print()
        
        # 值分布
        print("📊 值分布:")
        valid_bt = bt[~np.isnan(bt) & (bt != 0)]
        if len(valid_bt) > 0:
            percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
            print("   百分位数:")
            for p in percentiles:
                val = np.percentile(valid_bt, p)
                print(f"      {p:3d}%: {val:10.4f}")
        print()
        
        # 通道分析
        print("📡 通道分析:")
        if bt.ndim == 3:
            n_channels = bt.shape[2]
            print(f"   通道数: {n_channels}")
            print(f"   前10个通道的平均值:")
            for ch in range(min(10, n_channels)):
                ch_data = bt[:, :, ch]
                valid_data = ch_data[~np.isnan(ch_data) & (ch_data != 0)]
                if len(valid_data) > 0:
                    print(f"      通道 {ch+1:2d}: min={valid_data.min():8.2f}, "
                          f"max={valid_data.max():8.2f}, "
                          f"mean={valid_data.mean():8.2f}")
        print()
        
        # 单位推断
        print("🔍 单位推断:")
        mean_val = np.nanmean(valid_bt) if len(valid_bt) > 0 else 0
        
        if 0 < mean_val < 50:
            print("   ⚠️  数据可能是以 °C (摄氏度) 为单位")
            print("   建议转换: BT(K) = BT(°C) + 273.15")
        elif 50 < mean_val < 400:
            print("   ✓ 数据可能已经是 K (开尔文) 单位")
        elif 400 < mean_val < 10000:
            print("   ⚠️  数据可能是缩放后的值（需要乘以缩放因子）")
            print("   检查文件中的 'scale_factor' 或 'add_offset' 属性")
        elif 10000 < mean_val:
            print("   ⚠️  数据可能是原始计数值，需要定标")
        else:
            print("   ⚠️  数据范围异常，请检查数据质量")
        print()
        
        # 检查属性
        print("📝 数据集属性:")
        bt_dataset = f['Data/Earth_Obs_BT']
        if bt_dataset.attrs:
            for key, value in bt_dataset.attrs.items():
                print(f"   {key}: {value}")
        else:
            print("   无属性信息")
        print()
        
        # 建议的质量控制范围
        print("💡 建议的质量控制参数:")
        if 0 < mean_val < 50:
            print("   --bt-min -50  --bt-max 50  (如果单位是°C)")
            print("   或先转换到K: BT_K = BT_C + 273.15")
        elif 50 < mean_val < 400:
            print("   --bt-min 100  --bt-max 400  (如果单位是K)")
        elif bt.dtype in [np.uint16, np.int16, np.uint32, np.int32]:
            print("   数据可能需要定标，检查:")
            print("   - scale_factor")
            print("   - add_offset")
            print("   - valid_min / valid_max")
        
        # 采样显示
        print("\n🔬 数据采样 (第1个扫描线，前5个角度，前5个通道):")
        sample = bt[0, :5, :5]
        print(sample)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python diagnose_bt_values.py <FY3F_file>")
        sys.exit(1)
    
    diagnose_brightness_temperature(sys.argv[1])

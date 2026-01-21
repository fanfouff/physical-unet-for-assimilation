#!/usr/bin/env python3
"""
快速测试脚本：验证FY-3D/FY-3F文件是否能正确读取
"""

import h5py
import numpy as np
import sys

def test_fy3d_file(filepath):
    """测试FY-3D/FY-3F文件读取"""
    print("="*70)
    print("FY-3D/FY-3F 文件结构测试")
    print("="*70)
    print(f"文件: {filepath}\n")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print("✓ 文件成功打开\n")
            
            # 1. 显示顶层结构
            print("1️⃣  顶层结构:")
            for key in f.keys():
                print(f"   ├── {key}")
            print()
            
            # 2. 检查亮温数据
            print("2️⃣  亮温数据:")
            bt_found = False
            bt_data = None
            bt_path = None
            
            if 'Data/Earth_Obs_BT' in f:
                bt_data = f['Data/Earth_Obs_BT']
                bt_path = 'Data/Earth_Obs_BT'
                bt_found = True
                print(f"   ✓ 找到: {bt_path}")
            elif 'Brightness_Temperature' in f:
                bt_data = f['Brightness_Temperature']
                bt_path = 'Brightness_Temperature'
                bt_found = True
                print(f"   ✓ 找到: {bt_path}")
            else:
                print("   ✗ 未找到亮温数据")
            
            if bt_found:
                print(f"   形状: {bt_data.shape}")
                print(f"   数据类型: {bt_data.dtype}")
                
                # 读取一小部分数据测试
                sample = bt_data[0, 0, :]
                print(f"   样本数据 (第一个像元): {sample}")
                print(f"   数据范围: [{np.nanmin(bt_data[:]):.2f}, {np.nanmax(bt_data[:]):.2f}] K")
            print()
            
            # 3. 检查经纬度
            print("3️⃣  地理定位数据:")
            lat_found = False
            lon_found = False
            
            if 'Geolocation/Latitude' in f:
                lat = f['Geolocation/Latitude'][:]
                lon = f['Geolocation/Longitude'][:]
                lat_found = True
                lon_found = True
                print(f"   ✓ 找到: Geolocation/Latitude")
                print(f"   ✓ 找到: Geolocation/Longitude")
            elif 'Latitude' in f:
                lat = f['Latitude'][:]
                lon = f['Longitude'][:]
                lat_found = True
                lon_found = True
                print(f"   ✓ 找到: Latitude")
                print(f"   ✓ 找到: Longitude")
            else:
                print("   ✗ 未找到经纬度数据")
            
            if lat_found:
                print(f"   纬度形状: {lat.shape}")
                print(f"   纬度范围: [{lat.min():.2f}, {lat.max():.2f}]°")
                print(f"   经度范围: [{lon.min():.2f}, {lon.max():.2f}]°")
            print()
            
            # 4. 检查时间数据
            print("4️⃣  时间数据:")
            if 'Geolocation/Scnlin_daycnt' in f:
                print(f"   ✓ 找到: Geolocation/Scnlin_daycnt")
                print(f"   ✓ 找到: Geolocation/Scnlin_mscnt")
            elif 'Obs_Time' in f:
                print(f"   ✓ 找到: Obs_Time")
            elif 'Time' in f:
                print(f"   ✓ 找到: Time")
            else:
                print("   ⚠ 未找到时间数据（将从文件名提取）")
            print()
            
            # 5. 检查质量标记
            print("5️⃣  质量控制数据:")
            if 'QA/Quality_Flag_Scnlin' in f:
                print(f"   ✓ 找到: QA/Quality_Flag_Scnlin")
            elif 'Quality_Flag' in f:
                print(f"   ✓ 找到: Quality_Flag")
            else:
                print("   ⚠ 未找到质量标记（将创建默认值）")
            print()
            
            # 6. 总结
            print("="*70)
            print("测试总结")
            print("="*70)
            
            if bt_found and lat_found and lon_found:
                print("✅ 所有必需数据都已找到，文件可以正常处理！")
                print("\n建议使用以下命令进行配准:")
                print(f"python collocation_fy3d_era5_fixed.py {filepath} <ERA5_file>")
            else:
                print("❌ 缺少必需数据:")
                if not bt_found:
                    print("   - 亮温数据")
                if not lat_found:
                    print("   - 纬度数据")
                if not lon_found:
                    print("   - 经度数据")
            
            return bt_found and lat_found and lon_found
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python test_fy3d_read.py <FY3D_file>")
        print("示例: python test_fy3d_read.py /data2/lrx/obs/FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF")
        sys.exit(1)
    
    filepath = sys.argv[1]
    success = test_fy3d_file(filepath)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

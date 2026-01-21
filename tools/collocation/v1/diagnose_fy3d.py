#!/usr/bin/env python3
"""
FY-3D L1C 文件格式诊断工具
"""

import os
import sys
import h5py
import struct

def check_file_exists(filepath):
    """检查文件是否存在"""
    print(f"\n{'='*70}")
    print("1. 检查文件是否存在")
    print('='*70)
    
    if os.path.exists(filepath):
        print(f"✓ 文件存在: {filepath}")
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"✓ 文件大小: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ 文件不存在: {filepath}")
        return False

def check_file_permissions(filepath):
    """检查文件权限"""
    print(f"\n{'='*70}")
    print("2. 检查文件权限")
    print('='*70)
    
    if os.access(filepath, os.R_OK):
        print(f"✓ 文件可读")
        return True
    else:
        print(f"✗ 文件不可读，请检查权限")
        return False

def check_hdf5_signature(filepath):
    """检查是否为有效的HDF5文件"""
    print(f"\n{'='*70}")
    print("3. 检查HDF5文件签名")
    print('='*70)
    
    try:
        with open(filepath, 'rb') as f:
            # HDF5 文件的前8个字节应该是: \x89HDF\r\n\x1a\n
            header = f.read(8)
            
            # 显示前16字节的十六进制
            f.seek(0)
            first_bytes = f.read(16)
            print(f"文件前16字节 (hex): {first_bytes.hex()}")
            print(f"文件前16字节 (repr): {repr(first_bytes)}")
            
            # 检查HDF5签名
            hdf5_signature = b'\x89HDF\r\n\x1a\n'
            if header == hdf5_signature:
                print(f"✓ 这是标准HDF5文件")
                return True
            else:
                print(f"✗ 不是标准HDF5文件")
                print(f"  期望签名: {hdf5_signature.hex()}")
                print(f"  实际签名: {header.hex()}")
                
                # 检查是否是HDF4文件
                if header[:4] == b'\x0e\x03\x13\x01':
                    print(f"⚠ 这可能是HDF4格式文件，不是HDF5！")
                    return False
                
                return False
    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        return False

def try_h5py_open(filepath):
    """尝试用h5py打开文件"""
    print(f"\n{'='*70}")
    print("4. 尝试用h5py打开文件")
    print('='*70)
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"✓ h5py成功打开文件")
            print(f"\n文件中的数据集和组:")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"  Group: {name}")
            
            f.visititems(print_structure)
            return True
    except Exception as e:
        print(f"✗ h5py无法打开文件")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {e}")
        return False

def check_pyhdf():
    """检查是否安装了pyhdf库（用于读取HDF4）"""
    print(f"\n{'='*70}")
    print("5. 检查HDF4读取支持")
    print('='*70)
    
    try:
        from pyhdf.SD import SD, SDC
        print("✓ pyhdf已安装（可以读取HDF4文件）")
        return True
    except ImportError:
        print("✗ pyhdf未安装")
        print("  如果文件是HDF4格式，需要安装: pip install python-hdf4")
        return False

def try_read_as_hdf4(filepath):
    """尝试作为HDF4文件读取"""
    print(f"\n{'='*70}")
    print("6. 尝试作为HDF4文件读取")
    print('='*70)
    
    try:
        from pyhdf.SD import SD, SDC
        
        hdf = SD(filepath, SDC.READ)
        datasets = hdf.datasets()
        
        print(f"✓ 成功作为HDF4文件打开！")
        print(f"\n文件中包含 {len(datasets)} 个数据集:")
        
        for name, info in list(datasets.items())[:10]:  # 只显示前10个
            print(f"  - {name}: shape={info[1]}, type={info[3]}")
        
        if len(datasets) > 10:
            print(f"  ... (共{len(datasets)}个数据集)")
        
        hdf.end()
        return True
        
    except ImportError:
        print("⚠ 无法导入pyhdf库")
        return False
    except Exception as e:
        print(f"✗ 无法作为HDF4文件读取")
        print(f"  错误: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python diagnose_fy3d.py <FY3D_L1C_file>")
        print("示例: python diagnose_fy3d.py /path/to/FY3D_MWTSX_*.L1c")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print("\n" + "="*70)
    print("FY-3D L1C 文件格式诊断工具")
    print("="*70)
    print(f"目标文件: {filepath}\n")
    
    results = {
        'exists': check_file_exists(filepath),
        'readable': False,
        'hdf5': False,
        'h5py_works': False,
        'hdf4': False
    }
    
    if results['exists']:
        results['readable'] = check_file_permissions(filepath)
        
        if results['readable']:
            results['hdf5'] = check_hdf5_signature(filepath)
            results['h5py_works'] = try_h5py_open(filepath)
            
            if not results['h5py_works']:
                # 如果h5py失败，尝试HDF4
                check_pyhdf()
                results['hdf4'] = try_read_as_hdf4(filepath)
    
    # 总结和建议
    print(f"\n{'='*70}")
    print("诊断总结和建议")
    print('='*70)
    
    if results['h5py_works']:
        print("\n✓ 文件可以用h5py正常读取，无需修改代码")
    elif results['hdf4']:
        print("\n⚠ 文件是HDF4格式，需要修改读取代码！")
        print("\n建议解决方案:")
        print("1. 安装pyhdf库:")
        print("   pip install python-hdf4")
        print("\n2. 使用pyhdf读取文件，而不是h5py")
        print("   我可以为您修改collocation_fy3d_era5.py脚本")
    elif not results['exists']:
        print("\n✗ 文件不存在，请检查路径")
    elif not results['readable']:
        print("\n✗ 文件权限问题，请运行:")
        print(f"   chmod +r {filepath}")
    else:
        print("\n✗ 无法识别的文件格式")
        print("   请确认这是正确的FY-3D L1C文件")

if __name__ == '__main__':
    main()

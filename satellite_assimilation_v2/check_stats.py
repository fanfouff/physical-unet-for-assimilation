#!/usr/bin/env python3
"""
检查stats.npz文件的内容
"""

import numpy as np
import sys

if len(sys.argv) < 2:
    print("用法: python check_stats.py /path/to/stats.npz")
    sys.exit(1)

stats_file = sys.argv[1]

print(f"检查文件: {stats_file}")
print("="*60)

try:
    stats = np.load(stats_file)
    
    print(f"文件类型: {type(stats)}")
    print(f"\n包含的键:")
    print("-"*60)
    
    for key in stats.files:
        value = stats[key]
        print(f"  {key}:")
        print(f"    形状: {value.shape}")
        print(f"    类型: {value.dtype}")
        if value.size < 10:
            print(f"    值: {value}")
        else:
            print(f"    范围: [{value.min():.4f}, {value.max():.4f}]")
        print()
    
    print("="*60)
    print("检查完成!")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

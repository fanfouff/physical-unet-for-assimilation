import h5py

# 打开HDF文件
file_path = '/data2/lrx/obs/FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF'
with h5py.File(file_path, 'r') as f:
    # 列出所有顶层变量/组
    print("顶层变量/组:")
    for key in f.keys():
        print(f"  {key}")
    
    # 递归查看所有变量
    def print_structure(name, obj):
        print(name)
    
    print("\n所有变量:")
    f.visititems(print_structure)

# -*- coding: utf-8 -*-
"""本机 -> Dell Tower 关键文件同步脚本（在中转目录 wb_transfer 内运行）。

用法（在 Dell Tower 上）：python C:/Users/ADMIN/wb_transfer/_wb_copy_to_dell.py
把中转目录里的文件复制到同步盘目标位置（覆盖）。
"""
import shutil
import os

SRC_DIR = 'C:/Users/ADMIN/wb_transfer'
DST_ROOT = 'E:/华为家庭存储/BUAA_research/BUAA2021/PSSE_Simulation'

files = [
    ('psse_dynamic.py',   'package/psse_dynamic.py'),
    ('identification.py', 'package/identification.py'),
    ('研究结论.md',        '研究结论.md'),
]

for name, rel in files:
    src = os.path.join(SRC_DIR, name)
    dst = os.path.join(DST_ROOT, rel)
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        print('OK   ', rel)
    except Exception as e:
        print('FAIL ', rel, '->', repr(e))

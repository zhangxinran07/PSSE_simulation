# -*- coding: utf-8 -*-
"""Dell Tower PSSE 环境探测（临时脚本，验证完可删）。

目的：在 Dell Tower 上验证 PSSE36/36.3 + Python3.13 能否 import psspy 并 psseinit 成功。
与主链路 package/psse_dynamic.py 的区别：
  - 路径用 Dell Tower 的 C 盘（本机 D 盘）
  - PSSE 版本 36.3（本机 36.2）
  - 用 os.add_dll_directory 加载 PSSBIN 的依赖 DLL（绕开 cmd 传反斜杠被转正斜杠的问题）
"""
import os
import sys

PSSBIN = r'C:\Program Files\PTI\PSSE36\36.3\PSSBIN'
PSSPY = r'C:\Program Files\PTI\PSSE36\36.3\PSSPY313'

os.environ['PATH'] = PSSBIN + ';' + os.environ['PATH']
try:
    os.add_dll_directory(PSSBIN)
    print('add_dll_directory OK:', PSSBIN)
except Exception as e:
    print('add_dll_directory failed:', e)

sys.path.append(PSSPY)

print('== importing psspy ... ==')
import psspy  # noqa: E402
print('== psspy import OK ==')

print('== psseinit(20000) ... ==')
ierr = psspy.psseinit(20000)
print('== psseinit(20000) =', ierr, '(0=成功) ==')

if ierr == 0:
    psspy.pssehalt_2()
    print('== pssehalt_2 done ==')
else:
    print('== psseinit 失败，ierr=', ierr, ' ==')

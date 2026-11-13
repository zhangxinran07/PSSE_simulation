# -*- coding: utf-8 -*-
"""6~13Hz multi-sine 探测激励的 PSSE 闭环验证。

目的(2026-11-06 张老师"改,做"):
  1. 验证注入 6~13Hz 有功扰动后, 母线电压 u 是否出现同频波动(激励能否"打得进去");
  2. 在给定 MW 幅值下, 实测电压 |u| 的波动幅值(max-min 与 6~13Hz 带通幅度)是否满足
     小干扰约束(<= 0.08), 从而标定"多少 MW -> 多少 |u| 波动"。

用法:
  python probing_closedloop.py [MW]     # 默认 300 MW, 可命令行覆盖

输出:
  - 各监视母线的 |u| max-min(总) 与 6~13Hz 带通波动幅值;
  - 电压 6~13Hz 频段能量占比(检验后处理 filter_cutoff=30 是否保留高频)。
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.append(os.path.join(ROOT, 'package'))

import psse_dynamic as ps_d
from scipy.signal import butter, filtfilt

BUSES = [3, 4, 15, 20, 21, 24, 25, 27, 28]
F_LO, F_HI = 6.0, 13.0
FS = 100.0  # 后处理 sample_time_step=0.01


def band_amplitude(ud, uq, fs, f_lo, f_hi):
    """对复数电压 u=ud+j*uq 做 [f_lo,f_hi] 带通, 返回该频段电压幅值波动的峰值(max|u_band|)。"""
    nyq = fs / 2
    b, a = butter(5, [f_lo / nyq, f_hi / nyq], btype='band')
    ud_b = filtfilt(b, a, ud)
    uq_b = filtfilt(b, a, uq)
    u_band = np.abs(ud_b + 1j * uq_b)
    return float(u_band.max()), float(u_band.std())


def main():
    mag = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    print('=== 6~13Hz probing 闭环验证, 幅值 = %.0f MW ===' % mag)

    kwargs = dict(
        disturbance_type='probing', disturbance_range=mag,
        study='IEEE39_RE', total_time=30,
        load_bus_monitor=BUSES,
        ambient_seed=1,
        probing_f_lo=F_LO, probing_f_hi=F_HI, probing_n_comp=8,
        filter_cutoff_frequency=30,
    )
    df, d_para, t = ps_d.one_time_simulation(kwargs, max_retries=5)
    if df is None:
        print('仿真失败')
        return

    cols = list(df.columns)
    print('\n母线 | 总|u|max-min | 6~13Hz带通峰值 | 6~13Hz带通std | 电压6~13Hz能量占比')
    for bus in BUSES:
        ud = df.values[:, [i for i in range(len(cols)) if ('BUS%d UD' % bus) in cols[i]]].reshape(len(df))
        uq = df.values[:, [i for i in range(len(cols)) if ('BUS%d UQ' % bus) in cols[i]]].reshape(len(df))
        u = np.abs(ud + 1j * uq)
        total_range = float(u.max() - u.min())
        bp_peak, bp_std = band_amplitude(ud, uq, FS, F_LO, F_HI)
        # 电压幅值信号的频段能量占比
        from numpy.fft import rfft, rfftfreq
        U = np.abs(rfft(u - u.mean()))
        f = rfftfreq(len(u), 1 / FS)
        band = (f >= F_LO) & (f <= F_HI)
        frac = 100 * (U[band] ** 2).sum() / (U ** 2).sum()
        flag = 'OK(<=0.08)' if total_range <= 0.08 else '超限'
        print(' %2d |   %.5f   |   %.5f    |   %.5f   |   %.2f%%   %s'
              % (bus, total_range, bp_peak, bp_std, frac, flag))

    print('\n[结论] 若带通峰值>0 且能量占比>1%%, 说明 6~13Hz 激励确实打进电压;')
    print('       再看总|u|max-min 是否 <=0.08 决定 MW 需上调/下调。')


if __name__ == '__main__':
    main()

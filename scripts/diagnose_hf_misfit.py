# -*- coding: utf-8 -*-
"""根因诊断: probing(6~13Hz) 高频激励下 mix 模型为何失配。

背景(2026-11-06): 闭环对照冒烟测试发现, probing 激励下 mix 8 参数辨识误差远大于
ambient(fun=8.5e-2 vs 2.2e-3), 与"6~13Hz 频域诊断最优"的理论预测相反。

本脚本区分两种根因:
  A. 模型失配/数值积分问题: 在真值参数下, obj_mix_iden(truth) 的代价就很大
     (即三阶电机递推核在 dt=0.01 下无法复现 PSSE dt=0.002 的 6~13Hz 响应);
  B. 辨识器未收敛: 真值代价很小, 但 trust-constr 没找到全局最优。

诊断口径:
  对同一组真值, 分别跑 ambient 与 probing 仿真, 在真值参数处计算
  obj_mix_iden(truth, u, P, Q) 与归一化残差 RMS, 对比两种激励的真值代价。
  - 若 probing 真值代价 >> ambient 真值代价 => 根因 A(模型/数值失配);
  - 若两者接近且都小 => 根因 B(辨识器收敛问题)。

用法: python diagnose_hf_misfit.py [g]    # g=组号(默认 0, 从 mix_ambient_vs_probing.pkl 读真值)
"""
import os, sys, pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, _HERE)
sys.path.append(os.path.join(ROOT, 'package'))

import mix_ambient_vs_probing as M      # 复用采样/仿真口径(模块级 chdir/env 设置无害)
from identification import obj_mix_iden


def resid_rms(truth, u, P, Q, dt):
    """在真值处算逐点残差 RMS(百分比, 相对 P 均值), 与 obj_mix_iden 同口径(头尾裁 k)。"""
    from identification import _core_con, _core_var, _trim_k
    a_a, b_a, H2_a, mva_a = truth[0]/0.75, truth[1], truth[2]/0.75, P[0]*truth[3]
    a_b, b_b, H2_b, mva_b = truth[4]/0.75, truth[5], truth[6]/0.75, P[0]*truth[7]
    ok_a, P_pa, Q_pa = _core_con(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb = _core_var(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return None, None, False
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n-k)
    rP = (P_pa + P_pb - P)[sl]
    rQ = (Q_pa + Q_pb - Q)[sl]
    rms = float(np.sqrt(np.mean(rP**2 + rQ**2)))
    rms_norm = rms / abs(P[sl].mean()) * 100
    return rms, rms_norm, True


def main():
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    res = pickle.load(open(M.PKL, 'rb'))
    rec = res[g]
    truth = np.array(rec['truth'])
    cc = rec['cc']
    print('=' * 80)
    print('根因诊断 g=%d: 真值 truth = %s' % (g, np.round(truth, 3)))
    print('=' * 80)

    for exc in ('ambient', 'probing'):
        print('\n[%s] 跑仿真...' % exc, flush=True)
        u, P, Q, t = M.run_sim(cc, exc, M.SEED + g)
        if u is None:
            print('  仿真失败')
            continue
        cost_truth = obj_mix_iden(truth, u, P, Q, M.DT)
        rms, rms_norm, ok = resid_rms(truth, u, P, Q, M.DT)
        print('  |u| max-min = %.4f   |u| std = %.4f' % (abs(u).max()-abs(u).min(), abs(u).std()))
        print('  真值代价 obj_mix_iden(truth) = %.4e' % cost_truth)
        print('  残差 RMS = %.5f (归一化 %.3f%%)  ok=%s' % (rms, rms_norm, ok))
        # 频段能量占比(看激励是否真打进去)
        from numpy.fft import rfft, rfftfreq
        U = np.abs(rfft(u - u.mean()))
        f = rfftfreq(len(u), 1/100.0)
        for lo, hi, label in [(0, 0.5, '<0.5Hz'), (0.5, 6, '0.5-6Hz'), (6, 13, '6-13Hz'), (13, 50, '>13Hz')]:
            band = (f >= lo) & (f < hi)
            print('    %s 电压能量占比 = %.2f%%' % (label, 100*(U[band]**2).sum()/(U**2).sum()))

    print('\n[判断] probing 真值代价 若 >> ambient 真值代价 => 模型/数值失配(根因A);')
    print('       若两者接近且都小 => 辨识器未收敛(根因B)。')


if __name__ == '__main__':
    main()

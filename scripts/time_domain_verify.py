# -*- coding: utf-8 -*-
"""
时域闭环验证：不同截止频率激励下，A/B 电机同名参数的「混淆夹角」与真实辨识混淆率。

背景：
  频域灵敏度谱已发现: H2 全频段共线(不可解耦), a/b 低频段正交(可解耦)、高频共线。
  本脚本用「真实时域辨识」验证这一结论是否成立:
    (1) 时域灵敏度夹角 collin(θ_a, θ_b): 归一化内积, ±1=共线(混淆), 0=正交(可分离)
    (2) 真实辨识混淆率: 从随机初值跑 obj_mix_iden, 统计 A 参数被辨到 B 的错配

激励合成:
  低通白噪声(5阶 butterworth, 同 generate_ambient_signals), 电压 u = 1 + A*n(t),
  幅值 A 控制在 max-min ≈ 0.06 p.u.(满足张老师 ≤0.08 约束)。
  截止频率 f_c 扫描 [0.1, 0.2, 0.5, 1, 2, 5] Hz。
  观测: 仅母线总功率 P = P_A + P_B(用辨识核合成, 模拟实测)。
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import _core_con, _core_var, _core_con_jac, _core_var_jac, obj_mix_iden, identify

DT = 0.01
TOTAL_T = 30.0
N = int(TOTAL_T / DT) + 1
TRUTH = np.array([303.2, 72.7, 0.142, 0.337, 270.5, 63.2, 0.174, 0.265])
NAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
P0 = 5.9956


def make_excitation(fc, A=0.03, seed=42, length=None, fs=100.0):
    """生成 u(t) = 1 + A*n(t), n 为截止频率 fc 的低通白噪声(过零起点)。"""
    n = length if length is not None else N
    np.random.seed(seed)
    nyq = fs / 2
    b, a = butter(5, fc / nyq, btype='low', analog=False)
    margin = 8 * n
    y = np.random.uniform(-1, 1, n + margin)
    yf = filtfilt(b, a, y)
    # 过零起点
    for j in range(margin):
        if yf[j] * yf[j + 1] <= 0:
            k = j if abs(yf[j]) <= abs(yf[j + 1]) else j + 1
            sig = yf[k:k + n]
            break
    sig = sig / np.max(np.abs(sig))  # 归一化到 [-1,1]
    u = 1.0 + A * sig
    return u + 0j


def synthesize_pq(u):
    """真值处合成总功率 P/Q(模拟实测总功率)。"""
    mva_a = P0 * TRUTH[3]; mva_b = P0 * TRUTH[7]
    ok_a, P_pa, Q_pa = _core_con(u, TRUTH[0], TRUTH[1], TRUTH[2], mva_a, DT)
    ok_b, P_pb, Q_pb = _core_var(u, TRUTH[4], TRUTH[5], TRUTH[6], mva_b, DT)
    assert ok_a and ok_b, '核发散'
    return P_pa + P_pb, Q_pa + Q_pb


def collin_between(u, P, i, j):
    """时域灵敏度向量 ∂P/∂θ_i 与 ∂P/∂θ_j 的归一化内积(混淆夹角余弦)。"""
    mva_a = P0 * TRUTH[3]; mva_b = P0 * TRUTH[7]
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, TRUTH[0], TRUTH[1], TRUTH[2], mva_a, DT)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, TRUTH[4], TRUTH[5], TRUTH[6], mva_b, DT)
    n = len(u)
    # 8 参数对 P 的逐点导数(物理口径)
    dP = np.zeros((n, 8)); dQ = np.zeros((n, 8))
    dP[:, 0:3] = dPa; dP[:, 3] = P0 * (P_pa / mva_a)
    dP[:, 4:7] = dPb; dP[:, 7] = P0 * (P_pb / mva_b)
    dQ[:, 0:3] = dQa; dQ[:, 3] = P0 * (Q_pa / mva_a)
    dQ[:, 4:7] = dQb; dQ[:, 7] = P0 * (Q_pb / mva_b)
    # 拼接 P/Q 两通道
    gi = np.concatenate([dP[:, i], dQ[:, i]]) * TRUTH[i]
    gj = np.concatenate([dP[:, j], dQ[:, j]]) * TRUTH[j]
    return float(np.dot(gi, gj) / (np.linalg.norm(gi) * np.linalg.norm(gj)))


def identify_once(u, P, Q, seed):
    """从随机初值跑一次 8 参数辨识, 返回最优参数。"""
    np.random.seed(seed)
    lb = np.array([50, 10, 0.05, 0.1, 50, 10, 0.05, 0.1])
    ub = np.array([600, 200, 0.5, 0.8, 600, 200, 0.5, 0.8])
    best = None
    for _ in range(3):  # 3 个随机初值取最优
        x0 = np.random.uniform(lb, ub)
        res = identify(u, P, Q, dt=DT, lb=lb, ub=ub, obj_func=obj_mix_iden,
                       tol=1e-10, max_trials=5)
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def main():
    cutoffs = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    print('=== (1) 时域灵敏度夹角 collin(A参数, B同名参数) ===')
    print(f'{"f_c(Hz)":>8} {"collin(a对)":>12} {"collin(b对)":>12} {"collin(H2对)":>12}')
    collin_rows = {}
    for fc in cutoffs:
        u = make_excitation(fc)
        P, Q = synthesize_pq(u)
        c_a = collin_between(u, P, 0, 4)   # a_a vs a_b
        c_b = collin_between(u, P, 1, 5)   # b_a vs b_b
        c_h = collin_between(u, P, 2, 6)   # H2_a vs H2_b
        collin_rows[fc] = (c_a, c_b, c_h)
        print(f'{fc:>8.2f} {c_a:>+12.4f} {c_b:>+12.4f} {c_h:>+12.4f}')

    print('\n=== (2) 真实辨识混淆率(错配 = 辨到对侧参数而非真值) ===')
    print(f'{"f_c(Hz)":>8} {"a错配率":>9} {"b错配率":>9} {"H2错配率":>9} {"a中位误差%":>11} {"b中位误差%":>11} {"H2中位误差%":>11}')
    for fc in cutoffs:
        u = make_excitation(fc)
        P, Q = synthesize_pq(u)
        errs = {k: [] for k in ['a', 'b', 'H2']}
        n_succ = 0
        for seed in range(8):
            try:
                x = identify_once(u, P, Q, seed)
            except Exception:
                continue
            n_succ += 1
            for k, (ia, ib, ta, tb) in enumerate({
                'a': (0, 4, TRUTH[0], TRUTH[4]),
                'b': (1, 5, TRUTH[1], TRUTH[5]),
                'H2': (2, 6, TRUTH[2], TRUTH[6]),
            }.items()):
                # 错配: 辨出的 A 参数更接近 B 真值, 或 B 参数更接近 A 真值
                mis_a = abs(x[ia] - tb) < abs(x[ia] - ta)
                mis_b = abs(x[ib] - ta) < abs(x[ib] - tb)
                errs[k].append((mis_a, mis_b,
                                abs(x[ia] - ta) / ta * 100, abs(x[ib] - tb) / tb * 100))
        if n_succ == 0:
            print(f'{fc:>8.2f}  辨识全失败')
            continue
        out = []
        for k in ['a', 'b', 'H2']:
            e = errs[k]
            mis = sum(1 for x in e if x[0] or x[1]) / len(e)
            med = np.median([x[2] for x in e] + [x[3] for x in e])
            out.append((mis, med))
        print(f'{fc:>8.2f} {out[0][0]:>9.0%} {out[1][0]:>9.0%} {out[2][0]:>9.0%} '
              f'{out[0][1]:>11.1f} {out[1][1]:>11.1f} {out[2][1]:>11.1f}')


if __name__ == '__main__':
    main()

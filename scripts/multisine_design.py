# -*- coding: utf-8 -*-
"""
multi-sine 最优激励设计对比：不同频段分配下的可辨识性结构。

结论基础（excitation_sweep.py）：
  ① H2 对 在任意单频下 corr≈1.0（结构性共线，无法靠频点解耦），
     但 H2 信息量随频率单调上升 → 要降低 H2 方差只能把能量往高频推。
  ② a 对 / b 对 单频 corr 最低也 ≥0.81/0.945，但真实宽带激励(0.29Hz峰值)给 0.635/0.848
     → 说明「多频组合」本身才是压低 a/b 共线的手段（不同频点 A/B 灵敏度幅值比不同）。
本脚本在小干扰约束 |u| 范围 ≤ 0.08 下，对比不同「频段分配」的 multi-sine 激励，
给出最优的频段/能量分配方案，并与真实激励结构对照。
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, _trim_k

DT = 0.01
TRUTH = np.array([303.2, 72.7, 0.142, 0.337, 270.5, 63.2, 0.174, 0.265])
NAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
P0 = 5.9956
PAIRS = [('a', 0, 4), ('b', 1, 5), ('H2', 2, 6), ('Pim', 3, 7)]

U_RANGE_TARGET = 0.08   # |u| max-min 目标（小干扰上限）


def make_multisine(f_lo, f_hi, n_comp, T=40.0, u_range=U_RANGE_TARGET,
                   phase_std_deg=25.0, seed=1):
    """对数均匀分布的 n_comp 个频点构成 multi-sine。

    相位调制: θ(t) = Σ θ_k sin(2π f_k t + φ_k)，各频点等幅，总 std 归一化到 phase_std_deg。
    幅值调制: A(t) = 1 + Σ A_k sin(2π f_k t + ψ_k)，总范围归一化到 u_range。
    返回 u = A·e^{jθ}，保证 |u| max-min ≈ u_range。
    """
    rng = np.random.RandomState(seed)
    freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), n_comp)
    n = int(round(T / DT)) + 1
    t = np.arange(n) * DT
    ph_k = rng.uniform(0, 2 * np.pi, n_comp)
    ps_k = rng.uniform(0, 2 * np.pi, n_comp)
    th = np.zeros(n); amp = np.zeros(n)
    for f, pk, sk in zip(freqs, ph_k, ps_k):
        th += np.sin(2 * np.pi * f * t + pk)
        amp += np.sin(2 * np.pi * f * t + sk)
    # 归一化相位到目标 std
    th = th / th.std() * np.deg2rad(phase_std_deg)
    # 归一化幅值到目标 |u| 范围
    amp = amp / (amp.max() - amp.min()) * u_range
    u = (1.0 + amp) * np.exp(1j * th)
    return u, t, freqs


def build_jac(u):
    mva_a = P0 * TRUTH[3]; mva_b = P0 * TRUTH[7]
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, TRUTH[0], TRUTH[1], TRUTH[2], mva_a, DT)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, TRUTH[4], TRUTH[5], TRUTH[6], mva_b, DT)
    if not (ok_a and ok_b):
        return None
    n = len(u)
    dP = np.empty((n, 8)); dQ = np.empty((n, 8))
    dP[:, 0:3] = dPa; dP[:, 3] = P0 * (P_pa / mva_a)
    dP[:, 4:7] = dPb; dP[:, 7] = P0 * (P_pb / mva_b)
    dQ[:, 0:3] = dQa; dQ[:, 3] = P0 * (Q_pa / mva_a)
    dQ[:, 4:7] = dQb; dQ[:, 7] = P0 * (Q_pb / mva_b)
    k = _trim_k(n); sl = slice(k, n - k)
    return dP[sl, :], dQ[sl, :], np.abs(u[sl])


def project(J, V):
    m = len(V); X = np.column_stack((np.ones(m), V, V ** 2))
    Qr = np.linalg.solve(X.T @ X, X.T)
    return J - X @ (Qr @ J)


def evaluate(u):
    r = build_jac(u)
    if r is None:
        return None
    JP, JQ, V = r
    Jall = np.vstack([project(JP, V) * TRUTH[None, :], project(JQ, V) * TRUTH[None, :]])
    F = Jall.T @ Jall
    ev = np.linalg.eigvalsh(F)
    out = dict(cond8=float(ev[-1] / ev[0]) if ev[0] > 1e-30 else np.inf)
    for nm, i, j in PAIRS:
        gi = Jall[:, i]; gj = Jall[:, j]
        ni = np.linalg.norm(gi); nj = np.linalg.norm(gj)
        out['corr_' + nm] = float(gi @ gj / (ni * nj))
        G = np.array([[ni * ni, gi @ gj], [gi @ gj, nj * nj]])
        e2 = np.linalg.eigvalsh(G)
        out['cond_' + nm] = float(e2[1] / e2[0]) if e2[0] > 1e-30 else np.inf
    out['info_H2'] = float(F[2, 2] + F[6, 6])
    out['info_a'] = float(F[0, 0] + F[4, 4])
    out['info_b'] = float(F[1, 1] + F[5, 5])
    return out


def main():
    # 频段分配方案 (f_lo, f_hi, n_comp, 标签)
    plans = [
        (0.02, 0.10, 5, '低频段 0.02-0.10Hz'),
        (0.10, 0.50, 5, '真实频段 0.10-0.50Hz'),
        (0.02, 0.50, 7, '低+真实 0.02-0.50Hz'),
        (0.02, 5.00, 10, '宽带 0.02-5.0Hz'),
        (0.30, 5.00, 7, '高频段 0.30-5.0Hz'),
        (0.02, 5.00, 15, '宽带密集 0.02-5.0Hz(15频点)'),
    ]
    print('===== multi-sine 频段分配对比 (|u|范围=%.2f, 相位std=25°) =====' % U_RANGE_TARGET)
    print()
    hdr = '%-26s | %8s %8s %8s | %9s %9s %9s' % (
        '方案', 'corr_a', 'corr_b', 'corr_H2', 'info_a', 'info_b', 'info_H2')
    print(hdr)
    print('-' * len(hdr))
    results = []
    for f_lo, f_hi, n_comp, tag in plans:
        u, t, freqs = make_multisine(f_lo, f_hi, n_comp)
        assert abs(abs(u).max() - abs(u).min() - U_RANGE_TARGET) < 0.01, '|u|范围失配'
        m = evaluate(u)
        if m is None:
            print('%-26s | 核发散' % tag)
            continue
        results.append((tag, m))
        print('%-26s | %+8.4f %+8.4f %+8.4f | %9.3g %9.3g %9.3g'
              % (tag, m['corr_a'], m['corr_b'], m['corr_H2'],
                 m['info_a'], m['info_b'], m['info_H2']))

    print()
    print('===== 与真实激励(30组中位数)对照 =====')
    print('真实数据:  corr_a=%+.4f  corr_b=%+.4f  corr_H2=%+.4f  cond8=%.3g'
          % (0.6350, 0.8481, 0.9856, 4.72e4))

    print()
    print('===== 各方案 8x8 条件数 =====')
    for tag, m in results:
        print('%-26s : cond8 = %.3g' % (tag, m['cond8']))


if __name__ == '__main__':
    main()

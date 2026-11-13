# -*- coding: utf-8 -*-
"""
激励信号可设计性诊断（决定性）：相位主导复数激励下的频域扫描。

背景（承接 real_data_collinearity.py 的关键发现）：
  真实 PSSE 的电压激励是「相位主导」——相角摆幅 std≈31°、|u| 摆幅仅 1.3%，
  相角波动能量是幅值波动的 ~727 倍。此前 freq_sensitivity*.py 全用「幅值调制」
  激励 u=1+eps·sin(2πft)，激励通道与真实不符，故其「a/b 低频正交、H2 恒共线」
  的结论不可直接套用。

本脚本回答「激励设计」的核心问题：在小干扰约束 max-min(|u|) ≤ 0.08 下，
  若把激励能量集中到单一频率 f（相位主导复数激励），A/B 同名参数的
  ① 共线度 corr 与 ② 每参数 Fisher 信息量 如何随 f 变化？
  据此判断：a 对 / b 对能否靠换频点解耦，H2 对的信息量峰值在哪个频段，
  以及是否需要 multi-sine（多个频点各自负责不同参数）。

激励形态（与真实数据结构对齐）：
  u(t) = (1 + epsA·sin(2πft)) · exp(j·epsTh·sin(2πft))
  epsA = 0.03 → |u| max-min = 0.06 ≤ 0.08（满足小干扰）
  epsTh= 0.40 → 相角摆幅 ±0.4 rad ≈ ±23°（与真实 std≈31° 同量级）
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

EPS_A = 0.03     # 幅值调制单边幅值 -> |u| 范围 0.06 <= 0.08
EPS_TH = 0.40    # 相位调制单边幅值(rad)


def make_excitation(f, n_cycle_min=4, T_min=12.0):
    """单频相位主导复数激励。仿真长度取 max(T_min, n_cycle_min/f) 保证低频也有足够周期。"""
    T_sim = max(T_min, n_cycle_min / f)
    n = int(round(T_sim / DT)) + 1
    t = np.arange(n) * DT
    amp = 1.0 + EPS_A * np.sin(2 * np.pi * f * t)
    ang = EPS_TH * np.sin(2 * np.pi * f * t)
    u = amp * np.exp(1j * ang)
    return u, t


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


def main():
    freqs = np.logspace(np.log10(0.02), np.log10(5.0), 44)
    rows = []
    for f in freqs:
        u, t = make_excitation(f)
        r = build_jac(u)
        if r is None:
            rows.append(dict(f=f, ok=False))
            continue
        JP, JQ, V = r
        JPe = project(JP, V) * TRUTH[None, :]
        JQe = project(JQ, V) * TRUTH[None, :]
        Jall = np.vstack([JPe, JQe])
        F = Jall.T @ Jall
        ev = np.linalg.eigvalsh(F)
        row = dict(f=f, ok=True, cond8=float(ev[-1] / ev[0]) if ev[0] > 1e-30 else np.inf)
        # 每参数 Fisher 信息(对角, 相对尺度)
        for j, name in enumerate(NAMES):
            row['info_' + name] = float(F[j, j])
        # 各参数对的共线度 corr 与 2x2 条件数
        for nm, i, j in PAIRS:
            gi = Jall[:, i]; gj = Jall[:, j]
            ni = np.linalg.norm(gi); nj = np.linalg.norm(gj)
            if ni < 1e-30 or nj < 1e-30:
                row['corr_' + nm] = np.nan; row['cond_' + nm] = np.inf
            else:
                d = float(gi @ gj)
                row['corr_' + nm] = d / (ni * nj)
                G = np.array([[ni * ni, d], [d, nj * nj]])
                e2 = np.linalg.eigvalsh(G)
                row['cond_' + nm] = float(e2[1] / e2[0]) if e2[0] > 1e-30 else np.inf
        rows.append(row)

    df = pd.DataFrame(rows)
    ok = df[df.ok].copy()

    print('===== 相位主导复数激励 频域扫描 (epsA=%.2f, epsTh=%.2f, |u|范围=%.2f) ====='
          % (EPS_A, EPS_TH, 2 * EPS_A))
    print()
    print('  频率逐点：各参数对 corr / 2x2条件数 / 每参数信息量')
    hdr = '%8s | %8s %8s %8s | %10s %10s %10s | %9s %9s' % (
        'f(Hz)', 'corr_a', 'corr_b', 'corr_H2', 'cond_a', 'cond_b', 'cond_H2',
        'info_H2a', 'info_H2b')
    print(hdr)
    print('-' * len(hdr))
    for r in ok.itertuples():
        print('%8.3f | %+8.4f %+8.4f %+8.4f | %10.3g %10.3g %10.3g | %9.3g %9.3g'
              % (r.f, r.corr_a, r.corr_b, r.corr_H2, r.cond_a, r.cond_b, r.cond_H2,
                 r.info_H2_a, r.info_H2_b))

    print()
    print('===== 关键结论提取 =====')
    # 1) 各参数对 corr 最低的频点(最可解耦)
    for nm in ['a', 'b', 'H2']:
        c = ok['corr_' + nm]
        i = int(np.nanargmin(c.values))
        print('%-4s 对: corr 最低 = %+.4f @ f=%.3fHz ;  corr 最高 = %+.4f @ f=%.3fHz'
              % (nm, c.values[i], ok.f.values[i], np.nanmax(c.values),
                 ok.f.values[int(np.nanargmax(c.values))]))
    # 2) H2 信息量峰值频段
    ih = ok['info_H2_a'] + ok['info_H2_b']
    i = int(np.argmax(ih.values))
    print('H2 对总信息量峰值 @ f=%.3fHz (info_H2a=%.3g, info_H2b=%.3g)'
          % (ok.f.values[i], ok.info_H2_a.values[i], ok.info_H2_b.values[i]))
    # 3) a/b 信息量峰值频段
    ia = ok['info_a_a'] + ok['info_a_b']
    i = int(np.argmax(ia.values))
    print('a  对总信息量峰值 @ f=%.3fHz' % ok.f.values[i])
    ib = ok['info_b_a'] + ok['info_b_b']
    i = int(np.argmax(ib.values))
    print('b  对总信息量峰值 @ f=%.3fHz' % ok.f.values[i])
    # 4) 8x8 条件数最低频点(整体最优)
    i = int(np.nanargmin(ok.cond8.values))
    print('整体 8x8 条件数最低 = %.3g @ f=%.3fHz' % (ok.cond8.values[i], ok.f.values[i]))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'Sample_Generation', 'results', 'excitation_sweep_phase.csv')
    df.to_csv(out, index=False)
    print('\n已保存: %s' % out)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
同名参数对 (A/B) 的「可分离性」定量分析 —— 回答核心问题:
  「降低 corr(方向解耦)」与「提升信息量」到底哪个更能降低 A/B 混淆?

关键指标 (比 corr 更完整):
  对同名参数对 (i,j) 的 2x2 子 Gram 矩阵 G = [[‖g_i‖², g_i·g_j],[g_i·g_j, ‖g_j‖²]],
  特征值 λ_max ≥ λ_min ≥ 0:
    * corr = g_i·g_j/(‖g_i‖‖g_j‖) 只反映「方向」(λ_min/λ_max 的函数), 不含尺度。
    * λ_min = 区分 g_i 与 g_j 最难方向上的 Fisher 信息量 → 辨识该「差方向」的方差 ≈ 1/λ_min。
    * 因此 λ_min 才是「混淆」的直接后果度量: λ_min 越大 → a_a 与 a_b 越可分 → 混淆越小。

本脚本对比各激励方案下 a/b/H2 三对的 λ_min(对数) 与 λ_min/λ_max(方向可分离度),
并做「加权 A-最优」(给 a/b 高权重) 看能量是否向低频迁移, 从而判断多目标折中是否可行。
"""
import sys, os, time
import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, _trim_k

DT = 0.01
TRUTH = np.array([303.2, 72.7, 0.142, 0.337, 270.5, 63.2, 0.174, 0.265])
NAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
P0 = 5.9956
PAIRS = [('a', 0, 4), ('b', 1, 5), ('H2', 2, 6), ('Pim', 3, 7)]

PHASE_STD_DEG = 25.0
U_RANGE = 0.08
T_SIM = 40.0
LAM = 1e-8


def excitation_from_weights(A, freqs, seed=0):
    A = np.abs(np.asarray(A, float))
    s = A.sum(); A = A / s if s > 1e-12 else np.ones_like(A) / len(A)
    rng = np.random.RandomState(seed)
    n = int(round(T_SIM / DT)) + 1
    t = np.arange(n) * DT
    ph = rng.uniform(0, 2 * np.pi, len(freqs))
    ps = rng.uniform(0, 2 * np.pi, len(freqs))
    th = np.zeros(n); amp = np.zeros(n)
    for Ak, f, pk, sk in zip(A, freqs, ph, ps):
        th += Ak * np.sin(2 * np.pi * f * t + pk)
        amp += np.sin(2 * np.pi * f * t + sk)
    th = th / th.std() * np.deg2rad(PHASE_STD_DEG)
    amp = amp / (amp.max() - amp.min()) * U_RANGE
    return (1.0 + amp) * np.exp(1j * th), t


def jall_of(u):
    """投影后相对灵敏度 Jall (2n x 8)。"""
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
    V = np.abs(u[sl])
    X = np.column_stack((np.ones(len(V)), V, V ** 2))
    Qr = np.linalg.solve(X.T @ X, X.T)
    JP = dP[sl, :] - X @ (Qr @ dP[sl, :])
    JQ = dQ[sl, :] - X @ (Qr @ dQ[sl, :])
    return np.vstack([JP * TRUTH[None, :], JQ * TRUTH[None, :]])


def analyze(u):
    Jall = jall_of(u)
    if Jall is None:
        return None
    F = Jall.T @ Jall
    ev = np.linalg.eigvalsh(F)
    out = dict(cond8=float(ev[-1] / ev[0]) if ev[0] > 1e-30 else np.inf)
    Finv = np.linalg.inv(F + LAM * np.eye(8))
    out['var_ab'] = float(Finv[0, 0] + Finv[1, 1] + Finv[4, 4] + Finv[5, 5])
    out['var_H2'] = float(Finv[2, 2] + Finv[6, 6])
    for nm, i, j in PAIRS:
        gi = Jall[:, i]; gj = Jall[:, j]
        ni = float(gi @ gi); nj = float(gj @ gj); d = float(gi @ gj)
        G = np.array([[ni, d], [d, nj]])
        e = np.linalg.eigvalsh(G)
        out['corr_' + nm] = d / np.sqrt(ni * nj) if ni > 1e-30 and nj > 1e-30 else np.nan
        out['lmin_' + nm] = float(e[0])          # 最难区分方向的信息量
        out['sep_' + nm] = float(e[0] / e[1]) if e[1] > 1e-30 else 0.0   # 方向可分离度
    return out


def cost_weighted(A, freqs, w):
    """加权 A-最优: trace(W·diag(F⁻¹))。"""
    u, _ = excitation_from_weights(A, freqs)
    Jall = jall_of(u)
    if Jall is None:
        return 1e30
    F = Jall.T @ Jall
    try:
        Finv = np.linalg.inv(F + LAM * np.eye(8))
    except np.linalg.LinAlgError:
        return 1e30
    return float(np.sum(w * np.diag(Finv)))


def report(tag, u):
    m = analyze(u)
    if m is None:
        print('%-16s | 核发散' % tag); return None
    print('%-16s | a: corr=%+.3f lmin=%8.2g sep=%5.3f | b: corr=%+.3f lmin=%8.2g sep=%5.3f | '
          'H2: corr=%+.3f lmin=%8.2g | var_ab=%9.3g var_H2=%9.3g cond8=%9.3g'
          % (tag, m['corr_a'], m['lmin_a'], m['sep_a'], m['corr_b'], m['lmin_b'], m['sep_b'],
             m['corr_H2'], m['lmin_H2'], m['var_ab'], m['var_H2'], m['cond8']))
    return m


def main():
    cand = np.logspace(np.log10(0.02), np.log10(5.0), 12)
    print('指标说明: corr=方向共线度(-1..1); lmin=区分两参数最难方向的信息量(越大越好);')
    print('          sep=lmin/lmax 方向可分离度(1=正交, 0=共线); var_ab=a/b 四参数联合辨识方差。')
    print('=' * 118)

    # 基线方案
    for f_lo, f_hi, ncomp, tag in [(0.02, 0.10, 5, '等幅低频'),
                                    (0.10, 0.50, 5, '等幅真实频段'),
                                    (0.02, 5.0, 12, '等幅宽带'),
                                    (0.30, 5.0, 7, '等幅高频')]:
        ff = np.logspace(np.log10(f_lo), np.log10(f_hi), ncomp)
        report(tag, excitation_from_weights(np.ones(ncomp), ff)[0])

    # 等权 A-最优
    print('--- 优化: 等权 A-最优 ---')
    r1 = differential_evolution(lambda A: cost_weighted(A, cand, np.ones(8)),
                                [(0, 1)] * len(cand), args=(), seed=1,
                                popsize=8, maxiter=25, tol=1e-6, polish=True, updating='immediate')
    A1 = np.abs(r1.x); A1 /= A1.sum()
    report('A-最优等权', excitation_from_weights(A1, cand)[0])

    # 加权 A-最优: 给 a/b 四参数高权重(100), H2 与 Pim 低权重(1)
    print('--- 优化: 加权 A-最优 (a/b 权重=100, 其余=1) ---')
    w_ab = np.array([100, 100, 1, 1, 100, 100, 1, 1], float)
    r2 = differential_evolution(lambda A: cost_weighted(A, cand, w_ab),
                                [(0, 1)] * len(cand), args=(), seed=1,
                                popsize=8, maxiter=25, tol=1e-6, polish=True, updating='immediate')
    A2 = np.abs(r2.x); A2 /= A2.sum()
    report('A-最优加权a/b', excitation_from_weights(A2, cand)[0])

    # 打印加权优化的能量分配
    print('\n加权 a/b 最优的能量分配:')
    for f, a in zip(cand, A2):
        if a > 1e-3:
            print('  f=%6.3f Hz  w=%5.3f  %s' % (f, a, '#' * int(round(a * 50))))

    print('\n真实数据(30组中位数): corr_a=%+.4f corr_b=%+.4f corr_H2=%+.4f cond8=%.3g'
          % (0.6350, 0.8481, 0.9856, 4.72e4))


if __name__ == '__main__':
    main()

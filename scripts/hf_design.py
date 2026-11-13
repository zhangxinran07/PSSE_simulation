# -*- coding: utf-8 -*-
"""
最终激励频段设计: 验证「6~13Hz 转子动态敏感频段」multi-sine 的效果, 并重跑扩展频段 DE。

背景: hf_scan.py 发现 a/b 的 corr 在 6~8Hz 显著下降(b对0.37/a对0.69), 且 λ_min 在
      7~8Hz(a/b) 与 11Hz(H2) 达峰, 信息量峰值在 11~13Hz —— 这是转子磁链动态特征频段。
本脚本:
  ① 对比 6~13Hz 等幅 multi-sine 与既有方案;
  ② 在 0.5~20Hz 扩展候选频点上重跑 A-最优 DE, 看能量是否自发集中到 6~13Hz。
"""
import sys, os
import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, _trim_k

DT = 0.01
TRUTH = np.array([303.2, 72.7, 0.142, 0.337, 270.5, 63.2, 0.174, 0.265])
P0 = 5.9956
PAIRS = [('a', 0, 4), ('b', 1, 5), ('H2', 2, 6), ('Pim', 3, 7)]
PHASE_STD_DEG = 25.0
U_RANGE = 0.08
T_SIM = 40.0
LAM = 1e-8


def excitation_from_weights(A, freqs, seed=0):
    A = np.abs(np.asarray(A, float)); s = A.sum()
    A = A / s if s > 1e-12 else np.ones_like(A) / len(A)
    rng = np.random.RandomState(seed)
    n = int(round(T_SIM / DT)) + 1
    t = np.arange(n) * DT
    ph = rng.uniform(0, 2 * np.pi, len(freqs)); ps = rng.uniform(0, 2 * np.pi, len(freqs))
    th = np.zeros(n); amp = np.zeros(n)
    for Ak, f, pk, sk in zip(A, freqs, ph, ps):
        th += Ak * np.sin(2 * np.pi * f * t + pk); amp += np.sin(2 * np.pi * f * t + sk)
    th = th / th.std() * np.deg2rad(PHASE_STD_DEG)
    amp = amp / (amp.max() - amp.min()) * U_RANGE
    return (1.0 + amp) * np.exp(1j * th), t


def jall_of(u):
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
    V = np.abs(u[sl]); X = np.column_stack((np.ones(len(V)), V, V ** 2))
    Qr = np.linalg.solve(X.T @ X, X.T)
    JP = dP[sl, :] - X @ (Qr @ dP[sl, :]); JQ = dQ[sl, :] - X @ (Qr @ dQ[sl, :])
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
        e = np.linalg.eigvalsh(np.array([[ni, d], [d, nj]]))
        out['corr_' + nm] = d / np.sqrt(ni * nj)
        out['lmin_' + nm] = float(e[0])
    return out


def report(tag, u):
    m = analyze(u)
    if m is None:
        print('%-20s | 核发散' % tag); return None
    print('%-20s | corr a=%+.3f b=%+.3f H2=%+.3f | lmin a=%8.2g b=%8.2g H2=%8.2g | '
          'var_ab=%9.3g var_H2=%9.3g cond8=%9.3g'
          % (tag, m['corr_a'], m['corr_b'], m['corr_H2'], m['lmin_a'], m['lmin_b'], m['lmin_H2'],
             m['var_ab'], m['var_H2'], m['cond8']))
    return m


def main():
    print('===== 频段方案对比 (相位std=25°, |u|范围=0.08) =====')
    for f_lo, f_hi, ncomp, tag in [
        (6.0, 13.0, 8, '6-13Hz 等幅'),
        (6.0, 9.0, 6, '6-9Hz 等幅(a/b解耦)'),
        (9.0, 13.0, 6, '9-13Hz 等幅(H2)'),
        (6.0, 13.0, 12, '6-13Hz 密集'),
        (0.3, 20.0, 12, '0.3-20Hz 宽带'),
    ]:
        ff = np.logspace(np.log10(f_lo), np.log10(f_hi), ncomp)
        report(tag, excitation_from_weights(np.ones(ncomp), ff)[0])

    # 扩展频段 DE
    print('\n===== 扩展频段 A-最优 DE (候选 0.5~20Hz) =====')
    cand = np.logspace(np.log10(0.5), np.log10(20.0), 14)
    def cost(A):
        u, _ = excitation_from_weights(A, cand)
        Jall = jall_of(u)
        if Jall is None:
            return 1e30
        F = Jall.T @ Jall
        try:
            return float(np.trace(np.linalg.inv(F + LAM * np.eye(8))))
        except np.linalg.LinAlgError:
            return 1e30
    r = differential_evolution(cost, [(0, 1)] * len(cand), seed=1,
                               popsize=8, maxiter=30, tol=1e-6, polish=True, updating='immediate')
    A = np.abs(r.x); A /= A.sum()
    report('A-最优(0.5-20Hz)', excitation_from_weights(A, cand)[0])
    print('\n能量分配:')
    for f, a in zip(cand, A):
        if a > 1e-3:
            print('  f=%6.3f Hz  w=%5.3f  %s' % (f, a, '#' * int(round(a * 50))))


if __name__ == '__main__':
    main()

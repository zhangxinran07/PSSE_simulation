# -*- coding: utf-8 -*-
"""
高频扩展扫描: 单频相位主导激励下, a/b/H2 各对的「可分离信息量 λ_min」随频率(0.3~20Hz)变化。
确认信息量是否继续随频率单调上升, 找到最优激励频段上界。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, _trim_k

DT = 0.01
TRUTH = np.array([303.2, 72.7, 0.142, 0.337, 270.5, 63.2, 0.174, 0.265])
P0 = 5.9956
PAIRS = [('a', 0, 4), ('b', 1, 5), ('H2', 2, 6)]
EPS_A = 0.03
EPS_TH = 0.40


def make_excitation(f, n_cycle_min=6, T_min=12.0):
    T_sim = max(T_min, n_cycle_min / f)
    n = int(round(T_sim / DT)) + 1
    t = np.arange(n) * DT
    amp = 1.0 + EPS_A * np.sin(2 * np.pi * f * t)
    ang = EPS_TH * np.sin(2 * np.pi * f * t)
    return amp * np.exp(1j * ang), t


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
    V = np.abs(u[sl])
    X = np.column_stack((np.ones(len(V)), V, V ** 2))
    Qr = np.linalg.solve(X.T @ X, X.T)
    JP = dP[sl, :] - X @ (Qr @ dP[sl, :])
    JQ = dQ[sl, :] - X @ (Qr @ dQ[sl, :])
    return np.vstack([JP * TRUTH[None, :], JQ * TRUTH[None, :]])


def main():
    freqs = np.logspace(np.log10(0.3), np.log10(20.0), 30)
    print('%8s | %10s %10s %10s | %8s %8s %8s | %10s %10s' % (
        'f(Hz)', 'lmin_a', 'lmin_b', 'lmin_H2', 'corr_a', 'corr_b', 'corr_H2', 'info_a', 'info_b'))
    print('-' * 100)
    for f in freqs:
        u, t = make_excitation(f)
        Jall = jall_of(u)
        if Jall is None:
            continue
        F = Jall.T @ Jall
        info_a = F[0, 0] + F[4, 4]
        info_b = F[1, 1] + F[5, 5]
        vals = []
        for nm, i, j in PAIRS:
            gi = Jall[:, i]; gj = Jall[:, j]
            ni = float(gi @ gi); nj = float(gj @ gj); d = float(gi @ gj)
            G = np.array([[ni, d], [d, nj]])
            e = np.linalg.eigvalsh(G)
            corr = d / np.sqrt(ni * nj)
            vals.append((e[0], corr))
        print('%8.3f | %10.4g %10.4g %10.4g | %+8.4f %+8.4f %+8.4f | %10.4g %10.4g'
              % (f, vals[0][0], vals[1][0], vals[2][0],
                 vals[0][1], vals[1][1], vals[2][1], info_a, info_b))


if __name__ == '__main__':
    main()

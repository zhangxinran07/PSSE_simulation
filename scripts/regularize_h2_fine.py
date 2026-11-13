# -*- coding: utf-8 -*-
"""λ 精细扫描: 在 0.02~0.10 密集扫描 H2 差量 Tikhonov 正则, 锁定最终工程 λ, 给出 P_im/H2 精度。

复用 regularize_h2_diff.py 的 make_reg 逻辑(差量正则 obj + 解析梯度)。
额外输出: 逐参数中位误差 + 真值 H2 差量固有下限对照。
"""
import numpy as np
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import obj_mix_iden, obj_mix_jac, identify_multistart

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
ORI = os.path.join(RES, 'ori_data_mix_20261106_033919.pkl')
RRES = os.path.join(RES, 'result_all_mix_20261106_033919.pkl')
OUT = os.path.join(RES, 'regularize_h2_fine.txt')

lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
DT = 0.01
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
SEED = 20261107


def make_reg(lam_H, lam_P=0.0):
    def obj(para, u, P, Q, dt):
        return obj_mix_iden(para, u, P, Q, dt) + lam_H * (para[2] - para[6]) ** 2 + lam_P * (para[3] - para[7]) ** 2

    def jac(para, u, P, Q, dt):
        g = np.array(obj_mix_jac(para, u, P, Q, dt), dtype=float)
        g[2] += 2 * lam_H * (para[2] - para[6])
        g[6] -= 2 * lam_H * (para[2] - para[6])
        g[3] += 2 * lam_P * (para[3] - para[7])
        g[7] -= 2 * lam_P * (para[3] - para[7])
        return g
    return obj, jac


def run(LAM_H, K=12, n_groups=30):
    ori = pickle.load(open(ORI, 'rb'))
    truth_df = pickle.load(open(RRES, 'rb'))
    groups = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))[:n_groups]

    data = []
    for g in groups:
        u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)
        data.append((g, u, P, Q, truth))

    truths = np.array([d[4] for d in data])
    H2diff_true = np.median([abs(t[2] - t[6]) / abs(t[2] + t[6]) for t in truths])
    Pimdiff_true = np.median([abs(t[3] - t[7]) / abs(t[3] + t[7]) for t in truths])

    lines = []
    lines.append('=' * 110)
    lines.append('H2 差量正则 λ 精细扫描 (30 组, K=%d 起点) —— 锁定最终工程 λ' % K)
    lines.append('obj = obj_mix_iden + λ_H·(H2_a-H2_b)²')
    lines.append('=' * 110)
    lines.append('%-8s | %-10s %-10s %-10s | %-10s %-10s | %-9s %-9s %-9s' % (
        'λ_H', 'H2总量%', 'H2差量%', 'H2比值%', 'P_im总量%', 'P_im差量%', 'dyn6%', 'P_im中位%', 'H2中位%'))
    lines.append('-' * 110)

    rows = {}
    for lam in LAM_H:
        obj, jac = make_reg(lam)
        xs = []
        for (g, u, P, Q, truth) in data:
            best, _ = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj,
                                          tol=1e-8, n_restarts=K, jac_func=jac, seed=SEED + g)
            xs.append(np.asarray(best.x))
        xs = np.array(xs)
        E = np.abs(xs - truths) / np.abs(truths)

        SH = np.median([abs((x[2] + x[6]) - (t[2] + t[6])) / abs(t[2] + t[6]) for x, t in zip(xs, truths)]) * 100
        DH = np.median([abs((x[2] - x[6]) - (t[2] - t[6])) / abs(t[2] + t[6]) for x, t in zip(xs, truths)]) * 100
        RH = np.median([abs(x[2] / x[6] - t[2] / t[6]) / abs(t[2] / t[6]) for x, t in zip(xs, truths)]) * 100
        SP = np.median([abs((x[3] + x[7]) - (t[3] + t[7])) / abs(t[3] + t[7]) for x, t in zip(xs, truths)]) * 100
        DP = np.median([abs((x[3] - x[7]) - (t[3] - t[7])) / abs(t[3] + t[7]) for x, t in zip(xs, truths)]) * 100
        dyn6 = np.median(E[:, DYN]) * 100
        pim = np.median(E[:, PIM]) * 100
        h2 = np.median(E[:, H2]) * 100

        line = '%-8.4g | %-10.2f %-10.2f %-10.2f | %-10.2f %-10.2f | %-9.2f %-9.2f %-9.2f' % (
            lam, SH, DH, RH, SP, DP, dyn6, pim, h2)
        lines.append(line)
        rows[lam] = (SH, DH, RH, SP, DP, dyn6, pim, h2, xs)
        print(line, flush=True)

    lines.append('-' * 110)
    lines.append('※ 真值 H2 差量 |H2_a-H2_b|/总量 中位 = %.2f%% (正则收缩偏置下限, 不宜压到远低于此)' % (H2diff_true * 100))
    lines.append('※ 真值 P_im 差量 |P_ima-P_imb|/总量 中位 = %.2f%%' % (Pimdiff_true * 100))
    lines.append('=' * 110)
    lines.append('逐参数中位误差%%:')
    lines.append('%-8s  ' % 'λ_H' + ' '.join('%-8s' % p for p in PNAMES))
    for lam in LAM_H:
        xs = rows[lam][-1]
        E = np.abs(xs - truths) / np.abs(truths)
        lines.append('%-8.4g  ' % lam + ' '.join('%-8.2f' % m for m in np.median(E, axis=0) * 100))
    lines.append('=' * 110)

    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    LAM_H = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(LAM_H, K, n_groups)

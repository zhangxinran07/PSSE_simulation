# -*- coding: utf-8 -*-
"""两阶段辨识策略: 先强正则定「总量方向」, 再弱正则放「差量方向」。

背景: H2_a↔H2_b 存在「此消彼长」零空间(γ=9.85, r=-0.99, A2.19), 单一总功率观测下
      H2 差量最难辨识。差量正则 (H2_a-H2_b)² 是把差量往 0 收缩的软约束,
      但真值 |H2_a-H2_b|/总量 中位 ≈ 10%, 强正则会过度收缩、弱正则又收不住。

两阶段思路:
  stage1: 强正则 λ_strong(差量压到≈0) → 冻结零空间, 稳定辨识「总量方向」+ a/b/P_im
  stage2: 以 stage1 解为初值, 换弱正则 λ_weak → 只放差量方向微调, 不跑飞总量

对照:
  A = stage1(强正则, 差量被压 0)
  B = 两阶段(stage1→stage2)
  C = 直接弱正则(一步到位, 现状 A2.16)

回答: 「先稳总量再放差量」是否优于「一步弱正则」?
"""
import numpy as np
import os
import sys
import pickle
from scipy import optimize as opt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import obj_mix_iden, obj_mix_jac, identify_multistart

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
ORI = os.path.join(RES, 'ori_data_mix_20261106_033919.pkl')
RRES = os.path.join(RES, 'result_all_mix_20261106_033919.pkl')
OUT = os.path.join(RES, 'two_stage_identify.txt')

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


def run(K=12, n_groups=30, lam_weak=0.03, lam_strong=1.0):
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

    XA, XB, XC = [], [], []
    for (g, u, P, Q, truth) in data:
        bounds = list(zip(lb * 2, ub * 2))
        # stage1: 强正则多起点
        obj1, jac1 = make_reg(lam_strong)
        r1, _ = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj1,
                                    tol=1e-8, n_restarts=K, jac_func=jac1, seed=SEED + g)
        x1 = np.asarray(r1.x)
        # stage2: 弱正则, 以 x1 为初值单次 trust-constr 微调
        obj2, jac2 = make_reg(lam_weak)
        r2 = opt.minimize(obj2, x1, args=(u, P, Q, DT), jac=jac2, method='trust-constr',
                          bounds=bounds, tol=1e-8)
        x2 = np.asarray(r2.x)
        # 对照 C: 直接弱正则多起点
        objw, jacw = make_reg(lam_weak)
        rw, _ = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=objw,
                                    tol=1e-8, n_restarts=K, jac_func=jacw, seed=SEED + g)
        xw = np.asarray(rw.x)
        XA.append(x1)
        XB.append(x2)
        XC.append(xw)

    XA, XB, XC = np.array(XA), np.array(XB), np.array(XC)
    EA = np.abs(XA - truths) / np.abs(truths)
    EB = np.abs(XB - truths) / np.abs(truths)
    EC = np.abs(XC - truths) / np.abs(truths)

    lines = []
    lines.append('=' * 110)
    lines.append('两阶段辨识 vs 直接弱正则 (30 组, K=%d, λ_strong=%.3g, λ_weak=%.3g)' % (K, lam_strong, lam_weak))
    lines.append('=' * 110)
    lines.append('%-22s %-10s %-10s %-10s | %-10s %-10s | %-9s %-9s %-9s' % (
        '方案', 'H2总量%', 'H2差量%', 'H2比值%', 'P_im总量%', 'P_im差量%', 'dyn6%', 'P_im中位%', 'H2中位%'))
    lines.append('-' * 110)

    def row(tag, X):
        E = np.abs(X - truths) / np.abs(truths)
        SH = np.median([abs((x[2]+x[6])-(t[2]+t[6]))/abs(t[2]+t[6]) for x, t in zip(X, truths)]) * 100
        DH = np.median([abs((x[2]-x[6])-(t[2]-t[6]))/abs(t[2]+t[6]) for x, t in zip(X, truths)]) * 100
        RH = np.median([abs(x[2]/x[6]-t[2]/t[6])/abs(t[2]/t[6]) for x, t in zip(X, truths)]) * 100
        SP = np.median([abs((x[3]+x[7])-(t[3]+t[7]))/abs(t[3]+t[7]) for x, t in zip(X, truths)]) * 100
        DP = np.median([abs((x[3]-x[7])-(t[3]-t[7]))/abs(t[3]+t[7]) for x, t in zip(X, truths)]) * 100
        dyn6 = np.median(E[:, DYN]) * 100
        pim = np.median(E[:, PIM]) * 100
        h2 = np.median(E[:, H2]) * 100
        lines.append('%-22s %-10.2f %-10.2f %-10.2f | %-10.2f %-10.2f | %-9.2f %-9.2f %-9.2f' % (
            tag, SH, DH, RH, SP, DP, dyn6, pim, h2))

    row('A stage1(强 λ=%.3g)' % lam_strong, XA)
    row('B 两阶段(强→弱)', XB)
    row('C 直接弱(λ=%.3g)' % lam_weak, XC)

    lines.append('-' * 110)
    lines.append('※ 真值 H2 差量 |H2_a-H2_b|/总量 中位 = %.2f%% (差量辨识的偏置下限)' % (H2diff_true * 100))
    lines.append('=' * 110)
    lines.append('逐参数中位误差%%:')
    lines.append('%-22s  ' % '方案' + ' '.join('%-8s' % p for p in PNAMES))
    for tag, X in [('A stage1 强正则', XA), ('B 两阶段', XB), ('C 直接弱', XC)]:
        E = np.abs(X - truths) / np.abs(truths)
        lines.append('%-22s  ' % tag + ' '.join('%-8.2f' % m for m in np.median(E, axis=0) * 100))
    lines.append('=' * 110)

    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    lam_weak = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
    lam_strong = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    run(K, n_groups, lam_weak, lam_strong)

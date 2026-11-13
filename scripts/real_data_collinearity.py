# -*- coding: utf-8 -*-
"""
真实数据（复数电压 u，相位主导激励）下的可辨识性结构 —— 决定性诊断。

背景：
  上一轮发现真实 PSSE 数据的电压激励是「相位主导」：|u| 几乎不变(std 1.3%)，
  而相角摆幅 std≈31°(uq std 0.48 >> ud std 0.14)。此前所有频域/时域诊断用的都是
  「幅值调制」激励 u=1+eps*sin(2πft)+0j，激励通道与真实不匹配，结论不可靠。

本脚本在 30 组真实复数电压 u 上，按 obj_mix_iden 的真实口径（含 [1,V,V^2] 回归投影），
  计算 A/B 同名参数的混淆结构：P 通道 / Q 通道 / 合并通道 各自的相关系数与 2x2 条件数，
  以及 8x8 整体条件数 + 最小特征值方向。输出取 30 组中位数（稳健）。

关键对照：
  - 幅值调制合成激励(proj_collinearity.py) → P 通道 a/b 共线、Q 通道 a/b 反相关
  - 真实相位主导激励(本脚本) → ???（这才是辨识真正面对的结构）
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, _trim_k

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_PKL = os.path.join(PROJ, 'Sample_Generation', 'results', 'result_all_mix_20261106_003107_u0.060.pkl')
ORI_PKL = os.path.join(PROJ, 'Sample_Generation', 'results', 'ori_data_mix_20261106_003107.pkl')

NAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
PAIRS = [('a 对', 0, 4), ('b 对', 1, 5), ('H2对', 2, 6), ('P_im对', 3, 7)]


def build_jac(u, P0, truth, dt=0.01):
    a_a, b_a, H2_a, P_ima = truth[0], truth[1], truth[2], truth[3]
    a_b, b_b, H2_b, P_imb = truth[4], truth[5], truth[6], truth[7]
    mva_a = P0 * P_ima; mva_b = P0 * P_imb
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    assert ok_a and ok_b, '核发散'
    n = len(u)
    dP = np.empty((n, 8)); dQ = np.empty((n, 8))
    dP[:, 0:3] = dPa; dP[:, 3] = P0 * (P_pa / mva_a)
    dP[:, 4:7] = dPb; dP[:, 7] = P0 * (P_pb / mva_b)
    dQ[:, 0:3] = dQa; dQ[:, 3] = P0 * (Q_pa / mva_a)
    dQ[:, 4:7] = dQb; dQ[:, 7] = P0 * (Q_pb / mva_b)
    k = _trim_k(n); sl = slice(k, n - k)
    return dP[sl, :], dQ[sl, :], np.abs(u[sl])


def project(J, V):
    m = len(V)
    X = np.column_stack((np.ones(m), V, V ** 2))
    Qr = np.linalg.solve(X.T @ X, X.T)
    return J - X @ (Qr @ J)


def pair_corr_cond(Jn, i, j):
    gi = Jn[:, i]; gj = Jn[:, j]
    ni = np.linalg.norm(gi); nj = np.linalg.norm(gj)
    if ni < 1e-30 or nj < 1e-30:
        return np.nan, np.nan
    d = float(np.dot(gi, gj))
    corr = d / (ni * nj)
    G = np.array([[ni * ni, d], [d, nj * nj]])
    ev = np.linalg.eigvalsh(G)
    cond = float(ev[1] / ev[0]) if ev[0] > 1e-30 else np.inf
    return corr, cond


def main():
    res = pd.read_pickle(RES_PKL)
    ori = pd.read_pickle(ORI_PKL)
    n_groups = len(res)

    # 收集各组的指标
    keys = []
    for ch in ['P', 'Q', 'PQ']:
        for name, i, j in PAIRS:
            keys.append((ch, name, 'corr'))
            keys.append((ch, name, 'cond'))
    for ch in ['P', 'Q']:
        keys.append((ch, '', 'norm'))
    keys.append(('ALL', '', 'cond8'))

    acc = {k: [] for k in keys}

    for g in range(n_groups):
        row = res.iloc[g]
        truth = np.array([row['a_a_real'], row['b_a_real'], row['H2_a_real'], row['P_ima_real'],
                          row['a_b_real'], row['b_b_real'], row['H2_b_real'], row['P_imb_real']])
        ud = ori['%d ud' % g].values; uq = ori['%d uq' % g].values
        P = ori['%d P' % g].values
        u = ud + 1j * uq
        P0 = P[0]
        JP, JQ, V = build_jac(u, P0, truth)
        JPe = project(JP, V) * truth[None, :]
        JQe = project(JQ, V) * truth[None, :]
        Jall = np.vstack([JPe, JQe])

        for name, i, j in PAIRS:
            for ch, Jn in [('P', JPe), ('Q', JQe), ('PQ', Jall)]:
                corr, cond = pair_corr_cond(Jn, i, j)
                acc[(ch, name, 'corr')].append(corr)
                acc[(ch, name, 'cond')].append(cond)
        acc[('P', '', 'norm')].append(np.linalg.norm(JPe, 'fro'))
        acc[('Q', '', 'norm')].append(np.linalg.norm(JQe, 'fro'))
        F = Jall.T @ Jall
        ev = np.linalg.eigvalsh(F)
        acc[('ALL', '', 'cond8')].append(ev[-1] / ev[0])

    def med(x):
        return np.median([v for v in x if np.isfinite(v)]) if any(np.isfinite(v) for v in x) else np.inf

    print('===== 真实复数激励（相位主导）下的可辨识性结构，%d 组中位数 =====' % n_groups)
    print('通道相对灵敏度总范数:  P=%.3g   Q=%.3g' % (med(acc[('P', '', 'norm')]), med(acc[('Q', '', 'norm')])))
    print()
    print('%-6s | %-6s | %10s | %12s' % ('通道', '参数对', 'corr', '2x2条件数'))
    print('-' * 46)
    for ch in ['P', 'Q', 'PQ']:
        for name, i, j in PAIRS:
            c = med(acc[(ch, name, 'corr')])
            cd = med(acc[(ch, name, 'cond')])
            print('%-6s | %-6s | %+10.4f | %12.3g' % (ch, name, c, cd))
    print('-' * 46)
    print('合并 8x8 条件数: %.3g' % med(acc[('ALL', '', 'cond8')]))

    # 第 0 组最小特征值方向
    row = res.iloc[0]
    truth = np.array([row['a_a_real'], row['b_a_real'], row['H2_a_real'], row['P_ima_real'],
                      row['a_b_real'], row['b_b_real'], row['H2_b_real'], row['P_imb_real']])
    ud = ori['0 ud'].values; uq = ori['0 uq'].values; P = ori['0 P'].values
    u = ud + 1j * uq; P0 = P[0]
    JP, JQ, V = build_jac(u, P0, truth)
    Jall = np.vstack([project(JP, V) * truth[None, :], project(JQ, V) * truth[None, :]])
    F = Jall.T @ Jall
    ev, evec = np.linalg.eigh(F)
    print('\n第0组最小特征值方向(载荷前5):')
    v0 = evec[:, 0]
    for k in np.argsort(-np.abs(v0))[:5]:
        print('   %-7s %+.3f' % (NAMES[k], v0[k]))


if __name__ == '__main__':
    main()

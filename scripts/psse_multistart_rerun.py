# -*- coding: utf-8 -*-
"""PSSE 真实数据上的多起点辨识重跑: 单次(现状) vs 多起点取全局最优。

背景(2026-11-07, 张老师实测): 8 参数模型存在 2 个局部极小 + 1 个全局极小, 必须多起点。
辨识核心(obj_mix_iden/_core_con/_core_var)为纯 numpy/numba, 本机即可对 PSSE 已采样的
ori_data_mix 直接重跑, 无需 PSSE。对比:
  A. 单次 identify(max_trials=5)      —— main_simu.simu_iden_mix 现状(返回首个通过距离校验的解)
  B. 多起点 identify_multistart(K)     —— K 个独立初值取目标值最小(对抗非凸)

输出 Sample_Generation/results/psse_multistart_rerun.txt
"""
import numpy as np
import pandas as pd
import os
import sys
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import obj_mix_iden, obj_mix_jac, identify, identify_multistart

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
ORI = os.path.join(RES, 'ori_data_mix_20261106_033919.pkl')
RRES = os.path.join(RES, 'result_all_mix_20261106_033919.pkl')
OUT = os.path.join(RES, 'psse_multistart_rerun.txt')

lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
DT = 0.01
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]   # 6 个动态参数(不含 P_im 比例)
SEED = 20261107


def relerr(x, xt):
    return np.abs(np.asarray(x) - np.asarray(xt)) / np.abs(np.asarray(xt))


def run(K=6, n_groups=30):
    ori = pickle.load(open(ORI, 'rb'))
    truth_df = pickle.load(open(RRES, 'rb'))
    groups = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))
    groups = groups[:n_groups]

    rows = []
    print('=' * 100, flush=True)
    print('PSSE 真实数据多起点重跑: 单次(现状) vs 多起点 K=%d' % K, flush=True)
    print('=' * 100, flush=True)
    for g in groups:
        ud = ori['%d ud' % g].values
        uq = ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        u = ud + 1j * uq
        truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)

        # A. 单次(现状)
        np.random.seed(SEED + g)
        res1, t1 = identify(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj_mix_iden,
                            tol=1e-8, max_trials=5, jac_func=obj_mix_jac)
        # B. 多起点
        resK, tK = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj_mix_iden,
                                       tol=1e-8, n_restarts=K, jac_func=obj_mix_jac, seed=SEED + g)

        e1 = relerr(res1.x, truth)
        eK = relerr(resK.x, truth)

        h2r = truth[2] / truth[6]                       # H2_a / H2_b
        near_lb = (truth[0] < lb[0] * 2 * 1.05) or (truth[4] < lb[0] * 2 * 1.05)   # a 贴下界
        near_h2 = 0.9 < h2r < 1.1
        flag = '近不可辨识' if (near_h2 or near_lb) else '可辨识'
        rows.append((g, e1, eK, res1.fun, resK.fun, flag, h2r))
        print('  g=%02d %s H2a/H2b=%.2f | 单次 dyn=%.2f%% fun=%.2e | K%d dyn=%.2f%% fun=%.2e'
              % (g, flag, h2r, np.median(e1[DYN]) * 100, res1.fun,
                 K, np.median(eK[DYN]) * 100, resK.fun), flush=True)

    E1 = np.array([r[1] for r in rows])
    EK = np.array([r[2] for r in rows])
    flags = np.array([r[5] for r in rows])

    def stat(E):
        return np.median(E[:, DYN]) * 100, np.percentile(E[:, DYN], q=90) * 100

    lines = []
    lines.append('=' * 100)
    lines.append('PSSE 真实数据(%d 组)重跑: 单次 identify vs 多起点 identify_multistart(K=%d)' % (len(groups), K))
    lines.append('=' * 100)
    lines.append('%-12s %-14s %-14s' % ('', '单次(现状)', '多起点 K=%d' % K))
    lines.append('-' * 100)
    for tag, E in [('动态6参数', E1), (' ', EK)]:
        pass
    for name, E in [('单次(现状)', E1), ('多起点K%d' % K, EK)]:
        med = np.median(E, axis=0) * 100
        p90 = np.percentile(E, axis=0, q=90) * 100
        lines.append('%s:' % name)
        lines.append('  中位%% ' + ' '.join('%-9.2f' % m for m in med))
        lines.append('  P90%%  ' + ' '.join('%-9.2f' % m for m in p90))
        lines.append('  动态6参数 中位=%.2f%%  P90=%.2f%%' % (np.median(E[:, DYN]) * 100, np.percentile(E[:, DYN], q=90) * 100))
    lines.append('-' * 100)
    # 逐参数 ratio_std(散点): 多起点是否收窄 identified/truth 的比值散度
    lines.append('identified/truth 比值散度 ratio_std(越小=散点越干净):')
    lines.append('%-8s %-14s %-14s' % ('参数', '单次ratio_std', 'K%d ratio_std' % K))
    for j, p in enumerate(PNAMES):
        x1 = np.array([r[1][j] for r in rows])
        xK = np.array([r[2][j] for r in rows])
        # 用 (x-xt)/xt 的相对误差 std 近似散点(等价于 ratio-1 的 std)
        s1 = np.std(x1)
        sK = np.std(xK)
        lines.append('%-8s %-14.3f %-14.3f' % (p, s1, sK))
    lines.append('-' * 100)
    # 分区统计: 可辨识 vs 近不可辨识
    for flag in ['可辨识', '近不可辨识']:
        m = flags == flag
        if m.sum() == 0:
            continue
        e1m, eKm = E1[m], EK[m]
        lines.append('%s区(%d 组): 单次dyn 中位=%.2f%% P90=%.2f%% | K%d dyn 中位=%.2f%% P90=%.2f%%'
                     % (flag, m.sum(), np.median(e1m[:, DYN]) * 100, np.percentile(e1m[:, DYN], 90) * 100,
                        K, np.median(eKm[:, DYN]) * 100, np.percentile(eKm[:, DYN], 90) * 100))
    lines.append('=' * 100)
    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(K, n_groups)

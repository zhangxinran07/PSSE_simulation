# -*- coding: utf-8 -*-
"""PSSE 真实数据上的 P_im / H2 辨识精度专项诊断：naive(fun 最小) vs oracle(误差最小)。

回答张老师「把 P_im 和 H2 辨识准确」(2026-11-07) 的第二步 —— 在真实数据上量化损失来源:

  1. 逐参数误差: 多起点 naive(取 fun 最小) 下 P_im/H2 是否仍是瓶颈?
  2. naive vs oracle 差距: 同一批 K 个 restart 解里, 若「按真值误差挑最小」能好多少?
     -> 差距 = 结构失配(次暂态/堵转/频率依赖)扭曲 fun 表面, 导致 fun 最小 ≠ 误差最小的损失。
  3. 组合量分解(仅对 naive): P_im/H2 的总量(可辨识) vs 差量(零空间)误差。
  4. A/B 带符号误差相关性: 坐实「此消彼长」是否也发生在真实数据上。

口径与 psse_multistart_rerun.py 严格对齐:
  参数顺序 [a_a,b_a,H2_a,P_ima,a_b,b_b,H2_b,P_imb];
  trust-constr + obj_mix_jac; identify(max_trials=1) 逐个 restart 收集, 再分别挑 naive/oracle。
"""
import numpy as np
import os
import sys
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import obj_mix_iden, obj_mix_jac, identify

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
ORI = os.path.join(RES, 'ori_data_mix_20261106_033919.pkl')
RRES = os.path.join(RES, 'result_all_mix_20261106_033919.pkl')
OUT = os.path.join(RES, 'diagnose_pim_h2_psse.txt')

lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
DT = 0.01
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
SEED = 20261107


def relerr(x, xt):
    return np.abs(np.asarray(x) - np.asarray(xt)) / np.abs(np.asarray(xt))


def run_all_restarts(u, P, Q, truth, K, g):
    """跑 K 个独立 restart, 收集全部解, 返回 (naive_x, naive_fun, oracle 各版本)。"""
    xs, funs = [], []
    for r in range(K):
        np.random.seed(SEED + g + r * 1000)
        res, _ = identify(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj_mix_iden,
                          tol=1e-8, max_trials=1, jac_func=obj_mix_jac)
        xs.append(np.asarray(res.x))
        funs.append(res.fun)
    xs = np.array(xs)
    funs = np.array(funs)
    E = np.array([relerr(x, truth) for x in xs])   # (K, 8)

    i_fun = int(np.argmin(funs))
    i_all = int(np.argmin(np.median(E, axis=1)))                       # 全部 8 参数中位误差最小
    i_pim = int(np.argmin(np.median(E[:, PIM], axis=1)))               # 仅 P_im 误差最小
    i_h2 = int(np.argmin(np.median(E[:, H2], axis=1)))                 # 仅 H2 误差最小
    return {
        'naive': (xs[i_fun], funs[i_fun]),
        'oracle_all': (xs[i_all], funs[i_all]),
        'oracle_pim': (xs[i_pim], funs[i_pim]),
        'oracle_h2': (xs[i_h2], funs[i_h2]),
    }


def run(K=8, n_groups=30):
    ori = pickle.load(open(ORI, 'rb'))
    truth_df = pickle.load(open(RRES, 'rb'))
    groups = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))
    groups = groups[:n_groups]

    rec = {k: [] for k in ['naive', 'oracle_all', 'oracle_pim', 'oracle_h2']}
    rows = []   # (g, truth, naive_x, naive_e, all_e 列表)
    print('=' * 100, flush=True)
    print('PSSE 真实数据 P_im/H2 专项诊断: naive(fun 最小) vs oracle(误差最小), K=%d, N=%d' % (K, len(groups)), flush=True)
    print('=' * 100, flush=True)
    for g in groups:
        ud = ori['%d ud' % g].values
        uq = ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        u = ud + 1j * uq
        truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)

        sol = run_all_restarts(u, P, Q, truth, K, g)
        for k in rec:
            x, _ = sol[k]
            rec[k].append(relerr(x, truth))

        naive_x, naive_fun = sol['naive']
        naive_e = relerr(naive_x, truth)
        rows.append((g, truth, naive_x, naive_e))
        print('  g=%02d  naive dyn=%.2f%%  P_im={%.2f%%,%.2f%%}  H2={%.2f%%,%.2f%%}  fun=%.2e'
              % (g, np.median(naive_e[DYN]) * 100, naive_e[3] * 100, naive_e[7] * 100,
                 naive_e[2] * 100, naive_e[6] * 100, naive_fun), flush=True)

    # ---- 逐参数 naive vs oracle 汇总 ----
    En = {k: np.array(rec[k]) for k in rec}
    lines = []
    lines.append('=' * 100)
    lines.append('PSSE 真实数据(%d 组) P_im/H2 专项诊断: naive vs oracle (K=%d)' % (len(groups), K))
    lines.append('=' * 100)
    lines.append('')
    lines.append('【1. 逐参数中位误差: naive(fun 最小) vs oracle(真值误差最小)】')
    lines.append('%-12s ' % '方案' + ' '.join('%-9s' % p for p in PNAMES))
    lines.append('-' * 100)
    for k, tag in [('naive', 'naive fun-min'), ('oracle_all', 'oracle 全参最优'),
                   ('oracle_pim', 'oracle P_im最优'), ('oracle_h2', 'oracle H2最优')]:
        med = np.median(En[k], axis=0) * 100
        lines.append('%-12s ' % tag + ' '.join('%-9.2f' % m for m in med))
    lines.append('-' * 100)
    lines.append('%-12s %-18s %-18s %-18s' % ('方案', '动态6参数中位', 'P_im中位', 'H2中位'))
    for k, tag in [('naive', 'naive fun-min'), ('oracle_all', 'oracle 全参最优'),
                   ('oracle_pim', 'oracle P_im最优'), ('oracle_h2', 'oracle H2最优')]:
        E = En[k]
        lines.append('%-12s %-18.2f %-18.2f %-18.2f'
                     % (tag, np.median(E[:, DYN]) * 100, np.median(E[:, PIM]) * 100,
                        np.median(E[:, H2]) * 100))
    lines.append('')
    lines.append('  ※ oracle = 同一批 K 个 restart 解里「按真值误差挑最小」, 是任何无真值方法的上界;')
    lines.append('    naive 与 oracle 的差距 = 结构失配扭曲 fun 表面(取 fun 最小≠误差最小)的净损失。')
    lines.append('')
    lines.append('【2. naive 组合量分解: 总量(可辨识) vs 差量(零空间)】')
    lines.append('   (真实数据有结构失配, 总量也非零误差, 但差量应显著更差)')
    lines.append('-' * 64)
    SP, DP, SH, DH, RP, RH = [], [], [], [], [], []
    for (g, t, x, e) in rows:
        SP.append(abs((x[3] + x[7]) - (t[3] + t[7])) / abs(t[3] + t[7]))
        DP.append(abs((x[3] - x[7]) - (t[3] - t[7])) / abs(t[3] + t[7]))
        SH.append(abs((x[2] + x[6]) - (t[2] + t[6])) / abs(t[2] + t[6]))
        DH.append(abs((x[2] - x[6]) - (t[2] - t[6])) / abs(t[2] + t[6]))
        RP.append(abs(x[3] / x[7] - t[3] / t[7]) / abs(t[3] / t[7]))
        RH.append(abs(x[2] / x[6] - t[2] / t[6]) / abs(t[2] / t[6]))
    SP, DP, SH, DH = np.array(SP), np.array(DP), np.array(SH), np.array(DH)
    RP, RH = np.array(RP), np.array(RH)
    lines.append('%-24s %-14s %-14s' % ('组合量', '中位误差%', 'P90误差%'))
    lines.append('-' * 64)
    for name, arr in [('P_im 总量(P_ima+P_imb)', SP), ('P_im 差量(P_ima-P_imb)', DP),
                      ('P_im 比值(P_ima/P_imb)', RP), ('H2 总量(H2_a+H2_b)', SH),
                      ('H2 差量(H2_a-H2_b)', DH), ('H2 比值(H2_a/H2_b)', RH)]:
        lines.append('%-24s %-14.2f %-14.2f' % (name, np.median(arr) * 100, np.percentile(arr, 90) * 100))
    lines.append('-' * 64)
    lines.append('')
    lines.append('【3. A/B 带符号误差相关性(此消彼长实证, 仅 naive)】')
    sP_a = np.array([(r[2][3] - r[1][3]) / r[1][3] for r in rows])
    sP_b = np.array([(r[2][7] - r[1][7]) / r[1][7] for r in rows])
    sH_a = np.array([(r[2][2] - r[1][2]) / r[1][2] for r in rows])
    sH_b = np.array([(r[2][6] - r[1][6]) / r[1][6] for r in rows])
    lines.append('  P_ima vs P_imb 带符号误差相关系数 r = %+.3f' % np.corrcoef(sP_a, sP_b)[0, 1])
    lines.append('  H2_a  vs H2_b  带符号误差相关系数 r = %+.3f' % np.corrcoef(sH_a, sH_b)[0, 1])
    lines.append('  ※ 强负相关(r<-0.5) = 真实数据上也存在「总量守恒式病态分配/此消彼长」。')
    lines.append('')
    lines.append('=' * 100)
    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(K, n_groups)

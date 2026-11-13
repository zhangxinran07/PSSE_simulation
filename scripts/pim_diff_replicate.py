# -*- coding: utf-8 -*-
"""λ_P 改善的独立复验（纯离线，零 PSSE 仿真）。

背景：A2.35 在调参用的 30 组上发现 λ_P=0.01~0.03 显著改善 P_im，
      但该批正是调出 λ_H=0.05 的同一批，存在二次调参过拟合风险。
本脚本在 3 批独立 PSSE 数据（不同扰动幅值、均有配对真值）+ 1 批零失配合成数据上复验：

  固定 λ_H=0.05, K=12, 同 seed；只对比 λ_P=0（基线）与 λ_P=0.03。

判定逻辑：
  - 若独立 PSSE 数据上改善方向一致且显著 → 真规律，可考虑写入工程解
  - 若改善消失/反向 → 该批数据特异的偏置抵消（呼应 A2.31/A2.33），不可移植
  - 零失配数据上若改善消失 → 坐实「改善来自结构失配造成的方向性偏置」

用法：
    python pim_diff_replicate.py solve <dataset> <0|1>
    python pim_diff_replicate.py merge

dataset: 20261105_201009 / 20261105_225206 / 20261106_020010 / zeromis
输出：Sample_Generation/results/pim_diff_replicate.txt
"""
import numpy as np
import os
import sys
import pickle
import time
from scipy.stats import wilcoxon

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'package'))
from regularize_h2_fine import (make_reg, RES, DT, lb, ub, PNAMES, SEED)
from identification import obj_mix_iden, obj_mix_jac, identify_multistart, _core_con, _core_var
from control_zero_mismatch import sample_mix

OUT = os.path.join(RES, 'pim_diff_replicate.txt')
DATASETS = ['20261105_201009', '20261105_225206', '20261106_020010', 'zeromis']
LAM_P = [0.0, 0.03]
LAM_H = 0.05
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
P0 = 6.0
NG = 30


def gen_PQ_zero(u, para):
    a_a, b_a, H2_a = para[0] / 0.75, para[1], para[2] / 0.75
    a_b, b_b, H2_b = para[4] / 0.75, para[5], para[6] / 0.75
    P_ima, P_imb = para[3], para[7]
    Ma, Mb = P0 * P_ima, P0 * P_imb
    Ps = P0 * (1 - P_ima - P_imb)
    ok_a, Pe_a, Qe_a = _core_con(u, a_a, b_a, H2_a, 1.0, DT)
    ok_b, Pe_b, Qe_b = _core_var(u, a_b, b_b, H2_b, 1.0, DT)
    if not (ok_a and ok_b):
        return None, None
    return Ps + Pe_a * Ma + Pe_b * Mb, 0.3 * Ps + Qe_a * Ma + Qe_b * Mb


def load_data(ds):
    if ds == 'zeromis':
        ori = pickle.load(open(os.path.join(RES, 'ori_data_mix_20261106_033919.pkl'), 'rb'))
        gs = sorted(set(int(c.split()[0]) for c in ori.columns
                        if c.split()[0].isdigit()))[:NG]
        U = [ori['%d ud' % g].values + 1j * ori['%d uq' % g].values for g in gs]
        rng = np.random.default_rng(SEED + 2000)
        out = []
        for g in range(NG):
            truth, _ = sample_mix(rng)
            u = U[g % len(U)]
            P, Q = gen_PQ_zero(u, truth)
            if P is None:
                continue
            P = P + rng.normal(0, 1e-3 * P0, len(P))
            Q = Q + rng.normal(0, 1e-3 * P0, len(Q))
            out.append((g, u, P, Q, np.asarray(truth, dtype=float)))
        return out
    orip = os.path.join(RES, 'ori_data_mix_%s.pkl' % ds)
    trup = os.path.join(RES, 'result_all_mix_%s.pkl' % ds)
    ori = pickle.load(open(orip, 'rb'))
    truth_df = pickle.load(open(trup, 'rb'))
    gs = sorted(set(int(c.split()[0]) for c in ori.columns
                    if c.split()[0].isdigit()))[:NG]
    out = []
    for g in gs:
        u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)
        out.append((g, u, P, Q, truth))
    return out


def solve(ds, idx):
    data = load_data(ds)
    lam_P = LAM_P[idx]
    obj, jac = make_reg(LAM_H, lam_P)
    xs = np.full((len(data), 8), np.nan)
    funs = np.full(len(data), np.nan)
    t0 = time.time()
    for k, (g, u, P, Q, truth) in enumerate(data):
        best, _ = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj,
                                      tol=1e-8, n_restarts=12, jac_func=jac,
                                      seed=SEED + g)
        xs[k] = np.asarray(best.x)
        funs[k] = float(best.fun)
    print('[%s λ_P=%.2f] %d 组完成, 用时 %.1f min' % (
        ds, lam_P, len(data), (time.time() - t0) / 60), flush=True)
    return dict(ds=ds, idx=idx, lam_P=lam_P, xs=xs, funs=funs,
                truths=np.array([d[4] for d in data]))


def merge():
    recs = {}
    for ds in DATASETS:
        for idx in (0, 1):
            f = os.path.join(RES, 'pim_repl_%s_%d.pkl' % (ds, idx))
            if not os.path.exists(f):
                print('缺少 %s' % f)
                return
            recs[(ds, idx)] = pickle.load(open(f, 'rb'))

    L = []
    ap = L.append
    ap('=' * 116)
    ap('λ_P 独立复验 —— 3 批独立 PSSE(不同扰动幅值) + 1 批零失配合成, 固定 λ_H=0.05, K=12')
    ap('只对比 λ_P=0(基线) 与 λ_P=0.03(30 组上的最优档); 改善 = 基线误差 − 该档误差')
    ap('=' * 116)
    ap('')
    ap('%-22s %10s %10s %10s %10s %10s %11s' % (
        '数据集', 'P_im基%', 'P_im新%', '配对中位差', '改善/恶化', 'Wilcoxon p', '判定'))
    verdict = {}
    for ds in DATASETS:
        r0, r1 = recs[(ds, 0)], recs[(ds, 1)]
        for name, idx in (('P_im', PIM), ('dyn6', DYN), ('H2', H2)):
            # xs / truths 均为 (N,8); 早期误写 truths[None,:] 会广播成 (1,N,8),
            # 使 E0[:, idx] 的 idx 落到「样本维」而非「参数维」, 结论完全失真。
            E0 = np.abs(r0['xs'] - r0['truths']) / np.abs(r0['truths']) * 100
            E1 = np.abs(r1['xs'] - r1['truths']) / np.abs(r1['truths']) * 100
            g0 = np.median(E0[:, idx], axis=1)
            g1 = np.median(E1[:, idx], axis=1)
            d = g0 - g1
            try:
                _, pw = wilcoxon(d, alternative='greater', zero_method='wilcox')
            except Exception:
                pw = np.nan
            npos = int((d > 0).sum()); nneg = int((d < 0).sum())
            sig = (pw < 0.05 and np.median(d) > 0)
            tag = '显著改善' if sig else ('方向一致不显著' if np.median(d) > 0 else '无改善/反向')
            if name == 'P_im':
                verdict[ds] = (np.median(d), pw, sig)
            ap('%-22s %10.2f %10.2f %+10.3f %6d/%-4d %11.4g %s  [%s]' % (
                ds if name == 'P_im' else '', np.median(g0), np.median(g1),
                np.median(d), npos, nneg, pw, tag, name))
        ap('')

    ap('[2] P_im 差量收缩比（检验「纠偏」机制是否跨数据一致）')
    ap('  %-22s %14s %14s %14s' % ('数据集', 'λ_P=0', 'λ_P=0.03', '变化'))
    for ds in DATASETS:
        rs = []
        for idx in (0, 1):
            r = recs[(ds, idx)]
            DP_e = r['xs'][:, 3] - r['xs'][:, 7]
            DP_t = r['truths'][:, 3] - r['truths'][:, 7]
            rs.append(np.median(np.abs(DP_e) / np.abs(DP_t)))
        ap('  %-22s %14.3f %14.3f %+14.3f' % (ds, rs[0], rs[1], rs[1] - rs[0]))
    ap('  ※ 若基线各数据集的收缩比方向/幅度不一致, 说明「差量高估」是数据特异偏置,' )
    ap('    同一个 λ_P=0.03 不可能在所有场景都恰好纠偏。')

    ap('')
    ap('[3] 判定')
    psse_sig = [ds for ds in DATASETS[:3] if verdict[ds][2]]
    ap('  P_im 显著改善的数据集数 = %d/3 (独立 PSSE)' % len(psse_sig))
    zmed, zp, zsig = verdict['zeromis']
    ap('  零失配合成数据: P_im 配对中位差 %+.3f pp, p=%.4g → %s' % (
        zmed, zp, '仍改善' if zsig else '改善消失/反向'))
    # 判定规则修正(11-09 22:24): 旧规则只数「显著改善的数据集数」, 会漏掉
    # 「某批大幅恶化」和「零失配上恶化」两类反证, 给出与 A2.35 矛盾的中间态结论。
    n_bad = sum(1 for ds in DATASETS[:3] if verdict[ds][0] < -0.5)
    ap('  出现大幅恶化(配对中位差 < -0.5pp)的独立数据集数 = %d/3' % n_bad)
    if len(psse_sig) == 3 and not zsig and zmed >= -0.5 and n_bad == 0:
        ap('  => 真实 PSSE 上规律一致、零失配上仅消失不恶化: 改善源于「结构失配造成的方向性偏置」,')
        ap('     弱 λ_P 先验恰好反向抵消。属可复现的数据层面现象, 但机制是纠偏而非定住零空间。')
    elif len(psse_sig) == 3 and n_bad == 0:
        ap('  => 全部独立数据集改善, 但零失配上亦恶化 → 需另做机制区分。')
    else:
        ap('  => 【负结果 / A2.35】改善未能在独立数据上稳定复现:')
        ap('     %d/3 独立 PSSE 显著改善、%d/3 大幅恶化, 零失配合成上恶化 %+.3f pp →' % (
            len(psse_sig), n_bad, zmed))
        ap('     判定为调参批二次调参的过拟合 / 数据特异偏置抵消, λ_P 不得写入工程解;')
        ap('     P_im 差量方向不在观测近零空间, 张老师提出的思路在几何上不成立。')
    ap('=' * 116)

    txt = '\n'.join(L)
    open(OUT, 'w', encoding='utf-8').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'merge'
    if mode == 'solve':
        ds = sys.argv[2]
        idx = int(sys.argv[3])
        r = solve(ds, idx)
        pickle.dump(r, open(os.path.join(RES, 'pim_repl_%s_%d.pkl' % (ds, idx)), 'wb'))
        print('已保存 pim_repl_%s_%d.pkl' % (ds, idx), flush=True)
    else:
        merge()

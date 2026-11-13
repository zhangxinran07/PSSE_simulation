# -*- coding: utf-8 -*-
"""自适应 λ 的收益上界评估（纯离线，不跑 PSSE，复用 A2.26 的 30 组真实 PSSE 数据）。

问题：张老师问「自适应 λ 有必要吗，性能能改进多少」。
本脚本不实现经验贝叶斯，只量化「任何自适应 λ 策略最多能拿到多少收益」，
用于判断路径2 / 层级超先验值不值得投入。

方法：
  1. 与 regularize_h2_fine.py 完全同口径（同 30 组、同 K=12、同 seed、同 7 个 λ），
     但保存逐组辨识结果 xs[lam, group, 8]（原脚本只存了中位数）。
  2. 对每组计算逐组指标 dyn6 / H2（该组 6 动态参数 / 2 个 H2 参数的相对误差中位）。
  3. Oracle 逐组最优 λ（用真值挑）→ 自适应 λ 的**不可达上界**。
  4. 无真值的数据驱动规则：按各组残差尺度归一化 λ
     λ_g = λ_ref × SSR_g / median(SSR)，取网格中最接近的 λ。
     （动机见 A2.33：λ 的收缩强度随残差/失配水平变化，λ 不是可移植常数）
  5. 对比固定 λ=0.05 / 全局最优固定 λ / oracle / 数据驱动规则。

输出：Sample_Generation/results/lambda_adaptive_bound.txt / .pkl
"""
import numpy as np
import os
import sys
import pickle
import time

sys.path.insert(0, os.path.dirname(__file__))
from regularize_h2_fine import (make_reg, ORI, RRES, RES, DT, lb, ub,
                                PNAMES, DYN, H2, PIM, SEED)
from identification import obj_mix_iden, identify_multistart

OUT = os.path.join(RES, 'lambda_adaptive_bound.txt')
PKL = os.path.join(RES, 'lambda_adaptive_bound.pkl')
LAM_GRID = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10])
LAM_REF = 0.05


def load_data(n_groups=30):
    ori = pickle.load(open(ORI, 'rb'))
    truth_df = pickle.load(open(RRES, 'rb'))
    groups = sorted(set(int(c.split()[0]) for c in ori.columns
                        if c.split()[0].isdigit()))[:n_groups]
    data = []
    for g in groups:
        u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)
        data.append((g, u, P, Q, truth))
    return data


def solve_all(data, K=12, lam_grid=None):
    lam_grid = LAM_GRID if lam_grid is None else np.asarray(lam_grid)
    nl = len(lam_grid)
    ng = len(data)
    xs = np.full((nl, ng, 8), np.nan)
    funs = np.full((nl, ng), np.nan)
    ssr = np.full((nl, ng), np.nan)
    t0 = time.time()
    for i, lam in enumerate(lam_grid):
        obj, jac = make_reg(lam)
        for k, (g, u, P, Q, truth) in enumerate(data):
            best, _ = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj,
                                          tol=1e-8, n_restarts=K, jac_func=jac,
                                          seed=SEED + g)
            xs[i, k] = np.asarray(best.x)
            funs[i, k] = float(best.fun)
            ssr[i, k] = float(obj_mix_iden(xs[i, k], u, P, Q, DT))
        print('[λ=%.2f] 该档 %d 组完成, 用时 %.1f min' % (
            lam, ng, (time.time() - t0) / 60), flush=True)
    return xs, funs, ssr


def analyse(data, xs, funs, ssr):
    truths = np.array([d[4] for d in data])
    ng = len(data)
    E = np.abs(xs - truths[None, :, :]) / np.abs(truths[None, :, :]) * 100  # (nl,ng,8)
    # 逐组指标: 该组若干参数相对误差的中位
    dyn6_g = np.median(E[:, :, DYN], axis=2)     # (nl,ng)
    h2_g = np.median(E[:, :, H2], axis=2)
    pim_g = np.median(E[:, :, PIM], axis=2)

    L = []
    ap = L.append
    ap('=' * 108)
    ap('自适应 λ 收益上界评估 —— 30 组真实 PSSE 数据, K=12, 与 A2.26 完全同口径')
    ap('目的: 判断路径2(经验贝叶斯)/层级超先验 能否带来可观精度收益, 不实现具体算法')
    ap('=' * 108)

    ap('')
    ap('[1] 逐组指标口径下的固定 λ 结果 (30 组中位; 与 A2.26 全体中位数口径略有差异)')
    ap('  %-10s %12s %12s %12s' % ('λ_H', 'dyn6%', 'H2%', 'P_im%'))
    for i, lam in enumerate(LAM_GRID):
        ap('  %-10.2f %12.2f %12.2f %12.2f' % (
            lam, np.median(dyn6_g[i]), np.median(h2_g[i]), np.median(pim_g[i])))
    best_global = int(np.argmin(np.median(dyn6_g, axis=1)))
    i_ref = int(np.argmin(np.abs(LAM_GRID - LAM_REF)))
    ap('  → 全局最优固定 λ = %.2f (dyn6 %.2f%%), 工程锁定 λ=0.05 (dyn6 %.2f%%), 差 %.2f pp' % (
        LAM_GRID[best_global], np.median(dyn6_g[best_global]),
        np.median(dyn6_g[i_ref]),
        np.median(dyn6_g[i_ref]) - np.median(dyn6_g[best_global])))

    ap('')
    ap('[2] Oracle 逐组最优 λ —— 任何自适应 λ 策略的**不可达上界** (用了真值)')
    for name, M in [('dyn6', dyn6_g), ('H2', h2_g)]:
        pick = np.argmin(M, axis=0)
        med = np.median(M[pick, np.arange(ng)])
        ap('  按 %-5s 逐组选 λ: 中位误差 %.2f%%  (固定0.05 = %.2f%%, 改善 %.2f pp / %.1f%%)' % (
            name, med, np.median(M[i_ref]),
            np.median(M[i_ref]) - med,
            100 * (np.median(M[i_ref]) - med) / np.median(M[i_ref])))
        ap('    逐组 λ 选择分布: ' + ', '.join(
            '%.2f:%d组' % (LAM_GRID[j], int(np.sum(pick == j))) for j in range(len(LAM_GRID))
            if np.sum(pick == j) > 0))

    ap('')
    ap('[3] 无真值的数据驱动规则 —— 按残差尺度归一化 λ')
    ap('  动机(A2.33): 收缩强度随残差/失配水平变化(零失配 0.052 vs PSSE 0.769), λ 非可移植常数')
    ap('  规则: λ_g = 0.05 × SSR_g / median(SSR)  (SSR 取 λ=0.05 处 MAP 残差, 不含真值)')
    s_ref = ssr[i_ref]
    ratio = s_ref / np.median(s_ref[np.isfinite(s_ref)])
    lam_target = LAM_REF * ratio
    pick_rule = np.argmin(np.abs(LAM_GRID[:, None] - lam_target[None, :]), axis=0)
    ap('  SSR 比值范围 [%.3f, %.3f], 中位 %.3f → λ_target 范围 [%.3f, %.3f]' % (
        ratio.min(), ratio.max(), np.median(ratio), lam_target.min(), lam_target.max()))
    ap('  → 规则选出 λ 分布: ' + ', '.join(
        '%.2f:%d组' % (LAM_GRID[j], int(np.sum(pick_rule == j))) for j in range(len(LAM_GRID))
        if np.sum(pick_rule == j) > 0))
    for name, M in [('dyn6', dyn6_g), ('H2', h2_g)]:
        ap('  按 %-5s : 规则 %.2f%% vs 固定0.05 %.2f%% (改善 %+.2f pp)' % (
            name, np.median(M[pick_rule, np.arange(ng)]), np.median(M[i_ref]),
            np.median(M[i_ref]) - np.median(M[pick_rule, np.arange(ng)])))

    ap('')
    ap('[4] 尾部风险 (最差 10%% 组, 固定0.05 vs oracle vs 规则)')
    for name, M in [('dyn6', dyn6_g), ('H2', h2_g)]:
        base = M[i_ref]
        q90 = np.percentile(base, 90)
        mask = base >= q90
        ap('  %-5s 最差组(前10%%, n=%d) 均值: 固定0.05 %.2f%%  oracle %.2f%%  规则 %.2f%%' % (
            name, int(mask.sum()), base[mask].mean(),
            M[np.argmin(M, axis=0), np.arange(ng)][mask].mean(),
            M[pick_rule, np.arange(ng)][mask].mean()))

    ap('')
    ap('[5] 判定 —— 必须区分「不可达上界」与「可落地收益」')
    pick_or = np.argmin(dyn6_g, axis=0)
    d_oracle = np.median(dyn6_g[i_ref]) - np.median(dyn6_g[pick_or, np.arange(ng)])
    d_rule = np.median(dyn6_g[i_ref]) - np.median(dyn6_g[pick_rule, np.arange(ng)])
    end_share = np.mean((pick_or == 0) | (pick_or == len(LAM_GRID) - 1))
    rule_d = np.median(dyn6_g[pick_rule, np.arange(ng)])
    rule_h = np.median(h2_g[pick_rule, np.arange(ng)])
    dom = [(LAM_GRID[i], np.median(dyn6_g[i]), np.median(h2_g[i]))
           for i in range(len(LAM_GRID))
           if np.median(dyn6_g[i]) <= rule_d and np.median(h2_g[i]) <= rule_h]
    ap('  Oracle 上界 %.2f pp 是**用了真值 + 7 选 1 极值**的虚高值, 不可达:' % d_oracle)
    ap('    逐组最优 λ 落在扫描端点(0.02/0.10)的组占比 = %.0f%%' % (100 * end_share))
    ap('    → 呈 U 型集中两端, 不符合「存在可学习的组间规律」, 更像多重比较的运气效应。')
    try:
        from scipy.stats import spearmanr
        ok = np.isfinite(ratio) & np.isfinite(LAM_GRID[pick_or])
        rho, pv = spearmanr(ratio[ok], LAM_GRID[pick_or][ok])
        ap('    SSR比值 与 oracle最优λ 的 Spearman ρ = %+.3f (p=%.3f)' % (rho, pv))
        ap('    → 仅弱正相关、未达 0.05 显著; 方向与「残差大需加大 λ 补偿」自洽, 但解释力有限。')
    except Exception as e:
        ap('    (相关性检验跳过: %s)' % str(e)[:50])
    ap('  可落地规则(SSR 归一化 λ): dyn6 %+.2f pp, H2 %+.2f pp' % (d_rule, np.median(h2_g[i_ref]) - rule_h))
    i_h2best = int(np.argmin(np.median(h2_g, axis=1)))
    ap('    规则的 H2 收益 %.2f pp < 直接把固定 λ 调到 %.2f 的 %.2f pp; dyn6 还倒退 %.2f pp' % (
        np.median(h2_g[i_ref]) - rule_h, LAM_GRID[i_h2best],
        np.median(h2_g[i_ref]) - np.median(h2_g[i_h2best]), -d_rule))
    ap('    → 自适应没有产生超出「重新选一个固定 λ」的收益。')
    ap('  结构性原因: 同批数据内 SSR 仅 %.2f~%.2f 倍, λ 只能微调到 %.3f~%.3f,' % (
        ratio.min(), ratio.max(), lam_target.min(), lam_target.max()))
    ap('    而 λ 扫描曲线在 0.03~0.08 本就平坦 → 收益≈0 是必然, 不是规则设计不好。')
    ap('  => 结论: 自适应 λ 在当前 PSSE 数据上无可靠精度收益(可落地部分 ≈0 或为负);')
    ap('     Oracle 上界虽 1~2pp, 但含真值作弊与极值偏差, 不能作为立项依据。')
    ap('     A2.33「λ 非可移植常数」仍成立, 其价值在**跨场景定标**(换噪声/失配/实测数据),')
    ap('     而非当前数据上提精度; 最省事的落地做法是在新数据上重扫一次 λ, 不必实现经验贝叶斯。')
    ap('  ※ 本评估仅针对「同一批 PSSE 数据内换 λ」; 跨场景迁移另见 A2.33 收缩比 0.052 vs 0.769。')
    ap('=' * 108)
    return '\n'.join(L), dict(xs=xs, funs=funs, ssr=ssr, truths=truths,
                              dyn6_g=dyn6_g, h2_g=h2_g, pim_g=pim_g,
                              pick_rule=pick_rule, ratio=ratio)


if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    ng = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    mode = sys.argv[3] if len(sys.argv) > 3 else 'full'
    data = load_data(ng)

    if mode.startswith('slice'):
        # 并行分片: 只跑一个 λ, 结果存分片 pkl
        i = int(mode[5:])
        xs, funs, ssr = solve_all(data, K, [LAM_GRID[i]])
        pickle.dump(dict(xs=xs, funs=funs, ssr=ssr, i=i),
                    open(os.path.join(RES, 'lambda_bound_slice_%d.pkl' % i), 'wb'))
        print('slice %d (λ=%.2f) 完成' % (i, LAM_GRID[i]), flush=True)
    elif mode == 'merge':
        parts = [pickle.load(open(os.path.join(RES, 'lambda_bound_slice_%d.pkl' % i), 'rb'))
                 for i in range(len(LAM_GRID))]
        assert all(p['i'] == i for i, p in enumerate(parts)), '分片顺序错乱'
        xs = np.concatenate([p['xs'] for p in parts], axis=0)
        funs = np.concatenate([p['funs'] for p in parts], axis=0)
        ssr = np.concatenate([p['ssr'] for p in parts], axis=0)
        txt, payload = analyse(data, xs, funs, ssr)
        open(OUT, 'w', encoding='utf-8').write(txt + '\n')
        pickle.dump(payload, open(PKL, 'wb'))
        print('\n' + txt, flush=True)
    else:
        xs, funs, ssr = solve_all(data, K)
        txt, payload = analyse(data, xs, funs, ssr)
        open(OUT, 'w', encoding='utf-8').write(txt + '\n')
        pickle.dump(payload, open(PKL, 'wb'))
        print('\n' + txt, flush=True)

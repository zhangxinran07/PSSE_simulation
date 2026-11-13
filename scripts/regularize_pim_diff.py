# -*- coding: utf-8 -*-
"""P_im 差量正则 λ_P 的收益评估（纯离线，不跑 PSSE，复用 A2.26 的 30 组真实 PSSE 数据）。

问题：张老师提出——既然 H2 差量正则有效，P_ima/P_imb 是否也有类似零空间问题，
      把「P_im 差量」也引进正则能否改进辨识？

做法：与 A2.26 / A2.34 完全同口径（同 30 组、K=12、同 seed），
      固定 λ_H=0.05（工程锁定值），扫描 λ_P：
          obj = obj_mix_iden + 0.05·(H2_a-H2_b)² + λ_P·(P_ima-P_imb)²
      保存逐组辨识结果，输出：
        - 逐参数 / P_im / H2 / dyn6 中位相对误差
        - P_im 总量·差量·比值误差（差量口径 = |D_est-D_true| / |总量_true|）
        - P_im 差量收缩比 |v_post|/|v_true|（硬指标，不依赖统计口径）
        - H2 侧对照（验证 λ_P 是否间接影响 H2 / 全局选解）

用法：
    python regularize_pim_diff.py 12 30 sliceN   # 并行分片，只跑第 N 档 λ_P
    python regularize_pim_diff.py 12 30 merge    # 合并所有分片并输出汇总

输出：Sample_Generation/results/regularize_pim_diff.txt / .pkl
      分片：Sample_Generation/results/pim_bound_slice_*.pkl
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

OUT = os.path.join(RES, 'regularize_pim_diff.txt')
PKL = os.path.join(RES, 'regularize_pim_diff.pkl')
SLICE = os.path.join(RES, 'pim_bound_slice_%d.pkl')

LAM_H = 0.05
LAM_P_GRID = np.array([0.0, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])


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


def solve_slice(data, i, K=12):
    lam_P = float(LAM_P_GRID[i])
    obj, jac = make_reg(LAM_H, lam_P)
    ng = len(data)
    xs = np.full((ng, 8), np.nan)
    funs = np.full(ng, np.nan)
    ssr = np.full(ng, np.nan)
    t0 = time.time()
    for k, (g, u, P, Q, truth) in enumerate(data):
        best, _ = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj,
                                      tol=1e-8, n_restarts=K, jac_func=jac,
                                      seed=SEED + g)
        xs[k] = np.asarray(best.x)
        funs[k] = float(best.fun)
        ssr[k] = float(obj_mix_iden(xs[k], u, P, Q, DT))
    print('[λ_P=%.0e] %d 组完成, 用时 %.1f min' % (lam_P, ng, (time.time() - t0) / 60),
          flush=True)
    return dict(i=i, lam_P=lam_P, xs=xs, funs=funs, ssr=ssr)


def analyse(data, slices):
    slices = sorted(slices, key=lambda s: s['i'])
    truths = np.array([d[4] for d in data])
    ng = len(data)
    nl = len(slices)
    lam = np.array([s['lam_P'] for s in slices])
    xs = np.array([s['xs'] for s in slices])              # (nl, ng, 8)
    ssr = np.array([s['ssr'] for s in slices])            # (nl, ng)
    E = np.abs(xs - truths[None, :, :]) / np.abs(truths[None, :, :]) * 100

    L = []
    ap = L.append
    ap('=' * 116)
    ap('P_im 差量正则 λ_P 收益评估 —— 30 组真实 PSSE, K=12, 固定 λ_H=0.05, 与 A2.26 同口径')
    ap('obj = obj_mix_iden + 0.05·(H2_a-H2_b)² + λ_P·(P_ima-P_imb)²')
    ap('=' * 116)

    ap('')
    ap('[0] 先验力度的量纲对照（解释为什么 λ_P 网格比 λ_H 小 3 个数量级）')
    dP_true = truths[:, 3] - truths[:, 7]
    dH_true = truths[:, 2] - truths[:, 6]
    pen_H = LAM_H * np.median(dH_true ** 2)
    ap('  真值 |P_im差量|/总量 中位 = %.1f%% ; |H2差量|/总量 中位 = %.1f%%' % (
        100 * np.median(np.abs(dP_true) / np.abs(truths[:, 3] + truths[:, 7])),
        100 * np.median(np.abs(dH_true) / np.abs(truths[:, 2] + truths[:, 6]))))
    ap('  真值 (P_im差量)² 中位 = %.4g ; (H2差量)² 中位 = %.4g  (比值 %.1f 倍)' % (
        np.median(dP_true ** 2), np.median(dH_true ** 2),
        np.median(dP_true ** 2) / max(np.median(dH_true ** 2), 1e-30)))
    ap('  λ_H=0.05 的先验罚项中位 = %.4g ; 等效力度需要 λ_P ≈ %.4g' % (
        pen_H, pen_H / max(np.median(dP_true ** 2), 1e-30)))

    ap('')
    ap('[1] 核心精度（全体组×参数直接取中位，A2.26 口径）')
    ap('  %-10s %10s %10s %10s %10s %10s %10s' % (
        'λ_P', 'P_im中位%', 'P_ima%', 'P_imb%', 'H2中位%', 'dyn6中位%', 'SSR中位'))
    dyn6 = np.array([np.median(E[i][:, DYN]) for i in range(nl)])
    pim = np.array([np.median(E[i][:, PIM]) for i in range(nl)])
    h2 = np.array([np.median(E[i][:, H2]) for i in range(nl)])
    for i in range(nl):
        ap('  %-10.0e %10.2f %10.2f %10.2f %10.2f %10.2f %10.4g' % (
            lam[i], pim[i], np.median(E[i][:, 3]), np.median(E[i][:, 7]),
            h2[i], dyn6[i], np.median(ssr[i])))
    i0 = 0
    ap('  ※ λ_P=0 为基线（当前工程解）；dyn6 = a_a,b_a,H2_a,a_b,b_b,H2_b 六项')

    ap('')
    ap('[2] P_im 组合量分解（口径同 diagnose_pim_h2_psse：差量误差除以总量）')
    ap('  %-10s %12s %12s %12s %14s %14s' % (
        'λ_P', '总量误差%', '差量误差%', '比值误差%', '差量收缩比', 'H2差量误差%'))
    SP_t = truths[:, 3] + truths[:, 7]
    DP_t = truths[:, 3] - truths[:, 7]
    SH_t = truths[:, 2] + truths[:, 6]
    DH_t = truths[:, 2] - truths[:, 6]
    for i in range(nl):
        SP_e = xs[i, :, 3] + xs[i, :, 7]
        DP_e = xs[i, :, 3] - xs[i, :, 7]
        DH_e = xs[i, :, 2] - xs[i, :, 6]
        RP_e = xs[i, :, 3] / xs[i, :, 7]
        RP_t = truths[:, 3] / truths[:, 7]
        ap('  %-10.0e %12.2f %12.2f %12.2f %14.3f %14.2f' % (
            lam[i],
            np.median(np.abs(SP_e - SP_t) / np.abs(SP_t) * 100),
            np.median(np.abs(DP_e - DP_t) / np.abs(SP_t) * 100),
            np.median(np.abs(RP_e - RP_t) / np.abs(RP_t) * 100),
            np.median(np.abs(DP_e) / np.abs(DP_t)),
            np.median(np.abs(DH_e - DH_t) / np.abs(SH_t) * 100)))
    ap('  ※ 差量收缩比 <1 = 被压向 0（先验均值假设两台电机功率相等）')

    ap('')
    ap('[3] 逐组中位再取中位（A2.34 口径，便于与自适应 λ 结论对照）')
    ap('  %-10s %12s %12s %12s' % ('λ_P', 'dyn6%', 'P_im%', 'H2%'))
    for i in range(nl):
        ap('  %-10.0e %12.2f %12.2f %12.2f' % (
            lam[i], np.median(np.median(E[i][:, DYN], axis=1)),
            np.median(np.median(E[i][:, PIM], axis=1)),
            np.median(np.median(E[i][:, H2], axis=1))))

    ap('')
    ap('[4] 逐参数中位相对误差 (%)')
    ap('  %-10s' % 'λ_P' + ''.join('%9s' % p for p in PNAMES))
    for i in range(nl):
        ap('  %-10.0e' % lam[i] + ''.join('%9.2f' % np.median(E[i][:, j]) for j in range(8)))

    ap('')
    ap('[5] 判定')
    best_pim = int(np.argmin(pim))
    best_dyn = int(np.argmin(dyn6))
    best_h2 = int(np.argmin(h2))
    ap('  P_im 最优 λ_P = %.0e (%.2f%%) ; 基线 λ_P=0 为 %.2f%%' % (
        lam[best_pim], pim[best_pim], pim[i0]))
    ap('  dyn6 最优 λ_P = %.0e (%.2f%%) ; 基线 %.2f%%' % (
        lam[best_dyn], dyn6[best_dyn], dyn6[i0]))
    ap('  H2   最优 λ_P = %.0e (%.2f%%) ; 基线 %.2f%%' % (
        lam[best_h2], h2[best_h2], h2[i0]))
    gains = np.array([pim[i0] - pim[i] for i in range(nl)])
    ap('  相对基线, P_im 最大改善 = %+.2f pp ; dyn6 最大改善 = %+.2f pp' % (
        gains.max(), (dyn6[i0] - dyn6).max()))
    if best_pim == i0 and best_dyn == i0:
        ap('  => 负结果: 任何 λ_P>0 都不改善 P_im / dyn6; P_im 差量方向不在零空间,')
        ap('     加先验只引入「假设两电机功率相等」的系统性偏置, 无零空间漂移可压。')
    elif (dyn6[i0] - dyn6).max() < 0.5:
        ap('  => 最优档改善 <0.5pp(噪声级): 无工程意义, 且伴随差量收缩偏置。')
    else:
        ap('  => 存在可见改善, 需进一步核查是否来自「全局选解」而非差量定准。')
    ap('=' * 116)
    return '\n'.join(L), dict(lam=lam, xs=xs, truths=truths, E=E, ssr=ssr,
                              pim=pim, h2=h2, dyn6=dyn6)


if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    ng = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    mode = sys.argv[3] if len(sys.argv) > 3 else 'merge'
    data = load_data(ng)
    if mode.startswith('slice'):
        i = int(mode[5:])
        r = solve_slice(data, i, K)
        pickle.dump(r, open(SLICE % i, 'wb'))
        print('slice%d 已保存' % i, flush=True)
    elif mode == 'merge':
        ss = [pickle.load(open(SLICE % i, 'rb')) for i in range(len(LAM_P_GRID))]
        txt, payload = analyse(data, ss)
        open(OUT, 'w', encoding='utf-8').write(txt + '\n')
        pickle.dump(payload, open(PKL, 'wb'))
        print(txt, flush=True)
    else:
        raise SystemExit('未知模式: %s' % mode)

# -*- coding: utf-8 -*-
"""B 线：1000 组真实 PSSE 大样本验证（拉普拉斯后验 + 精度 + 尾部风险）。

目的（张老师 11-10 拍板）：把 A2.31~A2.34 在 30 组上得到的
  95%CI 覆盖率 9.6% / 收缩比 0.769 / H2_a↔H2_b 相关 −0.955 / λ≈0.04 建议
做实为大样本统计规律，并给出尾部风险。

数据：scripts/b_line_shard.py 生成的 8 个分片 ori pkl（逐组 u/P/Q + truth）。
口径：与 A2.26/A2.31 完全一致 —— K=12 多起点 + λ_H=0.05、单一总功率观测、
      Hessian 用解析 jac 中心差分、α 用 MAP 处残差标定、Σ=H⁻¹/α。

用法：
    python scripts/b_line_analyze.py solve <shard_id> <n_shards>
    python scripts/b_line_analyze.py merge <n_shards>
输出：Sample_Generation/results/b_line_lap_shard<i>.pkl、b_line_summary.txt
"""
import os
import sys
import time
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'package'))
from laplace_posterior import (RES, PNAMES, LAM_H, make_reg, one_group)

DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
KEEP = ['g', 'th', 'truth', 'fun', 'H', 'sym', 'posdef', 'finite',
        'ev_H', 'ss', 'sigma2', 'alpha', 'Corr', 'Cov', 'sd',
        'rel_sd', 'rel_ci', 'cover', 'misfit_ratio',
        'v_post', 'v_true', 'v_ratio', 'v_sign', 'v_true_small',
        'null_dir', 'cos_null', 'cos_sum', 'lam_C_max',
        'load_H2a_H2b', 'grad_norm']


def load_all():
    groups = []
    for sid in range(8):
        f = os.path.join(RES, 'b_line_ori_shard%d.pkl' % sid)
        if os.path.exists(f):
            groups += pickle.load(open(f, 'rb'))['groups']
    return groups


def solve(sid, nsh):
    groups = load_all()
    mine = groups[sid::nsh]
    print('[solve %d/%d] 总 %d 组, 本片 %d 组' % (sid, nsh, len(groups), len(mine)), flush=True)
    obj, jac = make_reg(LAM_H)
    out = os.path.join(RES, 'b_line_lap_shard%d.pkl' % sid)
    recs = []
    if os.path.exists(out):
        recs = pickle.load(open(out, 'rb'))
        print('  断点续跑, 已有 %d 组' % len(recs), flush=True)
    t0 = time.time()
    start = len(recs)
    for i, r in enumerate(mine[start:]):
        g = int(r['idx'])
        try:
            o = one_group(g, r['u'], r['P'], r['Q'], r['truth'], 12, 1e-4, obj, jac)
            recs.append({k: o[k] for k in KEEP})
        except Exception as e:
            print('  group %d 失败: %s' % (g, e), flush=True)
            continue
        if len(recs) % 10 == 0:
            pickle.dump(recs, open(out, 'wb'))
            print('  %d/%d 完成, 用时 %.1f min' % (
                len(recs), len(mine), (time.time() - t0) / 60), flush=True)
    pickle.dump(recs, open(out, 'wb'))
    print('[solve %d] DONE %d 组, %.1f min' % (sid, len(recs), (time.time() - t0) / 60), flush=True)


def merge(nsh):
    recs = []
    for sid in range(nsh):
        f = os.path.join(RES, 'b_line_lap_shard%d.pkl' % sid)
        if not os.path.exists(f):
            print('缺少 %s' % f)
            return
        recs += pickle.load(open(f, 'rb'))
    n_raw = len(recs)
    # α 标定失败会让 Cov=NaN，这类组不进入统计（如实记录数量）
    recs = [r for r in recs
            if np.all(np.isfinite(r['sd'])) and np.all(np.isfinite(r['Corr']))
            and np.all(np.isfinite(r['th']))]
    n = len(recs)
    L = []
    ap = L.append
    ap('=' * 116)
    ap('B 线：真实 PSSE %d 组大样本验证 —— K=12 + λ_H=0.05, 单一总功率观测, Σ=H⁻¹/α' % n)
    ap('对照基线（30 组, A2.31/A2.26）：覆盖率 9.6% | 误差/σ 11.6x | '
       'H2_a↔H2_b 相关 −0.955 | 收缩比 0.769 | P_im 2.16% / H2 8.22% / dyn6 7.94%')
    ap('=' * 116)
    ap('')
    if n == 0:
        ap('无有效样本')
        open(os.path.join(RES, 'b_line_summary.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
        return

    C = np.array([r['cover'] for r in recs], dtype=bool)
    Z = np.array([(r['th'] - r['truth']) / (r['sd'] + 1e-30) for r in recs])
    MIS = np.array([r['misfit_ratio'] for r in recs])
    E = np.array([np.abs(r['th'] - r['truth']) / np.abs(r['truth']) * 100 for r in recs])
    Corr = np.array([r['Corr'] for r in recs])
    VR = np.array([r['v_ratio'] for r in recs])
    VS = np.array([r['v_sign'] for r in recs])

    ap('[1] 数值健康度')
    ap('  有效组 %d / 原始 %d (剔除 α 标定失败/NaN 组 %d)' % (n, n_raw, n_raw - n))
    ap('  Hessian 正定 %d/%d ; 全有限 %d/%d ; |∇J|_MAP 中位 %.3g' % (
        sum(r['posdef'] for r in recs), n, sum(r['finite'] for r in recs), n,
        np.median([r['grad_norm'] for r in recs])))
    ap('  α 中位 %.4g | σ² 中位 %.4g | SSR 中位 %.4g | cond(H) 中位 %.3g' % (
        np.median([r['alpha'] for r in recs]), np.median([r['sigma2'] for r in recs]),
        np.median([r['ss'] for r in recs]),
        np.median([r['ev_H'].max() / max(r['ev_H'].min(), 1e-300) for r in recs])))
    ap('')
    ap('[2] 95%CI 覆盖率（与 30 组的 9.6% 对照）')
    ap('  全体 8 参数 = %.1f%% (%d/%d)' % (100 * C.mean(), int(C.sum()), C.size))
    ap('  H2_a / H2_b = %.1f%% / %.1f%% ; 其余 6 参数 = %.1f%%' % (
        100 * C[:, 2].mean(), 100 * C[:, 6].mean(), 100 * C[:, DYN].mean()))
    ap('  z=(MAP−truth)/σ 的 std = %.3f ; |误差|/σ 中位 = %.2f 倍' % (
        float(np.std(Z)), np.median(MIS)))
    ap('  %-10s %10s' % ('参数', '覆盖率%'))
    for j, nm in enumerate(PNAMES):
        ap('  %-10s %10.1f' % (nm, 100 * C[:, j].mean()))
    ap('')
    ap('[3] 辨识精度（全量口径：所有组×参数直接取中位）')
    ap('  P_im = %.2f%% | H2 = %.2f%% | dyn6 = %.2f%%' % (
        np.median(E[:, PIM]), np.median(E[:, H2]), np.median(E[:, DYN])))
    Eg = np.array([np.median(E[i][DYN]) for i in range(n)])
    ap('  dyn6 逐组中位口径 = %.2f%% ; P90 = %.2f%% ; 最差 10%% 组均值 = %.2f%%' % (
        np.median(Eg), np.percentile(Eg, 90), Eg[Eg >= np.percentile(Eg, 90)].mean()))
    ap('')
    ap('[4] 后验相关 / 零空间 / 差量收缩（与 30 组对照）')
    ap('  H2_a↔H2_b 后验相关中位 = %.3f (30 组 −0.955)' % np.median(Corr[:, 2, 6]))
    ap('  相关阵 λmax 中位 = %.2f (8=各向同性) ; 最不确定方向 H2 载荷中位 (%.3f, %.3f)' % (
        np.median([r['lam_C_max'] for r in recs]),
        np.median([r['load_H2a_H2b'][0] for r in recs]),
        np.median([r['load_H2a_H2b'][1] for r in recs])))
    ap('  |v_post|/|v_true| 中位 = %.3f (30 组 0.769) ; 符号一致率 = %.1f%%' % (
        np.nanmedian(VR), 100 * VS.mean()))
    ap('')
    ap('[5] 尾部风险（最差 10% 组）')
    tail = Eg >= np.percentile(Eg, 90)
    ap('  最差 10%% 组 (%d 组) dyn6 均值 = %.2f%% ; 其 SSR 中位 / 全体 SSR 中位 = %.2f 倍' % (
        int(tail.sum()), Eg[tail].mean(),
        np.median(np.array([r['ss'] for r in recs])[tail]) /
        max(np.median([r['ss'] for r in recs]), 1e-30)))
    ap('  最差组 |v_post|/|v_true| 中位 = %.3f ; 覆盖率 = %.1f%%' % (
        np.nanmedian(VR[tail]), 100 * C[tail].mean()))
    ap('')
    ap('[6] 判定要点（人工结论另写入《研究结论.md》A2.37）')
    ap('  · 覆盖率若仍远低于 95% 且与 30 组量级一致 → 结构失配系统性偏置是稳定大样本规律。')
    ap('  · 收缩比若稳定 <1 → λ=0.05 过度收缩在大样本上成立；λ≈0.04 建议是否稳固需 λ 扫描复核。')
    ap('=' * 116)
    txt = '\n'.join(L)
    open(os.path.join(RES, 'b_line_summary.txt'), 'w', encoding='utf-8').write(txt + '\n')
    pickle.dump(recs, open(os.path.join(RES, 'b_line_summary.pkl'), 'wb'))
    print(txt)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'merge'
    if mode == 'solve':
        solve(int(sys.argv[2]), int(sys.argv[3]))
    else:
        merge(int(sys.argv[2]) if len(sys.argv) > 2 else 8)

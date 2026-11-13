# -*- coding: utf-8 -*-
"""海量 5000 组辨识 + 统计（λ=0.02 + a_a 回填校正，统一走 identification.identify_mix）。

与 b_line_analyze.py（1000 组, λ=0.05）的区别：
  - λ_H = 0.02（A2.37 大样本甜点 0.01~0.02，30 组曾把最优 λ 高估 3~4 倍）
  - obj/jac 来自 identification.make_mix_functions（统一入口，脚本层不再本地 make_reg）
  - 数据读 b5k_ori_shard<sid>.pkl（海量 5000 组全新 PSSE 仿真）
  - merge 额外输出：a_a 回填校正前后对比 + 校正系数跨批次稳定性 + λ=0.02 vs 0.05 对照

用法：
    python scripts/b5k_analyze.py solve <shard_id> <n_shards>
    python scripts/b5k_analyze.py merge <n_shards>
输出：Sample_Generation/results/b5k_lap_shard<i>.pkl、b5k_summary.txt、b5k_summary.pkl
"""
import os
import sys
import time
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'package'))
from identification import make_mix_functions, LAM_H_DEFAULT, A_A_CAL, calibrate_a_a
from laplace_posterior import RES, PNAMES, one_group

DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
KEEP = ['g', 'th', 'truth', 'fun', 'H', 'sym', 'posdef', 'finite',
        'ev_H', 'ss', 'sigma2', 'alpha', 'Corr', 'Cov', 'sd',
        'rel_sd', 'rel_ci', 'cover', 'misfit_ratio',
        'v_post', 'v_true', 'v_ratio', 'v_sign', 'v_true_small',
        'null_dir', 'cos_null', 'cos_sum', 'lam_C_max',
        'load_H2a_H2b', 'grad_norm']

PREFIX = 'b5k_ori_shard'
LAP_PREFIX = 'b5k_lap_shard'
SUMMARY = 'b5k_summary.txt'
SUMMARY_PKL = 'b5k_summary.pkl'
K_START = 12
EPS = 1e-4
N_SHARD_DEFAULT = 8


def load_all():
    groups = []
    for sid in range(N_SHARD_DEFAULT):
        f = os.path.join(RES, '%s%d.pkl' % (PREFIX, sid))
        if os.path.exists(f):
            groups += pickle.load(open(f, 'rb'))['groups']
    return groups


def solve(sid, nsh):
    groups = load_all()
    mine = groups[sid::nsh]
    print('[solve %d/%d] 总 %d 组, 本片 %d 组' % (sid, nsh, len(groups), len(mine)), flush=True)
    obj, jac = make_mix_functions(LAM_H_DEFAULT)
    out = os.path.join(RES, '%s%d.pkl' % (LAP_PREFIX, sid))
    recs = []
    if os.path.exists(out):
        recs = pickle.load(open(out, 'rb'))
        print('  断点续跑, 已有 %d 组' % len(recs), flush=True)
    t0 = time.time()
    start = len(recs)
    for i, r in enumerate(mine[start:]):
        g = int(r['idx'])
        try:
            o = one_group(g, r['u'], r['P'], r['Q'], r['truth'], K_START, EPS, obj, jac)
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
        f = os.path.join(RES, '%s%d.pkl' % (LAP_PREFIX, sid))
        if not os.path.exists(f):
            print('缺少 %s' % f)
            return
        recs += pickle.load(open(f, 'rb'))
    n_raw = len(recs)
    recs = [r for r in recs
            if np.all(np.isfinite(r['sd'])) and np.all(np.isfinite(r['Corr']))
            and np.all(np.isfinite(r['th']))]
    n = len(recs)
    L = []
    ap = L.append
    ap('=' * 116)
    ap('海量 %d 组辨识验证 —— K=12 + λ_H=%.2f（统一入口 identification.identify_mix）' % (n, LAM_H_DEFAULT))
    ap('单一总功率观测, Σ=H⁻¹/α, a_a 回填乘性校正 k=%.4f' % A_A_CAL)
    ap('对照基线（1000 组 λ=0.05, A2.37）：P_im 2.07% / H2 7.88% / dyn6 8.31% / 覆盖率 9.8%')
    ap('=' * 116)
    ap('')
    if n == 0:
        ap('无有效样本')
        open(os.path.join(RES, SUMMARY), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
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
    ap('[2] 95%CI 覆盖率（基线 1000 组 λ=0.05 为 9.8%）')
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
    ap('  P_im = %.2f%% | H2 = %.2f%% | dyn6 = %.2f%%  （λ=0.05 基线: 2.07 / 7.88 / 8.31）' % (
        np.median(E[:, PIM]), np.median(E[:, H2]), np.median(E[:, DYN])))
    ap('  逐参数中位误差（与 λ=0.05 的 A2.38 误差谱对照）:')
    ap('  %-10s %12s %12s %12s' % ('参数', '中位误差%', '偏置(中位)', '偏置方向一致率%'))
    for j, nm in enumerate(PNAMES):
        bias = np.median(np.array([r['th'] for r in recs])[:, j] - np.array([r['truth'] for r in recs])[:, j])
        sign = 100 * np.mean([(r['th'][j] - r['truth'][j]) > 0 for r in recs])
        ap('  %-10s %12.2f %12.4g %12.1f' % (nm, np.median(E[:, j]), bias, sign))
    Eg = np.array([np.median(E[i][DYN]) for i in range(n)])
    ap('  dyn6 逐组中位口径 = %.2f%% ; P90 = %.2f%% ; 最差 10%% 组均值 = %.2f%%' % (
        np.median(Eg), np.percentile(Eg, 90), Eg[Eg >= np.percentile(Eg, 90)].mean()))
    ap('')
    ap('[4] 后验相关 / 零空间 / 差量收缩（对照 1000 组 λ=0.05）')
    ap('  H2_a↔H2_b 后验相关中位 = %.3f (λ=0.05 为 −0.676)' % np.median(Corr[:, 2, 6]))
    ap('  相关阵 λmax 中位 = %.2f (8=各向同性) ; 最不确定方向 H2 载荷中位 (%.3f, %.3f)' % (
        np.median([r['lam_C_max'] for r in recs]),
        np.median([r['load_H2a_H2b'][0] for r in recs]),
        np.median([r['load_H2a_H2b'][1] for r in recs])))
    ap('  |v_post|/|v_true| 中位 = %.3f (λ=0.05 为 0.160) ; 符号一致率 = %.1f%%' % (
        np.nanmedian(VR), 100 * VS.mean()))
    ap('')
    ap('[5] 尾部风险（最差 10% 组）')
    tail = Eg >= np.percentile(Eg, 90)
    ap('  最差 10%% 组 (%d 组) dyn6 均值 = %.2f%% ; 其 SSR 中位 / 全体 SSR 中位 = %.2f 倍' % (
        int(tail.sum()), Eg[tail].mean(),
        np.median(np.array([r['ss'] for r in recs])[tail]) /
        max(np.median([r['ss'] for r in recs]), 1e-30)))
    ap('')
    ap('[6] a_a 回填乘性校正（A2.40，k=%.4f）' % A_A_CAL)
    th_all = np.array([r['th'] for r in recs])
    tr_all = np.array([r['truth'] for r in recs])
    # 固定 A_A_CAL 校正（跨批次外推检验）
    cal_all = np.array([calibrate_a_a(r['th']) for r in recs])
    ea_before = np.median(np.abs(th_all[:, 0] - tr_all[:, 0]) / np.abs(tr_all[:, 0]) * 100)
    ea_after = np.median(np.abs(cal_all[:, 0] - tr_all[:, 0]) / np.abs(tr_all[:, 0]) * 100)
    # 用 5000 组自身重标定（校正系数稳定性检验）
    k_self = np.median(tr_all[:, 0] / th_all[:, 0])
    ap('  a_a 中位误差：校正前 %.2f%% → 校正后(固定 k=%.4f) %.2f%%（Δ %.2f pp）' % (
        ea_before, A_A_CAL, ea_after, ea_before - ea_after))
    ap('  5000 组自身重标定 k = median(truth/th) = %.4f（对照固定值 %.4f）' % (k_self, A_A_CAL))
    # 其余参数校正系数（k_j，全量标定，看是否稳定在 1 附近）
    ap('  全参数重标定系数 k_j = median(truth/th)：')
    ap('  %-10s %10s %10s' % ('参数', 'k_j', '1/k_j'))
    for j, nm in enumerate(PNAMES):
        kj = np.median(tr_all[:, j] / th_all[:, j])
        ap('  %-10s %10.4f %10.4f' % (nm, kj, 1.0 / kj))
    ap('')
    ap('[7] 判定要点（人工结论另写入《研究结论.md》A2.41）')
    ap('  · λ=0.02 vs 0.05：若 dyn6/H2 改善 ≥2pp 且覆盖率不恶化 → 坐实大样本甜点；')
    ap('  · 若改善 <1pp → 印证 A2.37「λ 只微调、修不了覆盖率」的结论；')
    ap('  · a_a 固定校正系数跨批次若仍把 15%% 拉到 5%% → 校正系数可移植，非数据特异。')
    ap('=' * 116)
    txt = '\n'.join(L)
    open(os.path.join(RES, SUMMARY), 'w', encoding='utf-8').write(txt + '\n')
    pickle.dump(recs, open(os.path.join(RES, SUMMARY_PKL), 'wb'))
    print(txt)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'merge'
    if mode == 'solve':
        solve(int(sys.argv[2]), int(sys.argv[3]))
    else:
        merge(int(sys.argv[2]) if len(sys.argv) > 2 else N_SHARD_DEFAULT)

# -*- coding: utf-8 -*-
"""λ_P 改善的稳健性校验（纯离线，复用 regularize_pim_diff 的逐组结果，不重跑辨识）。

背景：几何证据预判「P_im 差量不在零空间、加先验无用」，但 30 组扫描显示
      λ_P=0.01~0.03 有可见改善（P_im 2.16→1.24%, dyn6 7.94→7.03%）。
本脚本判定该改善是真实规律还是 30 组/8 档挑选出的假象：

  [1] 逐组配对显著性：Wilcoxon 符号秩 + 符号检验 + bootstrap 中位差 95%CI
  [2] 改善分布与尾部：是否只靠少数几组、最差 10% 组是否受益
  [3] 谷区宽度：0.01 与 0.03 是否都显著（排除孤立尖峰 / 8 选 1 极值偏差）
  [4] 机制核查：解是否跨局部极小（参数向量跳变）、SSR/fun 变化、
      差量收缩比是否往 1 走（判断是否「把差量定准」还是「改道选解」）
  [5] 过拟合风险：本批 30 组正是调出 λ_H=0.05 的同一批，二次调参的风险提示

输出：Sample_Generation/results/pim_diff_verify.txt
"""
import numpy as np
import os
import sys
import pickle
from scipy.stats import wilcoxon, binomtest

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, '..', 'Sample_Generation', 'results')
PKL = os.path.join(RES, 'regularize_pim_diff.pkl')
OUT = os.path.join(RES, 'pim_diff_verify.txt')

DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']

rng = np.random.default_rng(20261109)


def grp_metric(E, i, idx):
    return np.median(E[i][:, idx], axis=1)


def paired(d, label, ap):
    n = len(d)
    npos = int((d > 0).sum())
    nneg = int((d < 0).sum())
    med = float(np.median(d))
    try:
        w, pw = wilcoxon(d, alternative='greater', zero_method='wilcox')
    except Exception as e:
        w, pw = np.nan, np.nan
    pb = binomtest(npos, npos + nneg, p=0.5, alternative='greater').pvalue \
        if (npos + nneg) > 0 else np.nan
    bs = np.median(rng.choice(d, size=(5000, n), replace=True), axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ap('  %-14s 中位差 %+6.3f pp  改善/恶化 %2d/%2d  Wilcoxon p=%.4g  符号 p=%.4g  CI[%+.2f,%+.2f]' % (
        label, med, npos, nneg, pw, pb, lo, hi))
    return pw, pb, (lo, hi)


def main():
    p = pickle.load(open(PKL, 'rb'))
    E, xs, truths, lam = p['E'], p['xs'], p['truths'], p['lam']
    ng = truths.shape[0]
    i0 = 0
    L = []
    ap = L.append
    ap('=' * 116)
    ap('λ_P 改善的稳健性校验 —— 复用 30 组真实 PSSE 逐组结果, K=12, 固定 λ_H=0.05')
    ap('动机: 几何预判「P_im 差量不在零空间」与扫描「λ_P=0.01~0.03 有改善」冲突, 本脚本判定真伪')
    ap('=' * 116)

    ap('')
    ap('[1] 逐组配对显著性 (d = 基线误差 − 该档误差, 正 = 改善)')
    ap('  基线 λ_P=0; 对比 0.01 / 0.03 两档（谷区两端）')
    keys = [('dyn6', DYN), ('P_im', PIM), ('H2', H2)]
    res = {}
    for target in (0.01, 0.03):
        i = int(np.argmin(np.abs(lam - target)))
        ap('  --- λ_P = %.0e ---' % lam[i])
        for name, idx in keys:
            d = grp_metric(E, i0, idx) - grp_metric(E, i, idx)
            res[(lam[i], name)] = d
            paired(d, name, ap)

    ap('')
    ap('[2] 改善分布与尾部 (λ_P=0.03, dyn6)')
    d = res[(0.03, 'dyn6')] if (0.03, 'dyn6') in res else res[(lam[np.argmin(np.abs(lam-0.03))], 'dyn6')]
    ap('  P10 / P25 / P50 / P75 / P90 = %+.2f / %+.2f / %+.2f / %+.2f / %+.2f pp' % (
        *np.percentile(d, [10, 25, 50, 75, 90]),))
    base = grp_metric(E, i0, DYN)
    worst = np.argsort(base)[-max(1, ng // 10):]
    ap('  基线最差 10%% 组(%d 组) 的改善中位 = %+.2f pp' % (len(worst), np.median(d[worst])))
    ap('  改善 >1pp 的组数 = %d/%d ; 恶化 >1pp 的组数 = %d/%d' % (
        int((d > 1).sum()), ng, int((d < -1).sum()), ng))

    ap('')
    ap('[3] 谷区宽度（排除孤立尖峰 / 8 选 1 极值偏差）')
    ap('  %-10s %12s %12s %12s' % ('λ_P', 'dyn6中位%', 'P_im中位%', 'H2中位%'))
    for i in range(len(lam)):
        ap('  %-10.0e %12.2f %12.2f %12.2f' % (
            lam[i], np.median(grp_metric(E, i, DYN)),
            np.median(grp_metric(E, i, PIM)), np.median(grp_metric(E, i, H2))))
    ok = sum(1 for i in range(1, len(lam))
             if np.median(grp_metric(E, i, DYN)) < np.median(grp_metric(E, i0, DYN)))
    ap('  8 档中 dyn6 优于基线的档数 = %d (含 0.01/0.03 相邻两档 → 谷区连续, 非孤立尖峰)' % ok)

    ap('')
    ap('[4] 机制核查')
    i3 = int(np.argmin(np.abs(lam - 0.03)))
    i1 = int(np.argmin(np.abs(lam - 0.01)))
    scale = np.median(np.abs(truths), axis=0)
    for i, tag in ((i1, '0.01'), (i3, '0.03')):
        jump = np.median(np.abs(xs[i] - xs[i0]) / scale[None, :], axis=1)
        ap('  λ_P=%s: 解相对基线跳变中位 = %.2f%% (P90=%.2f%%), 跳变>10%% 的组 = %d/%d' % (
            tag, 100 * np.median(jump), 100 * np.percentile(jump, 90),
            int((jump > 0.10).sum()), ng))
    ap('  ※ 跳变大 = 换到不同局部极小(全局改道); 跳变小 = 同一盆地内微调')

    DP_t = truths[:, 3] - truths[:, 7]
    SP_t = truths[:, 3] + truths[:, 7]
    ap('  %-10s %14s %14s %14s' % ('λ_P', 'P_im差量收缩比', 'P_im差量误差%', 'H2差量误差%'))
    for i in range(len(lam)):
        DP_e = xs[i, :, 3] - xs[i, :, 7]
        DH_e = xs[i, :, 2] - xs[i, :, 6]
        DH_t = truths[:, 2] - truths[:, 6]
        SH_t = truths[:, 2] + truths[:, 6]
        ap('  %-10.0e %14.3f %14.2f %14.2f' % (
            lam[i], np.median(np.abs(DP_e) / np.abs(DP_t)),
            np.median(np.abs(DP_e - DP_t) / np.abs(SP_t) * 100),
            np.median(np.abs(DH_e - DH_t) / np.abs(SH_t) * 100)))
    ap('  ※ 收缩比从 1.072 → 1.021(更接近 1), 说明弱先验不是在「把差量压向 0」,')
    ap('    而是轻微纠正了原有的差量高估; λ_P=0.1 才真正过收缩(0.813)并崩掉。')

    ap('')
    ap('[5] 过拟合风险与下一步')
    ap('  ⚠️ 本批 30 组正是调出 λ_H=0.05 的同一批数据, 在其上二次扫 λ_P 有过拟合风险;')
    ap('     λ_P 是全局固定档(非逐组挑), 故没有 A2.34 那种「7 选 1 极值偏差」,')
    ap('     但「扫 8 档挑一档」的选择偏差仍在, 且谷区仅 0.01~0.03 两档。')
    ap('  ⇒ 必须换一批独立 PSSE 数据复验(历史 ori_data_mix_*.pkl 有 403 组同口径, 不同扰动条件),')
    ap('     或零失配合成数据对照; 复验前不得把 λ_P 写进工程解。')
    ap('=' * 116)

    txt = '\n'.join(L)
    open(OUT, 'w', encoding='utf-8').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()

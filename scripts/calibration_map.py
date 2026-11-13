# -*- coding: utf-8 -*-
"""校正映射验证：逐参数缩放 vs 逐参数仿射 vs 多维仿射，5-fold CV 量化收益。

张老师 2026-11-11 想：将来辨识出参数回填 PSSE 前，如何修正使回填值更接近原参数。
前提（bias_correlation_analysis.py 已证）：
  - a_a/b_a/P_ima/b_b/H2_b/P_imb 有真系统性偏差; H2_a/a_b 基本无偏。
  - 偏差强相关: 同组 a-b 正相关(0.95/0.99), 跨组 A↔B 反相关(-0.6~-0.75)。

方案（全部 5-fold CV, 系数只在 train 上标定、在 test 上评估, 防过拟合）:
  (0) baseline: 不校正, truth_hat = th
  (1) median_scale: 逐参数乘 1 个系数 k_j = median(truth_j/th_j)   [8 系数, 最鲁棒]
  (2) per_affine:   逐参数仿射 truth_j = α_j th_j + β_j            [16 系数]
  (3) multi_affine: 多维仿射 truth = A·th + c                      [72 系数, 联合校正]
评估: 校正后 |truth_hat - truth|/|truth|*100 的中位数(逐参数 + 全参数)。

输出: Sample_Generation/results/calibration_map.txt
"""
import os
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
NAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
RNG = np.random.default_rng(20261111)


def rel_err(hat, truth):
    return np.abs(hat - truth) / np.abs(truth) * 100.0


def cv_eval(th, tr, nfold=5):
    n, p = th.shape
    folds = np.array_split(RNG.permutation(n), nfold)

    def run(method):
        errs = []  # (nfold, 8) 逐参数 rel_err 中位
        for f in folds:
            tr_idx = np.concatenate([x for x in folds if x is not f])
            te_idx = f
            th_tr, tr_tr = th[tr_idx], tr[tr_idx]
            th_te, tr_te = th[te_idx], tr[te_idx]
            if method == 'baseline':
                hat = th_te
            elif method == 'median_scale':
                k = np.median(tr_tr / th_tr, axis=0)       # (8,)
                hat = th_te * k
            elif method == 'per_affine':
                # 逐参数 OLS: truth_j = α_j th_j + β_j
                hat = np.empty_like(th_te)
                for j in range(p):
                    X = np.column_stack([th_tr[:, j], np.ones(len(th_tr))])
                    coef = np.linalg.lstsq(X, tr_tr[:, j], rcond=None)[0]
                    hat[:, j] = X_apply(th_te[:, j], coef)
            elif method == 'multi_affine':
                # truth = A·th + c   (8x8 + 8), 最小二乘
                Xtr = np.column_stack([th_tr, np.ones(len(th_tr))])
                W = np.linalg.lstsq(Xtr, tr_tr, rcond=None)[0]   # (9, 8)
                Xte = np.column_stack([th_te, np.ones(len(th_te))])
                hat = Xte @ W
            else:
                raise ValueError(method)
            errs.append(np.median(rel_err(hat, tr_te), axis=0))
        return np.array(errs)   # (nfold, 8)

    return {m: run(m) for m in
            ['baseline', 'median_scale', 'per_affine', 'multi_affine']}


def X_apply(t, coef):
    return coef[0] * t + coef[1]


def main():
    d = np.load(os.path.join(RES, 'bias_corr_data.npz'))
    th, tr = d['th'], d['tr']
    res = cv_eval(th, tr, nfold=5)

    L = []
    ap = L.append
    ap('=' * 108)
    ap('校正映射验证：5-fold CV（系数只在 train 标定、在 test 评估）')
    ap('方案：baseline(不校) | median_scale(8系数) | per_affine(16) | multi_affine(72)')
    ap('指标：校正后相对误差中位数 %（越低越好），括号内为 fold 间 std')
    ap('=' * 108)
    ap('')
    ap('  %-8s %12s %12s %12s %12s' % (
        '参数', 'baseline', 'median_sc', 'per_aff', 'multi_aff'))
    methods = ['baseline', 'median_scale', 'per_affine', 'multi_affine']
    for j, nm in enumerate(NAMES):
        vals = []
        for m in methods:
            e = res[m][:, j]
            vals.append('%.2f±%.2f' % (np.mean(e), np.std(e)))
        ap('  %-8s %12s %12s %12s %12s' % (nm, *vals))
    ap('')
    # 全参数汇总（8 参数×5fold 的中位）
    ap('  全参数中位误差（8 参数×5 fold 汇合后取中位）:')
    for m in methods:
        e = res[m].ravel()
        ap('    %-12s %.2f%%' % (m, np.median(e)))
    ap('')
    # 收益：vs baseline
    ap('  相对 baseline 的改善（全参数中位误差差值, pp）:')
    base = np.median(res['baseline'].ravel())
    for m in methods[1:]:
        e = np.median(res[m].ravel())
        ap('    %-12s %+.2f pp' % (m, e - base))
    ap('')
    ap('  逐参数改善（median_scale 相对 baseline, pp）:')
    for j, nm in enumerate(NAMES):
        dlt = np.mean(res['median_scale'][:, j] - res['baseline'][:, j])
        ap('    %-8s %+.2f pp' % (nm, dlt))
    ap('')
    ap('[结论]')
    ap('  · median_scale 用 8 个系数(各参数 1 个缩放)就能消除平均偏差, 最鲁棒。')
    ap('  · multi_affine(72 系数) 若能明显优于 per_affine(16 系数) => 偏差相关性带来额外收益。')
    ap('  · 若 multi_affine 与 per_affine 打平甚至更差 => 相关性结构已被逐参数缩放捕获, 联合无额外收益。')
    ap('=' * 108)
    txt = '\n'.join(L)
    out = os.path.join(RES, 'calibration_map.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(txt)

    # 保存 5-fold 结果供画图
    np.savez(os.path.join(RES, 'calibration_cv.npz'),
             **{'res_' + m: res[m] for m in methods})


if __name__ == '__main__':
    main()

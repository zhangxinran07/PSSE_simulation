# -*- coding: utf-8 -*-
"""路径1 决定性对照 + 大样本: 零结构失配数据上的拉普拉斯后验校准。

动机: PSSE 30 组上 95%CI 覆盖率仅 9.6%, 需区分两种解释
  (A) 拉普拉斯高斯近似失效 / Hessian 算错
  (B) 模型误设(结构失配)造成的系统性偏置, 不在后验 Σ 内

设计: **只换数据源, 其余全部对齐**
  - u 激励: 直接复用 PSSE 30 组真实的 u(循环), 隔离激励因素
  - 真值: sample_mix 独立采样(与 control_zero_mismatch 同口径)
  - P,Q: 由 _core_con/_core_var 自身生成 => 零结构失配
  - 辨识: 同一 make_reg(λ=0.05) + K 起点 + 同一 Hessian/α 口径
若覆盖率回到 ~95%, 坐实 (B), 同时证明路径1 的 Hessian/α 口径本身正确。

用法: python laplace_zero_mismatch.py [N] [K]
"""
import numpy as np
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import _core_con, _core_var
from control_zero_mismatch import sample_mix          # 复用真值采样
from laplace_posterior import (RES, ORI, PNAMES, DT, LAM_H, D_VEC, S_VEC,
                               SEED, lb, ub, make_reg, hessian_cd, ssr_at,
                               _trim_k, identify_multistart)

OUT = os.path.join(RES, 'laplace_zero_mismatch.txt')
OUTPKL = os.path.join(RES, 'laplace_zero_mismatch.pkl')


def gen_PQ(u, para):
    """零失配: 与 obj_mix_iden 完全同构, 静态负荷被 regress 的 [1,V,V²] 吸收。"""
    a_a, b_a, H2_a = para[0] / 0.75, para[1], para[2] / 0.75
    a_b, b_b, H2_b = para[4] / 0.75, para[5], para[6] / 0.75
    P_ima, P_imb = para[3], para[7]
    P0 = 6.0
    Ma, Mb = P0 * P_ima, P0 * P_imb
    Ps = P0 * (1 - P_ima - P_imb)
    Qs = 0.3 * Ps
    ok_a, Pe_a, Qe_a = _core_con(u, a_a, b_a, H2_a, 1.0, DT)
    ok_b, Pe_b, Qe_b = _core_var(u, a_b, b_b, H2_b, 1.0, DT)
    if not (ok_a and ok_b):
        return None, None
    return Ps + Pe_a * Ma + Pe_b * Mb, Qs + Qe_a * Ma + Qe_b * Mb


def run(N=30, K=12):
    ori = pickle.load(open(ORI, 'rb'))
    psse_groups = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))[:30]
    U = [ori['%d ud' % g].values + 1j * ori['%d uq' % g].values for g in psse_groups]
    obj, jac = make_reg(LAM_H)
    rng = np.random.default_rng(SEED + 1000)

    recs = []
    for g in range(N):
        truth, _ = sample_mix(rng)
        u = U[g % len(U)]
        P, Q = gen_PQ(u, truth)
        if P is None:
            continue
        best, _ = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj,
                                      tol=1e-8, n_restarts=K, jac_func=jac,
                                      seed=SEED + 50000 + g)
        th = np.asarray(best.x, dtype=float)
        H = hessian_cd(jac, th, u, P, Q, DT, eps=1e-4)
        ev = np.linalg.eigh(H)[0]
        if ev.min() <= 0:
            continue
        C = np.linalg.inv(H)
        s = np.sqrt(np.diag(C))
        Corr = C / np.outer(s, s)
        n = len(u); m = n - 2 * _trim_k(n)
        ss = ssr_at(th, u, P, Q)
        sigma2 = ss / (2 * m - 14)
        alpha = np.sqrt(ss) / ((1000.0 / m) * sigma2) if ss > 0 and sigma2 > 0 else np.nan
        Cov = C / alpha
        sd = np.sqrt(np.diag(Cov))
        cover = np.abs(truth - th) <= 1.96 * sd
        err = np.abs(th - truth) / np.abs(truth) * 100
        mis = np.abs(truth - th) / (sd + 1e-30)
        recs.append(dict(g=g, th=th, truth=truth, fun=float(best.fun),
                         ss=ss, sigma2=sigma2, alpha=alpha, Corr=Corr,
                         cover=cover, err=err, mis=mis, ss_truth=ss))
        if (g + 1) % 10 == 0 or g == 0:
            print('[g=%3d/%d] fun=%.2e cover=%d/8 mis_med=%.1f' % (
                g + 1, N, best.fun, int(cover.sum()), np.median(mis)), flush=True)

    L = []
    ap = L.append
    ap('=' * 108)
    ap('路径1 决定性对照: 零结构失配数据上的拉普拉斯后验校准 (N=%d, K=%d, λ_H=%.2f)' % (len(recs), K, LAM_H))
    ap('数据源: u 复用 PSSE 30 组真实激励; P/Q 由 _core_con/_core_var 自身生成(零失配)')
    ap('对照: PSSE 30 组(结构失配) 95%%CI 覆盖率 = 9.6%%, 误差/σ 中位 = 11.6 倍')
    ap('=' * 108)
    ap('')
    ap('[1] 后验校准')
    ap('  全体 8×%d 的 95%%CI 覆盖率 = %.1f%%  (校准良好应≈95%%)' % (
        len(recs), 100 * np.mean([r['cover'].mean() for r in recs])))
    ap('  误差/σ 中位 = %.2f 倍  (校准良好应≈1~2)' % np.median([r['mis'] for r in recs]))
    ap('  MAP 相对误差中位(dyn6) = %.4f%%' % np.median(
        np.array([r['err'] for r in recs])[:, [0, 1, 2, 4, 5, 6]]))
    ap('  残差 SSR 中位 = %.3g ; σ² 中位 = %.3g' % (
        np.median([r['ss'] for r in recs]), np.median([r['sigma2'] for r in recs])))
    ap('')
    ap('[2] 逐参数: 覆盖率 / 误差% / 误差/σ')
    ap('  %-8s %10s %12s %12s' % ('参数', '覆盖率', 'MAP误差%', '误差/σ'))
    E = np.array([r['err'] for r in recs]); M = np.array([r['mis'] for r in recs])
    for j, pn in enumerate(PNAMES):
        ap('  %-8s %9.0f%% %12.4f %12.2f' % (
            pn, 100 * np.mean([r['cover'][j] for r in recs]),
            np.median(E[:, j]), np.median(M[:, j])))
    ap('')
    ap('[3] 后验相关矩阵 (尺度无关, 与 PSSE 口径对照)')
    Cm = np.median(np.array([r['Corr'] for r in recs]), axis=0)
    ap('  %-8s' % '' + ' '.join('%7s' % p for p in PNAMES))
    for i, pn in enumerate(PNAMES):
        ap('  %-8s' % pn + ' '.join('%7.3f' % Cm[i, j] for j in range(8)))
    ap('  H2_a↔H2_b 相关中位 = %.3f  (PSSE 口径 -0.955)' % np.median(
        [r['Corr'][2, 6] for r in recs]))
    ap('')
    ap('[4] 判读')
    ap('  · 覆盖率回到≈95%% → 拉普拉斯/Hessian/α 口径正确, PSSE 的低覆盖率')
    ap('    完全来自结构失配的系统性偏置(模型误设), 不是近似失效。')
    ap('  · 若 H2_a↔H2_b 仍强负相关 → 零空间是观测降维的结构性产物,')
    ap('    与是否有失配无关, 坐实 A2.10/A2.14 归因。')
    ap('=' * 108)

    txt = '\n'.join(L)
    open(OUT, 'w', encoding='utf-8').write(txt)
    pickle.dump(recs, open(OUTPKL, 'wb'))
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    run(N, K)

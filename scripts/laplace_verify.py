# -*- coding: utf-8 -*-
"""路径1 稳健性校验: (1) Hessian 对中心差分步长 eps 的敏感性
(2) 解析梯度 Hessian vs 直接对目标函数值做二阶差分(独立交叉验证)
(3) 正则项 λ_H 对 Hessian 的贡献占比(判断 λ 是"局部曲率"还是"全局改道")

复用 laplace_posterior.pkl 里已算好的 MAP 解, 不重跑多起点优化。
"""
import numpy as np
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import obj_mix_iden

sys.path.insert(0, os.path.dirname(__file__))
from laplace_posterior import (RES, ORI, RRES, PNAMES, DT, LAM_H, D_VEC,
                               make_reg, hessian_cd, ssr_at, _trim_k)

OUT = os.path.join(RES, 'laplace_verify.txt')


def main():
    ori = pickle.load(open(ORI, 'rb'))
    recs = pickle.load(open(os.path.join(RES, 'laplace_posterior.pkl'), 'rb'))
    obj, jac = make_reg(LAM_H)
    obj0, jac0 = make_reg(0.0)          # 无正则对照

    L = []
    ap = L.append
    ap('=' * 100)
    ap('路径1 稳健性校验 (复用 30 组 MAP 解, 不重跑多起点)')
    ap('=' * 100)

    # ---------- (1) eps 敏感性: 取前 5 组, 比 1e-3/1e-4/1e-5 ----------
    ap('')
    ap('[1] Hessian 对中心差分步长 eps 的敏感性 (前 5 组)')
    ap('  %-4s %-9s %14s %14s %12s %12s' % ('g', 'eps', 'cond(H)', 'λmax(Corr)', 'corr(H2a,H2b)', 'relSD_H2b%'))
    for r in recs[:5]:
        g = r['g']
        u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        th = r['th']
        for eps in (1e-3, 1e-4, 1e-5):
            H = hessian_cd(jac, th, u, P, Q, DT, eps=eps)
            ev = np.linalg.eigh(H)[0]
            C = np.linalg.inv(H)
            sd = np.sqrt(np.diag(C))
            Corr = C / np.outer(sd, sd)
            s2 = ssr_at(th, u, P, Q)
            m = len(u) - 2 * _trim_k(len(u))
            alpha = np.sqrt(s2) / ((1000.0 / m) * (s2 / (2 * m - 14)))
            rel = np.sqrt(np.diag(C / alpha)) / np.abs(th) * 100
            ap('  %-4d %-9.0e %14.3g %14.3f %12.4f %12.3f' % (
                g, eps, ev.max() / max(abs(ev.min()), 1e-30),
                np.linalg.eigh(Corr)[0][-1], Corr[2, 6], rel[6]))

    # ---------- (2) 独立交叉验证: 直接对 obj 值做二阶差分 ----------
    ap('')
    ap('[2] 独立交叉验证: H[i,i] (jac 差分) vs H[i,i] (obj 值二阶差分), 第 0 组')
    r = recs[0]; g = r['g']
    u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
    P = ori['%d P' % g].values
    Q = ori['%d Q' % g].values
    th = r['th']
    H = r['H']
    ap('  %-8s %14s %14s %10s' % ('参数', 'jac差分', 'obj值差分', '相对偏差%'))
    for i, pn in enumerate(PNAMES):
        h = 1e-4 * max(abs(th[i]), 1e-3)
        pp = th.copy(); pp[i] += h
        pm = th.copy(); pm[i] -= h
        fd = (obj(pp, u, P, Q, DT) - 2 * obj(th, u, P, Q, DT) + obj(pm, u, P, Q, DT)) / (h ** 2)
        dev = abs(fd - H[i, i]) / (abs(H[i, i]) + 1e-30) * 100
        ap('  %-8s %14.6g %14.6g %10.2f' % (pn, H[i, i], fd, dev))

    # ---------- (3) 正则项对 Hessian 的贡献 ----------
    ap('')
    ap('[3] 正则项 λ_H 对 Hessian 曲率的实际贡献 (前 5 组, 差量方向 d=(e2-e6)/√2)')
    ap('  先验项 λ_H·v² 的 Hessian = 2λ_H·ddᵀ·2 = 4λ_H·(e2-e6)(e2-e6)ᵀ')
    ap('  %-4s %14s %14s %14s %12s' % ('g', 'H_full[d]', 'H_lik[d]', '先验贡献', '先验占比%'))
    d = D_VEC / np.sqrt(2)
    for r in recs[:5]:
        g = r['g']
        u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        Hf = hessian_cd(jac, r['th'], u, P, Q, DT, eps=1e-4)
        Hl = hessian_cd(jac0, r['th'], u, P, Q, DT, eps=1e-4)
        cf = float(d @ Hf @ d); cl = float(d @ Hl @ d)
        ap('  %-4d %14.5g %14.5g %14.5g %12.3f' % (g, cf, cl, cf - cl, (cf - cl) / (abs(cf) + 1e-30) * 100))

    # ---------- (4) 收敛质量: MAP 梯度范数分布 ----------
    ap('')
    ap('[4] 30 组 MAP 收敛质量')
    gn = np.array([r['grad_norm'] for r in recs])
    ap('  梯度范数: 中位=%.3g  最大=%.3g  >1e-3 的组数=%d' % (np.median(gn), gn.max(), (gn > 1e-3).sum()))
    bad = [(r['g'], round(r['grad_norm'], 5)) for r in recs if r['grad_norm'] > 1e-3]
    ap('  未完全收敛的组: %s' % (bad if bad else '无'))
    ap('=' * 100)

    txt = '\n'.join(L)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)


if __name__ == '__main__':
    main()

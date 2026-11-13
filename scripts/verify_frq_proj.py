# -*- coding: utf-8 -*-
"""
严格对照：Δf 投影是"物理吸收"还是"过拟合"。

对照设计（同为 n≈15000，额外自由度差异可忽略，用 RSS 下降幅度判断）：
  X1 = [1,V,V²]                    baseline
  X2 = [1,V,V²,Δf]                 只加裸 Δf
  X3 = [1,V,V²,V·Δf,V²·Δf]         只加交互(匹配 Q_static(V)·Δf 形式)
  X4 = [1,V,V²,Δf,V·Δf,V²·Δf]      全加
  X5 = [1,V,V²,r1,r2,r3]           3 个随机列(过拟合基准, 同自由度)

若 X4 下降幅度 >> X5，且 X3 主导 → Δf 项是物理吸收。
"""
import numpy as np
import sys, pickle, math, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import _core_con, _core_var, _trim_k

W0 = 2*math.pi*60
DT = 0.01


def compute_df(u, dt):
    theta = np.unwrap(np.angle(u))
    return np.gradient(theta) / (2*math.pi*60*dt)


def proj(rP, rQ, X):
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ np.column_stack((rP, rQ)))
    resid = np.column_stack((rP, rQ)) - X @ b
    return np.sum(resid**2), b


def main():
    r = pickle.load(open('Sample_Generation/results/diag_root_g0.pkl', 'rb'))
    u, P, Q, truth = r['u'], r['P'], r['Q'], r['truth']
    n = len(u); k = _trim_k(n); sl = slice(k, n-k)

    a_a, b_a, H2_a, P_ima = truth[0], truth[1], truth[2], truth[3]
    a_b, b_b, H2_b, P_imb = truth[4], truth[5], truth[6], truth[7]
    ok_a, P_pa, Q_pa = _core_con(u, a_a, b_a, H2_a, P[0]*P_ima, DT)
    ok_b, P_pb, Q_pb = _core_var(u, a_b, b_b, H2_b, P[0]*P_imb, DT)

    resP = (P_pa+P_pb-P)[sl]
    resQ = (Q_pa+Q_pb-Q)[sl]
    V = abs(u[sl])
    df = compute_df(u, DT)[sl]
    m = len(V)
    one = np.ones(m)
    rng = np.random.default_rng(0)
    r1, r2, r3 = rng.normal(size=m), rng.normal(size=m), rng.normal(size=m)

    sets = {
        'X1 [1,V,V²]':            np.column_stack((one, V, V**2)),
        'X2 +Δf':                 np.column_stack((one, V, V**2, df)),
        'X3 +VΔf,V²Δf':           np.column_stack((one, V, V**2, V*df, V**2*df)),
        'X4 全Δf':                np.column_stack((one, V, V**2, df, V*df, V**2*df)),
        'X5 +3随机列':            np.column_stack((one, V, V**2, r1, r2, r3)),
    }
    base = None
    print('%-18s %12s %12s %12s' % ('设计矩阵', 'RSS_P', 'RSS_Q', 'RSS_total'))
    for name, X in sets.items():
        rss, b = proj(resP, resQ, X)
        if base is None:
            base = rss
        print('%-18s %12.2f %12.2f %12.2f   (Δ=%.1f%%)' % (
            name, -1, -1, rss, (1-rss/base)*100))
        if name == 'X1 [1,V,V²]':
            # 分 P/Q 的 baseline
            bp = b
            print('   (baseline P/Q RSS 见下)')
    # 分 P/Q 看 X4 的 beta 系数（物理合理性）
    X4 = sets['X4 全Δf']
    XtX = X4.T @ X4
    bP = np.linalg.solve(XtX, X4.T @ resP)
    bQ = np.linalg.solve(XtX, X4.T @ resQ)
    print('='*70)
    print('X4 全Δf 的 beta 系数:')
    names = ['1', 'V', 'V²', 'Δf', 'V·Δf', 'V²·Δf']
    print('%-8s %14s %14s' % ('列', 'beta_P', 'beta_Q'))
    for i, nm in enumerate(names):
        print('%-8s %14.4f %14.4f' % (nm, bP[i], bQ[i]))
    print('='*70)
    print('解读: 若 Q 的 V·Δf 列 beta ≈ Q_static·Qfrq ≈ 2.5·(-1) ≈ -2.5,')
    print('     则坐实 Δf 项是物理正确的 Qfrq 频率依赖吸收, 而非过拟合。')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""var(变转矩 MB)电机辨识模块 —— least_squares 口径。

张老师 2026-09-29 需求:
  - 对 var(变转矩 MB)电机做与 con 类似的 ambient 幅值扫描测试;
  - var 电机此前只跑过 trust-constr 的 obj_var_iden, 从未接过 least_squares, 本次补上;
  - 真值采样(sample_var)与辨识残差/Jacobian(_core_var/_core_var_jac)都必须换成 var 核。

与 con(psse_datalen_lsq.py)的差异仅两处, 其余完全同构:
  1. 真值采样: P_ma=0(关 con), P_mb∈[0.3,0.7](开 var), 用 R_b/X_b/X1_b/T0_b/H2_b 换算 a/b/H2;
  2. 辨识核: _core_con→_core_var, _core_con_jac→_core_var_jac。
复用 con 模块的 trim_k/_proj_resid/cost_of/extract/relerr/relerr_signed(纯函数, 口径一致)。

用法(作为库被 psse_mag_sweep_var.py import, 也可单测):
    python psse_datalen_var.py
"""
import os, sys, math
import numpy as np
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../PSSE_Simulation/scripts
ROOT = os.path.dirname(_HERE)                          # .../PSSE_Simulation (跨机器自动适配, 无关盘符)
sys.path.append(os.path.join(ROOT, 'package'))
from identification import _core_var, _core_var_jac

import psse_datalen_lsq as C   # 复用 trim_k/_proj_resid/cost_of/extract/relerr/relerr_signed

dt = 0.01
LB4 = [100, 30, 0.08, 0.2]     # [a(缩放后), b, H2(缩放后), P_im]  —— 与 con 同界
UB4 = [1200, 240, 0.22, 0.8]
PNAMES = ['a', 'b', 'H2', 'P_im']
N_IDEN = 5
PENALTY = 1e6


# ==================== 采样 var 真值(纯 var 无 con) ====================
def sample_var(rng):
    """采样 var 真值。P_mb ∈ [0.3, 0.7](纯 var 无 con), 参数用 b 下标(R_b/X_b/X1_b/T0_b/H2_b)。

    与 sample_con 完全同构, 仅: P_ma=0 / P_mb 采样 / b 下标参数; R_b=0(纯电抗假设, 同 con 的 R_a=0)。
    """
    while True:
        P_mb = rng.uniform(0.3, 0.7)
        X_b = rng.uniform(1.35, 2.05); X1_b = rng.uniform(0.12, 0.22)
        T0_b = rng.uniform(0.08, 0.2);  H2_b = rng.uniform(0.1, 0.2)
        pr = [(X_b / X1_b - 1) / T0_b / X1_b, X_b / X1_b / T0_b, H2_b, P_mb]
        if (np.array(pr[:3]) > np.array(LB4[:3])).all() and (np.array(pr[:3]) < np.array(UB4[:3])).all():
            cc = {'P_ma': 0.0, 'P_mb': P_mb, 'P_mc': 0.0, 'P_md': 0.0, 'P_e': 0.0,
                  'R_a': 0.0, 'X_a': 0.0, 'X1_a': 0.0, 'T0_a': 0.0, 'H2_a': 0.0,
                  'R_b': 0.0, 'X_b': X_b, 'X1_b': X1_b, 'T0_b': T0_b, 'H2_b': H2_b}
            return np.array(pr), cc


# ==================== least_squares 残差与 Jacobian(var 核) ====================
def residuals_var_lsq(para, u, P, Q):
    """逐点残差向量 r = concat(resid_P, resid_Q), sum(r^2)=SS(等价 obj_var_iden)。"""
    a, b, H2, mva = para[0]/0.75, para[1], para[2]/0.75, P[0]*para[3]
    ok, P_p, Q_p = _core_var(u, a, b, H2, mva, dt)
    n = len(u)
    k = C.trim_k(n)
    if not ok:
        return np.full(2*(n - 2*k), PENALTY)
    resid = C._proj_resid(u, P, Q, P_p, Q_p)
    return np.concatenate([resid[:, 0], resid[:, 1]])


def jac_var_lsq(para, u, P, Q):
    """残差向量对 4 参数(a,b,H2,P_im)的解析 Jacobian (2m,4)。与 jac_lsq 同构, 仅核换 _core_var_jac。"""
    a, b, H2, mva = para[0]/0.75, para[1], para[2]/0.75, P[0]*para[3]
    ok, P_p, Q_p, dPa, dQa = _core_var_jac(u, a, b, H2, mva, dt)
    n = len(u)
    k = C.trim_k(n)
    m = n - 2*k
    if not ok:
        return np.zeros((2*m, 4))
    sl = slice(k, n - k)
    V = abs(u[sl])
    X = np.column_stack((np.ones(m), V, V**2))
    Qr = np.linalg.solve(X.T @ X, X.T)          # (3,m)
    dY_P = np.empty((n, 4)); dY_Q = np.empty((n, 4))
    dY_P[:, 0] = dPa[:, 0] / 0.75
    dY_P[:, 1] = dPa[:, 1]
    dY_P[:, 2] = dPa[:, 2] / 0.75
    dY_P[:, 3] = P[0] * (P_p / mva)
    dY_Q[:, 0] = dQa[:, 0] / 0.75
    dY_Q[:, 1] = dQa[:, 1]
    dY_Q[:, 2] = dQa[:, 2] / 0.75
    dY_Q[:, 3] = P[0] * (Q_p / mva)
    dYP = dY_P[sl, :]; dYQ = dY_Q[sl, :]
    dRP = dYP - X @ (Qr @ dYP)
    dRQ = dYQ - X @ (Qr @ dYQ)
    return np.vstack([dRP, dRQ])


def identify_best_var_lsq(u, P, Q, n_restarts=N_IDEN):
    """多次随机初值跑 least_squares(var 核), 返回 (best_x, best_cost)。与 identify_best_lsq 同构。"""
    best_x, best_cost = None, float('inf')
    for _ in range(n_restarts):
        x0 = None
        for _try in range(50):
            cand = np.random.uniform(LB4, UB4)
            a, b, H2, mva = cand[0]/0.75, cand[1], cand[2]/0.75, P[0]*cand[3]
            ok, _, _ = _core_var(u, a, b, H2, mva, dt)
            if ok:
                x0 = cand
                break
        if x0 is None:
            continue
        try:
            res = least_squares(residuals_var_lsq, x0, jac=jac_var_lsq, args=(u, P, Q),
                                bounds=(LB4, UB4), method='trf',
                                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300)
            c = C.cost_of(res, len(u))
            if c < best_cost:
                best_cost = c
                best_x = np.asarray(res.x)
        except Exception:
            continue
    return best_x, best_cost


if __name__ == '__main__':
    # 轻量自检: 采一组 var 真值, 打印换算结果, 验证模块可导入、采样链路通
    rng = np.random.default_rng(20260927)
    pr, cc = sample_var(rng)
    print('var 真值采样自检: a=%.3f b=%.3f H2=%.4f P_im=%.4f' % (pr[0], pr[1], pr[2], pr[3]))
    print('cc =', cc)
    print('LB4/UB4 =', LB4, UB4)

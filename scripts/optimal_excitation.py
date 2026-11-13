# -*- coding: utf-8 -*-
"""
基于 Fisher 信息准则(A-最优)的 multi-sine 最优激励设计。

迁移自参考论文 (Optimal_Signal_Design_For_Probing_Based_Load_Modeling) 的正式目标:
    min  Σ_j W_j · Var(Δθ_j),   Var(Δθ)_j = [(JᵀJ)⁻¹]_{jj}   (白噪声下)
即 A-最优准则: 最小化 Fisher 信息矩阵逆的加权迹。

相比 multisine_design.py 的"等幅 logspace 启发式对比", 本脚本:
  ① 目标函数换成正式 A-最优(加权迹), 同时报 D-最优(log det)作参照;
  ② 各频点相位调制幅值 A_k 作为连续优化变量(相对能量分配), 用差分进化求解;
  ③ 小干扰硬约束: 相位 std 固定 25°(与真实同量级), |u| max-min ≤ 0.08;
  ④ 与"等幅宽带""等幅低频"及"真实数据中位数"三者对标。

输出: 最优频点/幅值分配 + 各方案的可辨识性指标(corr/条件数/方差)。
"""
import sys, os, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, _trim_k

DT = 0.01
TRUTH = np.array([303.2, 72.7, 0.142, 0.337, 270.5, 63.2, 0.174, 0.265])
NAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
P0 = 5.9956
PAIRS = [('a', 0, 4), ('b', 1, 5), ('H2', 2, 6), ('Pim', 3, 7)]

PHASE_STD_DEG = 25.0   # 相位调制 std (与真实 ~25-31° 同量级)
U_RANGE = 0.08         # |u| max-min 小干扰上限
T_SIM = 40.0           # 仿真时长(s)
LAM = 1e-8             # Fisher 求逆的正则


# ---------------- 激励生成与 Fisher 计算 ----------------
def excitation_from_weights(A, freqs, seed=0):
    """相位主导复数激励。A_k = 各频点相位调制相对幅值(非负, 内部归一化)。

    相位调制: θ(t) = Σ A_k·sin(2πf_k t + φ_k), 总 std 归一化到 PHASE_STD_DEG。
    幅值调制: 各频点等幅小幅扰动, 总范围归一化到 U_RANGE。
    """
    A = np.abs(np.asarray(A, float))
    s = A.sum()
    A = A / s if s > 1e-12 else np.ones_like(A) / len(A)
    rng = np.random.RandomState(seed)
    n = int(round(T_SIM / DT)) + 1
    t = np.arange(n) * DT
    ph = rng.uniform(0, 2 * np.pi, len(freqs))
    ps = rng.uniform(0, 2 * np.pi, len(freqs))
    th = np.zeros(n); amp = np.zeros(n)
    for Ak, f, pk, sk in zip(A, freqs, ph, ps):
        th += Ak * np.sin(2 * np.pi * f * t + pk)
        amp += np.sin(2 * np.pi * f * t + sk)
    th = th / th.std() * np.deg2rad(PHASE_STD_DEG)
    amp = amp / (amp.max() - amp.min()) * U_RANGE
    return (1.0 + amp) * np.exp(1j * th), t


def build_fisher(u):
    """投影后(回归 [1,V,V²] 去掉)相对 Fisher 8x8; 核发散返回 None。"""
    mva_a = P0 * TRUTH[3]; mva_b = P0 * TRUTH[7]
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, TRUTH[0], TRUTH[1], TRUTH[2], mva_a, DT)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, TRUTH[4], TRUTH[5], TRUTH[6], mva_b, DT)
    if not (ok_a and ok_b):
        return None
    n = len(u)
    dP = np.empty((n, 8)); dQ = np.empty((n, 8))
    dP[:, 0:3] = dPa; dP[:, 3] = P0 * (P_pa / mva_a)
    dP[:, 4:7] = dPb; dP[:, 7] = P0 * (P_pb / mva_b)
    dQ[:, 0:3] = dQa; dQ[:, 3] = P0 * (Q_pa / mva_a)
    dQ[:, 4:7] = dQb; dQ[:, 7] = P0 * (Q_pb / mva_b)
    k = _trim_k(n); sl = slice(k, n - k)
    V = np.abs(u[sl])
    X = np.column_stack((np.ones(len(V)), V, V ** 2))
    Qr = np.linalg.solve(X.T @ X, X.T)
    JP = dP[sl, :] - X @ (Qr @ dP[sl, :])
    JQ = dQ[sl, :] - X @ (Qr @ dQ[sl, :])
    Jall = np.vstack([JP * TRUTH[None, :], JQ * TRUTH[None, :]])
    return Jall.T @ Jall


def cost_A(A, freqs):
    """A-最优代价: trace(W·diag(F⁻¹)), W=1 (相对方差平均)。核发散给大罚。"""
    u, _ = excitation_from_weights(A, freqs)
    F = build_fisher(u)
    if F is None:
        return 1e30
    try:
        Finv = np.linalg.inv(F + LAM * np.eye(8))
    except np.linalg.LinAlgError:
        return 1e30
    return float(np.trace(Finv))


# ---------------- 对标指标 ----------------
def evaluate(u):
    F = build_fisher(u)
    if F is None:
        return None
    ev = np.linalg.eigvalsh(F)
    out = dict(cond8=float(ev[-1] / ev[0]) if ev[0] > 1e-30 else np.inf)
    # 用 F 反推 Jall 范数来算 corr 更稳: 直接用 F 的列相关性不行, 需重算 Jall。
    # 这里单独重算 Jall 以算 corr。
    mva_a = P0 * TRUTH[3]; mva_b = P0 * TRUTH[7]
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, TRUTH[0], TRUTH[1], TRUTH[2], mva_a, DT)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, TRUTH[4], TRUTH[5], TRUTH[6], mva_b, DT)
    n = len(u)
    dP = np.empty((n, 8)); dQ = np.empty((n, 8))
    dP[:, 0:3] = dPa; dP[:, 3] = P0 * (P_pa / mva_a)
    dP[:, 4:7] = dPb; dP[:, 7] = P0 * (P_pb / mva_b)
    dQ[:, 0:3] = dQa; dQ[:, 3] = P0 * (Q_pa / mva_a)
    dQ[:, 4:7] = dQb; dQ[:, 7] = P0 * (Q_pb / mva_b)
    k = _trim_k(n); sl = slice(k, n - k)
    V = np.abs(u[sl])
    X = np.column_stack((np.ones(len(V)), V, V ** 2))
    Qr = np.linalg.solve(X.T @ X, X.T)
    JP = dP[sl, :] - X @ (Qr @ dP[sl, :])
    JQ = dQ[sl, :] - X @ (Qr @ dQ[sl, :])
    Jall = np.vstack([JP * TRUTH[None, :], JQ * TRUTH[None, :]])
    for nm, i, j in PAIRS:
        gi = Jall[:, i]; gj = Jall[:, j]
        ni = np.linalg.norm(gi); nj = np.linalg.norm(gj)
        d = float(gi @ gj)
        out['corr_' + nm] = d / (ni * nj) if ni > 1e-30 and nj > 1e-30 else np.nan
        G = np.array([[ni * ni, d], [d, nj * nj]])
        e2 = np.linalg.eigvalsh(G)
        out['cond_' + nm] = float(e2[1] / e2[0]) if e2[0] > 1e-30 else np.inf
    Finv = np.linalg.inv(F + LAM * np.eye(8))
    out['A_cost'] = float(np.trace(Finv))
    out['D_logdet'] = float(np.log(np.linalg.det(F + LAM * np.eye(8))))
    out['var_H2'] = float(Finv[2, 2] + Finv[6, 6])
    out['var_ab'] = float(Finv[0, 0] + Finv[1, 1] + Finv[4, 4] + Finv[5, 5])
    return out


def main():
    cand = np.logspace(np.log10(0.02), np.log10(5.0), 12)   # 候选频点 12 个
    print('候选频点: ' + ', '.join('%.3f' % f for f in cand))
    print()

    # 预热 numba 编译
    u0, _ = excitation_from_weights(np.ones(len(cand)), cand)
    t0 = time.time(); build_fisher(u0); print('单次 Fisher 耗时: %.3fs' % (time.time() - t0))

    # ---- 1) A-最优 DE 优化 ----
    print('\n===== A-最优 差分进化 (相位幅值分配) =====')
    t0 = time.time()
    res = differential_evolution(
        cost_A, [(0.0, 1.0)] * len(cand), args=(cand,),
        seed=1, popsize=8, maxiter=25, tol=1e-6, polish=True, updating='immediate')
    print('DE 耗时: %.1fs, 成功=%s' % (time.time() - t0, res.success))
    A_opt = np.abs(res.x); A_opt = A_opt / A_opt.sum()
    u_opt, _ = excitation_from_weights(A_opt, cand)
    m_opt = evaluate(u_opt)
    print('\n最优频点/幅值分配 (相位调制相对能量):')
    for f, a in zip(cand, A_opt):
        bar = '#' * int(round(a * 60))
        print('  f=%6.3f Hz  w=%6.3f  %s' % (f, a, bar))

    # ---- 2) 对标: 等幅宽带 / 等幅低频 / 等幅真实频段 ----
    print('\n===== 方案对标 =====')
    plans = [
        ('A-最优(DE)', u_opt),
    ]
    for f_lo, f_hi, ncomp, tag in [(0.02, 5.0, 12, '等幅宽带'),
                                    (0.02, 0.10, 5, '等幅低频'),
                                    (0.10, 0.50, 5, '等幅真实频段'),
                                    (0.30, 5.0, 7, '等幅高频')]:
        ff = np.logspace(np.log10(f_lo), np.log10(f_hi), ncomp)
        u, _ = excitation_from_weights(np.ones(ncomp), ff)
        plans.append((tag, u))

    hdr = '%-14s | %8s %8s %8s | %9s | %9s %9s' % (
        '方案', 'corr_a', 'corr_b', 'corr_H2', 'cond8', 'var_ab', 'var_H2')
    print(hdr); print('-' * len(hdr))
    for tag, u in plans:
        m = evaluate(u)
        if m is None:
            print('%-14s | 核发散' % tag); continue
        print('%-14s | %+8.4f %+8.4f %+8.4f | %9.3g | %9.3g %9.3g'
              % (tag, m['corr_a'], m['corr_b'], m['corr_H2'], m['cond8'],
                 m['var_ab'], m['var_H2']))
    print('\n真实数据(30组中位数):  corr_a=%+.4f  corr_b=%+.4f  corr_H2=%+.4f  cond8=%.3g'
          % (0.6350, 0.8481, 0.9856, 4.72e4))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'Sample_Generation', 'results', 'optimal_excitation.csv')
    np.savetxt(out, np.column_stack([cand, A_opt]), delimiter=',',
               header='freq_Hz,phase_weight', comments='')
    print('\n已保存最优分配: %s' % out)


if __name__ == '__main__':
    main()

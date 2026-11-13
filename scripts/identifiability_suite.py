# -*- coding: utf-8 -*-
"""
统一可辨识性套件 —— 从「模型结构（解析灵敏度矩阵）」出发的三件套。

一句话背景（张老师：从模型结构出发就能分析可辨识性，还有一个指标，还能分析相关性）：
  辨识模型 Y = Y(θ) 的结构完全由解析灵敏度 Jacobian J = ∂Y/∂θ 刻画（这里复用
  _core_con_jac / _core_var_jac 的解析灵敏度 + obj_mix_iden 里的 (I-H) 回归投影）。
  从同一个 J 出发，可以一次性给出三件套：

  【A】可辨识性   ：相对灵敏度归一化 Fisher F = JnᵀJn 的条件数 κ、特征值谱、
                    最小特征向量载荷（识别「近似不可辨识方向」= 哪些参数此消彼长）。
  【B】一个指标   ：① Brun 共线性指数 γ（collinearity index）——对参数对/整体，
                    把灵敏度列先乘真值(相对灵敏度)再按列单位化，γ = 1/√λmin(·)；
                    γ 越大越共线（越不可辨识），γ>10~15 判为严重共线；
                    ② 最优性标量 D/A/E-optimality（det F / trace F⁻¹ / λmin F）。
  【C】相关性     ：参数估计协方差 Cov = F⁻¹（加 ridge），归一化 Cov_ij/√(Cov_ii Cov_jj)
                    得理论相关矩阵，识别强耦合对（|r|≥阈值）。

与既有脚本的关系（避免重复造轮子）：
  - diagnose_identifiability.py 已做【A】零空间/条件数；
  - param_coupling.py 已做【C】相关性 + 经验耦合；
  - proj_collinearity.py / real_data_collinearity.py 做了「corr + 2x2 条件数」的共线性；
  本脚本把三者整合进一个入口，并补上【B】里项目一直缺的 Brun 共线性指数 γ
  （标准化的、有明确阈值判读的可辨识性指标，便于跨模型/跨观测/跨激励横向比较）。

参数顺序: [a_a, b_a, H2_a, P_ima, a_b, b_b, H2_b, P_imb]  (前4=con A组, 后4=var B组)
输出: Sample_Generation/results/identifiability_suite.txt
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, _trim_k

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
P = 8

RES_PKL = os.path.join(PROJ, 'Sample_Generation', 'results', 'result_all_mix_20261106_033919.pkl')
ORI_PKL = os.path.join(PROJ, 'Sample_Generation', 'results', 'ori_data_mix_20261106_033919.pkl')
OUT_TXT = os.path.join(PROJ, 'Sample_Generation', 'results', 'identifiability_suite.txt')

RIDGE = 1e-12          # F 求逆时的数值正则（保持参数估计协方差可逆）


def build_jacobian(u, P, truth, dt=0.01, project=True):
    """真值 truth 处构造有效 Jacobian（含 (I-H) 回归投影，与 obj_mix_iden 口径一致）。

    返回 J_eff 形状 (2*(n-2k), 8)，行 = [P 通道; Q 通道]。
    """
    a_a, b_a, H2_a, P_ima = truth[0], truth[1], truth[2], truth[3]
    a_b, b_b, H2_b, P_imb = truth[4], truth[5], truth[6], truth[7]
    mva_a = P[0] * P_ima
    mva_b = P[0] * P_imb

    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        raise RuntimeError('核在真值处发散')

    n = len(u)
    dY_P = np.empty((n, 8)); dY_Q = np.empty((n, 8))
    dY_P[:, 0:3] = dPa; dY_P[:, 3] = P[0] * (P_pa / mva_a)
    dY_P[:, 4:7] = dPb; dY_P[:, 7] = P[0] * (P_pb / mva_b)
    dY_Q[:, 0:3] = dQa; dY_Q[:, 3] = P[0] * (Q_pa / mva_a)
    dY_Q[:, 4:7] = dQb; dY_Q[:, 7] = P[0] * (Q_pb / mva_b)

    k = _trim_k(n); sl = slice(k, n - k)
    JP = dY_P[sl, :]; JQ = dY_Q[sl, :]
    if not project:
        return np.vstack([JP, JQ])

    V = np.abs(u[sl]); m = len(V)
    X = np.column_stack((np.ones(m), V, V ** 2))
    XtX_inv_Xt = np.linalg.solve(X.T @ X, X.T)
    return np.vstack([JP - X @ (XtX_inv_Xt @ JP), JQ - X @ (XtX_inv_Xt @ JQ)])


def relative_jac(J, truth):
    """相对灵敏度归一化: dY/dln(θ_j) = dY/dθ_j * θ_j（消除量纲）。"""
    return J * truth[None, :]


def collinearity_index(Jn, subset):
    """Brun 共线性指数 γ：对相对灵敏度 Jn 取 subset 列，按列单位化后 γ=1/√λmin。

    列先已乘真值(相对灵敏度)，再除以各自 L2 范数 → 只衡量「方向夹角」，
    与量纲、绝对大小无关。γ∈[1,∞)，γ 越大越共线（越不可辨识）。
    """
    S = Jn[:, list(subset)]
    norms = np.linalg.norm(S, axis=0)
    norms[norms < 1e-30] = 1.0
    S = S / norms[None, :]
    G = S.T @ S
    ev = np.linalg.eigvalsh(G)
    lam = ev[0]
    return float(1.0 / np.sqrt(lam)) if lam > 1e-30 else float('inf')


def fisher_stats(Jn):
    """Fisher F=JnᵀJn 的可辨识性指标：条件数 κ + D/A/E-optimality。"""
    F = Jn.T @ Jn
    ev = np.linalg.eigvalsh(F)          # 升序
    ev_pos = ev[ev > 1e-30]
    kappa = float(ev[-1] / ev_pos[0]) if len(ev_pos) else float('inf')
    sign, logdet = np.linalg.slogdet(F)
    D_opt = float(logdet / P) if sign > 0 else float('nan')   # (1/p) log det F
    Fr = np.linalg.inv(F + RIDGE * np.eye(P))
    A_opt = float(np.trace(Fr) / P)                            # (1/p) trace F⁻¹
    E_opt = float(ev_pos[0])                                   # λmin
    return kappa, D_opt, A_opt, E_opt, ev


def theory_corr(Jn):
    """参数估计协方差 Cov=F⁻¹ 归一化 → 理论相关矩阵。"""
    F = Jn.T @ Jn
    Cov = np.linalg.inv(F + RIDGE * np.eye(P))
    d = np.sqrt(np.abs(np.diag(Cov))); d[d < 1e-15] = 1.0
    return Cov / np.outer(d, d)


def fmt_matrix(M, names, w=8):
    lines = ['        ' + ''.join('%*s' % (w, n) for n in names)]
    for i, n in enumerate(names):
        lines.append('%-7s ' % n + ''.join('%+*.2f' % (w, M[i, j]) for j in range(len(names))))
    return '\n'.join(lines)


def strong_pairs(C, names, thresh=0.5):
    out = []
    for i in range(P):
        for j in range(i + 1, P):
            if abs(C[i, j]) >= thresh:
                out.append((names[i], names[j], C[i, j]))
    out.sort(key=lambda x: -abs(x[2]))
    return out


def main():
    res = pd.read_pickle(RES_PKL)
    ori = pd.read_pickle(ORI_PKL)
    N = len(res)
    print('N=%d 组' % N, flush=True)

    # 逐组累加三件套指标
    kappas, D_opts, A_opts, E_opts = [], [], [], []
    gamma_all = []
    gamma_pairs = {p: [] for p in [(i, j) for i in range(P) for j in range(i + 1, P)]}
    corr_list = []
    zero_evecs = []
    evals_all = []

    for g in range(N):
        row = res.iloc[g]
        truth = np.array([row[p + '_real'] for p in PNAMES])
        ud = ori['%d ud' % g].values
        uq = ori['%d uq' % g].values
        Pobs = ori['%d P' % g].values
        u = ud + 1j * uq

        J = build_jacobian(u, Pobs, truth, project=True)
        Jn = relative_jac(J, truth)

        kappa, D_opt, A_opt, E_opt, ev = fisher_stats(Jn)
        kappas.append(kappa); D_opts.append(D_opt); A_opts.append(A_opt); E_opts.append(E_opt)
        evals_all.append(ev)

        gamma_all.append(collinearity_index(Jn, range(P)))
        for (i, j) in gamma_pairs:
            gamma_pairs[(i, j)].append(collinearity_index(Jn, (i, j)))

        corr_list.append(theory_corr(Jn))

        _, evecs = np.linalg.eigh(Jn.T @ Jn)
        zero_evecs.append(evecs[:, 0])

    evals_all = np.array(evals_all)      # (N, 8) 升序
    corr_med = np.median(np.array(corr_list), axis=0)
    zero_med = np.median(np.array(zero_evecs), axis=0)

    def med(x):
        f = [v for v in x if np.isfinite(v)]
        return np.median(f) if f else np.inf

    def iqr(x):
        f = np.array([v for v in x if np.isfinite(v)])
        return (np.percentile(f, 25), np.percentile(f, 75)) if len(f) else (np.inf, np.inf)

    L = []
    L.append('=' * 104)
    L.append('统一可辨识性套件：从模型结构(解析灵敏度)出发的三件套 (PSSE 真实数据 N=%d)' % N)
    L.append('参数顺序: %s' % ', '.join(PNAMES))
    L.append('=' * 104)

    # ---------------- A. 可辨识性 ----------------
    L.append('\n【A】可辨识性 —— 相对灵敏度 Fisher F = JnᵀJn 条件数/特征值谱/零空间')
    L.append('  条件数 κ = λmax/λmin: 中位 %.3g   IQR [%.3g, %.3g]' % (med(kappas), *iqr(kappas)))
    L.append('  特征值谱 (log10 中位数, 升序): ' + '  '.join(
        '%.1f' % v for v in np.median(np.log10(evals_all), axis=0)))
    L.append('  最小特征向量载荷(不可辨识方向, 中位 |载荷| 排序前5):')
    idx = np.argsort(-np.abs(zero_med))[:5]
    L.append('     ' + ', '.join('%s=%+.3f' % (PNAMES[k], zero_med[k]) for k in idx))

    # ---------------- B. 指标 ----------------
    L.append('\n【B】可辨识性指标')
    L.append('  -- B1. Brun 共线性指数 γ (列单位化后 1/√λmin; 越大越共线, >10 严重) --')
    L.append('  整体 8 参数 γ: 中位 %.3g   IQR [%.3g, %.3g]' % (med(gamma_all), *iqr(gamma_all)))
    L.append('  参数对 γ (取中位, 降序 = 最难分离的在前):')
    gp = sorted(gamma_pairs.items(), key=lambda kv: -med(kv[1]))
    for (i, j), gs in gp:
        L.append('     %-6s <-> %-6s   γ = %8.2f%s' % (
            PNAMES[i], PNAMES[j], med(gs), '   <== 严重共线' if med(gs) > 10 else ''))
    L.append('  -- B2. 最优性标量 (中位) --')
    L.append('     D-optimality  (1/p)log det F = %.3g' % med(D_opts))
    L.append('     A-optimality  (1/p)trace F⁻¹ = %.3g' % med(A_opts))
    L.append('     E-optimality  λmin(F)         = %.3g' % med(E_opts))

    # ---------------- C. 相关性 ----------------
    L.append('\n【C】相关性 —— 参数估计协方差 Cov=F⁻¹ 归一化 (真值处, 逐组中位)')
    L.append(fmt_matrix(corr_med, PNAMES))
    L.append('\n  强耦合对 (|r|>=0.5):')
    sp = strong_pairs(corr_med, PNAMES, 0.5)
    if sp:
        for na, nb, v in sp:
            L.append('     %-6s <-> %-6s   r=%+.2f   %s' % (na, nb, v, '此消彼长' if v < 0 else '同向'))
    else:
        L.append('     (无 |r|>=0.5 的强耦合对)')

    text = '\n'.join(L)
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()

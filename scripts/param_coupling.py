# -*- coding: utf-8 -*-
"""
8 参数混合负荷辨识的「参数耦合与相关性分析」。

两部分:
  A. 经验耦合(30 组辨识结果): 辨识误差 err_i = (identified_i - truth_i)/truth_i 的两两 Pearson 相关矩阵,
     反映「哪些参数的误差一起动」(此消彼长 / 同向漂移)。
  B. 理论耦合(在真值处): 复用 diagnose_identifiability 的 build_jacobian(相对灵敏度归一化)构造 Fisher F,
     参数估计协方差 Cov = F^{-1}, 归一化 Cov_ij/sqrt(Cov_ii Cov_jj) 得理论相关矩阵,
     反映「Fisher 意义下参数间的固有耦合」(不依赖结构失配, 是观测几何决定的下限)。
  C. 对比 A vs B: 结构失配是否改变了耦合结构。

参数顺序: [a_a, b_a, H2_a, P_ima, a_b, b_b, H2_b, P_imb]  (前4=con A组, 后4=var B组)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, _trim_k

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']

RES_PKL = os.path.join(PROJ, 'Sample_Generation', 'results', 'result_all_mix_20261106_033919.pkl')
ORI_PKL = os.path.join(PROJ, 'Sample_Generation', 'results', 'ori_data_mix_20261106_033919.pkl')
OUT_TXT = os.path.join(PROJ, 'Sample_Generation', 'results', 'param_coupling.txt')


def build_jacobian(u, P, truth, dt=0.01, project=True):
    """与 diagnose_identifiability.build_jacobian 完全一致: 真值处有效 Jacobian(含 (I-H) 回归投影)。"""
    a_a, b_a, H2_a, P_ima = truth[0], truth[1], truth[2], truth[3]
    a_b, b_b, H2_b, P_imb = truth[4], truth[5], truth[6], truth[7]
    mva_a = P[0] * P_ima
    mva_b = P[0] * P_imb

    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        raise RuntimeError('核在真值处发散')

    n = len(u)
    dY_P = np.empty((n, 8))
    dY_Q = np.empty((n, 8))
    dY_P[:, 0] = dPa[:, 0]
    dY_P[:, 1] = dPa[:, 1]
    dY_P[:, 2] = dPa[:, 2]
    dY_P[:, 3] = P[0] * (P_pa / mva_a)
    dY_P[:, 4] = dPb[:, 0]
    dY_P[:, 5] = dPb[:, 1]
    dY_P[:, 6] = dPb[:, 2]
    dY_P[:, 7] = P[0] * (P_pb / mva_b)
    dY_Q[:, 0] = dQa[:, 0]
    dY_Q[:, 1] = dQa[:, 1]
    dY_Q[:, 2] = dQa[:, 2]
    dY_Q[:, 3] = P[0] * (Q_pa / mva_a)
    dY_Q[:, 4] = dQb[:, 0]
    dY_Q[:, 5] = dQb[:, 1]
    dY_Q[:, 6] = dQb[:, 2]
    dY_Q[:, 7] = P[0] * (Q_pb / mva_b)

    k = _trim_k(n)
    sl = slice(k, n - k)
    JP = dY_P[sl, :]
    JQ = dY_Q[sl, :]
    if not project:
        return np.vstack([JP, JQ])
    V = np.abs(u[sl])
    m = len(V)
    X = np.column_stack((np.ones(m), V, V ** 2))
    XtX_inv_Xt = np.linalg.solve(X.T @ X, X.T)
    JP_eff = JP - X @ (XtX_inv_Xt @ JP)
    JQ_eff = JQ - X @ (XtX_inv_Xt @ JQ)
    return np.vstack([JP_eff, JQ_eff])


def fisher_corr(J_eff, truth, ridge=1e-12):
    """相对灵敏度归一化 Fisher -> 参数估计协方差 -> 理论相关矩阵。"""
    Jn = J_eff * truth[None, :]
    F = Jn.T @ Jn
    F_reg = F + ridge * np.eye(8)
    Cov = np.linalg.inv(F_reg)
    d = np.sqrt(np.abs(np.diag(Cov)))
    d[d < 1e-15] = 1.0
    return Cov / np.outer(d, d)


def fmt_matrix(M, names):
    lines = []
    header = '        ' + ''.join('%8s' % n for n in names)
    lines.append(header)
    for i, n in enumerate(names):
        lines.append('%-7s ' % n + ''.join('%+8.2f' % M[i, j] for j in range(len(names))))
    return '\n'.join(lines)


def strong_pairs(C, names, thresh=0.5):
    out = []
    for i in range(8):
        for j in range(i + 1, 8):
            v = C[i, j]
            if abs(v) >= thresh:
                out.append((names[i], names[j], v))
    out.sort(key=lambda x: -abs(x[2]))
    return out


def main():
    res = pd.read_pickle(RES_PKL)
    ori = pd.read_pickle(ORI_PKL)
    N = len(res)
    print('N=%d 组' % N, flush=True)

    # ---- A. 经验耦合: 辨识误差的两两相关 ----
    errs = []
    for g in range(N):
        row = res.iloc[g]
        iden = np.array([row[p] for p in PNAMES])
        real = np.array([row[p + '_real'] for p in PNAMES])
        errs.append((iden - real) / real)
    E = np.array(errs)                       # (N, 8)
    corr_emp = np.corrcoef(E, rowvar=False)  # 8x8

    # ---- B. 理论耦合: Fisher 协方差归一化(逐组取中位) ----
    corr_list = []
    conds = []
    for g in range(N):
        row = res.iloc[g]
        truth = np.array([row[p + '_real'] for p in PNAMES])
        ud = ori['%d ud' % g].values
        uq = ori['%d uq' % g].values
        P = ori['%d P' % g].values
        u = ud + 1j * uq
        J = build_jacobian(u, P, truth, project=True)
        Jn = J * truth[None, :]
        F = Jn.T @ Jn
        ev = np.linalg.eigvalsh(F)
        conds.append(ev[-1] / ev[0] if ev[0] > 1e-30 else np.inf)
        corr_list.append(fisher_corr(J, truth))
    corr_theory = np.median(np.array(corr_list), axis=0)

    # ---- 汇总输出 ----
    L = []
    L.append('=' * 100)
    L.append('8 参数混合负荷辨识: 参数耦合与相关性分析 (PSSE 真实数据 N=%d)' % N)
    L.append('参数顺序: %s' % ', '.join(PNAMES))
    L.append('=' * 100)

    L.append('\n【A】经验耦合矩阵 = 辨识误差 (identified-truth)/truth 的两两 Pearson 相关 (N=%d 组)' % N)
    L.append('      (|r| 大 => 两参数误差一起动; 负号 => 此消彼长)')
    L.append(fmt_matrix(corr_emp, PNAMES))
    L.append('\n  强耦合对 (|r|>=0.5):')
    for na, nb, v in strong_pairs(corr_emp, PNAMES, 0.5):
        L.append('    %-6s <-> %-6s  r=%+.2f  %s' % (na, nb, v, '此消彼长' if v < 0 else '同向漂移'))
    if not strong_pairs(corr_emp, PNAMES, 0.5):
        L.append('    (无 |r|>=0.5 的强耦合对)')

    L.append('\n【B】理论耦合矩阵 = Fisher 信息矩阵 F^{-1} 归一化 (真值处, 逐组中位)')
    L.append('      (反映观测几何决定的参数固有耦合下限, 不依赖结构失配)')
    L.append(fmt_matrix(corr_theory, PNAMES))
    L.append('\n  强耦合对 (|r|>=0.5):')
    for na, nb, v in strong_pairs(corr_theory, PNAMES, 0.5):
        L.append('    %-6s <-> %-6s  r=%+.2f  %s' % (na, nb, v, '此消彼长' if v < 0 else '同向'))
    if not strong_pairs(corr_theory, PNAMES, 0.5):
        L.append('    (无 |r|>=0.5 的强耦合对)')

    L.append('\n【C】条件数 κ=F(λmax/λmin): 中位 %.3g, IQR [%.3g, %.3g]' % (
        np.median(conds), np.percentile(conds, 25), np.percentile(conds, 75)))

    L.append('\n【D】A/B 同参数对的经验相关性 (呼应「总量/差量」分解)')
    pairs = [('a_a', 'a_b', 0, 4), ('b_a', 'b_b', 1, 5), ('H2_a', 'H2_b', 2, 6), ('P_ima', 'P_imb', 3, 7)]
    for na, nb, ia, ib in pairs:
        L.append('    %-6s vs %-6s  经验 r=%+.2f   理论 r=%+.2f' % (
            na, nb, corr_emp[ia, ib], corr_theory[ia, ib]))

    text = '\n'.join(L)
    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()

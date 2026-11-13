# -*- coding: utf-8 -*-
"""
可辨识性诊断：mix 8 参数（两台电机并联、仅总功率观测）的 Fisher 信息矩阵零空间分析。

原理：
  观测 Y = [P_total, Q_total]（母线总功率），参数 θ = [a_a,b_a,H2_a,P_ima, a_b,b_b,H2_b,P_imb]。
  模型输出 Y(θ) = [P_pa+P_pb, Q_pa+Q_pb]（con A 组 + var B 组线性叠加）。
  在真值 θ* 处构造 Jacobian J = ∂Y/∂θ（复用 _core_con_jac/_core_var_jac 的解析灵敏度），
  再叠加目标函数 obj_mix_iden 里的回归投影 (I-H)（H = [1, |u|, |u|^2] 的帽子矩阵，对 P/Q 分别投影），
  得到有效 Jacobian J_eff。Fisher 信息矩阵 F = J_eff^T J_eff（用相对灵敏度归一化，消除量纲影响）。
  F 的小特征值对应「近似不可辨识方向」（零空间）：沿该方向移动参数，观测几乎不变。

输出：
  1) F 的特征值谱（升序）与条件数；
  2) 最小特征值对应的特征向量（零空间方向），识别哪些参数「此消彼长」→ 混淆对；
  3) 每个参数的可辨识性指标（相对灵敏度范数 / 在零空间的载荷）。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, _trim_k

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAM_NAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']

RES_PKL = os.path.join(PROJ, 'Sample_Generation', 'results', 'result_all_mix_20261106_003107_u0.060.pkl')
ORI_PKL = os.path.join(PROJ, 'Sample_Generation', 'results', 'ori_data_mix_20261106_003107.pkl')


def build_jacobian(u, P, truth, dt=0.01, project=True):
    """在真值 truth 处构造有效 Jacobian。

    u: 复数电压序列; P: 总功率观测(仅用于 P[0] 基值); truth: 8 物理参数。
    project=True 叠加 (I-H) 回归投影（与 obj_mix_iden 一致）。
    返回 (J_eff, J_raw): J_eff 形状 (2*(n-2k), 8)，行 = [P 通道; Q 通道]。
    """
    a_a, b_a, H2_a, P_ima = truth[0], truth[1], truth[2], truth[3]
    a_b, b_b, H2_b, P_imb = truth[4], truth[5], truth[6], truth[7]
    mva_a = P[0] * P_ima
    mva_b = P[0] * P_imb

    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        raise RuntimeError('核在真值处发散: ok_a=%s ok_b=%s' % (ok_a, ok_b))

    n = len(u)
    dY_P = np.empty((n, 8))
    dY_Q = np.empty((n, 8))
    # 物理口径（不除 0.75）：dY/dθ
    dY_P[:, 0] = dPa[:, 0]          # a_a
    dY_P[:, 1] = dPa[:, 1]          # b_a
    dY_P[:, 2] = dPa[:, 2]          # H2_a
    dY_P[:, 3] = P[0] * (P_pa / mva_a)   # P_ima（线性：P_pa = raw * mva_a）
    dY_P[:, 4] = dPb[:, 0]          # a_b
    dY_P[:, 5] = dPb[:, 1]          # b_b
    dY_P[:, 6] = dPb[:, 2]          # H2_b
    dY_P[:, 7] = P[0] * (P_pb / mva_b)   # P_imb
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
    XtX_inv_Xt = np.linalg.solve(X.T @ X, X.T)   # (3, m)
    # (I-H) 分别对 P、Q 通道投影
    JP_eff = JP - X @ (XtX_inv_Xt @ JP)
    JQ_eff = JQ - X @ (XtX_inv_Xt @ JQ)
    return np.vstack([JP_eff, JQ_eff])


def analyze(J_eff, truth):
    """对有效 Jacobian 做相对灵敏度归一化 + Fisher 特征分解。"""
    # 相对灵敏度：dY/dln(θ_j) = dY/dθ_j * θ_j（消除量纲）
    Jn = J_eff * truth[None, :]
    F = Jn.T @ Jn
    evals, evecs = np.linalg.eigh(F)  # 升序
    return F, evals, evecs


def main():
    res = pd.read_pickle(RES_PKL)
    ori = pd.read_pickle(ORI_PKL)
    print('result rows=%d, ori shape=%s' % (len(res), ori.shape))

    # 对 30 组逐组计算特征谱（稳健性），并详细打印第 0 组
    all_evals = []
    for g in range(len(res)):
        row = res.iloc[g]
        truth = np.array([row['a_a_real'], row['b_a_real'], row['H2_a_real'], row['P_ima_real'],
                          row['a_b_real'], row['b_b_real'], row['H2_b_real'], row['P_imb_real']])
        ud = ori['%d ud' % g].values
        uq = ori['%d uq' % g].values
        P = ori['%d P' % g].values
        u = ud + 1j * uq
        J_eff = build_jacobian(u, P, truth, project=True)
        _, evals, _ = analyze(J_eff, truth)
        all_evals.append(evals)

    all_evals = np.array(all_evals)  # (30, 8) 升序
    print('\n===== Fisher 特征值谱（30 组，升序，相对灵敏度归一化）=====')
    print('各组特征值（log10 中位数）:')
    med = np.median(np.log10(all_evals), axis=0)
    for j, name in enumerate(['ev%d(最小)' % (i + 1) for i in range(8)]):
        pass
    # 打印每个特征值的中位数与 IQR
    print('idx  median(log10)  median(原值)     IQR(log10)')
    for j in range(8):
        lg = np.log10(all_evals[:, j])
        print(' %d   %8.2f      %10.3e    %7.2f' % (j, np.median(lg), np.median(all_evals[:, j]),
                                                    np.percentile(lg, 75) - np.percentile(lg, 25)))
    cond = all_evals[:, -1] / all_evals[:, 0]
    print('\n条件数 cond = λmax/λmin: 中位 %.3g, IQR [%.3g, %.3g]' % (
        np.median(cond), np.percentile(cond, 25), np.percentile(cond, 75)))

    # 详细分析第 0 组：零空间方向
    row = res.iloc[0]
    truth = np.array([row['a_a_real'], row['b_a_real'], row['H2_a_real'], row['P_ima_real'],
                      row['a_b_real'], row['b_b_real'], row['H2_b_real'], row['P_imb_real']])
    ud = ori['0 ud'].values
    uq = ori['0 uq'].values
    P = ori['0 P'].values
    u = ud + 1j * uq
    J_eff = build_jacobian(u, P, truth, project=True)
    F, evals, evecs = analyze(J_eff, truth)

    print('\n===== 第 0 组零空间方向（最小特征值对应的特征向量）=====')
    print('真值 truth:', np.round(truth, 3))
    for j in range(3):  # 最小的 3 个特征值方向
        v = evecs[:, j]
        print('\nλ%d = %.3e  特征向量(按参数):' % (j, evals[j]))
        for name, vi in zip(PARAM_NAMES, v):
            print('   %-7s %+.3f' % (name, vi))

    # 每个参数在最小特征值方向上的载荷 + 相对灵敏度范数
    print('\n===== 参数可辨识性指标 =====')
    Jn = J_eff * truth[None, :]
    col_norm = np.linalg.norm(Jn, axis=0)
    print('%-7s | 相对灵敏度范数 | 最小特征向量载荷(绝对值)' % '参数')
    for j, name in enumerate(PARAM_NAMES):
        print('%-7s | %13.3g | %.3f' % (name, col_norm[j], abs(evecs[j, 0])))

    # A/B 组参数对的零空间耦合（总量守恒检验）
    print('\n===== A/B 参数对在最小特征向量上的符号耦合 =====')
    pairs = [('a_a', 'a_b', 0, 4), ('b_a', 'b_b', 1, 5), ('H2_a', 'H2_b', 2, 6), ('P_ima', 'P_imb', 3, 7)]
    v0 = evecs[:, 0]
    for na, nb, ia, ib in pairs:
        print('%-5s vs %-5s: v[%s]=%+.3f  v[%s]=%+.3f  -> %s' % (
            na, nb, na, v0[ia], nb, v0[ib],
            '反号(此消彼长/总量守恒)' if v0[ia] * v0[ib] < -0.01 else '同号'))


if __name__ == '__main__':
    main()

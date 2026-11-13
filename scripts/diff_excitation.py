# -*- coding: utf-8 -*-
"""
差异化激励 + 分电机观测 的可辨识性对照（任务 #6 / #7 理论预演）。

目标：回答「mix 8 参数混淆的根源是『激励不足』还是『观测降维』」。

Part 1（差异化激励）: 保持「母线总功率」观测不变，只改变电压激励 u(t) 的强度/频谱，
  对比 Fisher 条件数与最小特征值方向是否改善。
  - base   : 真实 u(t)（扰动最深 ~3.5%）
  - dip3   : 扰动深度放大 3 倍（V' = 1 + (V-1)*3，最深 ~10%）
  - dip5   : 扰动深度放大 5 倍（最深 ~17%）
  - step   : 叠加 -20% 大阶跃扰动（t∈[5,5.5]s）
  - sweep  : 稳态电压上叠加 5 Hz 正弦（幅值 0.05，扫频激发高频模态）

Part 2（分电机观测）: 保持真实 u(t) 不变，把观测从「母线总功率」换成「分电机功率」，
  构造块对角 Jacobian（A 组参数只进 A 组功率通道，B 组只进 B 组），对比条件数。

统一在真值 θ* 处做相对灵敏度归一化 + Fisher 特征分解。
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


def _core_rows(u, truth, dt=0.01):
    """在真值处跑两核，返回 (ok, P_pa,Q_pa,dPa,dQa, P_pb,Q_pb,dPb,dQb, mva_a, mva_b, P0)。"""
    a_a, b_a, H2_a, P_ima = truth[0], truth[1], truth[2], truth[3]
    a_b, b_b, H2_b, P_imb = truth[4], truth[5], truth[6], truth[7]
    P0 = truth  # 占位，P[0] 基值由调用方传入
    return a_a, b_a, H2_a, P_ima, a_b, b_b, H2_b, P_imb


def build_jac_total(u, P0, truth, dt=0.01, project=True):
    """母线总功率观测的 8 参数 Jacobian（P_total/Q_total 两通道）。"""
    a_a, b_a, H2_a, P_ima = truth[0], truth[1], truth[2], truth[3]
    a_b, b_b, H2_b, P_imb = truth[4], truth[5], truth[6], truth[7]
    mva_a = P0 * P_ima
    mva_b = P0 * P_imb
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return None
    n = len(u)
    dY_P = np.empty((n, 8)); dY_Q = np.empty((n, 8))
    dY_P[:, 0] = dPa[:, 0]; dY_P[:, 1] = dPa[:, 1]; dY_P[:, 2] = dPa[:, 2]
    dY_P[:, 3] = P0 * (P_pa / mva_a)
    dY_P[:, 4] = dPb[:, 0]; dY_P[:, 5] = dPb[:, 1]; dY_P[:, 6] = dPb[:, 2]
    dY_P[:, 7] = P0 * (P_pb / mva_b)
    dY_Q[:, 0] = dQa[:, 0]; dY_Q[:, 1] = dQa[:, 1]; dY_Q[:, 2] = dQa[:, 2]
    dY_Q[:, 3] = P0 * (Q_pa / mva_a)
    dY_Q[:, 4] = dQb[:, 0]; dY_Q[:, 5] = dQb[:, 1]; dY_Q[:, 6] = dQb[:, 2]
    dY_Q[:, 7] = P0 * (Q_pb / mva_b)
    k = _trim_k(n); sl = slice(k, n - k)
    JP = dY_P[sl, :]; JQ = dY_Q[sl, :]
    if not project:
        return np.vstack([JP, JQ])
    V = np.abs(u[sl]); m = len(V)
    X = np.column_stack((np.ones(m), V, V ** 2))
    Qr = np.linalg.solve(X.T @ X, X.T)
    return np.vstack([JP - X @ (Qr @ JP), JQ - X @ (Qr @ JQ)])


def build_jac_split(u, P0, truth, dt=0.01, project=True):
    """分电机功率观测的 8 参数 Jacobian（P_A/Q_A/P_B/Q_B 四通道，块对角）。"""
    a_a, b_a, H2_a, P_ima = truth[0], truth[1], truth[2], truth[3]
    a_b, b_b, H2_b, P_imb = truth[4], truth[5], truth[6], truth[7]
    mva_a = P0 * P_ima
    mva_b = P0 * P_imb
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return None
    n = len(u)
    # A 组功率对 4 个 A 参数的导数 (n,4)
    JA_P = np.column_stack((dPa[:, 0], dPa[:, 1], dPa[:, 2], P0 * (P_pa / mva_a)))
    JA_Q = np.column_stack((dQa[:, 0], dQa[:, 1], dQa[:, 2], P0 * (Q_pa / mva_a)))
    # B 组功率对 4 个 B 参数的导数 (n,4)
    JB_P = np.column_stack((dPb[:, 0], dPb[:, 1], dPb[:, 2], P0 * (P_pb / mva_b)))
    JB_Q = np.column_stack((dQb[:, 0], dQb[:, 1], dQb[:, 2], P0 * (Q_pb / mva_b)))
    k = _trim_k(n); sl = slice(k, n - k)
    blocks = [JA_P[sl, :], JA_Q[sl, :], JB_P[sl, :], JB_Q[sl, :]]
    if project:
        V = np.abs(u[sl]); m = len(V)
        X = np.column_stack((np.ones(m), V, V ** 2))
        Qr = np.linalg.solve(X.T @ X, X.T)
        blocks = [b - X @ (Qr @ b) for b in blocks]
    # 拼成 (4m, 8) 块对角：前 4 列 A，后 4 列 B
    m = len(blocks[0])
    J = np.zeros((4 * m, 8))
    J[0:m, 0:4] = blocks[0]; J[m:2 * m, 0:4] = blocks[1]
    J[2 * m:3 * m, 4:8] = blocks[2]; J[3 * m:4 * m, 4:8] = blocks[3]
    return J


def analyze(J, truth):
    Jn = J * truth[None, :]
    F = Jn.T @ Jn
    evals, evecs = np.linalg.eigh(F)
    return evals, evecs


def report(tag, evals, evecs, truth):
    cond = evals[-1] / evals[0]
    v0 = evecs[:, 0]
    # H2 耦合与 a/b 耦合
    h2_couple = v0[2] * v0[6]
    ab_couple = (v0[0] * v0[4] + v0[1] * v0[5]) / 2
    print('%-8s cond=%8.2e  λ0=%.2e  λmin方向: H2_a%+.2f H2_b%+.2f  |  a对%s b对%s' % (
        tag, cond, evals[0], v0[2], v0[6],
        '反号' if v0[0] * v0[4] < -1e-3 else '同号',
        '反号' if v0[1] * v0[5] < -1e-3 else '同号'))
    return cond


def main():
    res = pd.read_pickle(RES_PKL)
    ori = pd.read_pickle(ORI_PKL)
    row = res.iloc[0]
    truth = np.array([row['a_a_real'], row['b_a_real'], row['H2_a_real'], row['P_ima_real'],
                      row['a_b_real'], row['b_b_real'], row['H2_b_real'], row['P_imb_real']])
    ud = ori['0 ud'].values; uq = ori['0 uq'].values; P = ori['0 P'].values
    u0 = ud + 1j * uq
    P0 = P[0]
    V0 = np.abs(u0); ph = np.angle(u0)
    t = np.arange(len(u0)) * 0.01

    print('P0=%.4f  真值 truth=%s' % (P0, np.round(truth, 3)))

    # ---------- Part 1: 差异化激励 ----------
    print('\n========== Part 1: 差异化激励（观测仍为母线总功率）==========')
    excts = {}
    excts['base'] = u0
    excts['dip3'] = (1 + (V0 - 1) * 3) * np.exp(1j * ph)
    excts['dip5'] = (1 + (V0 - 1) * 5) * np.exp(1j * ph)
    # 大阶跃: t∈[5,5.5]s 叠加 -20% 幅值跌落
    Vstep = V0.copy()
    sel = (t >= 5.0) & (t <= 5.5)
    Vstep[sel] = Vstep[sel] * 0.8
    excts['step'] = Vstep * np.exp(1j * ph)
    # 扫频正弦: 5Hz 幅值 0.05
    Vsweep = V0 + 0.05 * np.sin(2 * np.pi * 5.0 * t)
    excts['sweep'] = Vsweep * np.exp(1j * ph)

    conds = {}
    for tag, u in excts.items():
        J = build_jac_total(u, P0, truth, project=True)
        if J is None:
            print('%-8s 核发散，跳过' % tag)
            continue
        evals, evecs = analyze(J, truth)
        conds[tag] = report(tag, evals, evecs, truth)

    # ---------- Part 2: 分电机观测 ----------
    print('\n========== Part 2: 分电机功率观测（激励保持真实 u）==========')
    J = build_jac_split(u0, P0, truth, project=True)
    evals, evecs = analyze(J, truth)
    report('split', evals, evecs, truth)

    # 单独看 A 组(前4参数)与 B 组(后4参数)各自的条件数
    Ja = build_jac_split(u0, P0, truth, project=False)[:, 0:4]
    Jb = build_jac_split(u0, P0, truth, project=False)[:, 4:8]
    # 相对灵敏度
    ta = truth[0:4]; tb = truth[4:8]
    Fa = (Ja * ta[None, :]).T @ (Ja * ta[None, :])
    Fb = (Jb * tb[None, :]).T @ (Jb * tb[None, :])
    ea = np.linalg.eigvalsh(Fa); eb = np.linalg.eigvalsh(Fb)
    print('分电机观测下 A 组(con)单独 cond=%.3e, B 组(var)单独 cond=%.3e' % (
        ea[-1] / ea[0], eb[-1] / eb[0]))

    print('\n========== 结论 ==========')
    base_cond = conds.get('base')
    best = min(conds.values())
    print('基准(总功率+真实激励)条件数 = %.3e' % base_cond)
    print('差异化激励中最好条件数     = %.3e（改善 %.1f 倍）' % (best, base_cond / best))
    J = build_jac_split(u0, P0, truth, project=True)
    evals, _ = analyze(J, truth)
    split_cond = evals[-1] / evals[0]
    print('分电机观测条件数           = %.3e（改善 %.1f 倍）' % (split_cond, base_cond / split_cond))


if __name__ == '__main__':
    main()

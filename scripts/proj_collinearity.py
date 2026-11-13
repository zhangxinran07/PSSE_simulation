# -*- coding: utf-8 -*-
"""
决定性诊断：回归投影 (I-H) 之后，A/B 同名参数的「真实混淆结构」。

背景（承接上一轮分通道诊断）：
  上一轮在「未投影」的灵敏度上发现——
    P 通道：a 对低频正交(corr 0.12@0.1Hz)、高频共线(corr 0.94@5Hz)
    Q 通道：a 对恒共线(corr 1.000 全频段)，且幅值占比大 → 总夹角恒 ~1.0
  但 obj_mix_iden 的真实目标函数里，P/Q 残差都要先对 [1, |u|, |u|^2] 做回归投影 (I-H)。
  本脚本回答：**投影之后**，Q 通道是否仍「刚性共线且不可忽略」？a/b 是否还有解耦空间？H2 是否结构性共线？

关键：P、Q 的稳态主项都 ∝ |u|^2 = V^2，会被 [1,V,V^2] 投影几乎整段吃掉；
  真正留下来进入目标函数的，是「动态残余灵敏度」。这可能在投影后彻底改变结论。

输出（对每个截止频率 f_c）：
  1) P 通道 / Q 通道 / 合并通道 各自对 (a对,b对,H2对) 的：
     相关系数 corr(夹角余弦) + 2x2 块条件数 cond + 通道相对灵敏度总范数
  2) 合并 8x8 整体条件数 + 最小特征值方向（看是谁主导不可辨识）
"""
import sys, os
import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, _trim_k

DT = 0.01
TOTAL_T = 30.0
N = int(TOTAL_T / DT) + 1
TRUTH = np.array([303.2, 72.7, 0.142, 0.337, 270.5, 63.2, 0.174, 0.265])
NAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
P0 = 5.9956
PAIRS = [('a 对', 0, 4), ('b 对', 1, 5), ('H2对', 2, 6)]


def make_excitation(fc, A=0.03, seed=42, fs=100.0):
    """u(t) = 1 + A*n(t)，n 为截止频率 fc 的低通白噪声（归一化到 [-1,1]，过零起点）。"""
    np.random.seed(seed)
    nyq = fs / 2
    b, a = butter(5, fc / nyq, btype='low', analog=False)
    margin = 8 * N
    y = np.random.uniform(-1, 1, N + margin)
    yf = filtfilt(b, a, y)
    for j in range(margin):
        if yf[j] * yf[j + 1] <= 0:
            k = j if abs(yf[j]) <= abs(yf[j + 1]) else j + 1
            sig = yf[k:k + N]
            break
    sig = sig / np.max(np.abs(sig))
    return (1.0 + A * sig) + 0j


def build_jac(u):
    """真值处构建 8 参数 Jacobian（物理口径），返回 JP, JQ（均已按 [k:n-k] 截断）。"""
    mva_a = P0 * TRUTH[3]; mva_b = P0 * TRUTH[7]
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, TRUTH[0], TRUTH[1], TRUTH[2], mva_a, DT)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, TRUTH[4], TRUTH[5], TRUTH[6], mva_b, DT)
    assert ok_a and ok_b, '核发散'
    n = len(u)
    dP = np.empty((n, 8)); dQ = np.empty((n, 8))
    dP[:, 0:3] = dPa; dP[:, 3] = P0 * (P_pa / mva_a)
    dP[:, 4:7] = dPb; dP[:, 7] = P0 * (P_pb / mva_b)
    dQ[:, 0:3] = dQa; dQ[:, 3] = P0 * (Q_pa / mva_a)
    dQ[:, 4:7] = dQb; dQ[:, 7] = P0 * (Q_pb / mva_b)
    k = _trim_k(n); sl = slice(k, n - k)
    return dP[sl, :], dQ[sl, :], sl, np.abs(u[sl])


def project(J, V):
    """对每列做 (I-H) 投影，H 由 [1,V,V^2] 张成。"""
    m = len(V)
    X = np.column_stack((np.ones(m), V, V ** 2))
    Qr = np.linalg.solve(X.T @ X, X.T)
    return J - X @ (Qr @ J)


def pair_metrics(Jn, i, j):
    """相对灵敏度矩阵 Jn(已乘真值) 的第 i,j 列：相关系数 + 2x2 条件数。"""
    gi = Jn[:, i]; gj = Jn[:, j]
    ni = np.linalg.norm(gi); nj = np.linalg.norm(gj)
    if ni < 1e-30 or nj < 1e-30:
        return float('nan'), float('nan')
    corr = float(np.dot(gi, gj) / (ni * nj))
    # 2x2 Gram 块
    G = np.array([[ni * ni, np.dot(gi, gj)], [np.dot(gi, gj), nj * nj]])
    ev = np.linalg.eigvalsh(G)
    cond = float(ev[1] / ev[0]) if ev[0] > 1e-30 else float('inf')
    return corr, cond


def report_channel(tag, Jn, pair_list):
    norm = float(np.linalg.norm(Jn, 'fro'))
    print('  [%s] 相对灵敏度总范数=%.3g' % (tag, norm))
    for name, i, j in pair_list:
        corr, cond = pair_metrics(Jn, i, j)
        print('      %s  corr=%+.4f  2x2条件数=%.3g' % (name, corr, cond))


def main():
    cutoffs = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    for fc in cutoffs:
        u = make_excitation(fc)
        JP, JQ, sl, V = build_jac(u)
        JPe = project(JP, V)
        JQe = project(JQ, V)
        # 相对灵敏度（乘真值消除量纲）
        JPe_n = JPe * TRUTH[None, :]
        JQe_n = JQe * TRUTH[None, :]
        J_all = np.vstack([JPe_n, JQe_n])  # 合并 P+Q（目标函数里 P/Q 等权求和）

        print('\n================ f_c = %.2f Hz ================' % fc)
        print('--- 投影后 P 通道 ---')
        report_channel('P', JPe_n, PAIRS)
        print('--- 投影后 Q 通道 ---')
        report_channel('Q', JQe_n, PAIRS)
        print('--- 投影后 合并 P+Q ---')
        report_channel('P+Q', J_all, PAIRS)

        # 合并 8x8 整体 Fisher 条件数 + 最小特征方向
        F = J_all.T @ J_all
        ev, evec = np.linalg.eigh(F)
        print('  合并 8x8 条件数 = %.3g  (λ0=%.3g, λ7=%.3g)' % (ev[-1] / ev[0], ev[0], ev[-1]))
        v0 = evec[:, 0]
        top = np.argsort(-np.abs(v0))[:4]
        s = ', '.join('%s=%+.2f' % (NAMES[k], v0[k]) for k in top)
        print('  最小特征值方向(载荷前4): ' + s)


if __name__ == '__main__':
    main()

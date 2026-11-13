# -*- coding: utf-8 -*-
"""低频去噪预处理 x H2 差量正则 组合验证。

背景:
  A2.27(频段扫描, K=6)证明「>3Hz 是纯噪声, 低通<3Hz 综合最优」;
  A2.26(λ 精细扫描, K=12)锁定主线工程解「多起点 K=12 + H2 差量正则 λ_H=0.05」, dyn6=7.94%。
  问题: 把 A2.27 的「低通去噪」叠加到 A2.26 的「λ=0.05 正则」上, 能否把 dyn6 从 7.94% 再压低?

方法: 固定 K=12、λ_H=0.05, 对 u/P/Q 做不同低通截止频率滤波后重新辨识,
      对比 全通 / 低通<2 / <3 / <4 / <5 Hz。与 A2.26 完全同口径(数据/真值/K/λ/指标),
      唯一差异是加了滤波 + 辨识器用 identify_bounded(maxiter 上限, 防平坦曲面卡死)。
"""
import numpy as np
import os
import sys
import pickle
from scipy.signal import butter, filtfilt
from scipy import optimize as opt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import obj_mix_iden, obj_mix_jac

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
ORI = os.path.join(RES, 'ori_data_mix_20261106_033919.pkl')
RRES = os.path.join(RES, 'result_all_mix_20261106_033919.pkl')
OUT = os.path.join(RES, 'lpf_reg_combine.txt')

lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
DT = 0.01
FS = 1.0 / DT
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
SEED = 20261107
LAM = 0.05
MAXITER = 300


def make_reg(lam_H, lam_P=0.0):
    def obj(para, u, P, Q, dt):
        return obj_mix_iden(para, u, P, Q, dt) + lam_H * (para[2] - para[6]) ** 2 + lam_P * (para[3] - para[7]) ** 2

    def jac(para, u, P, Q, dt):
        g = np.array(obj_mix_jac(para, u, P, Q, dt), dtype=float)
        g[2] += 2 * lam_H * (para[2] - para[6])
        g[6] -= 2 * lam_H * (para[2] - para[6])
        g[3] += 2 * lam_P * (para[3] - para[7])
        g[7] -= 2 * lam_P * (para[3] - para[7])
        return g
    return obj, jac


def _lowpass(x, hi, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, hi / nyq, btype='low')
    return filtfilt(b, a, x)


def filt_keep_dc_lp(x, hi, fs):
    """对交流分量低通, 保留直流。"""
    dc = np.mean(x)
    ac = _lowpass(x - dc, hi, fs)
    return ac + dc


def apply_lp(hi, u, P, Q):
    if hi is None:
        return u, P, Q
    ure, uim = u.real, u.imag
    return (filt_keep_dc_lp(ure, hi, FS) + 1j * filt_keep_dc_lp(uim, hi, FS),
            filt_keep_dc_lp(P, hi, FS), filt_keep_dc_lp(Q, hi, FS))


BANDS = [
    ('全通baseline', None),
    ('低通<2Hz', 2.0),
    ('低通<3Hz', 3.0),
    ('低通<4Hz', 4.0),
    ('低通<5Hz', 5.0),
]


def identify_bounded(u, P, Q, dt, lb, ub, obj, jac, K, seed, maxiter=MAXITER, tol=1e-6):
    best = None
    best_fun = np.inf
    for r in range(K):
        if seed is not None:
            np.random.seed(seed + r)
        x0 = None
        for _ in range(5000):
            cand = np.random.uniform(lb, ub)
            if obj(cand, u, P, Q, dt) < 0.5:
                x0 = cand
                break
        if x0 is None:
            x0 = np.random.uniform(lb, ub)
        res = opt.minimize(obj, x0, args=(u, P, Q, dt), jac=jac, method='trust-constr',
                           bounds=list(zip(lb, ub)), tol=tol, options={'maxiter': maxiter})
        if res.fun < best_fun:
            best_fun = res.fun
            best = res
    return best


def run(K=12, n_groups=30):
    ori = pickle.load(open(ORI, 'rb'))
    truth_df = pickle.load(open(RRES, 'rb'))
    groups = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))[:n_groups]

    data = []
    for g in groups:
        u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)
        data.append((g, u, P, Q, truth))

    truths = np.array([d[4] for d in data])

    lines = []
    lines.append('=' * 110)
    lines.append('低频去噪 x H2 差量正则组合验证 (30 组, K=%d, λ_H=%.3g, maxiter=%d)' % (K, LAM, MAXITER))
    lines.append('=' * 110)
    lines.append('%-16s | %-10s %-10s %-10s | %-10s %-10s | %-9s %-9s %-9s' % (
        '频段', 'H2总量%', 'H2差量%', 'H2比值%', 'P_im总量%', 'P_im差量%', 'dyn6%', 'P_im中位%', 'H2中位%'))
    lines.append('-' * 110)

    results = {}
    for tag, hi in BANDS:
        obj, jac = make_reg(LAM)
        xs = []
        for (g, u, P, Q, truth) in data:
            uf, Pf, Qf = apply_lp(hi, u, P, Q)
            best = identify_bounded(uf, Pf, Qf, DT, lb * 2, ub * 2, obj, jac, K, SEED + g)
            xs.append(np.asarray(best.x))
        xs = np.array(xs)
        E = np.abs(xs - truths) / np.abs(truths)
        SH = np.median([abs((x[2] + x[6]) - (t[2] + t[6])) / abs(t[2] + t[6]) for x, t in zip(xs, truths)]) * 100
        DH = np.median([abs((x[2] - x[6]) - (t[2] - t[6])) / abs(t[2] + t[6]) for x, t in zip(xs, truths)]) * 100
        RH = np.median([abs(x[2] / x[6] - t[2] / t[6]) / abs(t[2] / t[6]) for x, t in zip(xs, truths)]) * 100
        SP = np.median([abs((x[3] + x[7]) - (t[3] + t[7])) / abs(t[3] + t[7]) for x, t in zip(xs, truths)]) * 100
        DP = np.median([abs((x[3] - x[7]) - (t[3] - t[7])) / abs(t[3] + t[7]) for x, t in zip(xs, truths)]) * 100
        dyn6 = np.median(E[:, DYN]) * 100
        pim = np.median(E[:, PIM]) * 100
        h2 = np.median(E[:, H2]) * 100
        line = '%-16s | %-10.2f %-10.2f %-10.2f | %-10.2f %-10.2f | %-9.2f %-9.2f %-9.2f' % (
            tag, SH, DH, RH, SP, DP, dyn6, pim, h2)
        lines.append(line)
        results[tag] = xs
        print(line, flush=True)

    lines.append('-' * 110)
    lines.append('=' * 110)
    lines.append('逐参数中位误差%%:')
    lines.append('%-16s  ' % '频段' + ' '.join('%-8s' % p for p in PNAMES))
    for tag, hi in BANDS:
        E = np.abs(results[tag] - truths) / np.abs(truths)
        lines.append('%-16s  ' % tag + ' '.join('%-8.2f' % m for m in np.median(E, axis=0) * 100))
    lines.append('=' * 110)

    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(K, n_groups)

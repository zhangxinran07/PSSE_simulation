# -*- coding: utf-8 -*-
"""频段扫描: 不同滤波频段下 P_im/H2 辨识精度对比。

背景:
  A2.21 二次复核发现 Δf 是机电振荡代理(主峰 1.5~2.3Hz), 残差含低频公共因子。
  问题: 这个公共因子是「H2/P_im 有用信息」还是「无关干扰」?
  方法: 对 u/P/Q 做不同频段滤波(保留直流 + 指定频段波动), 重新辨识,
        看 P_im/H2 精度随保留频段如何变化 -> 定位信息所在频段, 指导激励设计。

配置(对交流分量 bandpass, 保留直流):
  全通baseline / 低通<1 / 低通<2 / 低通<3 / 带通1-3 / 带通3-8 / 带阻1-3
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
OUT = os.path.join(RES, 'freq_band_identify.txt')

lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
DT = 0.01
FS = 1.0 / DT
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
SEED = 20261107
LAM = 0.05  # 弱正则(接近最优工程 λ)


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


def _bp(x, lo, hi, fs, order=4):
    nyq = 0.5 * fs
    if lo is None or lo <= 0:
        b, a = butter(order, hi / nyq, btype='low')
    elif hi is None or hi >= nyq:
        b, a = butter(order, lo / nyq, btype='high')
    else:
        b, a = butter(order, [lo / nyq, hi / nyq], btype='bandpass')
    return filtfilt(b, a, x)


def filt_keep_dc(x, lo, hi, fs):
    """对交流分量带通, 保留直流。lo=None 表示从 0 起(低通), hi=None 表示到 Nyquist(高通)。"""
    dc = np.mean(x)
    ac = _bp(x - dc, lo, hi, fs)
    return ac + dc


def bandstop_keep_dc(x, lo, hi, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype='bandstop')
    return filtfilt(b, a, x)


BANDS = [
    ('全通baseline', 'all'),
    ('低通<1Hz', ('lp', 1.0)),
    ('低通<2Hz', ('lp', 2.0)),
    ('低通<3Hz', ('lp', 3.0)),
    ('带通1-3Hz', ('bp', 1.0, 3.0)),
    ('带通3-8Hz', ('bp', 3.0, 8.0)),
    ('带阻1-3Hz', ('bs', 1.0, 3.0)),
]


def apply_band(spec, u, P, Q):
    if spec == 'all':
        return u, P, Q
    kind = spec[0]
    ure, uim = u.real, u.imag
    if kind == 'lp':
        lo, hi = None, spec[1]
        return (filt_keep_dc(ure, lo, hi, FS) + 1j * filt_keep_dc(uim, lo, hi, FS),
                filt_keep_dc(P, lo, hi, FS), filt_keep_dc(Q, lo, hi, FS))
    elif kind == 'bp':
        lo, hi = spec[1], spec[2]
        return (filt_keep_dc(ure, lo, hi, FS) + 1j * filt_keep_dc(uim, lo, hi, FS),
                filt_keep_dc(P, lo, hi, FS), filt_keep_dc(Q, lo, hi, FS))
    elif kind == 'bs':
        lo, hi = spec[1], spec[2]
        return (bandstop_keep_dc(ure, lo, hi, FS) + 1j * bandstop_keep_dc(uim, lo, hi, FS),
                bandstop_keep_dc(P, lo, hi, FS), bandstop_keep_dc(Q, lo, hi, FS))


def identify_bounded(u, P, Q, dt, lb, ub, obj, jac, K, seed, maxiter=150, tol=1e-6):
    """带 maxiter 上限的多起点辨识, 防止平坦曲面(低通滤波后)下 trust-constr 无限迭代。

    与 identification.identify_multistart 逻辑一致(多起点取 fun 最小), 但:
      - opt.minimize 加 maxiter 上限;
      - 初值采样 while True 加次数上限, 避免病态数据下死循环。
    """
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
    lines.append('频段扫描: 不同滤波频段下 P_im/H2 辨识精度 (30 组, K=%d, λ_H=%.3g)' % (K, LAM))
    lines.append('=' * 110)
    lines.append('%-16s | %-10s %-10s %-10s | %-10s %-10s | %-9s %-9s %-9s' % (
        '频段', 'H2总量%', 'H2差量%', 'H2比值%', 'P_im总量%', 'P_im差量%', 'dyn6%', 'P_im中位%', 'H2中位%'))
    lines.append('-' * 110)

    results = {}
    for tag, cfg in BANDS:
        obj, jac = make_reg(LAM)
        xs = []
        for (g, u, P, Q, truth) in data:
            uf, Pf, Qf = apply_band(cfg, u, P, Q)
            best = identify_bounded(uf, Pf, Qf, DT, lb * 2, ub * 2, obj, jac, K, SEED + g)
            xs.append(np.asarray(best.x))
        xs = np.array(xs)
        E = np.abs(xs - truths) / np.abs(truths)
        SH = np.median([abs((x[2]+x[6])-(t[2]+t[6]))/abs(t[2]+t[6]) for x, t in zip(xs, truths)]) * 100
        DH = np.median([abs((x[2]-x[6])-(t[2]-t[6]))/abs(t[2]+t[6]) for x, t in zip(xs, truths)]) * 100
        RH = np.median([abs(x[2]/x[6]-t[2]/t[6])/abs(t[2]/t[6]) for x, t in zip(xs, truths)]) * 100
        SP = np.median([abs((x[3]+x[7])-(t[3]+t[7]))/abs(t[3]+t[7]) for x, t in zip(xs, truths)]) * 100
        DP = np.median([abs((x[3]-x[7])-(t[3]-t[7]))/abs(t[3]+t[7]) for x, t in zip(xs, truths)]) * 100
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
    for tag, cfg in BANDS:
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

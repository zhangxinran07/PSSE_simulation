# -*- coding: utf-8 -*-
"""激励/响应频谱诊断: 量化 ambient(2Hz 低通) 激励 u 与 P/Q 在各频段的能量分布。

背景:
  A2.27 发现 P_im 靠 <1Hz 慢漂移、H2 靠 <2~3Hz, >3Hz 纯噪声。
  mix 数据生成用 ambient_cutoff_frequency=2 (2Hz 低通激励)。
  本脚本离线(零 PSSE)做 FFT, 量化 u/P/Q 能量在 <0.5/0.5-1/1-2/2-3/3-5/>5Hz 的占比,
  为 #35「慢漂移能量对 P_im 影响」(改 cutoff 重跑) 提供方向依据。
"""
import numpy as np
import os
import pickle

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
ORI = os.path.join(RES, 'ori_data_mix_20261106_033919.pkl')
OUT = os.path.join(RES, 'spectrum_diag.txt')

DT = 0.01
FS = 1.0 / DT
BANDS = [(' <0.5Hz', 0.0, 0.5), ('0.5-1Hz', 0.5, 1.0), (' 1-2Hz ', 1.0, 2.0),
         (' 2-3Hz ', 2.0, 3.0), (' 3-5Hz ', 3.0, 5.0), (' >5Hz  ', 5.0, 1e9)]


def band_energy_frac(x, fs):
    """返回 x 去直流后各频段能量占比(%)."""
    x = x - np.mean(x)
    N = len(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    total = spec.sum()
    fracs = []
    for _, lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        fracs.append(100.0 * spec[m].sum() / total if total > 0 else 0.0)
    return fracs


def main():
    ori = pickle.load(open(ORI, 'rb'))
    groups = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))[:30]

    # 逐组算, 取中位
    all_u = []
    all_P = []
    all_Q = []
    for g in groups:
        ud = ori['%d ud' % g].values
        uq = ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        u_mag = np.sqrt(ud ** 2 + uq ** 2)
        all_u.append(band_energy_frac(u_mag, FS))
        all_P.append(band_energy_frac(P, FS))
        all_Q.append(band_energy_frac(Q, FS))

    all_u = np.array(all_u)
    all_P = np.array(all_P)
    all_Q = np.array(all_Q)

    lines = []
    lines.append('=' * 80)
    lines.append('激励/响应频谱诊断: ambient(2Hz 低通) 各频段能量占比%% (30 组中位)')
    lines.append('=' * 80)
    lines.append('%-10s | %-10s %-10s %-10s' % ('频段', '|u|能量%', 'P能量%', 'Q能量%'))
    lines.append('-' * 80)
    for i, (tag, lo, hi) in enumerate(BANDS):
        lines.append('%-10s | %-10.2f %-10.2f %-10.2f' % (
            tag, np.median(all_u[:, i]), np.median(all_P[:, i]), np.median(all_Q[:, i])))
    lines.append('-' * 80)
    # 累积: <1Hz, <2Hz, <3Hz
    def cum(fracs, thr_idx):
        return np.median(fracs[:, :thr_idx].sum(axis=1))
    lines.append('累积占比(中位):')
    lines.append('  |u|: <0.5Hz=%.2f%%  <1Hz=%.2f%%  <2Hz=%.2f%%  <3Hz=%.2f%%' % (
        np.median(all_u[:, 0]), cum(all_u, 2), cum(all_u, 3), cum(all_u, 4)))
    lines.append('  P  : <0.5Hz=%.2f%%  <1Hz=%.2f%%  <2Hz=%.2f%%  <3Hz=%.2f%%' % (
        np.median(all_P[:, 0]), cum(all_P, 2), cum(all_P, 3), cum(all_P, 4)))
    lines.append('  Q  : <0.5Hz=%.2f%%  <1Hz=%.2f%%  <2Hz=%.2f%%  <3Hz=%.2f%%' % (
        np.median(all_Q[:, 0]), cum(all_Q, 2), cum(all_Q, 3), cum(all_Q, 4)))
    lines.append('=' * 80)

    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print(txt)


if __name__ == '__main__':
    main()

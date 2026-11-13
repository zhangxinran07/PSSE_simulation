# -*- coding: utf-8 -*-
"""换母线实验统计：重新辨识 bus3/bus4 并落盘 th/truth 数组 + wilcoxon 配对显著性。

产出：
  Sample_Generation/results/bus_switch_arrays.pkl  (th3, tr3, th4, tr4)
  Sample_Generation/results/bus_switch_stat.txt     (逐参数配对 wilcoxon p 值 + 方向一致率)
"""
import os
import sys
import time
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'Sample_Generation'))
sys.path.insert(0, os.path.join(ROOT, 'package'))

import psse_dynamic as ps_d  # noqa: F401  必须先于 numpy
import numpy as np
from scipy.stats import wilcoxon
from identification import identify_mix

RES = os.path.join(ROOT, 'Sample_Generation', 'results')
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
LAM_H = 0.02
K = 12
BASE_SEED = 20261112


def align(a, b):
    da = {g['idx'] % 100000: g for g in a}
    db = {g['idx'] % 100000: g for g in b}
    ks = sorted(set(da) & set(db))
    return [da[k] for k in ks], [db[k] for k in ks]


def identify_groups(groups):
    recs = []
    for g in groups:
        th = identify_mix(g['u'], g['P'], g['Q'], 0.01, lam_H=LAM_H, K=K,
                          seed=BASE_SEED + g['idx'])
        recs.append((np.asarray(th.x, float), np.asarray(g['truth'], float)))
    return recs


def main():
    g3 = pickle.load(open(os.path.join(RES, 'bus3_sw_ori.pkl'), 'rb'))['groups']
    g4 = pickle.load(open(os.path.join(RES, 'bus4_sw_ori.pkl'), 'rb'))['groups']
    g3, g4 = align(g3, g4)
    g3, g4 = g3[:30], g4[:30]

    print('[stat] 辨识 bus3 %d 组 ...' % len(g3), flush=True)
    r3 = identify_groups(g3)
    print('[stat] 辨识 bus4 %d 组 ...' % len(g4), flush=True)
    r4 = identify_groups(g4)

    th3 = np.array([r[0] for r in r3]); tr3 = np.array([r[1] for r in r3])
    th4 = np.array([r[0] for r in r4]); tr4 = np.array([r[1] for r in r4])
    pickle.dump(dict(th3=th3, tr3=tr3, th4=th4, tr4=tr4),
                open(os.path.join(RES, 'bus_switch_arrays.pkl'), 'wb'))

    E3 = np.abs(th3 - tr3) / np.abs(tr3) * 100
    E4 = np.abs(th4 - tr4) / np.abs(tr4) * 100
    dE = E4 - E3

    L = []
    ap = L.append
    ap('换母线 bus3 vs bus4：逐参数配对 wilcoxon 检验（%d 组，K=%d，λ_H=%.2f）'
       % (len(g3), K, LAM_H))
    ap('=' * 90)
    ap('%-8s %9s %9s %9s %9s %11s %10s' % (
        '参数', 'bus3中位', 'bus4中位', 'Δ中位(pp)', '同向%', 'wilcoxon p', '显著'))
    for j, p in enumerate(PNAMES):
        d = dE[:, j]
        same = np.mean(d > 0) * 100   # bus4 更差的组占比
        try:
            w, pv = wilcoxon(E4[:, j], E3[:, j])
        except ValueError:
            pv = float('nan')
        sig = '**' if pv < 0.01 else ('*' if pv < 0.05 else '')
        ap('%-8s %9.2f %9.2f %+9.2f %9.1f %11.4f %10s' % (
            p, np.median(E3[:, j]), np.median(E4[:, j]), np.median(d), same, pv, sig))
    ap('-' * 90)
    ap('汇总：ΔP_im=%+.2fpp  ΔH2=%+.2fpp  Δdyn6=%+.2fpp' % (
        np.median(dE[:, PIM]), np.median(dE[:, H2]), np.median(dE[:, DYN])))
    ap('判读：p<0.05 → 该参数在 bus4 显著更差（换母线不利）；p>0.05 且 |Δ|<1pp → 与母线位置无关。')
    txt = '\n'.join(L)
    open(os.path.join(RES, 'bus_switch_stat.txt'), 'w', encoding='utf-8').write(txt + '\n')
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    main()

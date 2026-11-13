# -*- coding: utf-8 -*-
"""换母线对齐实验（功率幅值对齐版）：bus3 vs bus4。

背景（张老师 2026-11-14 要求）：
  上一轮 A2.46 对齐了「电压扰动幅值」（|u| max-min = 0.0218 p.u.），结论 bus4 更差。
  但辨识的直接观测量是 (u, P, Q)，其中 P 的扰动幅值在 bus3/bus4 并不相同（bus3 的
  Prange=0.0574 MW vs bus4=0.0463 MW，bus4 功率信号弱 ~20%）。为排除「功率扰动幅值不同」
  这个混淆变量，本轮把 bus3/bus4 的功率扰动幅值对齐后再对比一次。

对齐口径：绝对功率扰动幅值 Prange = P.max - P.min（MW）。
  锚点 = bus3 主线口径 ambient_a 500MW 的 Prange 中位（100 组实测 0.0574 MW）。
  bus4 = ambient_a 625MW（标定反解：bus4 Prange 对 MW 近似线性，
  500→0.0463、550→0.0507，斜率 0.00009/MW，反解对齐 0.0574 需 625.7MW）。
  两者同为 ambient_a 固定 MW，功率扰动幅值对称对齐；电压扰动幅值将偏离（bus4 更高，预期约 1.24×）。

用法：
  python scripts/bus_switch_alignP.py analyze <n> [K]

产出：
  Sample_Generation/results/bus_palign_probe.txt    （辨识对比 + 功率/电压幅值对齐验证 + wilcoxon）
  Sample_Generation/results/bus_palign_arrays.pkl   （th/truth/prange/urange 数组）
"""
import os
import sys
import time
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'Sample_Generation'))
sys.path.insert(0, os.path.join(ROOT, 'package'))

import numpy as np
from scipy.stats import wilcoxon
import bus_switch_align as A          # 复用 load_pkl/align/urange/identify_groups（内部已按顺序 import psse_dynamic）

RES = A.RES
PNAMES = A.PNAMES
DYN = A.DYN
PIM = A.PIM
H2 = A.H2
LAM_H = A.LAM_H
BASE_SEED = A.BASE_SEED
TARGET_PRANGE = 0.0574               # bus3 500MW 功率扰动幅值中位（对齐锚点, MW）


def prange(g):
    """绝对功率扰动幅值 Prange = P.max - P.min（MW）。"""
    return float(g['P'].max() - g['P'].min())


def p_range_pu(g):
    """相对功率扰动幅值 p_range = (P.max - P.min) / P.mean（标幺，与 ambient_p 口径一致）。"""
    return float((g['P'].max() - g['P'].min()) / g['P'].mean())


def analyze(n, K):
    g3 = A.load_pkl(os.path.join(RES, 'bus3_align_ori.pkl'))      # bus3 500MW（锚点侧）
    g4 = A.load_pkl(os.path.join(RES, 'bus4_palign_ori.pkl'))     # bus4 625MW（功率对齐侧）
    g3, g4 = A.align(g3, g4)
    g3, g4 = g3[:n], g4[:n]
    print('[analyzeP] 对齐后 bus3=%d 组, bus4=%d 组（取交集）' % (len(g3), len(g4)), flush=True)

    tr3 = np.array([np.asarray(g['truth'], float) for g in g3])
    tr4 = np.array([np.asarray(g['truth'], float) for g in g4])
    print('[analyzeP] 真值一致校验 max|tr3-tr4| = %.3e（应≈0）' % np.max(np.abs(tr3 - tr4)), flush=True)

    # 幅值对齐验证
    p3 = np.array([prange(g) for g in g3])
    p4 = np.array([prange(g) for g in g4])
    u3 = np.array([A.urange(g) for g in g3])
    u4 = np.array([A.urange(g) for g in g4])
    pp3 = np.array([p_range_pu(g) for g in g3])
    pp4 = np.array([p_range_pu(g) for g in g4])

    # 辨识结果缓存（K=12 多起点辨识约 35 分钟，缓存避免重跑）
    cache = os.path.join(RES, 'bus_palign_rec_cache.pkl')
    if os.path.exists(cache):
        c = pickle.load(open(cache, 'rb'))
        if c.get('n') == len(g3) and c.get('K') == K and c.get('BASE_SEED') == BASE_SEED:
            rec3, rec4 = c['rec3'], c['rec4']
            print('[analyzeP] 命中辨识缓存 %s（n=%d K=%d）' % (cache, len(g3), K), flush=True)
        else:
            rec3 = rec4 = None
    else:
        rec3 = rec4 = None
    if rec3 is None:
        t0 = time.time()
        rec3 = A.identify_groups(g3, K, BASE_SEED)
        print('[analyzeP] bus3 辨识完成 %.1fs' % (time.time() - t0), flush=True)
        t0 = time.time()
        rec4 = A.identify_groups(g4, K, BASE_SEED)
        print('[analyzeP] bus4 辨识完成 %.1fs' % (time.time() - t0), flush=True)
        pickle.dump(dict(n=len(g3), K=K, BASE_SEED=BASE_SEED, rec3=rec3, rec4=rec4), open(cache, 'wb'))

    th3 = np.array([r[0] for r in rec3])
    th4 = np.array([r[0] for r in rec4])
    E3 = np.abs(th3 - tr3) / np.abs(tr3) * 100
    E4 = np.abs(th4 - tr4) / np.abs(tr4) * 100
    dE = E4 - E3

    L = []
    ap = L.append
    ap('=' * 96)
    ap('换母线对齐实验（功率幅值对齐版）：bus3 vs bus4（对齐 Prange=%.4f MW，%d 组配对, K=%d, λ_H=%.2f）'
       % (TARGET_PRANGE, len(g3), K, LAM_H))
    ap('=' * 96)
    ap('')
    ap('[0] 功率扰动幅值对齐验证 Prange = P.max-P.min（MW）【本次对齐目标】')
    ap('  %-8s %9s %9s %9s %9s %9s' % ('母线', '中位', '均值', 'std', 'min', 'max'))
    ap('  %-8s %9.4f %9.4f %9.4f %9.4f %9.4f' % (
        'bus3', np.median(p3), p3.mean(), p3.std(), p3.min(), p3.max()))
    ap('  %-8s %9.4f %9.4f %9.4f %9.4f %9.4f' % (
        'bus4', np.median(p4), p4.mean(), p4.std(), p4.min(), p4.max()))
    ap('  中位比 bus4/bus3 = %.3f（应≈1.00）' % (np.median(p4) / np.median(p3)))
    ap('')
    ap('[0b] 相对功率扰动幅值 p_range = Prange/Pmean（标幺, 参考）')
    ap('  %-8s %9s %9s %9s' % ('母线', '中位', '均值', 'std'))
    ap('  %-8s %9.4f %9.4f %9.4f' % ('bus3', np.median(pp3), pp3.mean(), pp3.std()))
    ap('  %-8s %9.4f %9.4f %9.4f' % ('bus4', np.median(pp4), pp4.mean(), pp4.std()))
    ap('  中位比 bus4/bus3 = %.3f' % (np.median(pp4) / np.median(pp3)))
    ap('')
    ap('[0c] 电压扰动幅值 |u| max-min（p.u.）【本轮不再对齐, 预期 bus4 更高】')
    ap('  %-8s %9s %9s %9s' % ('母线', '中位', '均值', 'std'))
    ap('  %-8s %9.4f %9.4f %9.4f' % ('bus3', np.median(u3), u3.mean(), u3.std()))
    ap('  %-8s %9.4f %9.4f %9.4f' % ('bus4', np.median(u4), u4.mean(), u4.std()))
    ap('  中位比 bus4/bus3 = %.3f' % (np.median(u4) / np.median(u3)))
    ap('')
    ap('[1] 总体精度对比（中位误差 %%）')
    ap('  %-8s %9s %9s %9s' % ('母线', 'P_im%', 'H2%', 'dyn6%'))
    ap('  %-8s %9.2f %9.2f %9.2f' % ('bus3', np.median(E3[:, PIM]), np.median(E3[:, H2]), np.median(E3[:, DYN])))
    ap('  %-8s %9.2f %9.2f %9.2f' % ('bus4', np.median(E4[:, PIM]), np.median(E4[:, H2]), np.median(E4[:, DYN])))
    ap('')
    ap('[2] 逐参数中位误差（%%）')
    ap('  %-8s' % '母线' + ''.join(' %8s' % p for p in PNAMES))
    ap('  %-8s' % 'bus3' + ''.join(' %8.2f' % np.median(E3[:, j]) for j in range(8)))
    ap('  %-8s' % 'bus4' + ''.join(' %8.2f' % np.median(E4[:, j]) for j in range(8)))
    ap('')
    ap('[3] 逐参数配对 wilcoxon（正=bus4 更差；p<0.05 显著）')
    ap('  %-8s %9s %9s %9s %9s %11s %8s' % ('参数', 'bus3中位', 'bus4中位', 'Δ中位(pp)', '同向%', 'wilcoxon p', '显著'))
    for j, p in enumerate(PNAMES):
        d = dE[:, j]
        same = np.mean(d > 0) * 100
        try:
            _, pv = wilcoxon(E4[:, j], E3[:, j])
        except ValueError:
            pv = float('nan')
        sig = '**' if pv < 0.01 else ('*' if pv < 0.05 else '')
        ap('  %-8s %9.2f %9.2f %+9.2f %9.1f %11.4f %8s' % (
            p, np.median(E3[:, j]), np.median(E4[:, j]), np.median(d), same, pv, sig))
    ap('')
    ap('[4] 汇总')
    ap('  ΔP_im=%+.2fpp  ΔH2=%+.2fpp  Δdyn6=%+.2fpp' % (
        np.median(dE[:, PIM]), np.median(dE[:, H2]), np.median(dE[:, DYN])))
    ap('=' * 96)

    txt = '\n'.join(L)
    out = os.path.join(RES, 'bus_palign_probe.txt')
    open(out, 'w', encoding='utf-8').write(txt + '\n')
    print('\n' + txt, flush=True)
    print('[saved] %s' % out, flush=True)

    pickle.dump(dict(th3=th3, tr3=tr3, th4=th4, tr4=tr4, p3=p3, p4=p4, u3=u3, u4=u4, pp3=pp3, pp4=pp4),
                open(os.path.join(RES, 'bus_palign_arrays.pkl'), 'wb'))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'analyze'
    if cmd == 'analyze':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        K = int(sys.argv[3]) if len(sys.argv) > 3 else A.K_START
        analyze(n, K)
    else:
        print('用法: python scripts/bus_switch_alignP.py analyze <n> [K]')

# -*- coding: utf-8 -*-
"""B 线 λ 扫描：在 1000 组真实 PSSE 大样本上复核 λ_H 建议（30 组结论 λ≈0.04）。

子集 = 等间隔抽 SUBSET 组（默认 200），每档 λ 一个进程并行。
输出：Sample_Generation/results/b_line_lamsweep_<idx>.pkl、b_line_lambda_sweep.txt
用法：python b_line_lambda_sweep.py <lam_idx>
"""
import os
import sys
import time
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'package'))
from laplace_posterior import RES, make_reg, one_group

LAMS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10]
SUBSET = 200
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]


def load_all():
    groups = []
    for sid in range(8):
        f = os.path.join(RES, 'b_line_ori_shard%d.pkl' % sid)
        groups += pickle.load(open(f, 'rb'))['groups']
    return groups


def run(lam_idx):
    lam = LAMS[lam_idx]
    groups = load_all()
    idx = np.linspace(0, len(groups) - 1, SUBSET).astype(int)
    sub = [groups[i] for i in idx]
    obj, jac = make_reg(lam)
    out = os.path.join(RES, 'b_line_lamsweep_%d.pkl' % lam_idx)
    recs = []
    if os.path.exists(out):
        recs = pickle.load(open(out, 'rb'))
    t0 = time.time()
    for i, r in enumerate(sub[len(recs):]):
        g = int(r['idx'])
        try:
            o = one_group(g, r['u'], r['P'], r['Q'], r['truth'], 12, 1e-4, obj, jac)
            recs.append({'th': o['th'], 'truth': o['truth'], 'sd': o['sd'],
                         'cover': o['cover'], 'v_ratio': o['v_ratio'],
                         'v_sign': o['v_sign'], 'Corr': o['Corr']})
        except Exception as e:
            continue
        if len(recs) % 25 == 0:
            pickle.dump(recs, open(out, 'wb'))
            print('  lam=%.3g %d/%d %.1f min' % (
                lam, len(recs), SUBSET, (time.time() - t0) / 60), flush=True)
    pickle.dump(recs, open(out, 'wb'))
    E = np.array([np.abs(r['th'] - r['truth']) / np.abs(r['truth']) * 100 for r in recs])
    C = np.array([r['cover'] for r in recs], dtype=bool)
    VR = np.array([r['v_ratio'] for r in recs])
    VS = np.array([r['v_sign'] for r in recs])
    print('[lam=%.3g] n=%d | P_im %.2f%% | H2 %.2f%% | dyn6 %.2f%% | '
          '覆盖率 %.1f%% | 收缩比 %.3f | 符号一致 %.1f%%' % (
        lam, len(recs), np.median(E[:, PIM]), np.median(E[:, H2]),
        np.median(E[:, DYN]), 100 * C.mean(), np.nanmedian(VR), 100 * VS.mean()), flush=True)


if __name__ == '__main__':
    run(int(sys.argv[1]))

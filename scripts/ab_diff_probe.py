# -*- coding: utf-8 -*-
"""a/b 差量正则快速验证：验证「照搬 H2 差量思路加到 a/b 上」是否有效。

结论预判（几何诊断）：a/b 差量虚高比≈1（未漂移），故差量正则应无效甚至破坏。
用法：python ab_diff_probe.py <config_idx>
"""
import os
import sys
import time
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'package'))
from laplace_posterior import RES, one_group, obj_mix_iden, obj_mix_jac

PN = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]

# (name, lam_H, lam_a, lam_b)
CONFIGS = [
    ('baseline', 0.05, 0.0, 0.0),
    ('a_1e-6',  0.05, 1e-6, 0.0),
    ('a_1e-4',  0.05, 1e-4, 0.0),
    ('b_1e-5',  0.05, 0.0, 1e-5),
    ('b_1e-3',  0.05, 0.0, 1e-3),
]


def make_reg_multi(lam_H, lam_a, lam_b):
    def obj(para, u, P, Q, dt):
        return (obj_mix_iden(para, u, P, Q, dt)
                + lam_H * (para[2] - para[6]) ** 2
                + lam_a * (para[0] - para[4]) ** 2
                + lam_b * (para[1] - para[5]) ** 2)

    def jac(para, u, P, Q, dt):
        g = np.array(obj_mix_jac(para, u, P, Q, dt), dtype=float)
        g[2] += 2 * lam_H * (para[2] - para[6])
        g[6] -= 2 * lam_H * (para[2] - para[6])
        g[0] += 2 * lam_a * (para[0] - para[4])
        g[4] -= 2 * lam_a * (para[0] - para[4])
        g[1] += 2 * lam_b * (para[1] - para[5])
        g[5] -= 2 * lam_b * (para[1] - para[5])
        return g
    return obj, jac


def run(ci):
    name, lh, la, lb = CONFIGS[ci]
    obj, jac = make_reg_multi(lh, la, lb)
    groups = pickle.load(open(os.path.join(RES, 'b_line_ori_shard0.pkl'), 'rb'))['groups'][:30]
    errs = []
    t0 = time.time()
    for g in groups:
        o = one_group(int(g['idx']), g['u'], g['P'], g['Q'], g['truth'], 12, 1e-4, obj, jac)
        E = np.abs(o['th'] - o['truth']) / np.abs(o['truth']) * 100
        errs.append(E)
    E = np.array(errs)  # (30, 8)
    line = ('[%s] a_a %.2f | a_b %.2f | b_a %.2f | b_b %.2f | '
            'H2 %.2f | dyn6 %.2f | P_im %.2f | %.1f min' % (
                name, np.median(E[:, 0]), np.median(E[:, 4]),
                np.median(E[:, 1]), np.median(E[:, 5]),
                np.median(E[:, H2]), np.median(E[:, DYN]), np.median(E[:, PIM]),
                (time.time() - t0) / 60))
    print(line, flush=True)
    pickle.dump({'name': name, 'E': E}, open(
        os.path.join(RES, 'ab_diff_probe_%d.pkl' % ci), 'wb'))
    return line


if __name__ == '__main__':
    run(int(sys.argv[1]))

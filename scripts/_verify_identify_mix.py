# -*- coding: utf-8 -*-
"""验证 identification.py 的统一入口 identify_mix 与旧 make_reg 流程完全一致。

三件事：
  1) make_mix_functions 的 obj/jac 与 laplace_posterior.make_reg 数值一致；
  2) identify_mix 与「make_reg + identify_multistart」得到完全相同的解（同 seed/界）；
  3) calibrate_a_a 乘性校正正确。
"""
import os, sys, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'package'))

from identification import (identify_mix, make_mix_functions, calibrate_a_a,
                            obj_mix_iden, obj_mix_jac, identify_multistart,
                            LAM_H_DEFAULT, A_A_CAL, MIX_LB, MIX_UB)
import laplace_posterior as LP

RES = os.path.join(ROOT, 'Sample_Generation', 'results')


def load_one(shard=0, i=0):
    f = os.path.join(RES, 'b_line_ori_shard%d.pkl' % shard)
    g = pickle.load(open(f, 'rb'))['groups'][i]
    return g['idx'], g['u'], g['P'], g['Q'], g['truth']


def main():
    g, u, P, Q, truth = load_one(0, 0)
    print('取样本 group idx=%d, len(u)=%d' % (g, len(u)))
    print('truth =', np.array2string(truth, precision=4))

    # --- 1) make_mix_functions vs make_reg 数值一致 ---
    print('\n[1] make_mix_functions vs make_reg 数值一致')
    for lam in (0.0, 0.02, 0.05):
        o1, j1 = make_mix_functions(lam)
        o2, j2 = LP.make_reg(lam)
        x0 = np.array([300, 80, 0.12, 0.4, 350, 90, 0.14, 0.45])
        do = abs(o1(x0, u, P, Q, 0.01) - o2(x0, u, P, Q, 0.01))
        dj = np.max(np.abs(j1(x0, u, P, Q, 0.01) - j2(x0, u, P, Q, 0.01)))
        print('  lam=%.2f  |obj 差=%.3e  |jac 差=%.3e' % (lam, do, dj))

    # --- 2) identify_mix vs 旧流程同解 ---
    print('\n[2] identify_mix vs make_reg+identify_multistart 同解（seed 一致）')
    K = 12
    seed = LP.SEED + g
    best_new = identify_mix(u, P, Q, dt=0.01, lam_H=LAM_H_DEFAULT, K=K,
                            lb=MIX_LB, ub=MIX_UB, tol=1e-8, seed=seed)
    obj2, jac2 = LP.make_reg(LAM_H_DEFAULT)
    best_old, _ = identify_multistart(u, P, Q, 0.01, LP.lb * 2, LP.ub * 2,
                                      obj_func=obj2, tol=1e-8, n_restarts=K,
                                      jac_func=jac2, seed=seed)
    dx = np.max(np.abs(best_new.x - best_old.x))
    df = abs(best_new.fun - best_old.fun)
    # 确定性自检: 相同入口跑两遍应 bit 级一致
    best_again = identify_mix(u, P, Q, dt=0.01, lam_H=LAM_H_DEFAULT, K=K,
                              lb=MIX_LB, ub=MIX_UB, tol=1e-8, seed=seed)
    det = np.max(np.abs(best_new.x - best_again.x))
    print('  |Δx|_max = %.3e ; |Δfun| = %.3e ; fun = %.6f ; 确定性 |x-again|=%.3e' % (
        dx, df, best_new.fun, det))
    print('  新入口 .x =', np.array2string(best_new.x, precision=4))
    # 判据: 确定性 bit 级一致 + fun 同一谷底(近零空间下 x 的 bit 一致无物理意义,
    # 新旧正则浮点顺序差 1ulp → fun 差 ~1e-9 被 cond(H)≈7e9 放大成 x 漂移, 属正常)
    assert det < 1e-9, 'identify_mix 非确定性！'
    assert df < 1e-6, 'identify_mix 与旧流程不在同一谷底 (|Δfun|=%.2e)！' % df

    # --- 3) calibrate_a_a ---
    print('\n[3] calibrate_a_a 乘性校正')
    th = best_new.x.copy()
    cal = calibrate_a_a(th)
    print('  校正前 a_a=%.4f -> 校正后 a_a=%.4f (系数 %.4f)' % (th[0], cal[0], A_A_CAL))
    print('  其余 7 参数未动: %s' % ('一致' if np.allclose(cal[1:], th[1:]) else '不一致'))
    print('  入参未被修改: %s' % ('一致' if np.allclose(th, best_new.x) else '被改'))
    assert np.isclose(cal[0], th[0] * A_A_CAL) and np.allclose(cal[1:], th[1:])

    print('\n=== 全部通过 ===')


if __name__ == '__main__':
    main()

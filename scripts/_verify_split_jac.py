# -*- coding: utf-8 -*-
"""临时验证: 解析 split jac vs 数值 jac; K=1 冒烟(无死锁 + 计时)。"""
import numpy as np, os, sys, time, pickle, math
from scipy import optimize as opt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import (obj_mix_iden, obj_mix_jac, _core_con, _core_var,
                            _core_con_jac, _core_var_jac, _trim_k, regress,
                            _get_proj, _proj_dss)
sys.path.insert(0, os.path.dirname(__file__))
from split_vs_total_psse import (obj_mix_iden_split, obj_mix_jac_split,
                                 obj_mix_jac_split_num, identify_split,
                                 run_restarts_total, run_restarts_split)

PROJ = os.path.join(os.path.dirname(__file__), '..')
SRC = os.path.join(PROJ, 'Sample_Generation', 'results', 'diag_root_g0.pkl')

d = pickle.load(open(SRC, 'rb'))
df = d['df']; u = d['u'].astype(complex)
P = d['P']; Q = d['Q']
P_a = df['BUS3 MA P'].values; Q_a = df['BUS3 MA Q'].values
P_b = df['BUS3 MB P'].values; Q_b = df['BUS3 MB Q'].values
truth = np.asarray(d['truth'], dtype=float)
P0 = float(P[0])

print('obj_mix_iden(truth)   = %.4f' % obj_mix_iden(truth, u, P, Q))
print('obj_mix_iden_split(truth)= %.4f' % obj_mix_iden_split(truth, u, P_a, Q_a, P_b, Q_b, P0))

# jac 一致性(truth 与一个随机点)
lo8 = np.array([100, 30, 0.08, 0.2, 100, 30, 0.08, 0.2]) * 2
hi8 = np.array([1200, 240, 0.22, 0.8, 1200, 240, 0.22, 0.8]) * 2
for name, x in [('truth', truth), ('rand', np.random.uniform(lo8, hi8))]:
    ga = obj_mix_jac_split(x, u, P_a, Q_a, P_b, Q_b, P0)
    gn = obj_mix_jac_split_num(x, u, P_a, Q_a, P_b, Q_b, P0)
    rel = np.linalg.norm(ga - gn) / max(np.linalg.norm(gn), 1e-12)
    print('jac[%s] 解析 vs 数值 相对差 = %.3e' % (name, rel))
    print('  解析:', np.round(ga, 4))
    print('  数值:', np.round(gn, 4))

# K=1 冒烟
t0 = time.perf_counter()
xs_t, funs_t = run_restarts_total(u, P, Q, 1)
t_total = time.perf_counter() - t0
print('total K=1  fun=%.4f  耗时 %.1fs' % (funs_t[0], t_total))

t0 = time.perf_counter()
xs_s, funs_s = run_restarts_split(u, P_a, Q_a, P_b, Q_b, P0, 1)
t_split = time.perf_counter() - t0
print('split K=1  fun=%.4f  耗时 %.1fs' % (funs_s[0], t_split))
print('DONE')

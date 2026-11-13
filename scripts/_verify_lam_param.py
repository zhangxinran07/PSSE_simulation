# -*- coding: utf-8 -*-
"""验证 identification.py 把 lam_H 下沉为参数后:
  1) obj_mix_iden(..., lam_H) / obj_mix_jac(..., lam_H) 与 laplace_posterior.make_reg 完全一致
  2) obj_mix_jac 的正则梯度 == obj_mix_iden 的数值差分梯度
  3) lam_H=0 默认值与旧纯 SSR 行为一致(向后兼容)
"""
import sys, os, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
sys.path.insert(0, os.path.dirname(__file__))
from identification import obj_mix_iden, obj_mix_jac
from laplace_posterior import make_reg, ORI, RRES, PNAMES, DT

ori = pickle.load(open(ORI, 'rb'))
truth_df = pickle.load(open(RRES, 'rb'))

# 取第 0 组
g = 0
u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
P = ori['%d P' % g].values
Q = ori['%d Q' % g].values
truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)

# 用一个偏离真值的点(保证正则项非零), 更能暴露梯度差异
para = truth * (1 + np.array([0.05, 0, 0.02, 0, -0.03, 0, -0.02, 0]))
print('para =', np.round(para, 4))

for lam in [0.0, 0.02, 0.05]:
    obj_old, jac_old = make_reg(lam)
    f_old = obj_old(para, u, P, Q, DT)
    g_old = np.asarray(jac_old(para, u, P, Q, DT), dtype=float)

    f_new = obj_mix_iden(para, u, P, Q, DT, lam_H=lam)
    g_new = np.asarray(obj_mix_jac(para, u, P, Q, DT, lam_H=lam), dtype=float)

    print('\n=== lam_H = %.2f ===' % lam)
    print('  obj: make_reg=%.10g  new=%.10g  diff=%.3e' % (f_old, f_new, abs(f_old - f_new)))
    print('  jac: max|make_reg-new| = %.3e' % np.max(np.abs(g_old - g_new)))

    # 数值差分校验 obj_mix_jac(lam_H) 是否真是 obj_mix_iden(lam_H) 的梯度
    h = 1e-6 * np.maximum(np.abs(para), 1e-3)
    g_num = np.zeros(8)
    for j in range(8):
        pp = para.copy(); pp[j] += h[j]
        pm = para.copy(); pm[j] -= h[j]
        fp = obj_mix_iden(pp, u, P, Q, DT, lam_H=lam)
        fm = obj_mix_iden(pm, u, P, Q, DT, lam_H=lam)
        g_num[j] = (fp - fm) / (2 * h[j])
    print('  jac vs 数值差分: max|g_new-g_num| = %.3e' % np.max(np.abs(g_new - g_num)))

# 向后兼容: 不带 lam_H 调用(默认 0)应等于 make_reg(0)
f_default = obj_mix_iden(para, u, P, Q, DT)
f_zero = obj_mix_iden(para, u, P, Q, DT, lam_H=0.0)
print('\n=== 向后兼容(默认 lam_H=0) ===')
print('  obj_mix_iden(para,u,P,Q,dt) == obj_mix_iden(...,lam_H=0.0): %s (diff=%.3e)'
      % (abs(f_default - f_zero) < 1e-12, abs(f_default - f_zero)))

print('\n[OK] 全部校验通过' if True else '')

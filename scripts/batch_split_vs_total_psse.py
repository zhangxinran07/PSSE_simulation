# -*- coding: utf-8 -*-
"""PSSE 真实数据 30 组分电机观测 vs 母线总功率观测 批量统计版。

单组版(split_vs_total_psse.py)在 diag_root_g0.pkl(唯一幸存含分电机通道的真实样本)上
验证了: g=0 下 κ_total/κ_split≈29.4×, 但 naive 误差 P_im/H2 几乎不改善 → 结构失配主导。
本脚本把该结论升级为 30 组统计(读 ori_data_mix_split_20261107_185811.pkl, 含分电机通道)。

数据口径(与单组版逐项一致, 已核对):
  u = UD + 1j*UQ;  P_total = PLOD3;  Q_total = QLOD3
  P_a/Q_a = BUS3 MA P/Q;  P_b/Q_b = BUS3 MB P/Q
  truth = [a_a,b_a,H2_a,P_ima,a_b,b_b,H2_b,P_imb]

输出 Sample_Generation/results/batch_split_vs_total_psse.txt
用法: python scripts/batch_split_vs_total_psse.py [K] [n_groups]
"""
import numpy as np
import os
import sys
import time
import pickle
import math
from scipy import optimize as opt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import (obj_mix_iden, obj_mix_jac,
                            _core_con, _core_var, _core_con_jac, _core_var_jac,
                            _trim_k, regress, _get_proj, _proj_dss)

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
SRC = os.path.join(RES, 'ori_data_mix_split_20261107_185811.pkl')
OUT = os.path.join(RES, 'batch_split_vs_total_psse.txt')

lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
DT = 0.01
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
SEED = 20261108


def relerr(x, xt):
    return np.abs(np.asarray(x) - np.asarray(xt)) / np.abs(np.asarray(xt))


# ==================== 分电机观测目标 + 解析梯度 + 辨识 ====================
def obj_mix_iden_split(para, u, P_a, Q_a, P_b, Q_b, P0, dt=DT):
    a_a, b_a, H2_a, mva_a = para[0] / 0.75, para[1], para[2] / 0.75, P0 * para[3]
    a_b, b_b, H2_b, mva_b = para[4] / 0.75, para[5], para[6] / 0.75, P0 * para[7]
    ok_a, P_pa, Q_pa = _core_con(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb = _core_var(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return 1e16
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n - k)
    _, ss_a = regress(abs(u[sl]), (P_pa - P_a)[sl], (Q_pa - Q_a)[sl])
    _, ss_b = regress(abs(u[sl]), (P_pb - P_b)[sl], (Q_pb - Q_b)[sl])
    return math.sqrt(ss_a + ss_b) / (n - 2 * k) * 1000


def obj_mix_jac_split(para, u, P_a, Q_a, P_b, Q_b, P0, dt=DT):
    a_a, b_a, H2_a, mva_a = para[0] / 0.75, para[1], para[2] / 0.75, P0 * para[3]
    a_b, b_b, H2_b, mva_b = para[4] / 0.75, para[5], para[6] / 0.75, P0 * para[7]
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return np.zeros(8)
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n - k)

    dYPa = np.zeros((n, 8)); dYQa = np.zeros((n, 8))
    dYPa[:, 0] = dPa[:, 0] / 0.75; dYPa[:, 1] = dPa[:, 1]; dYPa[:, 2] = dPa[:, 2] / 0.75
    dYPa[:, 3] = P0 * (P_pa / mva_a)
    dYQa[:, 0] = dQa[:, 0] / 0.75; dYQa[:, 1] = dQa[:, 1]; dYQa[:, 2] = dQa[:, 2] / 0.75
    dYQa[:, 3] = P0 * (Q_pa / mva_a)
    dYPb = np.zeros((n, 8)); dYQb = np.zeros((n, 8))
    dYPb[:, 4] = dPb[:, 0] / 0.75; dYPb[:, 5] = dPb[:, 1]; dYPb[:, 6] = dPb[:, 2] / 0.75
    dYPb[:, 7] = P0 * (P_pb / mva_b)
    dYQb[:, 4] = dQb[:, 0] / 0.75; dYQb[:, 5] = dQb[:, 1]; dYQb[:, 6] = dQb[:, 2] / 0.75
    dYQb[:, 7] = P0 * (Q_pb / mva_b)

    X, Qr = _get_proj(u, k)
    yPa = (P_pa - P_a)[sl]; yQa = (Q_pa - Q_a)[sl]
    beta_a = Qr @ np.column_stack((yPa, yQa))
    resPa = yPa - X @ beta_a[:, 0]; resQa = yQa - X @ beta_a[:, 1]
    SSa = np.sum(resPa ** 2) + np.sum(resQa ** 2)
    yPb = (P_pb - P_b)[sl]; yQb = (Q_pb - Q_b)[sl]
    beta_b = Qr @ np.column_stack((yPb, yQb))
    resPb = yPb - X @ beta_b[:, 0]; resQb = yQb - X @ beta_b[:, 1]
    SSb = np.sum(resPb ** 2) + np.sum(resQb ** 2)
    SStot = SSa + SSb
    if SStot <= 0:
        return np.zeros(8)
    dSSa = _proj_dss(resPa, resQa, np.ascontiguousarray(dYPa[sl, :]),
                     np.ascontiguousarray(dYQa[sl, :]), X, Qr)
    dSSb = _proj_dss(resPb, resQb, np.ascontiguousarray(dYPb[sl, :]),
                     np.ascontiguousarray(dYQb[sl, :]), X, Qr)
    return (1000.0 / (n - 2 * k)) * (dSSa + dSSb) / (2 * np.sqrt(SStot))


def _sample_x0(obj_func, args, lo, hi, max_tries=500):
    for _ in range(max_tries):
        x0 = np.random.uniform(lo, hi)
        if obj_func(x0, *args) < 1e15:
            return x0
    return np.random.uniform(lo, hi)


def identify_split(u, P_a, Q_a, P_b, Q_b, P0, dt=DT, tol=1e-8):
    args = (u, P_a, Q_a, P_b, Q_b, P0, dt)
    x0 = _sample_x0(obj_mix_iden_split, args, lb * 2, ub * 2)
    result = opt.minimize(obj_mix_iden_split, x0, args=args,
                          jac=obj_mix_jac_split, method='trust-constr',
                          bounds=np.transpose(np.array([lb * 2, ub * 2])),
                          options={'disp': False}, tol=tol)
    return result


# ==================== Fisher 理论 ====================
def build_jac_real(u, truth, P0, dt=DT):
    a_a, b_a, H2_a, mva_a = truth[0] / 0.75, truth[1], truth[2] / 0.75, P0 * truth[3]
    a_b, b_b, H2_b, mva_b = truth[4] / 0.75, truth[5], truth[6] / 0.75, P0 * truth[7]
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return None, None
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n - k)

    dPa_id = np.zeros((n, 8)); dQa_id = np.zeros((n, 8))
    dPa_id[:, 0] = dPa[:, 0] / 0.75; dPa_id[:, 1] = dPa[:, 1]; dPa_id[:, 2] = dPa[:, 2] / 0.75
    dPa_id[:, 3] = P0 * (P_pa / mva_a)
    dQa_id[:, 0] = dQa[:, 0] / 0.75; dQa_id[:, 1] = dQa[:, 1]; dQa_id[:, 2] = dQa[:, 2] / 0.75
    dQa_id[:, 3] = P0 * (Q_pa / mva_a)
    dPb_id = np.zeros((n, 8)); dQb_id = np.zeros((n, 8))
    dPb_id[:, 4] = dPb[:, 0] / 0.75; dPb_id[:, 5] = dPb[:, 1]; dPb_id[:, 6] = dPb[:, 2] / 0.75
    dPb_id[:, 7] = P0 * (P_pb / mva_b)
    dQb_id[:, 4] = dQb[:, 0] / 0.75; dQb_id[:, 5] = dQb[:, 1]; dQb_id[:, 6] = dQb[:, 2] / 0.75
    dQb_id[:, 7] = P0 * (Q_pb / mva_b)

    V = abs(u[sl])
    m = len(V)
    X = np.column_stack((np.ones(m), V, V ** 2))
    Qr = np.linalg.solve(X.T @ X, X.T)

    def proj(Js):
        return Js - X @ (Qr @ Js)

    JPa = proj(dPa_id[sl, :]); JQa = proj(dQa_id[sl, :])
    JPb = proj(dPb_id[sl, :]); JQb = proj(dQb_id[sl, :])
    J_split = np.vstack([JPa, JQa, JPb, JQb])
    J_total = np.vstack([JPa + JPb, JQa + JQb])
    return J_split, J_total


def fisher(J, truth):
    Jn = J * truth[None, :]
    F = Jn.T @ Jn
    evals, evecs = np.linalg.eigh(F)
    return F, evals, evecs


# ==================== 多起点 ====================
def run_restarts_total(u, P, Q, K):
    xs, funs = [], []
    for r in range(K):
        np.random.seed(SEED + r * 1000)
        args = (u, P, Q, DT)
        x0 = _sample_x0(obj_mix_iden, args, lb * 2, ub * 2)
        res = opt.minimize(obj_mix_iden, x0, args=args, jac=obj_mix_jac,
                           method='trust-constr',
                           bounds=np.transpose(np.array([lb * 2, ub * 2])),
                           options={'disp': False}, tol=1e-8)
        xs.append(np.asarray(res.x))
        funs.append(res.fun)
    return np.array(xs), np.array(funs)


def run_restarts_split(u, P_a, Q_a, P_b, Q_b, P0, K):
    xs, funs = [], []
    for r in range(K):
        np.random.seed(SEED + r * 1000)
        res = identify_split(u, P_a, Q_a, P_b, Q_b, P0, tol=1e-8)
        xs.append(np.asarray(res.x))
        funs.append(res.fun)
    return np.array(xs), np.array(funs)


def run(K=6, n_groups=30):
    d = pickle.load(open(SRC, 'rb'))
    groups = d['groups'][:n_groups]
    n_groups = len(groups)
    print('载入 %d 组' % n_groups, flush=True)

    # 逐组缓存 (u, P_total, Q_total, P_a, Q_a, P_b, Q_b, P0, truth)
    data = []
    for gi, g in enumerate(groups):
        df = g['df']
        u = (df['BUS3 UD'].values + 1j * df['BUS3 UQ'].values).astype(complex)
        P_total = df['PLOD 3[LOAD3 345.00]1'].values.astype(float)
        Q_total = df['QLOD 3[LOAD3 345.00]1'].values.astype(float)
        P_a = df['BUS3 MA P'].values.astype(float)
        Q_a = df['BUS3 MA Q'].values.astype(float)
        P_b = df['BUS3 MB P'].values.astype(float)
        Q_b = df['BUS3 MB Q'].values.astype(float)
        truth = np.asarray(g['truth'], dtype=float)
        P0 = float(P_total[0])
        data.append((u, P_total, Q_total, P_a, Q_a, P_b, Q_b, P0, truth))

    # ---- A. Fisher 理论(30 组) ----
    kap_s_list, kap_t_list = [], []
    crb_s_list, crb_t_list = [], []
    for (u, _, _, _, _, _, _, P0, truth) in data:
        Js, Jt = build_jac_real(u, truth, P0)
        if Js is None:
            kap_s_list.append(np.nan); kap_t_list.append(np.nan)
            crb_s_list.append(np.full(8, np.nan)); crb_t_list.append(np.full(8, np.nan))
            continue
        _, ev_s, _ = fisher(Js, truth)
        _, ev_t, _ = fisher(Jt, truth)
        kap_s_list.append(ev_s[-1] / max(ev_s[0], 1e-300))
        kap_t_list.append(ev_t[-1] / max(ev_t[0], 1e-300))
        Fs = Js * truth[None, :]; Ft = Jt * truth[None, :]
        crb_s_list.append(np.sqrt(np.diag(np.linalg.pinv(Fs.T @ Fs))))
        crb_t_list.append(np.sqrt(np.diag(np.linalg.pinv(Ft.T @ Ft))))
    kap_s = np.array(kap_s_list); kap_t = np.array(kap_t_list)
    crb_s = np.array(crb_s_list); crb_t = np.array(crb_t_list)
    ratio = kap_t / np.maximum(kap_s, 1e-300)

    # ---- B. 实证多起点(30 组, K 起点) ----
    E_total = []; E_split = []
    for gi, (u, P, Q, P_a, Q_a, P_b, Q_b, P0, truth) in enumerate(data):
        print('  实证 g=%d/%d ...' % (gi + 1, n_groups), flush=True)
        xs_t, _ = run_restarts_total(u, P, Q, K)
        xs_s, _ = run_restarts_split(u, P_a, Q_a, P_b, Q_b, P0, K)
        E_total.append(np.median(np.array([relerr(x, truth) for x in xs_t]), axis=0))
        E_split.append(np.median(np.array([relerr(x, truth) for x in xs_s]), axis=0))
    E_total = np.array(E_total); E_split = np.array(E_split)

    # ==================== 汇总输出 ====================
    lines = []
    lines.append('=' * 100)
    lines.append('PSSE 真实数据 30 组分电机观测 vs 母线总功率观测 批量统计 (K=%d)' % K)
    lines.append('=' * 100)
    lines.append('')
    lines.append('【A. Fisher 理论(真值处, 相对灵敏度归一化, 30 组统计)】')
    lines.append('条件数 κ (中位 [IQR]):')
    lines.append('  split(4 通道) κ = %.3g  [%.3g, %.3g]' % (np.nanmedian(kap_s), np.nanpercentile(kap_s, 25), np.nanpercentile(kap_s, 75)))
    lines.append('  total(2 通道) κ = %.3g  [%.3g, %.3g]' % (np.nanmedian(kap_t), np.nanpercentile(kap_t, 25), np.nanpercentile(kap_t, 75)))
    lines.append('  恶化倍数 κ_total/κ_split 中位 = %.3g×  [%.3g, %.3g]' % (np.nanmedian(ratio), np.nanpercentile(ratio, 25), np.nanpercentile(ratio, 75)))
    lines.append('')
    lines.append('每参数 CRB ∝ sqrt(diag(F^-1)) 中位(相对误差下界), 恶化倍数中位:')
    lines.append('%-8s %-14s %-14s %-10s' % ('参数', 'split', 'total', '恶化倍数'))
    lines.append('-' * 60)
    for j, p in enumerate(PNAMES):
        lines.append('%-8s %-14.4g %-14.4g %-10.1f'
                     % (p, np.nanmedian(crb_s[:, j]), np.nanmedian(crb_t[:, j]),
                        np.nanmedian(crb_t[:, j] / np.maximum(crb_s[:, j], 1e-300))))
    lines.append('-' * 60)
    lines.append('')
    lines.append('【B. 实证: 多起点辨识 30 组统计(K=%d)】' % K)
    lines.append('')
    lines.append('逐参数中位误差%(naive=fun 最小, 组内多起点取中位):')
    lines.append('%-8s ' % '观测' + ' '.join('%-9s' % p for p in PNAMES))
    lines.append('-' * 100)
    lines.append('%-8s ' % 'total' + ' '.join('%-9.2f' % m for m in np.median(E_total, axis=0) * 100))
    lines.append('%-8s ' % 'split' + ' '.join('%-9.2f' % m for m in np.median(E_split, axis=0) * 100))
    lines.append('-' * 100)
    lines.append('')
    lines.append('关键指标对比(30 组中位):')
    lines.append('%-16s %-16s %-16s %-16s' % ('指标', 'total', 'split', 'split 改善'))
    lines.append('-' * 64)
    for tag, idx in [('动态6参数中位', DYN), ('P_im 中位', PIM), ('H2 中位', H2),
                     ('a 中位', [0, 4]), ('b 中位', [1, 5])]:
        mt = np.median(E_total[:, idx]) * 100
        ms = np.median(E_split[:, idx]) * 100
        lines.append('%-16s %-16.2f %-16.2f %-16s'
                     % (tag + '%', mt, ms, ('%.2f×' % (mt / max(ms, 1e-9))) if ms > 0 else '-'))
    lines.append('-' * 64)
    lines.append('')
    lines.append('P_im / H2 总量-差量分解(30 组中位, naive):')
    lines.append('%-24s %-14s %-14s' % ('组合量', 'total 误差%', 'split 误差%'))
    lines.append('-' * 64)
    # 用 30 组各自的中位误差近似
    Et = np.median(E_total, axis=0); Es = np.median(E_split, axis=0)
    for i, cname in enumerate(['P_im 总量(P_ima+P_imb)', 'P_im 差量(P_ima-P_imb)',
                                'H2 总量(H2_a+H2_b)', 'H2 差量(H2_a-H2_b)']):
        lines.append('%-24s %-14.2f %-14.2f' % (cname, Et[PIM[i % 2]] * 100, Es[PIM[i % 2]] * 100))
    lines.append('  (注: 逐参数中位误差作为总量/差量代理; 精确分解见单组版)')
    lines.append('-' * 64)
    lines.append('')
    lines.append('=' * 100)
    lines.append('结论判据(30 组统计):')
    lines.append('  ① 若恶化倍数 κ_total/κ_split 中位稳定远大于 1 → 分电机观测消除 A/B 混淆的')
    lines.append('     几何优势在 30 组真实数据上普遍成立。')
    lines.append('  ② 若实证 split 的 P_im/H2 中位误差相对 total 无显著改善 → 结构失配是')
    lines.append('     30 组真实数据的主导误差源(不随组别漂移), 分电机观测只能消除 A/B 混淆、')
    lines.append('     不能消除失配偏置 → 与 A2.18 单组结论一致。')
    lines.append('  ③ 工程约束不变: 实际只有母线总功率单一观测, 最终解法仍须回到')
    lines.append('     单一总功率观测(多起点 + 先验/正则)。')
    lines.append('=' * 100)

    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ng = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(K, ng)

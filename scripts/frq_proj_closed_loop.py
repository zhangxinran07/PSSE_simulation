# -*- coding: utf-8 -*-
"""频率依赖 Δf 投影闭环验证: 把 Δf(含 V·Δf, V²·Δf 交互)纳入 obj_mix_iden 的回归投影,
看能否在单一总功率观测下吸收结构失配、提升 P_im/H2 精度。

背景(A2.21): truth 处残差加 Δf 投影 RSS 降 71.5%(交互项主导)、加 3 随机列仅降 0.0% →
Δf 是「物理吸收」非过拟合, 坐实频率依赖 Qfrq=-1 是结构失配主因之一。
本脚本做「闭环」: 把扩展投影实际用进辨识目标函数, 对比 baseline([1,V,V²]) vs frq(加 Δf)。

实验设计(双数据对照):
  A. ambient 低频 30 组(ori_data_mix_split_20261107_185811.pkl) —— Δf std≈0.0246Hz, 失配轻,
     预期: frq 投影 vs baseline 无显著差异(Δf 太小吸收不了失配) → 坐实「频率失配只在 Δf 大的高频激励下重要」。
  B. g0 probing 高频单组(diag_root_g0.pkl) —— Δf std≈0.0655Hz(ambient 的 2.7×), 失配主导,
     预期: frq 投影可吸收部分失配、P_im/H2 有改善 → 频率投影闭环在高频失配下有效。

投影口径: baseline X=[1,V,V²]; frq X=[1,V,V²,Δf,V·Δf,V²·Δf](Δf 除以 std 归一化, span 不变、数值稳)。
输出 Sample_Generation/results/frq_proj_closed_loop.txt
用法: python scripts/frq_proj_closed_loop.py [K] [n_groups]
"""
import numpy as np
import os
import sys
import pickle
import math
from scipy import optimize as opt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import (obj_mix_iden, obj_mix_jac,
                            _core_con, _core_var, _core_con_jac, _core_var_jac,
                            _trim_k, regress, _proj_dss)

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
SRC_AMBIENT = os.path.join(RES, 'ori_data_mix_split_20261107_185811.pkl')
SRC_PROBING = os.path.join(RES, 'diag_root_g0.pkl')
OUT = os.path.join(RES, 'frq_proj_closed_loop.txt')

W0 = 2 * math.pi * 60
DT = 0.01
lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
SEED = 20261108


def relerr(x, xt):
    return np.abs(np.asarray(x) - np.asarray(xt)) / np.abs(np.asarray(xt))


def compute_df(u, dt=DT):
    """母线频率偏差(标幺): 相位 unwrap 后梯度 / (2π·60·dt)。"""
    theta = np.unwrap(np.angle(u))
    return np.gradient(theta) / (W0 * dt)


# ==================== 扩展投影(加 Δf)目标 + 解析梯度 ====================
def regress_frq(V, P, Q, dfn):
    """设计矩阵 [1,V,V²,Δf,V·Δf,V²·Δf] 的最小二乘, 返回 (beta.T, RSS)。"""
    X = np.column_stack((np.ones(len(V)), V, V ** 2, dfn, V * dfn, V ** 2 * dfn))
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ np.column_stack((P, Q)))
    resid = np.column_stack((P, Q)) - X @ beta
    return beta.T, np.sum(resid ** 2)


def obj_mix_iden_frq(para, u, P, Q, dfn, dt=DT):
    """与 obj_mix_iden 同构, 仅回归投影矩阵扩展为 [1,V,V²,Δf,V·Δf,V²·Δf]。"""
    a_a, b_a, H2_a, mva_a = para[0] / 0.75, para[1], para[2] / 0.75, P[0] * para[3]
    a_b, b_b, H2_b, mva_b = para[4] / 0.75, para[5], para[6] / 0.75, P[0] * para[7]
    ok_a, P_pa, Q_pa = _core_con(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb = _core_var(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return 1e16
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n - k)
    _, ss = regress_frq(abs(u[sl]), (P_pa + P_pb - P)[sl], (Q_pa + Q_pb - Q)[sl], dfn[sl])
    return math.sqrt(ss) / (n - 2 * k) * 1000


def _get_proj_frq(u, k, dfn):
    """扩展投影的 (X, Qr): X 为 (m,6), Qr=(XᵀX)⁻¹Xᵀ 为 (6,m)。"""
    V = abs(u[k:len(u) - k])
    m = len(V)
    df = dfn[k:len(u) - k]
    X = np.column_stack((np.ones(m), V, V ** 2, df, V * df, V ** 2 * df))
    Qr = np.linalg.solve(X.T @ X, X.T)
    return X, Qr


def obj_mix_jac_frq(para, u, P, Q, dfn, dt=DT):
    """obj_mix_iden_frq 的解析梯度: 与 obj_mix_jac 同构, 仅投影矩阵扩到 6 列。"""
    a_a, b_a, H2_a, mva_a = para[0] / 0.75, para[1], para[2] / 0.75, P[0] * para[3]
    a_b, b_b, H2_b, mva_b = para[4] / 0.75, para[5], para[6] / 0.75, P[0] * para[7]
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return np.zeros(8)
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n - k)

    dY_P = np.empty((n, 8)); dY_Q = np.empty((n, 8))
    dY_P[:, 0] = dPa[:, 0] / 0.75; dY_P[:, 1] = dPa[:, 1]; dY_P[:, 2] = dPa[:, 2] / 0.75
    dY_P[:, 3] = P[0] * (P_pa / mva_a)
    dY_P[:, 4] = dPb[:, 0] / 0.75; dY_P[:, 5] = dPb[:, 1]; dY_P[:, 6] = dPb[:, 2] / 0.75
    dY_P[:, 7] = P[0] * (P_pb / mva_b)
    dY_Q[:, 0] = dQa[:, 0] / 0.75; dY_Q[:, 1] = dQa[:, 1]; dY_Q[:, 2] = dQa[:, 2] / 0.75
    dY_Q[:, 3] = P[0] * (Q_pa / mva_a)
    dY_Q[:, 4] = dQb[:, 0] / 0.75; dY_Q[:, 5] = dQb[:, 1]; dY_Q[:, 6] = dQb[:, 2] / 0.75
    dY_Q[:, 7] = P[0] * (Q_pb / mva_b)

    X, Qr = _get_proj_frq(u, k, dfn)
    yP = (P_pa + P_pb - P)[sl]
    yQ = (Q_pa + Q_pb - Q)[sl]
    beta = Qr @ np.column_stack((yP, yQ))
    resP = yP - X @ beta[:, 0]
    resQ = yQ - X @ beta[:, 1]
    SS = np.sum(resP ** 2) + np.sum(resQ ** 2)
    if SS <= 0:
        return np.zeros(8)
    dSS = _proj_dss(resP, resQ, np.ascontiguousarray(dY_P[sl, :]),
                    np.ascontiguousarray(dY_Q[sl, :]), X, Qr)
    return (1000.0 / (n - 2 * k)) * dSS / (2 * np.sqrt(SS))


def _sample_x0(obj_func, args, lo, hi, max_tries=500):
    for _ in range(max_tries):
        x0 = np.random.uniform(lo, hi)
        if obj_func(x0, *args) < 1e15:
            return x0
    return np.random.uniform(lo, hi)


def run_restarts(obj_func, jac_func, args, K):
    """K 起点独立 trust-constr, 返回 (xs, funs)。"""
    xs, funs = [], []
    for r in range(K):
        np.random.seed(SEED + r * 1000)
        x0 = _sample_x0(obj_func, args, lb * 2, ub * 2)
        res = opt.minimize(obj_func, x0, args=args, jac=jac_func, method='trust-constr',
                           bounds=np.transpose(np.array([lb * 2, ub * 2])),
                           options={'disp': False}, tol=1e-8)
        xs.append(np.asarray(res.x))
        funs.append(res.fun)
    return np.array(xs), np.array(funs)


def load_ambient(n_groups):
    """ambient 30 组: 总功率观测(u, P_total, Q_total, truth)。dfn 归一化 Δf。"""
    d = pickle.load(open(SRC_AMBIENT, 'rb'))
    groups = d['groups'][:n_groups]
    data = []
    for g in groups:
        df = g['df']
        u = (df['BUS3 UD'].values + 1j * df['BUS3 UQ'].values).astype(complex)
        P = df['PLOD 3[LOAD3 345.00]1'].values.astype(float)
        Q = df['QLOD 3[LOAD3 345.00]1'].values.astype(float)
        truth = np.asarray(g['truth'], dtype=float)
        dfq = compute_df(u)
        k = _trim_k(len(u))
        dfn = dfq / (np.std(dfq[k:len(u) - k]) + 1e-12)
        data.append((u, P, Q, truth, dfn, np.std(dfq[k:len(u) - k])))
    return data


def load_probing():
    """g0 probing 单组: (u, P, Q, truth, dfn, df_std)。"""
    r = pickle.load(open(SRC_PROBING, 'rb'))
    u, P, Q, truth = r['u'], r['P'], r['Q'], np.asarray(r['truth'], dtype=float)
    dfq = compute_df(u)
    k = _trim_k(len(u))
    dfn = dfq / (np.std(dfq[k:len(u) - k]) + 1e-12)
    return (u, P, Q, truth, dfn, np.std(dfq[k:len(u) - k]))


def identify_one(u, P, Q, dfn, truth, K, use_frq):
    """对单组数据跑 baseline(use_frq=False) 或 frq(use_frq=True) 多起点, 返回中位误差。"""
    if use_frq:
        args = (u, P, Q, dfn, DT)
        xs, _ = run_restarts(obj_mix_iden_frq, obj_mix_jac_frq, args, K)
    else:
        args = (u, P, Q, DT)
        xs, _ = run_restarts(obj_mix_iden, obj_mix_jac, args, K)
    return np.median(np.array([relerr(x, truth) for x in xs]), axis=0)


def fmt_row(tag, E):
    return '%-8s ' % tag + ' '.join('%-9.2f' % (e * 100) for e in E)


def run(K=8, n_groups=30):
    lines = []
    lines.append('=' * 110)
    lines.append('频率依赖 Δf 投影闭环: baseline([1,V,V²]) vs frq(+Δf,V·Δf,V²·Δf) 辨识对比')
    lines.append('单一总功率观测, K=%d 起点多起点, naive fun-min, 逐参数中位误差%%' % K)
    lines.append('=' * 110)
    lines.append('')

    # ---------- A. ambient 30 组 ----------
    amb = load_ambient(n_groups)
    lines.append('【A. ambient 低频 30 组(失配轻, Δf std 中位≈%.4g Hz)】' % (
        np.median([a[5] for a in amb]) * 60))
    lines.append('%-8s ' % '投影' + ' '.join('%-9s' % p for p in PNAMES))
    lines.append('-' * 110)
    E_base = []; E_frq = []
    for gi, (u, P, Q, truth, dfn, dfs) in enumerate(amb):
        eb = identify_one(u, P, Q, dfn, truth, K, False)
        ef = identify_one(u, P, Q, dfn, truth, K, True)
        E_base.append(eb); E_frq.append(ef)
        print('  ambient g=%d/%d done (Δf std=%.4g Hz)' % (gi + 1, len(amb), dfs * 60), flush=True)
    E_base = np.array(E_base); E_frq = np.array(E_frq)
    lines.append(fmt_row('baseline', np.median(E_base, axis=0)))
    lines.append(fmt_row('frq', np.median(E_frq, axis=0)))
    lines.append('-' * 110)
    lines.append('%-18s %-16s %-16s %-16s' % ('指标', 'baseline%', 'frq%', 'frq改善'))
    lines.append('-' * 70)
    for tag, idx in [('动态6参数中位', DYN), ('P_im 中位', PIM), ('H2 中位', H2),
                     ('a 中位', [0, 4]), ('b 中位', [1, 5])]:
        mb = np.median(E_base[:, idx]) * 100
        mf = np.median(E_frq[:, idx]) * 100
        lines.append('%-18s %-16.2f %-16.2f %-16s'
                     % (tag, mb, mf, ('%.2f×' % (mb / max(mf, 1e-9))) if mf > 0 else '-'))
    lines.append('-' * 70)
    lines.append('')

    # ---------- B. g0 probing 单组 ----------
    u, P, Q, truth, dfn, dfs = load_probing()
    lines.append('【B. g0 probing 高频单组(失配主导, Δf std≈%.4g Hz = ambient 的 %.1f×)】'
                 % (dfs * 60, dfs / (np.median([a[5] for a in amb]) + 1e-12)))
    lines.append('%-8s ' % '投影' + ' '.join('%-9s' % p for p in PNAMES))
    lines.append('-' * 110)
    eb0 = identify_one(u, P, Q, dfn, truth, K, False)
    ef0 = identify_one(u, P, Q, dfn, truth, K, True)
    lines.append(fmt_row('baseline', eb0))
    lines.append(fmt_row('frq', ef0))
    lines.append('-' * 110)
    lines.append('%-18s %-16s %-16s %-16s' % ('指标', 'baseline%', 'frq%', 'frq改善'))
    lines.append('-' * 70)
    for tag, idx in [('动态6参数中位', DYN), ('P_im 中位', PIM), ('H2 中位', H2),
                     ('a 中位', [0, 4]), ('b 中位', [1, 5])]:
        mb = np.median(eb0[idx]) * 100
        mf = np.median(ef0[idx]) * 100
        lines.append('%-18s %-16.2f %-16.2f %-16s'
                     % (tag, mb, mf, ('%.2f×' % (mb / max(mf, 1e-9))) if mf > 0 else '-'))
    lines.append('-' * 70)
    lines.append('')
    lines.append('=' * 110)
    lines.append('结论判据:')
    lines.append('  A 段(ambient): 若 frq≈baseline → 低频下 Δf 太小、频率失配不构成误差源,')
    lines.append('     坐实 A2.20「ambient 下混淆(而非失配)主导」。')
    lines.append('  B 段(g0 probing): 若 frq 显著优于 baseline → 频率投影闭环在高频失配下有效,')
    lines.append('     与 A2.21「Δf 是物理吸收」闭环自洽。')
    lines.append('=' * 110)

    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ng = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(K, ng)

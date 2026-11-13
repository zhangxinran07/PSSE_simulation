# -*- coding: utf-8 -*-
"""PSSE 真实数据(g=0 单组)上的「分电机功率观测 vs 母线总功率观测」辨识对照。

回答张老师「多电机功率量测找找规律」(2026-11-08) 的真实数据版 —— 零失配版
(split_vs_total_power.py) 已证: 分电机观测可消除 A/B 混淆、条件数降约 2 个数量级、
单起点无局部极小; 本脚本用唯一幸存的含分电机通道的真实样本 diag_root_g0.pkl 验证:
结构失配下「分电机观测改善 P_im/H2 精度」是否依然成立。

数据口径(已逐项核实):
  u = ud + 1j*uq (母线 3 电压, 标幺)
  total 目标: P = PLOD3, Q = QLOD3 (母线总功率, 含静态负荷 P + 固定电容 Q)
  split 目标: P_a = BUS3 MA P, Q_a = BUS3 MA Q   (con 组恒转矩电机, 纯电机通道)
              P_b = BUS3 MB P, Q_b = BUS3 MB Q   (var 组变转矩电机, 纯电机通道)
  MA 对应 para[0:4](_core_con), MB 对应 para[4:8](_core_var);
  truth = [a_a,b_a,H2_a,P_ima,a_b,b_b,H2_b,P_imb]
  MC P/Q = 0(无第 3 台电机), S P = 静态负荷(约 2.01 MW 恒定), S Q = 0。
  Q 通道说明: MA Q+MB Q+MC Q+S Q = QLOD3 + 0.9333(与|u|^2 相关=1.0, 即固定并联电容
  Q_c∝V^2), 故 total 的 Q 目标含该电容, split 的 Q 目标是纯净电机无功; 二者都被
  obj 里的二次回归投影(span{1,|u|,|u|^2})吸收, 不影响电机参数对比口径。

方法:
  A. Fisher 理论(真值处): split(4 通道) vs total(2 通道) 的条件数 κ 与逐参数 CRB。
  B. 实证(K 起点): 同种子空间下 total(obj_mix_iden) vs split(obj_mix_iden_split) 多起点,
     分别取 naive(fun 最小)/oracle(真值误差最小), 对比逐参数误差、P_im/H2 总量-差量分解、
     A/B 带符号误差相关性(此消彼长)。

输出 Sample_Generation/results/split_vs_total_psse.txt
"""
import numpy as np
import os
import sys
import time
import math
import pickle
from scipy import optimize as opt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import (obj_mix_iden, obj_mix_jac,
                            _core_con, _core_var, _core_con_jac, _core_var_jac,
                            _trim_k, regress, _get_proj, _proj_dss)

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
SRC = os.path.join(RES, 'diag_root_g0.pkl')
OUT = os.path.join(RES, 'split_vs_total_psse.txt')

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


# ==================== 分电机观测目标函数 + 数值梯度 + 辨识 ====================
def obj_mix_iden_split(para, u, P_a, Q_a, P_b, Q_b, P0, dt=DT):
    """分电机观测目标: 4 通道残差(每台电机 P/Q 各自回归)之和的开方。基值 P0(=P[0])。"""
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


def obj_mix_jac_split_num(para, u, P_a, Q_a, P_b, Q_b, P0, dt=DT, eps=1e-6):
    """数值梯度(前向差分), 8 维。"""
    f0 = obj_mix_iden_split(para, u, P_a, Q_a, P_b, Q_b, P0, dt)
    g = np.zeros(8)
    for i in range(8):
        xp = np.array(para, dtype=float)
        h = eps * max(1.0, abs(para[i]))
        xp[i] += h
        g[i] = (obj_mix_iden_split(xp, u, P_a, Q_a, P_b, Q_b, P0, dt) - f0) / h
    return g


def _sample_x0(obj_func, args, lo, hi, max_tries=500):
    """采样一个有限(非 1e16 惩罚)的初始点。

    原 identify() 用「obj < 0.5」筛初值, 但真实数据结构失配下 obj(truth)≈4.9、
    随机点 min≈4.4, <0.5 永无满足 → 死循环。改为只排除发散/惩罚点。
    """
    for _ in range(max_tries):
        x0 = np.random.uniform(lo, hi)
        if obj_func(x0, *args) < 1e15:
            return x0
    return np.random.uniform(lo, hi)   # 兜底: 全部发散时也返回, 交由 trust-constr 自行处理


def obj_mix_jac_split(para, u, P_a, Q_a, P_b, Q_b, P0, dt=DT):
    """obj_mix_iden_split 的解析梯度(与 obj_mix_jac 同构, 分 con/var 两通道各自投影)。"""
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


def identify_split(u, P_a, Q_a, P_b, Q_b, P0, dt=DT, tol=1e-8):
    """单起点辨识(trust-constr, split 目标)。无 obj<0.5 初值筛选(避免真实数据死锁)。"""
    t_start = time.perf_counter()
    args = (u, P_a, Q_a, P_b, Q_b, P0, dt)
    x0 = _sample_x0(obj_mix_iden_split, args, lb * 2, ub * 2)
    result = opt.minimize(obj_mix_iden_split, x0, args=args,
                          jac=obj_mix_jac_split, method='trust-constr',
                          bounds=np.transpose(np.array([lb * 2, ub * 2])),
                          options={'disp': False}, tol=tol)
    return result, time.perf_counter() - t_start


# ==================== Fisher 理论: split / total 有效 Jacobian ====================
def build_jac_real(u, truth, P0, dt=DT):
    """在真值(辨识空间)处构造 split 与 total 的有效 Jacobian(相对灵敏度用)。
    返回 (J_split, J_total), 形状 (4m,8) 与 (2m,8)。"""
    a_a, b_a, H2_a, mva_a = truth[0] / 0.75, truth[1], truth[2] / 0.75, P0 * truth[3]
    a_b, b_b, H2_b, mva_b = truth[4] / 0.75, truth[5], truth[6] / 0.75, P0 * truth[7]
    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    assert ok_a and ok_b, 'truth 处 Jacobian 发散'

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

    J_split = np.vstack([JPa, JQa, JPb, JQb])      # (4m, 8)
    J_total = np.vstack([JPa + JPb, JQa + JQb])    # (2m, 8) = 秩降线性组合
    return J_split, J_total


def fisher(J, truth):
    """相对灵敏度归一化 + Fisher 特征分解。返回 (F, evals升序, evecs)。"""
    Jn = J * truth[None, :]
    F = Jn.T @ Jn
    evals, evecs = np.linalg.eigh(F)
    return F, evals, evecs


# ==================== 多起点辨识 ====================
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
        res, _ = identify_split(u, P_a, Q_a, P_b, Q_b, P0, tol=1e-8)
        xs.append(np.asarray(res.x))
        funs.append(res.fun)
    return np.array(xs), np.array(funs)


def summarize(name, xs, funs, truth):
    """对一批 K 个 restart 解, 汇总 naive(fun 最小)与 oracle(误差最小)的逐参数误差。"""
    E = np.array([relerr(x, truth) for x in xs])      # (K, 8)
    i_fun = int(np.argmin(funs))
    i_all = int(np.argmin(np.median(E, axis=1)))
    i_pim = int(np.argmin(np.median(E[:, PIM], axis=1)))
    i_h2 = int(np.argmin(np.median(E[:, H2], axis=1)))
    return {
        'naive': (xs[i_fun], funs[i_fun], E[i_fun]),
        'oracle_all': (xs[i_all], funs[i_all], E[i_all]),
        'oracle_pim': (xs[i_pim], funs[i_pim], E[i_pim]),
        'oracle_h2': (xs[i_h2], funs[i_h2], E[i_h2]),
        'E_all': E,
        'funs': funs,
    }


def run(K=12):
    d = pickle.load(open(SRC, 'rb'))
    df = d['df']
    u = d['u'].astype(complex)
    P_total = d['P']
    Q_total = d['Q']
    P_a = df['BUS3 MA P'].values
    Q_a = df['BUS3 MA Q'].values
    P_b = df['BUS3 MB P'].values
    Q_b = df['BUS3 MB Q'].values
    truth = np.asarray(d['truth'], dtype=float)
    P0 = float(P_total[0])
    print('P0 = %.4f' % P0, flush=True)

    lines = []
    lines.append('=' * 100)
    lines.append('PSSE 真实数据(g=0 单组) 分电机观测 vs 母线总功率观测 辨识对照 (K=%d)' % K)
    lines.append('truth = ' + '  '.join('%-8s=%.4f' % (p, t) for p, t in zip(PNAMES, truth)))
    lines.append('=' * 100)

    # ---- A. Fisher 理论 ----
    Js, Jt = build_jac_real(u, truth, P0)
    _, ev_s, _ = fisher(Js, truth)
    _, ev_t, evec_t = fisher(Jt, truth)
    kap_s = ev_s[-1] / max(ev_s[0], 1e-300)
    kap_t = ev_t[-1] / max(ev_t[0], 1e-300)
    Fs = Js * truth[None, :]; Ft = Jt * truth[None, :]
    crb_s = np.sqrt(np.diag(np.linalg.pinv(Fs.T @ Fs)))
    crb_t = np.sqrt(np.diag(np.linalg.pinv(Ft.T @ Ft)))
    lines.append('')
    lines.append('【A. Fisher 理论(真值处, 相对灵敏度归一化)】')
    lines.append('条件数 κ = λmax/λmin:')
    lines.append('  split(4 通道) κ = %.3g' % kap_s)
    lines.append('  total(2 通道) κ = %.3g' % kap_t)
    lines.append('  恶化倍数 κ_total/κ_split = %.3g ×' % (kap_t / max(kap_s, 1e-300)))
    lines.append('')
    lines.append('每参数相对误差下界 CRB ∝ sqrt(diag(F^-1)):')
    lines.append('%-8s %-14s %-14s %-10s' % ('参数', 'split', 'total', '恶化倍数'))
    lines.append('-' * 60)
    for j, p in enumerate(PNAMES):
        lines.append('%-8s %-14.4g %-14.4g %-10.1f'
                     % (p, crb_s[j], crb_t[j], crb_t[j] / max(crb_s[j], 1e-300)))
    lines.append('-' * 60)
    lines.append('total 最小特征向量(零空间方向, 逐参数载荷 |v|):')
    lines.append('  ' + '  '.join('%-7s' % p for p in PNAMES))
    lines.append('  ' + '  '.join('%-7.3f' % v for v in np.abs(evec_t[:, 0])))
    lines.append('  说明: 载荷大的参数 = 在「观测几乎不变」方向上主导, 即总功率观测下最难辨识。')

    # ---- B. 实证多起点 ----
    print('跑 total %d 起点...' % K, flush=True)
    xs_t, funs_t = run_restarts_total(u, P_total, Q_total, K)
    print('跑 split %d 起点...' % K, flush=True)
    xs_s, funs_s = run_restarts_split(u, P_a, Q_a, P_b, Q_b, P0, K)

    T = summarize('total', xs_t, funs_t, truth)
    S = summarize('split', xs_s, funs_s, truth)

    lines.append('')
    lines.append('【B. 实证: 多起点辨识(K=%d, 同种子空间)】' % K)
    lines.append('')
    lines.append('逐参数中位误差(naive=fun 最小):')
    lines.append('%-12s ' % '观测' + ' '.join('%-9s' % p for p in PNAMES))
    lines.append('-' * 100)
    for name, R in [('total', T), ('split', S)]:
        med = np.median(R['E_all'], axis=0) * 100
        lines.append('%-12s ' % name + ' '.join('%-9.2f' % m for m in med))
    lines.append('-' * 100)
    lines.append('')
    lines.append('关键指标对比(naive):')
    lines.append('%-16s %-16s %-16s %-16s' % ('指标', 'total', 'split', 'split 改善'))
    lines.append('-' * 64)
    Et = np.median(T['E_all'], axis=0)
    Es = np.median(S['E_all'], axis=0)
    for tag, idx in [('动态6参数中位', DYN), ('P_im 中位', PIM), ('H2 中位', H2),
                     ('a 中位', [0, 4]), ('b 中位', [1, 5])]:
        mt = np.median(Et[idx]) * 100
        ms = np.median(Es[idx]) * 100
        lines.append('%-16s %-16.2f %-16.2f %-16s'
                     % (tag + '%', mt, ms, ('%.1f×' % (mt / max(ms, 1e-9))) if ms > 0 else '-'))
    lines.append('-' * 64)
    lines.append('')
    lines.append('naive vs oracle 差距(结构失配扭曲 fun 表面, fun 最小≠误差最小的净损失):')
    lines.append('%-16s %-16s %-16s' % ('方案', '动态6中位%', 'P_im中位%'))
    for name, R in [('total', T), ('split', S)]:
        lines.append('  %-14s' % name)
        for k, tag in [('naive', 'naive fun-min'), ('oracle_all', 'oracle 全参最优'),
                       ('oracle_pim', 'oracle P_im最优')]:
            lines.append('    %-14s %-16.2f %-16.2f'
                         % (tag, np.median(R[k][2][DYN]) * 100, np.median(R[k][2][PIM]) * 100))
    lines.append('')

    # ---- P_im / H2 总量-差量分解 ----
    lines.append('【C. P_im / H2 总量(可辨识) vs 差量(零空间) 分解(naive)】')
    lines.append('%-24s %-14s %-14s' % ('组合量', 'total 误差%', 'split 误差%'))
    lines.append('-' * 64)
    for name, R in [('total', T), ('split', S)]:
        x = R['naive'][0]
        SP = abs((x[3] + x[7]) - (truth[3] + truth[7])) / abs(truth[3] + truth[7])
        DP = abs((x[3] - x[7]) - (truth[3] - truth[7])) / abs(truth[3] + truth[7])
        SH = abs((x[2] + x[6]) - (truth[2] + truth[6])) / abs(truth[2] + truth[6])
        DH = abs((x[2] - x[6]) - (truth[2] - truth[6])) / abs(truth[2] + truth[6])
        R['comb'] = (SP, DP, SH, DH)
    for i, cname in enumerate(['P_im 总量(P_ima+P_imb)', 'P_im 差量(P_ima-P_imb)',
                                'H2 总量(H2_a+H2_b)', 'H2 差量(H2_a-H2_b)']):
        ct = T['comb'][i] * 100
        cs = S['comb'][i] * 100
        lines.append('%-24s %-14.2f %-14.2f' % (cname, ct, cs))
    lines.append('-' * 64)
    lines.append('')

    # ---- A/B 带符号误差相关性(用 K 个 restart 的带符号误差求相关, 更稳健) ----
    lines.append('【D. A/B 带符号误差相关性(此消彼长实证)】')
    for name, xs_all in [('total', xs_t), ('split', xs_s)]:
        sPa = (xs_all[:, 3] - truth[3]) / truth[3]
        sPb = (xs_all[:, 7] - truth[7]) / truth[7]
        sHa = (xs_all[:, 2] - truth[2]) / truth[2]
        sHb = (xs_all[:, 6] - truth[6]) / truth[6]
        rP = np.corrcoef(sPa, sPb)[0, 1]
        rH = np.corrcoef(sHa, sHb)[0, 1]
        lines.append('  %-6s  P_ima vs P_imb 带符号误差 r = %+.3f   H2_a vs H2_b r = %+.3f'
                     % (name, rP, rH))
    lines.append('  ※ 强负相关(r<-0.5) = 该观测下存在「总量守恒式病态分配/此消彼长」。')
    lines.append('')
    lines.append('=' * 100)
    lines.append('结论判据:')
    lines.append('  ① 条件数 κ_total/κ_split = %.0f×' % (kap_t / max(kap_s, 1e-300)))
    lines.append('     → 分电机观测消除 A/B 混淆的几何优势, 在真实数据的真值处依然成立。')
    lines.append('  ② 实证(结构失配下): 若 split 的 P_im/H2 中位误差显著小于 total, 则')
    lines.append('     「多电机功率量测改善 P_im/H2 精度」在真实数据上成立; 反之说明结构失配')
    lines.append('     是当前真实数据的主导误差源, 分电机观测只能消除 A/B 混淆、不能消除失配偏置。')
    lines.append('  ③ 工程约束不变: 实际只有母线总功率单一观测, 分电机观测仅用于找规律,')
    lines.append('     最终解法仍须回到单一总功率观测(多起点 + 先验/正则)。')
    lines.append('=' * 100)

    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    run(K)

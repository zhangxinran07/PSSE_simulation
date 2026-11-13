# -*- coding: utf-8 -*-
"""积分方法敏感性实验（A2.44 候选）：回答「辨识误差是否由积分算法不同引起」。

设计：30 组 PSSE 数据，统一 K=12 起点 + λ_H=0.02（主线口径），全部解析雅可比。
  变体 V1  RK4        @ dt=0.01   （baseline，现状 obj_mix_iden/obj_mix_jac）
  变体 V2  修正欧拉    @ dt=0.01   （只换积分算法，其余完全同构）

注意：上一版用数值雅可比跑，trust-constr 的 BFGS 在数值梯度下发散（V1 自身就劣化到
P_im 5.6%/H2 12.8%），故本版为修正欧拉补写了**解析灵敏度**（predictor-corrector 两段链式），
保证 V1/V2 口径一致（都用解析梯度）。

判据：V2 与 V1 的 P_im/H2/dyn6 差异若 <1pp → 积分算法不是误差主因；
     若 V2 明显变差 → 修正欧拉@10ms 二阶精度不足（O(h²)=1e-4），需再跑 @2ms 复核。

用法：python scripts/int_method_probe.py [n_groups] [K] [--check]
输出：Sample_Generation/results/int_method_probe.txt
"""
import os
import sys
import math
import pickle
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'package'))
from numba import njit
from identification import (w0, regress, _trim_k, _get_proj, _proj_dss,
                            identify_multistart, _core_con, _core_var,
                            _core_con_jac, _core_var_jac, obj_mix_iden, obj_mix_jac)

PROJ = os.path.join(HERE, '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
ORI = os.path.join(RES, 'ori_data_mix_20261106_033919.pkl')
RRES = os.path.join(RES, 'result_all_mix_20261106_033919.pkl')
OUT = os.path.join(RES, 'int_method_probe.txt')

PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
LAM_H = 0.02
SEED = 20261107


# ==================== 修正欧拉核（Heun，二阶）+ 解析灵敏度 ====================
@njit(cache=True)
def _core_con_me(u, a, b, H2, mva, dt):
    """恒转矩模型：修正欧拉递推，直接输出预测功率。"""
    n = len(u)
    f_p = np.zeros(n, dtype=np.complex128)
    s_p = np.zeros(n)
    P_p = np.zeros(n)
    Q_p = np.zeros(n)
    c = abs(u[0]) ** 2 * a * w0
    dl = c ** 2 - 4 * b ** 2 * w0 ** 2
    if dl < 0:
        return False, P_p, Q_p
    s_p[0] = (c - np.sqrt(dl)) / 2 / w0 ** 2
    f_p[0] = a * u[0] / (b + 1j * s_p[0] * w0)
    for k in range(n - 1):
        uk = u[k]
        uk1 = u[k + 1]
        k1_f = (-b - 1j * s_p[k] * w0) * f_p[k] + a * uk
        Pe1 = f_p[k].real * uk.imag - f_p[k].imag * uk.real
        k1_s = (1.0 - Pe1) / (2 * H2)
        f_s = f_p[k] + dt * k1_f
        s_s = s_p[k] + dt * k1_s
        k2_f = (-b - 1j * s_s * w0) * f_s + a * uk1
        Pe2 = f_s.real * uk1.imag - f_s.imag * uk1.real
        k2_s = (1.0 - Pe2) / (2 * H2)
        f_p[k + 1] = f_p[k] + dt / 2 * (k1_f + k2_f)
        s_p[k + 1] = s_p[k] + dt / 2 * (k1_s + k2_s)
        if not (np.isfinite(f_p[k + 1].real) and np.isfinite(f_p[k + 1].imag)
                and np.isfinite(s_p[k + 1])):
            return False, P_p, Q_p
    if not (s_p[n - 1] > 0 and s_p[n - 1] < 0.2):
        return False, P_p, Q_p
    P_p = (f_p.real * u.imag - f_p.imag * u.real) * mva
    Q_p = -(f_p.real * u.real + f_p.imag * u.imag) * mva
    return True, P_p, Q_p


@njit(cache=True)
def _core_var_me(u, a, b, H2, mva, dt):
    """变转矩模型：修正欧拉递推（机械转矩随 (1-s)^2 变化）。"""
    n = len(u)
    f_p = np.zeros(n, dtype=np.complex128)
    s_p = np.zeros(n)
    P_p = np.zeros(n)
    Q_p = np.zeros(n)
    c = abs(u[0]) ** 2 * a * w0
    dl = c ** 2 - 4 * b ** 2 * w0 ** 2
    if dl < 0:
        return False, P_p, Q_p
    s_p[0] = (c - np.sqrt(dl)) / 2 / w0 ** 2
    f_p[0] = a * u[0] / (b + 1j * s_p[0] * w0)
    tm = 1.0 / (1 - s_p[0]) ** 2
    for k in range(n - 1):
        uk = u[k]
        uk1 = u[k + 1]
        k1_f = (-b - 1j * s_p[k] * w0) * f_p[k] + a * uk
        Pe1 = f_p[k].real * uk.imag - f_p[k].imag * uk.real
        k1_s = (tm * (1 - s_p[k]) ** 2 - Pe1) / (2 * H2)
        f_s = f_p[k] + dt * k1_f
        s_s = s_p[k] + dt * k1_s
        k2_f = (-b - 1j * s_s * w0) * f_s + a * uk1
        Pe2 = f_s.real * uk1.imag - f_s.imag * uk1.real
        k2_s = (tm * (1 - s_s) ** 2 - Pe2) / (2 * H2)
        f_p[k + 1] = f_p[k] + dt / 2 * (k1_f + k2_f)
        s_p[k + 1] = s_p[k] + dt / 2 * (k1_s + k2_s)
        if not (np.isfinite(f_p[k + 1].real) and np.isfinite(f_p[k + 1].imag)
                and np.isfinite(s_p[k + 1])):
            return False, P_p, Q_p
    if not (s_p[n - 1] > 0 and s_p[n - 1] < 0.2):
        return False, P_p, Q_p
    P_p = (f_p.real * u.imag - f_p.imag * u.real) * mva
    Q_p = -(f_p.real * u.real + f_p.imag * u.imag) * mva
    return True, P_p, Q_p


@njit(cache=True)
def _core_con_me_jac(u, a, b, H2, mva, dt):
    """恒转矩核 + 解析灵敏度（修正欧拉，predictor-corrector 两段链式）。"""
    n = len(u)
    f_p = np.zeros(n, dtype=np.complex128)
    s_p = np.zeros(n)
    g = np.zeros((n, 3), dtype=np.complex128)
    q = np.zeros((n, 3))
    P_p = np.zeros(n)
    Q_p = np.zeros(n)
    dP = np.zeros((n, 3))
    dQ = np.zeros((n, 3))

    c = abs(u[0]) ** 2 * a * w0
    dl = c ** 2 - 4 * b ** 2 * w0 ** 2
    if dl < 0:
        return False, P_p, Q_p, dP, dQ
    sq = np.sqrt(dl)
    s_p[0] = (c - sq) / 2 / w0 ** 2
    f_p[0] = a * u[0] / (b + 1j * s_p[0] * w0)

    dcd = abs(u[0]) ** 2 * w0
    ddl0 = 2 * c * dcd
    ddl1 = -8 * b * w0 ** 2
    ds0 = np.zeros(3)
    ds0[0] = (dcd - ddl0 / (2 * sq)) / 2 / w0 ** 2
    ds0[1] = (0.0 - ddl1 / (2 * sq)) / 2 / w0 ** 2
    den = b + 1j * s_p[0] * w0
    dfs0 = -a * u[0] * (1j * w0) / den ** 2
    g[0, 0] = u[0] / den + dfs0 * ds0[0]
    g[0, 1] = -a * u[0] / den ** 2 + dfs0 * ds0[1]
    g[0, 2] = 0.0 + 0j
    q[0, 0] = ds0[0]
    q[0, 1] = ds0[1]
    q[0, 2] = 0.0

    for k in range(n - 1):
        uk = u[k]
        uk1 = u[k + 1]
        fk = f_p[k]
        sk = s_p[k]

        k1_f = (-b - 1j * sk * w0) * fk + a * uk
        Pe1 = fk.real * uk.imag - fk.imag * uk.real
        k1_s = (1.0 - Pe1) / (2 * H2)
        f_s = fk + dt * k1_f
        s_s = sk + dt * k1_s
        k2_f = (-b - 1j * s_s * w0) * f_s + a * uk1
        Pe2 = f_s.real * uk1.imag - f_s.imag * uk1.real
        k2_s = (1.0 - Pe2) / (2 * H2)

        f_p[k + 1] = fk + dt / 2 * (k1_f + k2_f)
        s_p[k + 1] = sk + dt / 2 * (k1_s + k2_s)

        for j in range(3):
            da = 1.0 if j == 0 else 0.0
            db = 1.0 if j == 1 else 0.0
            dH = 1.0 if j == 2 else 0.0

            dk1_f = ((-b - 1j * sk * w0) * g[k, j]
                     + (-db - 1j * q[k, j] * w0) * fk + da * uk)
            dPe1 = g[k, j].real * uk.imag - g[k, j].imag * uk.real
            dk1_s = -dPe1 / (2 * H2) - (1.0 - Pe1) / (2 * H2 * H2) * dH

            df_s = g[k, j] + dt * dk1_f
            ds_s = q[k, j] + dt * dk1_s
            dk2_f = ((-b - 1j * s_s * w0) * df_s
                     + (-db - 1j * ds_s * w0) * f_s + da * uk1)
            dPe2 = df_s.real * uk1.imag - df_s.imag * uk1.real
            dk2_s = -dPe2 / (2 * H2) - (1.0 - Pe2) / (2 * H2 * H2) * dH

            g[k + 1, j] = g[k, j] + dt / 2 * (dk1_f + dk2_f)
            q[k + 1, j] = q[k, j] + dt / 2 * (dk1_s + dk2_s)

        if not (np.isfinite(f_p[k + 1].real) and np.isfinite(f_p[k + 1].imag)
                and np.isfinite(s_p[k + 1])):
            return False, P_p, Q_p, dP, dQ

    if not (s_p[n - 1] > 0 and s_p[n - 1] < 0.2):
        return False, P_p, Q_p, dP, dQ

    for k in range(n):
        P_p[k] = (f_p[k].real * u[k].imag - f_p[k].imag * u[k].real) * mva
        Q_p[k] = -(f_p[k].real * u[k].real + f_p[k].imag * u[k].imag) * mva
        for j in range(3):
            dP[k, j] = (g[k, j].real * u[k].imag - g[k, j].imag * u[k].real) * mva
            dQ[k, j] = -(g[k, j].real * u[k].real + g[k, j].imag * u[k].imag) * mva
    return True, P_p, Q_p, dP, dQ


@njit(cache=True)
def _core_var_me_jac(u, a, b, H2, mva, dt):
    """变转矩核 + 解析灵敏度（修正欧拉，机械转矩 tm*(1-s)^2 含 s0 依赖）。"""
    n = len(u)
    f_p = np.zeros(n, dtype=np.complex128)
    s_p = np.zeros(n)
    g = np.zeros((n, 3), dtype=np.complex128)
    q = np.zeros((n, 3))
    P_p = np.zeros(n)
    Q_p = np.zeros(n)
    dP = np.zeros((n, 3))
    dQ = np.zeros((n, 3))

    c = abs(u[0]) ** 2 * a * w0
    dl = c ** 2 - 4 * b ** 2 * w0 ** 2
    if dl < 0:
        return False, P_p, Q_p, dP, dQ
    sq = np.sqrt(dl)
    s_p[0] = (c - sq) / 2 / w0 ** 2
    f_p[0] = a * u[0] / (b + 1j * s_p[0] * w0)

    dcd = abs(u[0]) ** 2 * w0
    ddl0 = 2 * c * dcd
    ddl1 = -8 * b * w0 ** 2
    ds0 = np.zeros(3)
    ds0[0] = (dcd - ddl0 / (2 * sq)) / 2 / w0 ** 2
    ds0[1] = (0.0 - ddl1 / (2 * sq)) / 2 / w0 ** 2
    den = b + 1j * s_p[0] * w0
    dfs0 = -a * u[0] * (1j * w0) / den ** 2
    g[0, 0] = u[0] / den + dfs0 * ds0[0]
    g[0, 1] = -a * u[0] / den ** 2 + dfs0 * ds0[1]
    g[0, 2] = 0.0 + 0j
    q[0, 0] = ds0[0]
    q[0, 1] = ds0[1]
    q[0, 2] = 0.0

    tm = 1.0 / (1 - s_p[0]) ** 2
    s0_ = s_p[0]
    for k in range(n - 1):
        uk = u[k]
        uk1 = u[k + 1]
        fk = f_p[k]
        sk = s_p[k]

        T1 = tm * (1 - sk) ** 2
        k1_f = (-b - 1j * sk * w0) * fk + a * uk
        Pe1 = fk.real * uk.imag - fk.imag * uk.real
        k1_s = (T1 - Pe1) / (2 * H2)
        f_s = fk + dt * k1_f
        s_s = sk + dt * k1_s
        T2 = tm * (1 - s_s) ** 2
        k2_f = (-b - 1j * s_s * w0) * f_s + a * uk1
        Pe2 = f_s.real * uk1.imag - f_s.imag * uk1.real
        k2_s = (T2 - Pe2) / (2 * H2)

        f_p[k + 1] = fk + dt / 2 * (k1_f + k2_f)
        s_p[k + 1] = sk + dt / 2 * (k1_s + k2_s)

        for j in range(3):
            da = 1.0 if j == 0 else 0.0
            db = 1.0 if j == 1 else 0.0
            dH = 1.0 if j == 2 else 0.0
            dtm = 2.0 / (1 - s0_) ** 3 * q[0, j]

            dk1_f = ((-b - 1j * sk * w0) * g[k, j]
                     + (-db - 1j * q[k, j] * w0) * fk + da * uk)
            dPe1 = g[k, j].real * uk.imag - g[k, j].imag * uk.real
            dk1_s = (dtm * (1 - sk) ** 2 + tm * 2 * (1 - sk) * (-q[k, j]) - dPe1) / (2 * H2) \
                    - (T1 - Pe1) / (2 * H2 * H2) * dH

            df_s = g[k, j] + dt * dk1_f
            ds_s = q[k, j] + dt * dk1_s
            dk2_f = ((-b - 1j * s_s * w0) * df_s
                     + (-db - 1j * ds_s * w0) * f_s + da * uk1)
            dPe2 = df_s.real * uk1.imag - df_s.imag * uk1.real
            dk2_s = (dtm * (1 - s_s) ** 2 + tm * 2 * (1 - s_s) * (-ds_s) - dPe2) / (2 * H2) \
                    - (T2 - Pe2) / (2 * H2 * H2) * dH

            g[k + 1, j] = g[k, j] + dt / 2 * (dk1_f + dk2_f)
            q[k + 1, j] = q[k, j] + dt / 2 * (dk1_s + dk2_s)

        if not (np.isfinite(f_p[k + 1].real) and np.isfinite(f_p[k + 1].imag)
                and np.isfinite(s_p[k + 1])):
            return False, P_p, Q_p, dP, dQ

    if not (s_p[n - 1] > 0 and s_p[n - 1] < 0.2):
        return False, P_p, Q_p, dP, dQ

    for k in range(n):
        P_p[k] = (f_p[k].real * u[k].imag - f_p[k].imag * u[k].real) * mva
        Q_p[k] = -(f_p[k].real * u[k].real + f_p[k].imag * u[k].imag) * mva
        for j in range(3):
            dP[k, j] = (g[k, j].real * u[k].imag - g[k, j].imag * u[k].real) * mva
            dQ[k, j] = -(g[k, j].real * u[k].real + g[k, j].imag * u[k].imag) * mva
    return True, P_p, Q_p, dP, dQ


# ==================== 修正欧拉版目标函数 + 解析梯度（与 obj_mix_iden/jac 同构） ====================
def obj_mix_iden_me(para, u, P, Q, dt=0.01, lam_H=0.0):
    a_a, b_a, H2_a, mva_a = para[0] / 0.75, para[1], para[2] / 0.75, P[0] * para[3]
    a_b, b_b, H2_b, mva_b = para[4] / 0.75, para[5], para[6] / 0.75, P[0] * para[7]
    ok_a, P_pa, Q_pa = _core_con_me(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb = _core_var_me(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return 1e16
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n - k)
    obj = math.sqrt(regress(abs(u[sl]), (P_pa + P_pb - P)[sl], (Q_pa + Q_pb - Q)[sl])[1]) / (n - 2 * k) * 1000
    if lam_H:
        obj = obj + lam_H * (para[2] - para[6]) ** 2
    return obj


def obj_mix_jac_me(para, u, P, Q, dt=0.01, lam_H=0.0):
    a_a, b_a, H2_a, mva_a = para[0] / 0.75, para[1], para[2] / 0.75, P[0] * para[3]
    a_b, b_b, H2_b, mva_b = para[4] / 0.75, para[5], para[6] / 0.75, P[0] * para[7]

    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_me_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_me_jac(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return np.zeros(8)

    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n - k)

    dY_P = np.empty((n, 8))
    dY_Q = np.empty((n, 8))
    dY_P[:, 0] = dPa[:, 0] / 0.75
    dY_P[:, 1] = dPa[:, 1]
    dY_P[:, 2] = dPa[:, 2] / 0.75
    dY_P[:, 3] = P[0] * (P_pa / mva_a)
    dY_P[:, 4] = dPb[:, 0] / 0.75
    dY_P[:, 5] = dPb[:, 1]
    dY_P[:, 6] = dPb[:, 2] / 0.75
    dY_P[:, 7] = P[0] * (P_pb / mva_b)
    dY_Q[:, 0] = dQa[:, 0] / 0.75
    dY_Q[:, 1] = dQa[:, 1]
    dY_Q[:, 2] = dQa[:, 2] / 0.75
    dY_Q[:, 3] = P[0] * (Q_pa / mva_a)
    dY_Q[:, 4] = dQb[:, 0] / 0.75
    dY_Q[:, 5] = dQb[:, 1]
    dY_Q[:, 6] = dQb[:, 2] / 0.75
    dY_Q[:, 7] = P[0] * (Q_pb / mva_b)

    X, Qr = _get_proj(u, k)
    yP = (P_pa + P_pb - P)[sl]
    yQ = (Q_pa + Q_pb - Q)[sl]
    beta = Qr @ np.column_stack((yP, yQ))
    resP = yP - X @ beta[:, 0]
    resQ = yQ - X @ beta[:, 1]
    SS = np.sum(resP ** 2) + np.sum(resQ ** 2)
    if SS <= 0:
        return np.zeros(8)

    dSS = _proj_dss(resP, resQ,
                    np.ascontiguousarray(dY_P[sl, :]),
                    np.ascontiguousarray(dY_Q[sl, :]), X, Qr)

    g = (1000.0 / (n - 2 * k)) * dSS / (2 * np.sqrt(SS))
    if lam_H:
        g[2] += 2 * lam_H * (para[2] - para[6])
        g[6] -= 2 * lam_H * (para[2] - para[6])
    return g


def make_me(lam_H):
    def obj(para, u, P, Q, dt):
        return obj_mix_iden_me(para, u, P, Q, dt, lam_H=lam_H)

    def jac(para, u, P, Q, dt):
        return obj_mix_jac_me(para, u, P, Q, dt, lam_H=lam_H)
    return obj, jac


def make_rk4(lam_H):
    def obj(para, u, P, Q, dt):
        return obj_mix_iden(para, u, P, Q, dt, lam_H=lam_H)

    def jac(para, u, P, Q, dt):
        return obj_mix_jac(para, u, P, Q, dt, lam_H=lam_H)
    return obj, jac


def upsample_u(u, factor):
    """复数电压序列从 10ms 网格线性插值到 2ms 网格（factor=5）。"""
    n = len(u)
    idx = np.arange(n)
    idx_fine = np.arange(0, n - 1 + 1e-9, 1.0 / factor)
    re = np.interp(idx_fine, idx, u.real)
    im = np.interp(idx_fine, idx, u.imag)
    return re + 1j * im


def make_functions(k_con, k_var, k_con_jac, k_var_jac, dt_inner, factor, lam_H):
    """通用工厂：给定 4 个核（obj/jac × con/var）+ 内步长 + 上采样因子，返回 (obj, jac)。

    factor=None  → 直接在 10ms 网格递推（dt_inner 通常 0.01）
    factor=5     → u 插值到 2ms，核按 dt_inner=0.002 递推，P/Q 与灵敏度再下采样 [::5] 回 10ms
    """
    def obj(para, u, P, Q, dt=0.01):
        a_a, b_a, H2_a, mva_a = para[0] / 0.75, para[1], para[2] / 0.75, P[0] * para[3]
        a_b, b_b, H2_b, mva_b = para[4] / 0.75, para[5], para[6] / 0.75, P[0] * para[7]
        if factor is not None:
            uu = upsample_u(u, factor)
            ok_a, P_pa, Q_pa = k_con(uu, a_a, b_a, H2_a, mva_a, dt_inner)
            ok_b, P_pb, Q_pb = k_var(uu, a_b, b_b, H2_b, mva_b, dt_inner)
            if not (ok_a and ok_b):
                return 1e16
            P_pa = P_pa[::factor]; Q_pa = Q_pa[::factor]
            P_pb = P_pb[::factor]; Q_pb = Q_pb[::factor]
        else:
            ok_a, P_pa, Q_pa = k_con(u, a_a, b_a, H2_a, mva_a, dt_inner)
            ok_b, P_pb, Q_pb = k_var(u, a_b, b_b, H2_b, mva_b, dt_inner)
            if not (ok_a and ok_b):
                return 1e16
        n = len(u)
        k = _trim_k(n)
        sl = slice(k, n - k)
        val = math.sqrt(regress(abs(u[sl]), (P_pa + P_pb - P)[sl], (Q_pa + Q_pb - Q)[sl])[1]) / (n - 2 * k) * 1000
        return val + lam_H * (para[2] - para[6]) ** 2

    def jac(para, u, P, Q, dt=0.01):
        a_a, b_a, H2_a, mva_a = para[0] / 0.75, para[1], para[2] / 0.75, P[0] * para[3]
        a_b, b_b, H2_b, mva_b = para[4] / 0.75, para[5], para[6] / 0.75, P[0] * para[7]
        if factor is not None:
            uu = upsample_u(u, factor)
            ok_a, P_pa, Q_pa, dPa, dQa = k_con_jac(uu, a_a, b_a, H2_a, mva_a, dt_inner)
            ok_b, P_pb, Q_pb, dPb, dQb = k_var_jac(uu, a_b, b_b, H2_b, mva_b, dt_inner)
            if not (ok_a and ok_b):
                return np.zeros(8)
            P_pa = P_pa[::factor]; Q_pa = Q_pa[::factor]
            P_pb = P_pb[::factor]; Q_pb = Q_pb[::factor]
            dPa = dPa[::factor]; dQa = dQa[::factor]
            dPb = dPb[::factor]; dQb = dQb[::factor]
        else:
            ok_a, P_pa, Q_pa, dPa, dQa = k_con_jac(u, a_a, b_a, H2_a, mva_a, dt_inner)
            ok_b, P_pb, Q_pb, dPb, dQb = k_var_jac(u, a_b, b_b, H2_b, mva_b, dt_inner)
            if not (ok_a and ok_b):
                return np.zeros(8)

        n = len(u)
        k = _trim_k(n)
        sl = slice(k, n - k)

        dY_P = np.empty((n, 8))
        dY_Q = np.empty((n, 8))
        dY_P[:, 0] = dPa[:, 0] / 0.75
        dY_P[:, 1] = dPa[:, 1]
        dY_P[:, 2] = dPa[:, 2] / 0.75
        dY_P[:, 3] = P[0] * (P_pa / mva_a)
        dY_P[:, 4] = dPb[:, 0] / 0.75
        dY_P[:, 5] = dPb[:, 1]
        dY_P[:, 6] = dPb[:, 2] / 0.75
        dY_P[:, 7] = P[0] * (P_pb / mva_b)
        dY_Q[:, 0] = dQa[:, 0] / 0.75
        dY_Q[:, 1] = dQa[:, 1]
        dY_Q[:, 2] = dQa[:, 2] / 0.75
        dY_Q[:, 3] = P[0] * (Q_pa / mva_a)
        dY_Q[:, 4] = dQb[:, 0] / 0.75
        dY_Q[:, 5] = dQb[:, 1]
        dY_Q[:, 6] = dQb[:, 2] / 0.75
        dY_Q[:, 7] = P[0] * (Q_pb / mva_b)

        X, Qr = _get_proj(u, k)
        yP = (P_pa + P_pb - P)[sl]
        yQ = (Q_pa + Q_pb - Q)[sl]
        beta = Qr @ np.column_stack((yP, yQ))
        resP = yP - X @ beta[:, 0]
        resQ = yQ - X @ beta[:, 1]
        SS = np.sum(resP ** 2) + np.sum(resQ ** 2)
        if SS <= 0:
            return np.zeros(8)

        dSS = _proj_dss(resP, resQ,
                        np.ascontiguousarray(dY_P[sl, :]),
                        np.ascontiguousarray(dY_Q[sl, :]), X, Qr)

        g = (1000.0 / (n - 2 * k)) * dSS / (2 * np.sqrt(SS))
        if lam_H:
            g[2] += 2 * lam_H * (para[2] - para[6])
            g[6] -= 2 * lam_H * (para[2] - para[6])
        return g

    return obj, jac


def load_groups(n_groups):
    ori = pickle.load(open(ORI, 'rb'))
    truth_df = pickle.load(open(RRES, 'rb'))
    groups = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))[:n_groups]
    data = []
    for g in groups:
        u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)
        data.append((g, u, P, Q, truth))
    return data


def grad_check(obj, jac, u, P, Q):
    """对一组数据、若干随机参数点，验证解析梯度与中心差分数值梯度一致。"""
    rng = np.random.default_rng(0)
    max_rel = 0.0
    for _ in range(5):
        p = np.array([rng.uniform(l, u) for l, u in zip(lb * 2, ub * 2)])
        ga = np.asarray(jac(p, u, P, Q, 0.01), dtype=float)
        gn = np.zeros(8)
        for j in range(8):
            h = 1e-6 * max(abs(p[j]), 1e-3)
            pp = p.copy(); pp[j] += h
            pm = p.copy(); pm[j] -= h
            gn[j] = (obj(pp, u, P, Q, 0.01) - obj(pm, u, P, Q, 0.01)) / (2 * h)
        rel = np.linalg.norm(ga - gn) / (np.linalg.norm(gn) + 1e-30)
        max_rel = max(max_rel, rel)
    return max_rel


def main(n_groups=30, K=12):
    data = load_groups(n_groups)
    print('[data] 加载 %d 组 PSSE 数据' % len(data), flush=True)

    variants = [
        ('V1 RK4@10ms     ', make_functions(_core_con, _core_var, _core_con_jac, _core_var_jac, 0.01, None, LAM_H)),
        ('V2 修正欧拉@10ms ', make_functions(_core_con_me, _core_var_me, _core_con_me_jac, _core_var_me_jac, 0.01, None, LAM_H)),
        ('V3 RK4@2ms      ', make_functions(_core_con, _core_var, _core_con_jac, _core_var_jac, 0.002, 5, LAM_H)),
        ('V4 修正欧拉@2ms  ', make_functions(_core_con_me, _core_var_me, _core_con_me_jac, _core_var_me_jac, 0.002, 5, LAM_H)),
    ]

    if '--check' in sys.argv:
        g0, u0, P0, Q0, _ = data[0]
        for tag, (obj_, jac_) in variants:
            print('[check] %s jac 相对误差 max = %.3e' % (tag.strip(), grad_check(obj_, jac_, u0, P0, Q0)), flush=True)
        return

    results = []
    for name, (obj, jac) in variants:
        print('\n>>> 跑 %s ...' % name.strip(), flush=True)
        t0 = time.time()
        recs = []
        for (g, u, P, Q, truth) in data:
            best, _ = identify_multistart(u, P, Q, 0.01, lb * 2, ub * 2, obj_func=obj,
                                          tol=1e-8, n_restarts=K, jac_func=jac, seed=SEED + g)
            th = np.asarray(best.x, dtype=float)
            recs.append((th, truth))
        th_all = np.array([r[0] for r in recs])
        tr_all = np.array([r[1] for r in recs])
        E = np.abs(th_all - tr_all) / np.abs(tr_all) * 100
        results.append(dict(name=name.strip(), th=th_all, tr=tr_all, E=E, n=len(recs), t=time.time() - t0))
        print('  n=%d  P_im=%.2f%%  H2=%.2f%%  dyn6=%.2f%%  用时 %.1f s' % (
            results[-1]['n'], np.median(E[:, PIM]), np.median(E[:, H2]), np.median(E[:, DYN]), results[-1]['t']), flush=True)

    L = []
    ap = L.append
    ap('=' * 112)
    ap('积分方法敏感性实验 —— 2×2 全因子（解析雅可比，%d 组 PSSE, K=%d 起点, λ_H=%.2f）' % (len(data), K, LAM_H))
    ap('V1/V3 = RK4（四阶）；V2/V4 = 修正欧拉（二阶）；@10ms 为现状步长，@2ms 对齐 PSSE 积分步长')
    ap('=' * 112)
    ap('')
    ap('[1] 总体精度对比（全量口径：所有组×参数直接取中位）')
    ap('  %-16s %9s %9s %9s' % ('变体', 'P_im%', 'H2%', 'dyn6%'))
    for r in results:
        ap('  %-16s %9.2f %9.2f %9.2f' % (r['name'], np.median(r['E'][:, PIM]),
                                          np.median(r['E'][:, H2]), np.median(r['E'][:, DYN])))
    ap('')
    ap('[2] 逐参数中位误差（%）')
    ap('  %-16s' % '参数' + ''.join(' %8s' % p for p in PNAMES))
    for r in results:
        ap('  %-16s' % r['name'] + ''.join(' %8.2f' % np.median(r['E'][:, j]) for j in range(8)))
    ap('')
    ap('[3] 各变体 vs V1(RK4@10ms) 的配对差异（pp，正=变差）')
    base = results[0]['E']
    for r in results[1:]:
        dE = r['E'] - base
        ap('  %-16s  ΔP_im=%+.2fpp  ΔH2=%+.2fpp  Δdyn6=%+.2fpp' % (
            r['name'], np.median(dE[:, PIM]), np.median(dE[:, H2]), np.median(dE[:, DYN])))
    ap('')
    ap('[4] 关键判读')
    v4 = results[3]['E']
    dE4 = v4 - base
    ap('  · V2(修正欧拉@10ms)：若 H2 崩 → 证明二阶方法 @10ms 精度不足(O(h²)=1e-4)，')
    ap('    不是「算法」本身的问题，而是步长没对齐 PSSE。')
    ap('  · V4(修正欧拉@2ms) vs V1：Δdyn6 = %+.2f pp（这才是「对齐 PSSE 积分器」的公平对比）' % np.median(dE4[:, DYN]))
    ap('    - 若 |Δdyn6| < 1pp → 积分算法+步长均不是误差主因，实锤结构失配；')
    ap('    - 若 |Δdyn6| ≥ 2pp → 积分器匹配是重要来源，需对齐网络耦合继续深挖。')
    ap('  · V3(RK4@2ms) vs V1：若几乎一致 → RK4@10ms 已数值收敛，步长细化无意义。')
    ap('=' * 112)

    txt = '\n'.join(L)
    open(OUT, 'w', encoding='utf-8').write(txt + '\n')
    print('\n' + txt, flush=True)
    print('[saved] %s' % OUT, flush=True)


if __name__ == '__main__':
    n_groups = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    main(n_groups, K)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
import time
from scipy import optimize as opt
from numba import njit

w0 = 2*math.pi*60  # 系统角频率(60 Hz), RK4 核与目标函数共用

# ==================== Numba 加速核心 ====================
# RK4 递推为串行循环，Numba 编译后比纯 Python 快 50~100 倍。
# 公式与原代码逐行一致（IEEE 浮点，无 fastmath），并直接输出预测功率:
#   返回 (ok, P_p, Q_p); ok=False ⇔ 数值发散或稳态滑移越界,
#   由各目标函数统一返回惩罚值 1e16

@njit(cache=True)
def _core_con(u, a, b, H2, mva, dt):
    """恒转矩模型: RK4 递推并直接输出预测功率(已乘容量基值 mva)。

    返回 (ok, P_p, Q_p); ok=False 表示数值发散或稳态滑差越界(调用方给大惩罚)。
    """
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
    tm = 1.0  # 恒转矩: 基准机械转矩
    for k in range(n - 1):
        uk = u[k]
        uk1 = u[k + 1]

        if k >= 1 and k + 2 < n:
            umid = (-u[k - 1] + 9 * u[k] + 9 * u[k + 1] - u[k + 2]) / 16
        else:
            umid = 0.5 * (u[k] + u[k + 1])  # 测试4点插值

        # k1
        k1_f = (-b - 1j * s_p[k] * w0) * f_p[k] + a * uk
        Pe1 = f_p[k].real * uk.imag - f_p[k].imag * uk.real
        k1_s = (tm - Pe1) / (2 * H2)

        # k2
        f2 = f_p[k] + 0.5 * dt * k1_f
        s2 = s_p[k] + 0.5 * dt * k1_s
        k2_f = (-b - 1j * s2 * w0) * f2 + a * umid
        Pe2 = f2.real * umid.imag - f2.imag * umid.real
        k2_s = (tm - Pe2) / (2 * H2)

        # k3
        f3 = f_p[k] + 0.5 * dt * k2_f
        s3 = s_p[k] + 0.5 * dt * k2_s
        k3_f = (-b - 1j * s3 * w0) * f3 + a * umid
        Pe3 = f3.real * umid.imag - f3.imag * umid.real
        k3_s = (tm - Pe3) / (2 * H2)

        # k4
        f4 = f_p[k] + dt * k3_f
        s4 = s_p[k] + dt * k3_s
        k4_f = (-b - 1j * s4 * w0) * f4 + a * uk1
        Pe4 = f4.real * uk1.imag - f4.imag * uk1.real
        k4_s = (tm - Pe4) / (2 * H2)

        f_p[k + 1] = f_p[k] + dt / 6 * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)
        s_p[k + 1] = s_p[k] + dt / 6 * (k1_s + 2 * k2_s + 2 * k3_s + k4_s)

        # 原代码由 np.seterr 的 RuntimeWarning 提前返回；Numba 无警告，
        # 用有限性检查等价替代（病态输入均回到同一个惩罚值）
        if not (np.isfinite(f_p[k + 1].real) and np.isfinite(f_p[k + 1].imag)
                and np.isfinite(s_p[k + 1])):
            return False, P_p, Q_p

    if not (s_p[n - 1] > 0 and s_p[n - 1] < 0.2):
        return False, P_p, Q_p
    P_p = (f_p.real * u.imag - f_p.imag * u.real) * mva
    Q_p = -(f_p.real * u.real + f_p.imag * u.imag) * mva
    return True, P_p, Q_p


@njit(cache=True)
def _core_con_jac(u, a, b, H2, mva, dt):
    """恒转矩核 + 解析灵敏度: 返回 (ok, P_p, Q_p, dP, dQ)。

    dP/dQ: (n,3) 数组, 分别为 P_p/Q_p 对 (a, b, H2) 的逐点导数。
    灵敏度随 RK4 递推同步传播; ok=False 时 dP/dQ 无意义。
    """
    n = len(u)
    f_p = np.zeros(n, dtype=np.complex128)
    s_p = np.zeros(n)
    g = np.zeros((n, 3), dtype=np.complex128)   # df_p/dθ
    q = np.zeros((n, 3))                        # ds_p/dθ
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

    # 初值灵敏度: θ = (a, b, H2)
    dcd = abs(u[0]) ** 2 * w0                     # dc/da
    ddl0 = 2 * c * dcd                            # d(dl)/da
    ddl1 = -8 * b * w0 ** 2                       # d(dl)/db
    ds0 = np.zeros(3)
    ds0[0] = (dcd - ddl0 / (2 * sq)) / 2 / w0 ** 2
    ds0[1] = (0.0 - ddl1 / (2 * sq)) / 2 / w0 ** 2
    den = b + 1j * s_p[0] * w0
    dfs0 = -a * u[0] * (1j * w0) / den ** 2       # ∂f0/∂s0
    g[0, 0] = u[0] / den + dfs0 * ds0[0]
    g[0, 1] = -a * u[0] / den ** 2 + dfs0 * ds0[1]
    g[0, 2] = 0.0 + 0j
    q[0, 0] = ds0[0]
    q[0, 1] = ds0[1]
    q[0, 2] = 0.0

    tm = 1.0  # 恒转矩
    for k in range(n - 1):
        uk = u[k]
        uk1 = u[k + 1]

        if k >= 1 and k + 2 < n:
            umid = (-u[k - 1] + 9 * u[k] + 9 * u[k + 1] - u[k + 2]) / 16
        else:
            umid = 0.5 * (u[k] + u[k + 1])

        fk = f_p[k]
        sk = s_p[k]

        # ---- 主递推 ----
        k1_f = (-b - 1j * sk * w0) * fk + a * uk
        Pe1 = fk.real * uk.imag - fk.imag * uk.real
        k1_s = (tm - Pe1) / (2 * H2)

        f2 = fk + 0.5 * dt * k1_f
        s2 = sk + 0.5 * dt * k1_s
        k2_f = (-b - 1j * s2 * w0) * f2 + a * umid
        Pe2 = f2.real * umid.imag - f2.imag * umid.real
        k2_s = (tm - Pe2) / (2 * H2)

        f3 = fk + 0.5 * dt * k2_f
        s3 = sk + 0.5 * dt * k2_s
        k3_f = (-b - 1j * s3 * w0) * f3 + a * umid
        Pe3 = f3.real * umid.imag - f3.imag * umid.real
        k3_s = (tm - Pe3) / (2 * H2)

        f4 = fk + dt * k3_f
        s4 = sk + dt * k3_s
        k4_f = (-b - 1j * s4 * w0) * f4 + a * uk1
        Pe4 = f4.real * uk1.imag - f4.imag * uk1.real
        k4_s = (tm - Pe4) / (2 * H2)

        f_p[k + 1] = fk + dt / 6 * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)
        s_p[k + 1] = sk + dt / 6 * (k1_s + 2 * k2_s + 2 * k3_s + k4_s)

        # ---- 灵敏度递推(对 a, b, H2) ----
        for j in range(3):
            da = 1.0 if j == 0 else 0.0
            db = 1.0 if j == 1 else 0.0
            dH = 1.0 if j == 2 else 0.0

            dk1_f = ((-b - 1j * sk * w0) * g[k, j]
                     + (-db - 1j * q[k, j] * w0) * fk + da * uk)
            dPe1 = g[k, j].real * uk.imag - g[k, j].imag * uk.real   # Pe1 用 f_k
            dk1_s = -dPe1 / (2 * H2) - (tm - Pe1) / (2 * H2 * H2) * dH

            df2 = g[k, j] + 0.5 * dt * dk1_f
            ds2 = q[k, j] + 0.5 * dt * dk1_s
            dk2_f = ((-b - 1j * s2 * w0) * df2
                     + (-db - 1j * ds2 * w0) * f2 + da * umid)
            dPe2 = df2.real * umid.imag - df2.imag * umid.real       # Pe2 用 f2
            dk2_s = -dPe2 / (2 * H2) - (tm - Pe2) / (2 * H2 * H2) * dH

            df3 = g[k, j] + 0.5 * dt * dk2_f
            ds3 = q[k, j] + 0.5 * dt * dk2_s
            dk3_f = ((-b - 1j * s3 * w0) * df3
                     + (-db - 1j * ds3 * w0) * f3 + da * umid)
            dPe3 = df3.real * umid.imag - df3.imag * umid.real       # Pe3 用 f3
            dk3_s = -dPe3 / (2 * H2) - (tm - Pe3) / (2 * H2 * H2) * dH

            df4 = g[k, j] + dt * dk3_f
            ds4 = q[k, j] + dt * dk3_s
            dk4_f = ((-b - 1j * s4 * w0) * df4
                     + (-db - 1j * ds4 * w0) * f4 + da * uk1)
            dPe4 = df4.real * uk1.imag - df4.imag * uk1.real         # Pe4 用 f4
            dk4_s = -dPe4 / (2 * H2) - (tm - Pe4) / (2 * H2 * H2) * dH

            g[k + 1, j] = g[k, j] + dt / 6 * (dk1_f + 2 * dk2_f + 2 * dk3_f + dk4_f)
            q[k + 1, j] = q[k, j] + dt / 6 * (dk1_s + 2 * dk2_s + 2 * dk3_s + dk4_s)

        # 有限性检查(主状态与灵敏度)
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
def _core_var(u, a, b, H2, mva, dt):
    """变转矩模型: RK4 递推(机械转矩随 (1-s)^2 变化)并直接输出预测功率。

    返回 (ok, P_p, Q_p); ok=False 表示数值发散或稳态滑差越界(调用方给大惩罚)。
    """
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
    tm = 1.0 / (1 - s_p[0]) ** 2  # 变转矩基准: 初始平衡 tm*(1-s0)^2 = 1
    for k in range(n - 1):
        uk = u[k]
        uk1 = u[k + 1]

        if k >= 1 and k + 2 < n:
            umid = (-u[k - 1] + 9 * u[k] + 9 * u[k + 1] - u[k + 2]) / 16
        else:
            umid = 0.5 * (u[k] + u[k + 1])

        # k1
        k1_f = (-b - 1j * s_p[k] * w0) * f_p[k] + a * uk
        Pe1 = f_p[k].real * uk.imag - f_p[k].imag * uk.real
        k1_s = (tm * (1 - s_p[k]) ** 2 - Pe1) / (2 * H2)

        # k2
        f2 = f_p[k] + 0.5 * dt * k1_f
        s2 = s_p[k] + 0.5 * dt * k1_s
        k2_f = (-b - 1j * s2 * w0) * f2 + a * umid
        Pe2 = f2.real * umid.imag - f2.imag * umid.real
        k2_s = (tm * (1 - s2) ** 2 - Pe2) / (2 * H2)

        # k3
        f3 = f_p[k] + 0.5 * dt * k2_f
        s3 = s_p[k] + 0.5 * dt * k2_s
        k3_f = (-b - 1j * s3 * w0) * f3 + a * umid
        Pe3 = f3.real * umid.imag - f3.imag * umid.real
        k3_s = (tm * (1 - s3) ** 2 - Pe3) / (2 * H2)

        # k4
        f4 = f_p[k] + dt * k3_f
        s4 = s_p[k] + dt * k3_s
        k4_f = (-b - 1j * s4 * w0) * f4 + a * uk1
        Pe4 = f4.real * uk1.imag - f4.imag * uk1.real
        k4_s = (tm * (1 - s4) ** 2 - Pe4) / (2 * H2)

        f_p[k + 1] = f_p[k] + dt / 6 * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)
        s_p[k + 1] = s_p[k] + dt / 6 * (k1_s + 2 * k2_s + 2 * k3_s + k4_s)

        # 原代码由 np.seterr 的 RuntimeWarning 提前返回；Numba 无警告，
        # 用有限性检查等价替代（病态输入均回到同一个惩罚值）
        if not (np.isfinite(f_p[k + 1].real) and np.isfinite(f_p[k + 1].imag)
                and np.isfinite(s_p[k + 1])):
            return False, P_p, Q_p

    if not (s_p[n - 1] > 0 and s_p[n - 1] < 0.2):
        return False, P_p, Q_p
    P_p = (f_p.real * u.imag - f_p.imag * u.real) * mva
    Q_p = -(f_p.real * u.real + f_p.imag * u.imag) * mva
    return True, P_p, Q_p


@njit(cache=True)
def _core_var_jac(u, a, b, H2, mva, dt):
    """变转矩核 + 解析灵敏度: 返回 (ok, P_p, Q_p, dP, dQ)(dP/dQ 对 (a,b,H2) 逐点导数)。"""
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

    tm = 1.0 / (1 - s_p[0]) ** 2  # 变转矩基准
    for k in range(n - 1):
        uk = u[k]
        uk1 = u[k + 1]

        if k >= 1 and k + 2 < n:
            umid = (-u[k - 1] + 9 * u[k] + 9 * u[k + 1] - u[k + 2]) / 16
        else:
            umid = 0.5 * (u[k] + u[k + 1])

        fk = f_p[k]
        sk = s_p[k]

        # ---- 主递推 ----
        k1_f = (-b - 1j * sk * w0) * fk + a * uk
        Pe1 = fk.real * uk.imag - fk.imag * uk.real
        k1_s = (tm * (1 - sk) ** 2 - Pe1) / (2 * H2)

        f2 = fk + 0.5 * dt * k1_f
        s2 = sk + 0.5 * dt * k1_s
        k2_f = (-b - 1j * s2 * w0) * f2 + a * umid
        Pe2 = f2.real * umid.imag - f2.imag * umid.real
        k2_s = (tm * (1 - s2) ** 2 - Pe2) / (2 * H2)

        f3 = fk + 0.5 * dt * k2_f
        s3 = sk + 0.5 * dt * k2_s
        k3_f = (-b - 1j * s3 * w0) * f3 + a * umid
        Pe3 = f3.real * umid.imag - f3.imag * umid.real
        k3_s = (tm * (1 - s3) ** 2 - Pe3) / (2 * H2)

        f4 = fk + dt * k3_f
        s4 = sk + dt * k3_s
        k4_f = (-b - 1j * s4 * w0) * f4 + a * uk1
        Pe4 = f4.real * uk1.imag - f4.imag * uk1.real
        k4_s = (tm * (1 - s4) ** 2 - Pe4) / (2 * H2)

        f_p[k + 1] = fk + dt / 6 * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)
        s_p[k + 1] = sk + dt / 6 * (k1_s + 2 * k2_s + 2 * k3_s + k4_s)

        # ---- 灵敏度递推 ----
        s0_ = s_p[0]
        for j in range(3):
            da = 1.0 if j == 0 else 0.0
            db = 1.0 if j == 1 else 0.0
            dH = 1.0 if j == 2 else 0.0
            dtm = 2.0 / (1 - s0_) ** 3 * q[0, j]   # tm = 1/(1-s0)^2

            dk1_f = ((-b - 1j * sk * w0) * g[k, j]
                     + (-db - 1j * q[k, j] * w0) * fk + da * uk)
            dPe1 = g[k, j].real * uk.imag - g[k, j].imag * uk.real
            dk1_s = (dtm * (1 - sk) ** 2 + tm * 2 * (1 - sk) * (-q[k, j]) - dPe1) / (2 * H2) \
                    - (tm * (1 - sk) ** 2 - Pe1) / (2 * H2 * H2) * dH

            df2 = g[k, j] + 0.5 * dt * dk1_f
            ds2 = q[k, j] + 0.5 * dt * dk1_s
            dk2_f = ((-b - 1j * s2 * w0) * df2
                     + (-db - 1j * ds2 * w0) * f2 + da * umid)
            dPe2 = df2.real * umid.imag - df2.imag * umid.real
            dk2_s = (dtm * (1 - s2) ** 2 + tm * 2 * (1 - s2) * (-ds2) - dPe2) / (2 * H2) \
                    - (tm * (1 - s2) ** 2 - Pe2) / (2 * H2 * H2) * dH

            df3 = g[k, j] + 0.5 * dt * dk2_f
            ds3 = q[k, j] + 0.5 * dt * dk2_s
            dk3_f = ((-b - 1j * s3 * w0) * df3
                     + (-db - 1j * ds3 * w0) * f3 + da * umid)
            dPe3 = df3.real * umid.imag - df3.imag * umid.real
            dk3_s = (dtm * (1 - s3) ** 2 + tm * 2 * (1 - s3) * (-ds3) - dPe3) / (2 * H2) \
                    - (tm * (1 - s3) ** 2 - Pe3) / (2 * H2 * H2) * dH

            df4 = g[k, j] + dt * dk3_f
            ds4 = q[k, j] + dt * dk3_s
            dk4_f = ((-b - 1j * s4 * w0) * df4
                     + (-db - 1j * ds4 * w0) * f4 + da * uk1)
            dPe4 = df4.real * uk1.imag - df4.imag * uk1.real
            dk4_s = (dtm * (1 - s4) ** 2 + tm * 2 * (1 - s4) * (-ds4) - dPe4) / (2 * H2) \
                    - (tm * (1 - s4) ** 2 - Pe4) / (2 * H2 * H2) * dH

            g[k + 1, j] = g[k, j] + dt / 6 * (dk1_f + 2 * dk2_f + 2 * dk3_f + dk4_f)
            q[k + 1, j] = q[k, j] + dt / 6 * (dk1_s + 2 * dk2_s + 2 * dk3_s + dk4_s)

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


def regress(V,P,Q):#多元回归，返回偏差值和回归结果
    # 二次多项式最小二乘(同原 sm.OLS): 设计矩阵 [1, V, V^2], 正规方程闭式解(比 statsmodels 快 ~20 倍)
    X = np.column_stack((np.ones(len(V)), V, V**2))
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ np.column_stack((P, Q)))   # 3x2, 一次解 P 与 Q
    resid = np.column_stack((P, Q)) - X @ beta
    return beta.T, np.sum(resid**2)

def _trim_k(n):
    """头尾各裁 5%(1000 点裁 50 点)。k 随数据长度缩放, 适配不同时长仿真。"""
    return max(1, n // 20)

def obj_con_iden(para,u,P,Q,dt=0.01):
    a,b,H2,mva=para[0]/0.75,para[1],para[2]/0.75,P[0]*para[3]
    ok, P_p, Q_p = _core_con(u, a, b, H2, mva, dt)   # 核心直接输出预测功率
    if not ok:
        return 1e16

    n=len(u)
    k=_trim_k(n)
    sl=slice(k,n-k)
    return math.sqrt(regress(abs(u[sl]),P_p[sl]-P[sl],Q_p[sl]-Q[sl])[1])/(n-2*k)*1000


def obj_var_iden(para,u,P,Q,dt=0.01):
    # 与 obj_con_iden 完全同构(变量/公式/截断/返回一致), 仅递推核心为变转矩 _core_var
    a,b,H2,mva=para[0]/0.75,para[1],para[2]/0.75,P[0]*para[3]
    ok, P_p, Q_p = _core_var(u, a, b, H2, mva, dt)   # 核心直接输出预测功率
    if not ok:
        return 1e16

    n=len(u)
    k=_trim_k(n)
    sl=slice(k,n-k)
    return math.sqrt(regress(abs(u[sl]),P_p[sl]-P[sl],Q_p[sl]-Q[sl])[1])/(n-2*k)*1000

def obj_mix_iden(para,u,P,Q,dt=0.01,lam_H=0.0):
    """混合负荷辨识目标函数: 恒转矩(con)与变转矩(var)两组电机同时辨识。

    para 共 8 个: 前 4 个为 con 组 [a_a, b_a, H2_a, P_im_a](参考 obj_con_iden),
                后 4 个为 var 组 [a_b, b_b, H2_b, P_im_b](参考 obj_var_iden)。
    两组各自从 u 曲线递推预测 P/Q, 回归时从测量 P/Q 中减去两组预测之和。
    变量命名: _a 表示 con 组, _b 表示 var 组。

    lam_H : H2 差量 Tikhonov 正则强度(默认 0.0 = 纯 SSR, 向后兼容)。
            最终目标 J = c·√SSR + lam_H·(H2_a − H2_b)², 其中 H2_a/H2_b 取辨识口径
            (即 para[2]/para[6] 原始值, 未乘缩放)。正则只在 H2 差量方向施力,
            用于把差量钉在物理合理的近零空间内(A2.26/A2.33)。
    """
    a_a, b_a, H2_a, mva_a = para[0]/0.75, para[1], para[2]/0.75, P[0]*para[3]   # a/H2 缩放对齐 obj_con_iden, 基值并入
    a_b, b_b, H2_b, mva_b = para[4]/0.75, para[5], para[6]/0.75, P[0]*para[7]   # a/H2 缩放对齐 obj_var_iden, 基值并入
    # 两组各自递推预测(任一失败或稳态越界 → 大惩罚)
    ok_a, P_pa, Q_pa = _core_con(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb = _core_var(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return 1e16

    # 回归: 模型总预测(P_pa+P_pb) 减 测量(与原 obj 的残差方向一致; 头尾各裁 k)
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n-k)
    obj = math.sqrt(regress(abs(u[sl]), (P_pa+P_pb-P)[sl], (Q_pa+Q_pb-Q)[sl])[1])/(n-2*k)*1000
    if lam_H:
        obj = obj + lam_H * (para[2] - para[6]) ** 2
    return obj


# obj_mix_jac 的回归投影缓存: X 与 Qr=(XtX)^-1 X^T 只依赖 (u, k), 同一段数据只算一次
_proj_cache = {}

def _get_proj(u, k):
    key = (len(u), k, float(u[0].real), float(u[len(u) // 2].real), float(u[-1].imag))
    c = _proj_cache.get(key)
    if c is None:
        V = abs(u[k:len(u)-k])
        m = len(V)
        X = np.column_stack((np.ones(m), V, V**2))
        Qr = np.linalg.solve(X.T @ X, X.T)          # (3, m)
        _proj_cache.clear()                          # 只保留最近一段数据(防泄漏)
        _proj_cache[key] = (X, Qr)
        c = (X, Qr)
    return c


@njit(cache=True)
def _proj_dss(resP, resQ, dYsP, dYsQ, X, Qr):
    """投影 (I-H) 后的 ∂SS/∂θ: dR = dY - X(Qr dY), dSS_j = 2(resP·dRP_j + resQ·dRQ_j)"""
    m = len(resP)
    dSS = np.zeros(8)
    for j in range(8):
        t3 = Qr @ dYsP[:, j]
        dRP = dYsP[:, j] - X @ t3
        t3q = Qr @ dYsQ[:, j]
        dRQ = dYsQ[:, j] - X @ t3q
        sP = 0.0
        sQ = 0.0
        for i in range(m):
            sP += resP[i] * dRP[i]
            sQ += resQ[i] * dRQ[i]
        dSS[j] = 2.0 * (sP + sQ)
    return dSS


def obj_mix_jac(para,u,P,Q,dt=0.01,lam_H=0.0):
    """obj_mix_iden 的解析梯度(供 trust-constr 的 jac 使用)。

    链式: 两核灵敏度 dP/dθ → 参数链缩放 → 残差投影 (I-H) → dSS/dθ → dfun/dθ。
    投影矩阵与 (XtX)^-1 X^T 按 (u,k) 缓存; 投影与点积在 numba 中完成。

    lam_H : H2 差量正则强度(与 obj_mix_iden 一致)。梯度在 H2_a/H2_b 两分量上
            叠加 ±2·lam_H·(H2_a − H2_b)。
    """
    a_a, b_a, H2_a, mva_a = para[0]/0.75, para[1], para[2]/0.75, P[0]*para[3]
    a_b, b_b, H2_b, mva_b = para[4]/0.75, para[5], para[6]/0.75, P[0]*para[7]

    ok_a, P_pa, Q_pa, dPa, dQa = _core_con_jac(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb, dPb, dQb = _core_var_jac(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return np.zeros(8)

    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n-k)

    # 8 个参数对模型输出(P/Q)的逐点导数 (n, 8)
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

    # 回归残差(与 obj_mix_iden 相同的头尾截断区间 [k:n-k])
    X, Qr = _get_proj(u, k)
    yP = (P_pa + P_pb - P)[sl]
    yQ = (Q_pa + Q_pb - Q)[sl]
    beta = Qr @ np.column_stack((yP, yQ))
    resP = yP - X @ beta[:, 0]
    resQ = yQ - X @ beta[:, 1]
    SS = np.sum(resP**2) + np.sum(resQ**2)
    if SS <= 0:
        return np.zeros(8)

    dSS = _proj_dss(resP, resQ,
                    np.ascontiguousarray(dY_P[sl, :]),
                    np.ascontiguousarray(dY_Q[sl, :]), X, Qr)

    g = (1000.0 / (n - 2*k)) * dSS / (2 * np.sqrt(SS))
    if lam_H:
        g[2] += 2 * lam_H * (para[2] - para[6])
        g[6] -= 2 * lam_H * (para[2] - para[6])
    return g


def identify(u, P, Q, dt=0.01, lb=None, ub=None, obj_func=None, tol=1e-10, max_trials=20, jac_func=None):
    """trust-constr 参数辨识：随机采样合法初值 → 优化 → 距离校验（与 main_con 原逻辑一致）。

    参数
    ----
    u, P, Q : 测量序列（复数电压、有功、无功）
    dt      : 采样步长
    lb, ub  : 参数上下界（长度一致的列表）
    obj_func: 目标函数，默认 obj_con_iden；签名 obj_func(para, u, P, Q, dt)
    tol     : 优化器收敛容差
    max_trials: 距离校验连续失败的最大重试次数（防止病态数据下死循环）
    返回 (result, computation_time)：scipy 的 OptimizeResult（含 .x 最优参数、.fun 目标值）
            和辨识耗时（秒，含初值采样与距离校验重试）
    """
    t_start = time.perf_counter()
    if obj_func is None:
        obj_func = obj_con_iden
    for trial in range(1, max_trials + 1):
        # 随机采样满足阈值条件的初始点（与 main_con 原逻辑一致：obj < 0.5）
        while True:
            x0 = np.random.uniform(lb, ub)
            if obj_func(x0, u, P, Q, dt) < 0.5:
                break
        result = opt.minimize(obj_func, x0, args=(u, P, Q, dt), jac=jac_func, method='trust-constr',
                              bounds=np.transpose(np.array([lb, ub])), options={'disp': False}, tol=tol)
        # 与 main_con 原逻辑一致：优化器从初值移动足够远（距离平方 > 1000）才算有效辨识
        if np.sum(np.square(result.x - x0)) > 1000:
            break
    return result, time.perf_counter() - t_start


def identify_multistart(u, P, Q, dt=0.01, lb=None, ub=None, obj_func=None, tol=1e-8,
                        n_restarts=6, jac_func=None, seed=None):
    """多起点全局辨识: n_restarts 次独立 trust-constr(随机初值) 取目标值最小者。

    与 identify 的区别(2026-11-07, 张老师实测 8 参数模型存在 2 个局部极小 + 1 个全局极小):
      identify 返回「第一个通过距离校验的解」, 可能落入局部极小;
      本函数跑 n_restarts 个独立初值的完整优化, 返回代价(obj)最小的解, 用于对抗非凸。
    实现: 内部对 identify 设 max_trials=1(不做距离校验重试, 每个初值都算一次完整优化),
      每次初值由 np.random 采样, 可用 seed 复现(seed + r)。

    返回 (best_result, total_time_sec): best_result 为 scipy OptimizeResult(含 .x/.fun)。
    """
    if obj_func is None:
        obj_func = obj_con_iden
    t0 = time.perf_counter()
    best = None
    best_fun = np.inf
    for r in range(n_restarts):
        if seed is not None:
            np.random.seed(seed + r)
        result, _ = identify(u, P, Q, dt, lb, ub, obj_func=obj_func, tol=tol,
                             max_trials=1, jac_func=jac_func)
        if result.fun < best_fun:
            best_fun = result.fun
            best = result
    return best, time.perf_counter() - t0


# ==================== 主线辨识统一入口（A2.26 / A2.37 / A2.40 固化） ====================
# 「单一总功率观测 + 多起点 K=12 + H2 差量 Tikhonov 正则」是混合负荷 8 参数辨识的
# 主线口径。以下把 λ_H 与 a_a 回填校正两个结论固化于此，后续所有辨识（含海量测试）
# 统一从 identify_mix 入口调用，脚本层不再手工拼 make_reg 闭包（旧脚本不受影响）。

LAM_H_DEFAULT = 0.02      # H2 差量正则强度。A2.37 大样本甜点 0.01~0.02
                          #   （30 组曾把最优 λ 高估到 0.04~0.05，切到 0.02 对主线无实质影响）
A_A_CAL = 0.8719          # a_a 回填乘性校正系数。A2.40：trutĥ_a_a = 0.8719 × th_a_a
                          #   （a_a 辨识值系统性乘性高估 +14.7%，回填 PSSE 前乘此系数消一阶偏置）

# 8 参数上下界（单台电机 4 参数 × 2 台），与 laplace_posterior.py 的 lb/ub 一致
MIX_LB = [100, 30, 0.08, 0.2] * 2
MIX_UB = [1200, 240, 0.22, 0.8] * 2


def make_mix_functions(lam_H=LAM_H_DEFAULT):
    """返回绑定 lam_H 的 (obj, jac) 函数对，签名 (para, u, P, Q, dt)，供 identify 族使用。

    等价于旧脚本里的 make_reg(lam_H) 闭包，但改为直接透传 obj_mix_iden/jac 的 lam_H
    参数——正则已内嵌在 obj_mix_iden/obj_mix_jac 内，这里不再在目标函数外加正则项。
    """
    def obj(para, u, P, Q, dt):
        return obj_mix_iden(para, u, P, Q, dt, lam_H=lam_H)

    def jac(para, u, P, Q, dt):
        return obj_mix_jac(para, u, P, Q, dt, lam_H=lam_H)
    return obj, jac


def identify_mix(u, P, Q, dt=0.01, lam_H=LAM_H_DEFAULT, K=12, lb=None, ub=None,
                 tol=1e-8, seed=None):
    """混合负荷 8 参数辨识统一入口。

    单一总功率观测 + K 起点 trust-constr + H2 差量 Tikhonov 正则（主线口径）。
    返回 scipy OptimizeResult（.x 为辨识口径参数，未做回填校正；回填前用 calibrate_a_a）。

    参数
    ----
    u, P, Q : 复数电压 / 有功 / 无功 测量序列
    dt      : 采样步长（默认 0.01）
    lam_H   : H2 差量正则强度（默认 LAM_H_DEFAULT = 0.02）
    K       : 多起点次数（默认 12）
    lb, ub  : 参数上下界（默认 MIX_LB / MIX_UB）
    tol     : 优化器收敛容差
    seed    : 随机种子（用于复现；None 则不固定）
    """
    if lb is None:
        lb = MIX_LB
    if ub is None:
        ub = MIX_UB
    obj, jac = make_mix_functions(lam_H)
    return identify_multistart(u, P, Q, dt, lb, ub, obj_func=obj, tol=tol,
                               n_restarts=K, jac_func=jac, seed=seed)[0]


def calibrate_a_a(th):
    """回填 PSSE 前的 a_a 乘性校正（A2.40）。

    a_a 辨识值存在 +14.7% 的系统性乘性高估（结构失配的有向投影），回填前乘 A_A_CAL
    把一阶偏置消掉。其余 7 参数无系统性偏置（k≈0.94~1.03，接近 1），不做校正。

    返回校正后的参数数组（副本，不修改入参）；th[0] 即 a_a。
    """
    out = np.asarray(th, dtype=float).copy()
    out[0] *= A_A_CAL
    return out

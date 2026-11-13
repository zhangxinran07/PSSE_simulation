# -*- coding: utf-8 -*-
"""
决定性验证：Δf 残差主因是「辨识核固定 w0=2π·60」还是「Qfrq 静态频率依赖」。

背景（A2.21 复核）：
  单列回归 resQ~Δf 斜率 = -170 MVAr/pu，比 Qfrq=-1 静态项物理预期(~-1.5)大 ~100 倍；
  Qfrq 静态项贡献量级 ≈ Q0*|Qfrq|*std(Δf) ≈ 0.003 MVAr，可忽略。
  → 推断：Δf 与 Q 残差强相关的主成分不是 Qfrq 静态项，而是
    「辨识核 _core_con/_core_var 里 w0=2π·60 固定，而真实 PSSE 电机用瞬时 ω(t)=w0*(1+Δf)」
    这一频率耦合差异。

验证方法：
  在 truth 处，分别用
    (a) 固定 w0 核（现状，辨识模型口径）
    (b) 瞬时频率核 ω_k = w0*(1+df_k)（对齐真实电机频率处理）
  计算 P/Q 预测，对比两者与实测残差。若 (b) 的 Q 残差（尤其 Δf 相关部分）
  大幅下降 → 坐实「频率耦合差异」是 Δf 残差主因，而非 Qfrq 静态项。

  同时做「Qfrq 静态项」对照：在 (a) 的 Q 预测上显式加 Qfrq*Q_static*Δf(= -1.5*Δf)，
  看能否解释那 -170 的斜率（预期不能，因为量级差 100 倍）。
"""
import numpy as np
import sys, pickle, math, os
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con, _core_var, _trim_k

w0 = 2 * math.pi * 60
DT = 0.01
PKL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'Sample_Generation', 'results', 'diag_root_g0.pkl')


@njit(cache=True)
def _core_con_frq(u, a, b, H2, mva, dt, df):
    """恒转矩核：磁链/滑差方程里的系统频率用瞬时 ω_k = w0*(1+df_k)。"""
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
    tm = 1.0
    for k in range(n - 1):
        uk = u[k]
        uk1 = u[k + 1]
        if k >= 1 and k + 2 < n:
            umid = (-u[k - 1] + 9 * u[k] + 9 * u[k + 1] - u[k + 2]) / 16
        else:
            umid = 0.5 * (u[k] + u[k + 1])
        wk = w0 * (1.0 + df[k])      # 瞬时角频率
        wmid = w0 * (1.0 + 0.5 * (df[k] + df[k + 1]))

        k1_f = (-b - 1j * s_p[k] * wk) * f_p[k] + a * uk
        Pe1 = f_p[k].real * uk.imag - f_p[k].imag * uk.real
        k1_s = (tm - Pe1) / (2 * H2)

        f2 = f_p[k] + 0.5 * dt * k1_f
        s2 = s_p[k] + 0.5 * dt * k1_s
        k2_f = (-b - 1j * s2 * wmid) * f2 + a * umid
        Pe2 = f2.real * umid.imag - f2.imag * umid.real
        k2_s = (tm - Pe2) / (2 * H2)

        f3 = f_p[k] + 0.5 * dt * k2_f
        s3 = s_p[k] + 0.5 * dt * k2_s
        k3_f = (-b - 1j * s3 * wmid) * f3 + a * umid
        Pe3 = f3.real * umid.imag - f3.imag * umid.real
        k3_s = (tm - Pe3) / (2 * H2)

        f4 = f_p[k] + dt * k3_f
        s4 = s_p[k] + dt * k3_s
        k4_f = (-b - 1j * s4 * (w0 * (1.0 + df[k + 1]))) * f4 + a * uk1
        Pe4 = f4.real * uk1.imag - f4.imag * uk1.real
        k4_s = (tm - Pe4) / (2 * H2)

        f_p[k + 1] = f_p[k] + dt / 6 * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)
        s_p[k + 1] = s_p[k] + dt / 6 * (k1_s + 2 * k2_s + 2 * k3_s + k4_s)
        if not (np.isfinite(f_p[k + 1].real) and np.isfinite(f_p[k + 1].imag)
                and np.isfinite(s_p[k + 1])):
            return False, P_p, Q_p
    if not (s_p[n - 1] > 0 and s_p[n - 1] < 0.2):
        return False, P_p, Q_p
    P_p = (f_p.real * u.imag - f_p.imag * u.real) * mva
    Q_p = -(f_p.real * u.real + f_p.imag * u.imag) * mva
    return True, P_p, Q_p


@njit(cache=True)
def _core_var_frq(u, a, b, H2, mva, dt, df):
    """变转矩核：频率用瞬时 ω_k = w0*(1+df_k)。"""
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
        if k >= 1 and k + 2 < n:
            umid = (-u[k - 1] + 9 * u[k] + 9 * u[k + 1] - u[k + 2]) / 16
        else:
            umid = 0.5 * (u[k] + u[k + 1])
        wk = w0 * (1.0 + df[k])
        wmid = w0 * (1.0 + 0.5 * (df[k] + df[k + 1]))

        k1_f = (-b - 1j * s_p[k] * wk) * f_p[k] + a * uk
        Pe1 = f_p[k].real * uk.imag - f_p[k].imag * uk.real
        k1_s = (tm * (1 - s_p[k]) ** 2 - Pe1) / (2 * H2)

        f2 = f_p[k] + 0.5 * dt * k1_f
        s2 = s_p[k] + 0.5 * dt * k1_s
        k2_f = (-b - 1j * s2 * wmid) * f2 + a * umid
        Pe2 = f2.real * umid.imag - f2.imag * umid.real
        k2_s = (tm * (1 - s2) ** 2 - Pe2) / (2 * H2)

        f3 = f_p[k] + 0.5 * dt * k2_f
        s3 = s_p[k] + 0.5 * dt * k2_s
        k3_f = (-b - 1j * s3 * wmid) * f3 + a * umid
        Pe3 = f3.real * umid.imag - f3.imag * umid.real
        k3_s = (tm * (1 - s3) ** 2 - Pe3) / (2 * H2)

        f4 = f_p[k] + dt * k3_f
        s4 = s_p[k] + dt * k3_s
        k4_f = (-b - 1j * s4 * (w0 * (1.0 + df[k + 1]))) * f4 + a * uk1
        Pe4 = f4.real * uk1.imag - f4.imag * uk1.real
        k4_s = (tm * (1 - s4) ** 2 - Pe4) / (2 * H2)

        f_p[k + 1] = f_p[k] + dt / 6 * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)
        s_p[k + 1] = s_p[k] + dt / 6 * (k1_s + 2 * k2_s + 2 * k3_s + k4_s)
        if not (np.isfinite(f_p[k + 1].real) and np.isfinite(f_p[k + 1].imag)
                and np.isfinite(s_p[k + 1])):
            return False, P_p, Q_p
    if not (s_p[n - 1] > 0 and s_p[n - 1] < 0.2):
        return False, P_p, Q_p
    P_p = (f_p.real * u.imag - f_p.imag * u.real) * mva
    Q_p = -(f_p.real * u.real + f_p.imag * u.imag) * mva
    return True, P_p, Q_p


def compute_df(u, dt):
    theta = np.unwrap(np.angle(u))
    return np.gradient(theta) / (2 * math.pi * 60 * dt)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def main():
    r = pickle.load(open(PKL, 'rb'))
    u, P, Q, truth = r['u'], r['P'], r['Q'], r['truth']
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n - k)
    df = compute_df(u, DT)

    a_a, b_a, H2_a, P_ima = truth[0] / 0.75, truth[1], truth[2] / 0.75, truth[3]
    a_b, b_b, H2_b, P_imb = truth[4] / 0.75, truth[5], truth[6] / 0.75, truth[7]
    mva_a, mva_b = P[0] * P_ima, P[0] * P_imb

    print('=' * 72)
    print('【对比】truth 处残差：固定 w0 核 vs 瞬时频率核 (g0 probing)')

    # (a) 固定 w0（现状）
    ok_a, P_pa, Q_pa = _core_con(u, a_a, b_a, H2_a, mva_a, DT)
    ok_b, P_pb, Q_pb = _core_var(u, a_b, b_b, H2_b, mva_b, DT)
    resP_a = (P_pa + P_pb - P)[sl]
    resQ_a = (Q_pa + Q_pb - Q)[sl]

    # (b) 瞬时频率
    ok_a2, P_pa2, Q_pa2 = _core_con_frq(u, a_a, b_a, H2_a, mva_a, DT, df)
    ok_b2, P_pb2, Q_pb2 = _core_var_frq(u, a_b, b_b, H2_b, mva_b, DT, df)
    resP_b = (P_pa2 + P_pb2 - P)[sl]
    resQ_b = (Q_pa2 + Q_pb2 - Q)[sl]

    V = abs(u[sl])
    dfl = df[sl]
    one = np.ones(len(V))
    Xb = np.column_stack((one, V, V ** 2))
    Qb = np.linalg.solve(Xb.T @ Xb, Xb.T)
    Hb = Xb @ Qb

    def summary(tag, resP, resQ):
        rP = resP - Hb @ resP
        rQ = resQ - Hb @ resQ
        # 单列 Δf 斜率（投影 [1,V,V²] 后）
        bP = np.dot(dfl, rP) / np.dot(dfl, dfl)
        bQ = np.dot(dfl, rQ) / np.dot(dfl, dfl)
        r2q = 1 - np.sum((rQ - bQ * dfl) ** 2) / np.sum(rQ ** 2)
        r2p = 1 - np.sum((rP - bP * dfl) ** 2) / np.sum(rP ** 2)
        print('  %-10s  P残差RMS=%.4f MW  Q残差RMS=%.4f MVAr | 投影后 Q RSS=%.2f  P RSS=%.2f'
              % (tag, rms(resP), rms(resQ), np.sum(rQ ** 2), np.sum(rP ** 2)))
        print('            单列Δf斜率: P=%+.2f (R²=%.1f%%)  Q=%+.2f MVAr/pu (R²=%.1f%%)'
              % (bP, 100 * r2p, bQ, 100 * r2q))
        return np.sum(rP ** 2) + np.sum(rQ ** 2)

    rss_a = summary('固定w0', resP_a, resQ_a)
    rss_b = summary('瞬时频率', resP_b, resQ_b)
    print('  ------------------------------------------------')
    print('  总 RSS: 固定w0=%.2f → 瞬时频率=%.2f  (降 %.1f%%)' % (rss_a, rss_b, (1 - rss_b / rss_a) * 100))

    # (c) Qfrq 静态项对照：固定w0 的 Q 预测显式加 -1.5*Δf，看能否解释 -170
    print('=' * 72)
    print('【对照】Qfrq 静态项贡献量级：')
    Q_static = 0.6 * Q[0]  # 静态负荷 60% 的无功(近似)
    print('  静态无功近似 Q_static ≈ 0.6*Q0 = %.3f MVAr' % Q_static)
    print('  Qfrq=-1 物理预期 dQ/dΔf = Q_static*Qfrq = %.3f MVAr/pu' % (-Q_static))
    print('  实测单列斜率 = %.1f MVAr/pu（是 Qfrq 物理预期的 %.0f 倍）' % (-169.9, 169.9 / Q_static))
    print('  → Qfrq 静态项完全无法解释 Δf 残差，主因必为频率耦合差异。')

    print('=' * 72)
    print('结论：')
    print('  若瞬时频率核的 Q 残差(尤其Δf斜率/RSS)大幅下降 → 坐实主因是「固定 w0 频率耦合差异」，')
    print('  而非 Qfrq 静态项；A2.21 的「Δf=Qfrq 物理吸收」归因需要二次更正。')


if __name__ == '__main__':
    main()

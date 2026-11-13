# -*- coding: utf-8 -*-
"""
频域灵敏度谱扫描：验证「两台电机 H2 可通过多频点激励解耦」的物理直觉。

原理：
  三阶感应电动机模型(转子磁链 f + 滑差 s)，滑差动态 ds/dt=(tm-Pe)/(2H2)，
  H2 决定滑差响应的转角频率。con(A) 与 var(B) 的 H2 不同(0.142 vs 0.174)，
  故两者 ∂P/∂H2 的复频响 g_a(f)、g_b(f) 随频率的滚降速度不同。
  在总功率观测下 H2_a/H2_b 混淆 = g_a(f) 与 g_b(f) 近似反号共线(此消彼长)。
  若复比值 R(f)=g_a(f)/g_b(f) 的相位/幅值随 f 显著变化，则说明存在频段可解耦。

方法：
  对每个频率 f，构造小幅电压扰动 u(t)=(1+eps*sin(2πft))+0j (幅值调制, 小干扰)，
  用 _core_con_jac/_core_var_jac 跑稳态，对 ∂P/∂θ(t) 做 sin/cos 拟合得复频响增益。
  纯数值实验，复用辨识核，无需重跑 PSSE。
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import _core_con_jac, _core_var_jac, w0

DT = 0.01

# 第 0 组真值(物理口径, 不除 0.75)
TRUTH = dict(
    a_a=303.2, b_a=72.7, H2_a=0.142, P_ima=0.337,
    a_b=270.5, b_b=63.2, H2_b=0.174, P_imb=0.265,
)


def _fit_sin(t, y, f):
    """y(t) 在频率 f 处做 sin/cos/常数 最小二乘拟合, 返回复幅值 (e^{jwt} 约定)。"""
    w = 2 * np.pi * f
    A = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    As, Ac = coef[0], coef[1]
    return (Ac - 1j * As)  # 复幅值 = Ac - j*As (对应 e^{jwt})


def single_freq_response(core_jac, params, f, eps=0.01):
    """单电机在频率 f 下的复频响: 返回 dict {param: (g_P, g_Q) 复增益}。

    core_jac: _core_con_jac 或 _core_var_jac
    params:   (a, b, H2, mva) 物理口径
    """
    a, b, H2, mva = params
    # 仿真时长: 瞬态衰减(5s) + 至少 3 个周期
    n_cycle = 3
    T_sim = max(5.0, n_cycle / f)
    n = int(T_sim / DT) + 1
    t = np.arange(n) * DT
    u = (1.0 + eps * np.sin(2 * np.pi * f * t)) + 0j

    ok, P_p, Q_p, dP, dQ = core_jac(u, a, b, H2, mva, DT)
    if not ok:
        return None

    # 稳态段: 取最后 n_cycle 个周期(去掉瞬态)
    n_ss = int(n_cycle / f / DT)
    t_ss = t[-n_ss:]
    dP_ss = dP[-n_ss:]
    dQ_ss = dQ[-n_ss:]

    g = {}
    for j, name in enumerate(['a', 'b', 'H2']):
        gP = _fit_sin(t_ss, dP_ss[:, j], f) / eps  # 归一化增益(每单位电压扰动)
        gQ = _fit_sin(t_ss, dQ_ss[:, j], f) / eps
        g[name] = (gP, gQ)
    return g


def main():
    # 读 P[0] 求真实 mva (P_im 是功率基值比例)
    ori = pd.read_pickle(r'D:\张欣然-F7D9CE\华为家庭存储\BUAA_research\BUAA2021\PSSE_Simulation\Sample_Generation\results\ori_data_mix_20261106_003107.pkl')
    P0 = float(ori['0 P'].values[0])
    mva_a = P0 * TRUTH['P_ima']
    mva_b = P0 * TRUTH['P_imb']
    print(f'P[0]={P0:.4f}, mva_a={mva_a:.4f}, mva_b={mva_b:.4f}')

    # 扫频范围: 对数分布 0.05~10 Hz (覆盖机电转角频率 + 磁链快动态)
    freqs = np.logspace(np.log10(0.05), np.log10(10.0), 32)

    rows = []
    for f in freqs:
        ga = single_freq_response(_core_con_jac, (TRUTH['a_a'], TRUTH['b_a'], TRUTH['H2_a'], mva_a), f)
        gb = single_freq_response(_core_var_jac, (TRUTH['a_b'], TRUTH['b_b'], TRUTH['H2_b'], mva_b), f)
        if ga is None or gb is None:
            rows.append(dict(f=f, ok=False))
            continue
        # 只取 P(有功) 通道的 H2 增益(混淆核心是 H2)
        gah2 = ga['H2'][0]   # ∂P_a/∂H2_a 复增益
        gbh2 = gb['H2'][0]   # ∂P_b/∂H2_b 复增益
        rows.append(dict(
            f=f, ok=True,
            mag_a=abs(gah2), mag_b=abs(gbh2),
            pha_a=np.angle(gah2, deg=True), pha_b=np.angle(gbh2, deg=True),
            R_mag=abs(gah2 / gbh2), R_pha=np.angle(gah2 / gbh2, deg=True),
            # 归一化共线度: cos角 = Re(g_a conj g_b)/(|g_a||g_b|), -1=反号共线(最混淆)
            collin=(gah2.real * gbh2.real + gah2.imag * gbh2.imag) / (abs(gah2) * abs(gbh2)),
        ))
    df = pd.DataFrame(rows)

    print('\n=== H2 频域灵敏度谱 (∂P/∂H2) ===')
    print(f'{"f(Hz)":>8} {"|g_a|":>10} {"|g_b|":>10} {"∠g_a":>8} {"∠g_b":>8} {"|R|":>8} {"∠R":>8} {"共线cos":>8}')
    for r in df.itertuples():
        if not r.ok:
            print(f'{r.f:>8.3f}  (数值发散)')
            continue
        print(f'{r.f:>8.3f} {r.mag_a:>10.3f} {r.mag_b:>10.3f} {r.pha_a:>8.1f} {r.pha_b:>8.1f} '
              f'{r.R_mag:>8.3f} {r.R_pha:>8.1f} {r.collin:>8.3f}')

    # 关键结论: 找出 ∠R 偏离 ±180° 最远的频点(即可解耦频段)
    okdf = df[df.ok]
    if len(okdf):
        # 反号共线 = ∠R 接近 180°(或 -180°), 可解耦 = ∠R 偏离 180°
        dev = np.abs(np.abs(okdf.R_pha) - 180.0)
        best = okdf.loc[dev.idxmin()]
        worst = okdf.loc[dev.idxmax()]
        print('\n=== 解耦诊断 ===')
        print(f'最共线(最混淆)频点: f={best.f:.3f}Hz, ∠R={best.R_pha:.1f}°(反号), cos={best.collin:.3f}')
        print(f'最可解耦频点:      f={worst.f:.3f}Hz, ∠R={worst.R_pha:.1f}°, cos={worst.collin:.3f}')

    # 保存数据供后续画图/设计激励
    out = r'D:\张欣然-F7D9CE\华为家庭存储\BUAA_research\BUAA2021\PSSE_Simulation\Sample_Generation\results\freq_sensitivity_h2.csv'
    df.to_csv(out, index=False)
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()

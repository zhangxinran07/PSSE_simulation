# -*- coding: utf-8 -*-
"""
频域灵敏度谱扩展：对 a/b/H2 三个动态参数分别扫频，看 A/B 两台电机的
∂P/∂θ 复频响比值 R(f) 是否随频率变化(可解耦)还是恒定(不可解耦)。

关键问题: H2 已发现全频段共线(比值恒定)。验证 a/b 是否也如此,
并确认转角频率(相位滚降最陡处)在哪,以判断工程可用频段(≤2Hz)内的可解耦性。
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import _core_con_jac, _core_var_jac

DT = 0.01
TRUTH = dict(a_a=303.2, b_a=72.7, H2_a=0.142, P_ima=0.337,
             a_b=270.5, b_b=63.2, H2_b=0.174, P_imb=0.265)


def _fit_sin(t, y, f):
    w = 2 * np.pi * f
    A = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return (coef[1] - 1j * coef[0])


def single_freq_response(core_jac, params, f, eps=0.01):
    a, b, H2, mva = params
    n_cycle = 4
    T_sim = max(8.0, n_cycle / f)
    n = int(T_sim / DT) + 1
    t = np.arange(n) * DT
    u = (1.0 + eps * np.sin(2 * np.pi * f * t)) + 0j
    ok, P_p, Q_p, dP, dQ = core_jac(u, a, b, H2, mva, DT)
    if not ok:
        return None
    n_ss = int(n_cycle / f / DT)
    t_ss = t[-n_ss:]
    dP_ss = dP[-n_ss:]
    dQ_ss = dQ[-n_ss:]
    g = {}
    for j, name in enumerate(['a', 'b', 'H2']):
        gP = _fit_sin(t_ss, dP_ss[:, j], f) / eps
        gQ = _fit_sin(t_ss, dQ_ss[:, j], f) / eps
        g[name] = (gP, gQ)
    return g


def main():
    ori = pd.read_pickle(r'D:\张欣然-F7D9CE\华为家庭存储\BUAA_research\BUAA2021\PSSE_Simulation\Sample_Generation\results\ori_data_mix_20261106_003107.pkl')
    P0 = float(ori['0 P'].values[0])
    mva_a = P0 * TRUTH['P_ima']
    mva_b = P0 * TRUTH['P_imb']

    freqs = np.logspace(np.log10(0.02), np.log10(40.0), 44)
    rows = []
    for f in freqs:
        ga = single_freq_response(_core_con_jac, (TRUTH['a_a'], TRUTH['b_a'], TRUTH['H2_a'], mva_a), f)
        gb = single_freq_response(_core_var_jac, (TRUTH['a_b'], TRUTH['b_b'], TRUTH['H2_b'], mva_b), f)
        if ga is None or gb is None:
            rows.append(dict(f=f, ok=False))
            continue
        r = dict(f=f, ok=True)
        for name in ['a', 'b', 'H2']:
            gaP, _ = ga[name]
            gbP, _ = gb[name]
            r[f'mag_{name}a'] = abs(gaP)
            r[f'mag_{name}b'] = abs(gbP)
            r[f'pha_{name}a'] = np.angle(gaP, deg=True)
            r[f'pha_{name}b'] = np.angle(gbP, deg=True)
            r[f'R_{name}_mag'] = abs(gaP / gbP)
            r[f'R_{name}_pha'] = np.angle(gaP / gbP, deg=True)
        rows.append(r)
    df = pd.DataFrame(rows)
    okdf = df[df.ok]

    # 打印 a/b/H2 的比值相位随频率变化(关键: 相位是否偏离 0°/±180°)
    print('=== A/B 各参数 ∂P/∂θ 复频响比值 ∠R(f) 随频率变化 ===')
    print(f'{"f(Hz)":>8} {"∠R_a":>8} {"∠R_b":>8} {"∠R_H2":>8} {"|R_a|":>9} {"|R_b|":>9} {"|R_H2|":>9}')
    for r in okdf.itertuples():
        print(f'{r.f:>8.3f} {r.R_a_pha:>8.2f} {r.R_b_pha:>8.2f} {r.R_H2_pha:>8.2f} '
              f'{r.R_a_mag:>9.3f} {r.R_b_mag:>9.3f} {r.R_H2_mag:>9.3f}')

    # 相位变化幅度 = 可解耦程度
    print('\n=== 相位变化幅度(全频段 max-min, 越大越可解耦) ===')
    for name in ['a', 'b', 'H2']:
        pha = okdf[f'R_{name}_pha'].values
        mag = okdf[f'R_{name}_mag'].values
        print(f'∠R_{name}: 范围 [{pha.min():.2f}, {pha.max():.2f}]°, 跨度 {pha.max()-pha.min():.2f}°; '
              f'|R_{name}|: [{mag.min():.3f}, {mag.max():.3f}]')

    # 转角频率定位: ∂P/∂H2 相位 -45° 和 -90° 对应的频率
    for name in ['a', 'b', 'H2']:
        pha = okdf[f'pha_{name}a'].values  # A 组相位
        # 找相位降到 -45° 和 -90° 的最近频点
        def find_f(target):
            idx = np.argmin(np.abs(pha - target))
            return okdf.f.values[idx], pha[idx]
        f45, p45 = find_f(-45)
        f90, p90 = find_f(-90)
        print(f'{name}: 转角频率 ≈ {f45:.2f}Hz(∠={p45:.1f}°), {f90:.2f}Hz(∠={p90:.1f}°)')

    out = r'D:\张欣然-F7D9CE\华为家庭存储\BUAA_research\BUAA2021\PSSE_Simulation\Sample_Generation\results\freq_sensitivity_abc.csv'
    df.to_csv(out, index=False)
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()

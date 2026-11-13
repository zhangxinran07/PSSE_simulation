# -*- coding: utf-8 -*-
"""
截止频率扫描下的 Fisher 可辨识性分析（纯数值，无需重跑 PSSE）。

背景：
  频域灵敏度谱已发现三个参数的可解耦性截然不同：
    - H2: ∠R 恒≈0°(全频段共线) → 结构性不可解耦
    - a/b: 低频段 ∠R≈88°(正交可分离), 高频段→0°(共线)
  本脚本把「频域灵敏度谱」换算成「不同截止频率低通激励下的 Fisher 矩阵」，
  定量回答：把 ambient 激励截止频率从 2Hz 降到更低，能否改善 a/b 混淆(以及 H2 是否无论如何都无解)。

方法：
  对每个频点 f，计算 8 参数 × (P/Q) 的复频响 g(f)（相对灵敏度归一化）。
  理想低通白噪声激励(截止 f_c)的 Fisher 元素 = ∫_0^{f_c} Re[g_Pi·conj(g_Pj) + g_Qi·conj(g_Qj)] df。
  对多个 f_c 做 Fisher 分解，输出条件数、最小特征值方向、各参数对的夹角。
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import _core_con_jac, _core_var_jac

DT = 0.01
TRUTH = dict(a_a=303.2, b_a=72.7, H2_a=0.142, P_ima=0.337,
             a_b=270.5, b_b=63.2, H2_b=0.174, P_imb=0.265)
NAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']


def _fit_sin(t, y, f):
    w = 2 * np.pi * f
    A = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return (coef[1] - 1j * coef[0])


def full_freq_response(core_jac, params, f, eps=0.01):
    """单电机在频率 f 下的完整复频响: 返回 (g_vec, P0_response)。

    g_vec: 4 参数 [a,b,H2,P_im] 对 P 的复增益(已除 eps)。Q 通道另算。
    P_im 的频响 = ∂P/∂P_im = P_p(t)/P_im 在 f 处的正弦分量(即 P_p 的频响/P_im)。
    """
    a, b, H2, P_im, mva = params
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
    dP_ss = dP[-n_ss:]; dQ_ss = dQ[-n_ss:]
    P_p_ss = P_p[-n_ss:]; Q_p_ss = Q_p[-n_ss:]
    # 8 参数频响(对 P 和 Q): [a,b,H2] 用灵敏度, P_im 用 P_p/P_im
    gP = np.zeros(4, dtype=complex); gQ = np.zeros(4, dtype=complex)
    for j in range(3):
        gP[j] = _fit_sin(t_ss, dP_ss[:, j], f) / eps
        gQ[j] = _fit_sin(t_ss, dQ_ss[:, j], f) / eps
    # P_im: ∂P/∂P_im = P0*P_p/mva = P0*P_p/(P0*P_im) = P_p/P_im (此处 mva=P0*P_im 已含, 但复频响是 P_p 在 f 的正弦分量)
    # 注意: 频响意义上 ∂P/∂P_im 的正弦分量 = (P_p 的正弦分量)/P_im
    gP[3] = _fit_sin(t_ss, P_p_ss, f) / P_im / eps
    gQ[3] = _fit_sin(t_ss, Q_p_ss, f) / P_im / eps
    return gP, gQ


def main():
    ori = pd.read_pickle(r'D:\张欣然-F7D9CE\华为家庭存储\BUAA_research\BUAA2021\PSSE_Simulation\Sample_Generation\results\ori_data_mix_20261106_003107.pkl')
    P0 = float(ori['0 P'].values[0])
    mva_a = P0 * TRUTH['P_ima']
    mva_b = P0 * TRUTH['P_imb']

    freqs = np.logspace(np.log10(0.02), np.log10(8.0), 40)
    # gP_all[f_idx, 8], gQ_all[f_idx, 8]
    gP_all = np.zeros((len(freqs), 8), dtype=complex)
    gQ_all = np.zeros((len(freqs), 8), dtype=complex)
    ok_mask = np.ones(len(freqs), dtype=bool)

    for i, f in enumerate(freqs):
        ra = full_freq_response(_core_con_jac, (TRUTH['a_a'], TRUTH['b_a'], TRUTH['H2_a'], TRUTH['P_ima'], mva_a), f)
        rb = full_freq_response(_core_var_jac, (TRUTH['a_b'], TRUTH['b_b'], TRUTH['H2_b'], TRUTH['P_imb'], mva_b), f)
        if ra is None or rb is None:
            ok_mask[i] = False
            continue
        gP_all[i, 0:4] = ra[0]; gP_all[i, 4:8] = rb[0]
        gQ_all[i, 0:4] = ra[1]; gQ_all[i, 4:8] = rb[1]

    freqs = freqs[ok_mask]; gP_all = gP_all[ok_mask]; gQ_all = gQ_all[ok_mask]

    # 相对灵敏度归一化(乘参数真值)
    truth = np.array([TRUTH[n] for n in NAMES])
    gP_all = gP_all * truth[None, :]
    gQ_all = gQ_all * truth[None, :]

    # 不同截止频率下的 Fisher 频谱积分
    cutoffs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    print('=== 截止频率扫描: Fisher 条件数 & 最小特征值方向 ===')
    print(f'{"f_c(Hz)":>8} {"cond":>10} {"H2_a":>8} {"H2_b":>8} {"a_a":>8} {"a_b":>8} {"b_a":>8} {"b_b":>8} {"P_ima":>8} {"P_imb":>8}')
    results = {}
    for fc in cutoffs:
        sel = freqs <= fc
        if sel.sum() == 0:
            continue
        # 积分(梯形, 频率对数域已由 logspace 保证; 这里用简单求和近似, 因 logspace 间距非均匀用 trapz)
        # 用 log 频率积分更合理, 但为简单用等权求和(频点对数均匀分布)
        F = np.zeros((8, 8))
        for f_idx in np.where(sel)[0]:
            gp = gP_all[f_idx]; gq = gQ_all[f_idx]
            F += np.outer(gp.real, gp.real) + np.outer(gp.imag, gp.imag) \
                 + np.outer(gq.real, gq.real) + np.outer(gq.imag, gq.imag)
        evals, evecs = np.linalg.eigh(F)
        cond = evals[-1] / evals[0]
        v0 = evecs[:, 0]
        results[fc] = (cond, v0, evals)
        print(f'{fc:>8.2f} {cond:>10.2e} {v0[2]:>+8.3f} {v0[6]:>+8.3f} {v0[0]:>+8.3f} {v0[4]:>+8.3f} '
              f'{v0[1]:>+8.3f} {v0[5]:>+8.3f} {v0[3]:>+8.3f} {v0[7]:>+8.3f}')

    # 关键: H2 子空间 与 a/b 子空间的 Fisher 2x2 条件数随 f_c 变化
    print('\n=== 各参数对(A/B)的 Fisher 2x2 子矩阵条件数随截止频率 ===')
    print(f'{"f_c(Hz)":>8} {"cond(H2对)":>12} {"cond(a对)":>12} {"cond(b对)":>12}')
    for fc in cutoffs:
        sel = freqs <= fc
        F = np.zeros((8, 8))
        for f_idx in np.where(sel)[0]:
            gp = gP_all[f_idx]; gq = gQ_all[f_idx]
            F += np.outer(gp.real, gp.real) + np.outer(gp.imag, gp.imag) \
                 + np.outer(gq.real, gq.real) + np.outer(gq.imag, gq.imag)
        def subcond(i, j):
            M = F[np.ix_([i, j], [i, j])]
            e = np.linalg.eigvalsh(M)
            return e[-1] / e[0] if e[0] > 0 else np.inf
        print(f'{fc:>8.2f} {subcond(2,6):>12.2e} {subcond(0,4):>12.2e} {subcond(1,5):>12.2e}')

    # 保存
    out = r'D:\张欣然-F7D9CE\华为家庭存储\BUAA_research\BUAA2021\PSSE_Simulation\Sample_Generation\results\fisher_cutoff_sweep.csv'
    pd.DataFrame(results).to_csv(out)
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""P_im 与 H2 辨识精度专项诊断：总功率观测 + 多起点最优，误差结构分解。

回答张老师「把 P_im 和 H2 辨识准确」的第一步(2026-11-07):
  1. 逐参数误差: P_im(索引3,7) 与 H2(索引2,6) 是否真是瓶颈? 比 a/b 差多少?
  2. 组合量分解: 总量(P_ima+P_imb, H2_a+H2_b) vs 差量(P_ima-P_imb, H2_a-H2_b)
     -> 确认误差是否集中在零空间「差量方向」(此消彼长, 对应 A2.10/A2.14)。
  3. A/B 误差相关性: P_ima 高估时 P_imb 是否低估(负相关) = 此消彼长的实证。
  4. 大样本 N 坐实(比之前 30 组更多, 张老师要求「多做几组」)。

口径(与 control_zero_mismatch / split_vs_total_power 严格对齐):
  参数顺序 [a_a,b_a,H2_a,P_ima,a_b,b_b,H2_b,P_imb]; 零失配 gen_PQ;
  trust-constr + obj_mix_jac; 多起点 identify_multistart 取 fun 最小。
"""
import numpy as np
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
sys.path.insert(0, os.path.dirname(__file__))
from identification import obj_mix_iden, obj_mix_jac, identify_multistart
from control_zero_mismatch import sample_mix, gen_u, gen_PQ

DT = 0.01
P0 = 6.0
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]       # 动态 6 参数(不含 P_im)
PIM = [3, 7]                    # P_im 两个
H2 = [2, 6]                     # H2 两个
lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
SEED = 20261107
OUT = os.path.join(os.path.dirname(__file__), '..', 'Sample_Generation', 'results',
                   'diagnose_pim_h2.txt')


def relerr(x, xt):
    return np.abs(np.asarray(x) - np.asarray(xt)) / np.abs(np.asarray(xt))


def run(N, RESTARTS=8):
    rng = np.random.default_rng(SEED)
    errs = []                 # (N,8) 逐参数相对误差
    rows = []                 # (g, truth, x, fun)
    print('=' * 100, flush=True)
    print('P_im/H2 精度专项诊断: 总功率观测 + %d 起点最优, N=%d' % (RESTARTS, N), flush=True)
    print('=' * 100, flush=True)
    for g in range(N):
        truth, _ = sample_mix(rng)
        u = gen_u(np.random.default_rng(SEED + g))
        P, Q = gen_PQ(u, truth)
        best, t = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj_mix_iden,
                                      tol=1e-8, n_restarts=RESTARTS, jac_func=obj_mix_jac,
                                      seed=SEED * 10 + g)
        x = np.asarray(best.x)
        e = relerr(x, truth)
        errs.append(e)
        rows.append((g, truth, x, best.fun))
        print('  g=%02d  dyn中位=%.2f%%  P_im={%.2f%%,%.2f%%}  H2={%.2f%%,%.2f%%}  fun=%.1e  (%.1fs)'
              % (g, np.median(e[DYN]) * 100, e[3] * 100, e[7] * 100,
                 e[2] * 100, e[6] * 100, best.fun, t), flush=True)

    E = np.array(errs)

    # ---- 组合量分解: 总量 vs 差量 ----
    # 总量: S_P=P_ima+P_imb, S_H=H2_a+H2_b ; 差量: D_P=P_ima-P_imb, D_H=H2_a-H2_b
    SP, DP, SH, DH = [], [], [], []
    RP, RH = [], []           # 比值
    for g in range(N):
        t = rows[g][1]
        x = rows[g][2]
        SP.append(abs((x[3] + x[7]) - (t[3] + t[7])) / abs(t[3] + t[7]))
        DP.append(abs((x[3] - x[7]) - (t[3] - t[7])) / abs(t[3] + t[7]))   # 差量误差/总量
        SH.append(abs((x[2] + x[6]) - (t[2] + t[6])) / abs(t[2] + t[6]))
        DH.append(abs((x[2] - x[6]) - (t[2] - t[6])) / abs(t[2] + t[6]))
        RP.append(abs(x[3] / x[7] - t[3] / t[7]) / abs(t[3] / t[7]))
        RH.append(abs(x[2] / x[6] - t[2] / t[6]) / abs(t[2] / t[6]))
    SP, DP, SH, DH = np.array(SP), np.array(DP), np.array(SH), np.array(DH)
    RP, RH = np.array(RP), np.array(RH)

    # ---- A/B 误差相关性(此消彼长实证) ----
    # 用带符号误差(est-truth)/truth 看 A/B 是否反号
    sP_a = np.array([(rows[g][2][3] - rows[g][1][3]) / rows[g][1][3] for g in range(N)])
    sP_b = np.array([(rows[g][2][7] - rows[g][1][7]) / rows[g][1][7] for g in range(N)])
    sH_a = np.array([(rows[g][2][2] - rows[g][1][2]) / rows[g][1][2] for g in range(N)])
    sH_b = np.array([(rows[g][2][6] - rows[g][1][6]) / rows[g][1][6] for g in range(N)])
    corr_P = np.corrcoef(sP_a, sP_b)[0, 1]
    corr_H = np.corrcoef(sH_a, sH_b)[0, 1]

    # ---- 汇总 ----
    med = np.median(E, axis=0) * 100
    p90 = np.percentile(E, axis=0, q=90) * 100

    lines = []
    lines.append('=' * 100)
    lines.append('P_im/H2 辨识精度专项诊断汇总 (总功率观测 + %d 起点最优, N=%d)' % (RESTARTS, N))
    lines.append('=' * 100)
    lines.append('')
    lines.append('【1. 逐参数误差(多起点最优后)】')
    lines.append('%-10s ' % '参数' + ' '.join('%-10s' % p for p in PNAMES))
    lines.append('-' * 100)
    lines.append('%-10s ' % '中位%' + ' '.join('%-10.2f' % m for m in med))
    lines.append('%-10s ' % 'P90%' + ' '.join('%-10.2f' % m for m in p90))
    lines.append('-' * 100)
    lines.append('  分块中位误差: 动态6参数=%.2f%%  P_im={%.2f%%, %.2f%%}  H2={%.2f%%, %.2f%%}'
                 % (np.median(E[:, DYN]) * 100, med[3], med[7], med[2], med[6]))
    lines.append('')
    lines.append('【2. 组合量分解: 总量(可辨识) vs 差量(零空间/此消彼长)】')
    lines.append('%-24s %-12s %-12s %-12s' % ('组合量', '中位误差%', 'P90误差%', '结论'))
    lines.append('-' * 64)
    lines.append('%-24s %-12.2f %-12.2f %-12s' % ('P_im 总量(P_ima+P_imb)',
                  np.median(SP) * 100, np.percentile(SP, 90) * 100, ''))
    lines.append('%-24s %-12.2f %-12.2f %-12s' % ('P_im 差量(P_ima-P_imb)',
                  np.median(DP) * 100, np.percentile(DP, 90) * 100, ''))
    lines.append('%-24s %-12.2f %-12.2f %-12s' % ('P_im 比值(P_ima/P_imb)',
                  np.median(RP) * 100, np.percentile(RP, 90) * 100, ''))
    lines.append('%-24s %-12.2f %-12.2f %-12s' % ('H2 总量(H2_a+H2_b)',
                  np.median(SH) * 100, np.percentile(SH, 90) * 100, ''))
    lines.append('%-24s %-12.2f %-12.2f %-12s' % ('H2 差量(H2_a-H2_b)',
                  np.median(DH) * 100, np.percentile(DH, 90) * 100, ''))
    lines.append('%-24s %-12.2f %-12.2f %-12s' % ('H2 比值(H2_a/H2_b)',
                  np.median(RH) * 100, np.percentile(RH, 90) * 100, ''))
    lines.append('-' * 64)
    lines.append('  ※ 差量误差 = |D_est-D_truth| / 总量, 避免差量真值趋 0 时相对误差爆炸。')
    lines.append('  若「差量误差 >> 总量误差」, 坐实误差集中在零空间「差量方向」,')
    lines.append('  即 A/B 两台电机「此消彼长」, 需要先验/正则来定住差量。')
    lines.append('')
    lines.append('【3. A/B 误差相关性(此消彼长实证)】')
    lines.append('  P_ima 与 P_imb 带符号误差相关系数 r = %+.3f' % corr_P)
    lines.append('  H2_a  与 H2_b  带符号误差相关系数 r = %+.3f' % corr_H)
    lines.append('  ※ 强负相关(r<-0.5) = A 高估则 B 低估, 坐实「总量守恒式病态分配」。')
    lines.append('')
    lines.append('=' * 100)
    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    RESTARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run(N, RESTARTS)

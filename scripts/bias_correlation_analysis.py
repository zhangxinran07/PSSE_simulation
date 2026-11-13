# -*- coding: utf-8 -*-
"""偏差结构分析：8 参数系统性偏差的方向 / 相关性 / 真偏置 vs 样本量不足。

张老师 2026-11-11 提问：
  1) 8 个参数各自的符号偏差(辨识值 th - 真值 truth, 带方向)是什么?
  2) 这些偏差之间有没有相关性? 有的话能否联合校正映射?
  3) 这些偏差是"真系统性偏差"(结构失配, 不随 N 消失), 还是
     "样本量不足导致的估计偏差"(随 N 增大衰减到 0)?

数据: Sample_Generation/results/b_line_summary.pkl (1000 组 th/truth, 同口径)。

判据设计:
  - 符号一致率 + Wilcoxon 符号秩检验: 某参数若绝大多数组同号且 p 极小 => 方向性系统偏差。
  - bootstrap 子样本收敛: 对 N in {50,100,200,400,1000} 重抽样, 看 bias 均值随 N 是
    (a) 收敛到非零常数 μ (真偏置) 还是 (b) 按 1/sqrt(N) 衰减到 0 (纯随机/样本量)。
  - 相关性: bias 与 rel_bias 的 Pearson/Spearman 相关矩阵 => 判断能否联合校正。

输出: Sample_Generation/results/bias_correlation.txt (+ .png 收敛图)
"""
import os
import pickle
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
NAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
RNG = np.random.default_rng(20261111)


def main():
    recs = pickle.load(open(os.path.join(RES, 'b_line_summary.pkl'), 'rb'))
    th = np.array([r['th'] for r in recs], dtype=float)
    tr = np.array([r['truth'] for r in recs], dtype=float)
    n = len(th)
    bias = th - tr                       # (n, 8) 符号偏差(辨识口径)
    rel_bias = bias / tr * 100.0         # 相对偏差 %

    L = []
    ap = L.append
    ap('=' * 108)
    ap('偏差结构分析：%d 组真实 PSSE（K=12 + λ_H=0.05, 单一总功率观测）' % n)
    ap('口径：bias = th - truth（带符号, 同口径可直接相减）; rel_bias = bias/truth*100%%')
    ap('=' * 108)
    ap('')

    # ---------- [1] 逐参数符号偏差 ----------
    ap('[1] 逐参数符号偏差（带方向）与显著性')
    ap('  %-8s %12s %12s %12s %10s %12s %12s' % (
        '参数', '均值bias', '中位bias', '中位rel%', '一致率%', 'Wilcoxon_p', '判定'))
    sig = []
    for j, nm in enumerate(NAMES):
        b = bias[:, j]
        rb = rel_bias[:, j]
        mean_b = np.mean(b)
        med_b = np.median(b)
        med_rb = np.median(rb)
        # 符号一致率 = 与中位符号相同的比例（中位为 0 时取正值占比）
        if abs(med_b) > 1e-15:
            sign_agree = float(np.mean(np.sign(b) == np.sign(med_b)))
        else:
            sign_agree = float(np.mean(b > 0))
        w, p = stats.wilcoxon(b)
        # 判定：中位相对偏差超过 ±1% 且 p<0.001 且一致率>80% => 方向性系统偏差
        if abs(med_rb) >= 1.0 and p < 0.001 and sign_agree >= 0.80:
            ver = '系统(方向性)'
        elif abs(med_rb) >= 0.5 and p < 0.001:
            ver = '系统(弱)'
        elif p < 0.001:
            ver = '显著但小'
        else:
            ver = '不显著/随机'
        sig.append((nm, med_rb, p, sign_agree, ver))
        ap('  %-8s %12.4g %12.4g %12.2f %9.1f%% %12.2e %s' % (
            nm, mean_b, med_b, med_rb, sign_agree * 100, p, ver))
    ap('')
    ap('  ※ 判据：|中位rel|≥1%% 且 p<0.001 且 一致率≥80%% => 方向性系统偏差(真偏置)')
    ap('  ※ 一致率≈50%% 且 p 不显著 => 纯随机(样本量/噪声主导)')
    ap('')

    # ---------- [2] 偏差相关性 ----------
    ap('[2] 偏差相关性矩阵（Pearson, 基于符号偏差 bias）')
    Cb = np.corrcoef(bias.T)
    ap('  %-8s' % '' + ' '.join('%7s' % p for p in NAMES))
    for i, pn in enumerate(NAMES):
        ap('  %-8s' % pn + ' '.join('%7.2f' % Cb[i, j] for j in range(8)))
    ap('')
    ap('  [2b] 相对偏差 rel_bias 的 Pearson 相关（尺度对齐, 更直观）')
    Cr = np.corrcoef(rel_bias.T)
    ap('  %-8s' % '' + ' '.join('%7s' % p for p in NAMES))
    for i, pn in enumerate(NAMES):
        ap('  %-8s' % pn + ' '.join('%7.2f' % Cr[i, j] for j in range(8)))
    ap('')
    # 显著相关的参数对（|r|>=0.3）
    pairs = []
    for i in range(8):
        for j in range(i + 1, 8):
            r = Cr[i, j]
            if abs(r) >= 0.3:
                pairs.append((NAMES[i], NAMES[j], r))
    pairs.sort(key=lambda x: -abs(x[2]))
    ap('  |rel_bias 相关|≥0.3 的参数对（按强度）：')
    for a, b, r in pairs:
        ap('    %-5s ↔ %-5s  r = %+.2f' % (a, b, r))
    if not pairs:
        ap('    （无）')
    ap('')

    # ---------- [3] 真偏置 vs 样本量不足（bootstrap 子样本收敛） ----------
    ap('[3] 真系统性偏差 vs 样本量不足（bootstrap 子样本收敛）')
    ap('  %-8s %s' % ('参数', 'bias 均值(中位±95%CI) 随 N: 50 → 100 → 200 → 400 → 1000'))
    Ns = [50, 100, 200, 400, 1000]
    B = 400
    conv = {}
    for j, nm in enumerate(NAMES):
        row = []
        for N in Ns:
            draws = np.empty(B)
            for b in range(B):
                idx = RNG.integers(0, n, N)
                draws[b] = np.mean(bias[idx, j])
            row.append((np.median(draws), np.percentile(draws, 2.5), np.percentile(draws, 97.5)))
        conv[nm] = row
        seg = []
        for (m, lo, hi) in row:
            seg.append('%.3g[%.3g,%.3g]' % (m, lo, hi))
        # 收敛到非零? 看 N=1000 的 CI 是否不含 0
        m1000, lo1000, hi1000 = row[-1]
        nonzero = '收敛到非零(真偏置)' if (lo1000 > 0 or hi1000 < 0) else '含0(随机)'
        ap('  %-8s %s  → %s' % (nm, ' | '.join(seg), nonzero))
    ap('')
    ap('  ※ 真偏置：bias 均值随 N 增大收敛到稳定非零常数, CI 只变窄不移位。')
    ap('  ※ 样本量不足：bias 均值随 N 增大以 1/sqrt(N) 衰减到 0。')
    ap('')

    # 定量：bias 均值 = μ + k/sqrt(N) 拟合, 看截距 μ 是否显著非零
    ap('[3b] 截距拟合 bias_mean(N) = μ + k/√N（μ≠0 => 真偏置）')
    inv = 1.0 / np.sqrt(np.array(Ns, dtype=float))
    for j, nm in enumerate(NAMES):
        ys = np.array([conv[nm][k][0] for k in range(len(Ns))])
        A = np.column_stack([np.ones_like(inv), inv])
        mu, k = np.linalg.lstsq(A, ys, rcond=None)[0]
        ap('  %-8s μ = %+12.4g  k = %+12.4g' % (nm, mu, k))
    ap('')

    # ---------- 结论 ----------
    ap('[4] 联合校正映射的可行性小结（人工判定另写入《研究结论.md》）')
    ap('  · 哪些参数是真偏置、方向如何、量级多少 => 见 [1][3]。')
    ap('  · 偏差相关强度 => 见 [2b]。|r| 大则联合校正有意义, 否则逐参数校正即可。')
    ap('  · 真偏置 + 高相关 => 多维校正映射(仿射 truth≈A·th+c)值得做; 需 train/test 防过拟合。')
    ap('=' * 108)

    txt = '\n'.join(L)
    out = os.path.join(RES, 'bias_correlation.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(txt)

    # 保存数据供后续校正映射使用
    np.savez(os.path.join(RES, 'bias_corr_data.npz'),
             th=th, tr=tr, bias=bias, rel_bias=rel_bias,
             corr_bias=Cb, corr_rel=Cr)
    print('\n[saved] %s' % out)
    print('[saved] bias_corr_data.npz')


if __name__ == '__main__':
    main()

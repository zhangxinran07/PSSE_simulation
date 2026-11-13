# -*- coding: utf-8 -*-
"""Bus20 var 逐样本重跑辨识, 分析 a(及 b) 的偏差是"系统性"还是"散乱"。

用法:
    python analyze_a_bus20.py [mag1,mag2,...] [n_per_mag] [out_prefix]
  不传参数 => 全 10 档 x 100 样本。

输出:
  - 每档幅值: a 带符号偏差的 均值/中位/标准差/负号比例, 原始口径 + 去缩放(物理)口径;
  - 逐样本明细 CSV(含 a_est 物理值、a_sgn、b_sgn)。
"""
import os, sys, pickle, time, warnings
import numpy as np
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../PSSE_Simulation/scripts
ROOT = os.path.dirname(_HERE)                          # .../PSSE_Simulation (跨机器自动适配, 无关盘符)
HERE = _HERE
sys.path.append(os.path.join(ROOT, 'package'))
sys.path.insert(0, HERE)

import psse_datalen_var as V
import psse_datalen_lsq as L

D = r"C:\Users\Lenovo\WorkBuddy\2026-09-23-17-27-51\psse_datalen"
PKL = os.path.join(D, 'mag_sweep_bus20_var.pkl')

MAGS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def analyze(mags, n_per, out_prefix):
    data = pickle.load(open(PKL, 'rb'))
    rows = []          # 明细: mag,g,a_true,a_phys,a_sgn_raw,a_sgn_phys,b_true,b_est,b_sgn,...
    t0 = time.time()
    for mag in mags:
        gs = sorted(g for (m, g) in data.keys() if m == mag)
        gs = gs[:n_per] if n_per else gs
        for g in gs:
            rec = data[(mag, g)]
            truth = rec['truth']
            u, P, Q = rec['u'], rec['P'], rec['Q']
            x, cost = V.identify_best_var_lsq(u, P, Q)
            if x is None:
                rows.append([mag, g] + [np.nan] * 10)
                continue
            # 物理口径(去缩放): a_phys = x0/0.75, b=x1, H2_phys=x2/0.75, P_im=x3
            a_phys = x[0] / 0.75
            b_phys = x[1]
            h2_phys = x[2] / 0.75
            pim = x[3]
            est_phys = np.array([a_phys, b_phys, h2_phys, pim])
            # 原始口径(优化变量直接比 truth)
            sgn_raw = L.relerr_signed(x, truth) * 100       # [a,b,H2,P_im]
            err_raw = L.relerr(x, truth) * 100
            # 物理口径(去缩放)
            sgn_phys = L.relerr_signed(est_phys, truth) * 100
            err_phys = L.relerr(est_phys, truth) * 100
            rows.append([mag, g, truth[0], a_phys, truth[1], b_phys,
                         sgn_raw[0], sgn_raw[1], sgn_phys[0], sgn_phys[1],
                         sgn_phys[2], sgn_phys[3], float(cost)])
    dt = time.time() - t0
    rows = np.array(rows, dtype=float)

    # 明细落盘
    hdr = ['mag', 'g', 'a_true', 'a_phys', 'b_true', 'b_phys',
           'a_sgn_raw%', 'b_sgn_raw%', 'a_sgn_phys%', 'b_sgn_phys%', 'H2_sgn_phys%', 'Pim_sgn_phys%', 'cost']
    csv = os.path.join(HERE, out_prefix + '_detail.csv')
    np.savetxt(csv, rows, delimiter=',', header=','.join(hdr), comments='', fmt='%.6g')

    # 汇总
    lines = []
    lines.append('=' * 96)
    lines.append('Bus20 var 逐样本 a/b 偏差诊断: 系统性 vs 散乱  (n=%d/档, 共 %.1fs)' % (n_per, dt))
    lines.append('sgn_phys = 去缩放(物理)口径带符号偏差%%; 负=辨识偏小, 正=偏大')
    lines.append('=' * 96)
    lines.append('%-6s %-6s | %-28s | %-28s' % ('幅值', 'n', 'a 偏差(物理口径)%%', 'b 偏差(物理口径)%%'))
    lines.append('%-6s %-6s | %-9s %-9s %-9s %-8s | %-9s %-9s %-9s %-8s' %
                 ('', '', '均值', '中位', 'std', '负号%', '均值', '中位', 'std', '负号%'))
    lines.append('-' * 96)
    summary = {}
    for mag in mags:
        m = rows[rows[:, 0] == mag]
        if len(m) == 0:
            continue
        a_phys = m[:, 8]   # a_sgn_phys%
        b_phys = m[:, 9]   # b_sgn_phys%
        a_raw = m[:, 6]
        a_ok = a_phys[~np.isnan(a_phys)]
        b_ok = b_phys[~np.isnan(b_phys)]
        araw_ok = a_raw[~np.isnan(a_raw)]
        if len(a_ok) == 0:
            continue
        neg_a = float((a_ok < 0).mean() * 100)
        neg_b = float((b_ok < 0).mean() * 100)
        neg_araw = float((araw_ok < 0).mean() * 100)
        lines.append('%-6d %-6d | %+8.1f %+8.1f %8.1f %6.0f%% | %+8.1f %+8.1f %8.1f %6.0f%%' %
                     (mag, len(a_ok), a_ok.mean(), np.median(a_ok), a_ok.std(), neg_a,
                      b_ok.mean(), np.median(b_ok), b_ok.std(), neg_b))
        summary[mag] = dict(mean_a=a_ok.mean(), med_a=np.median(a_ok), std_a=a_ok.std(),
                            neg_a=neg_a, mean_b=b_ok.mean(), med_b=np.median(b_ok),
                            std_b=b_ok.std(), neg_b=neg_b, neg_a_raw=neg_araw)

    lines.append('-' * 96)
    lines.append('')
    lines.append('【结论判据】')
    lines.append('  系统性偏大/偏小: 负号比例接近 0%% 或 100%%(符号一致), std 远小于 |mean|')
    lines.append('  散乱(局部极值):   负号比例约 50%%(符号随机), std 大、mean 近 0')
    lines.append('')
    # 全局符号一致性
    all_a = rows[~np.isnan(rows[:, 8]), 8]
    all_b = rows[~np.isnan(rows[:, 9]), 9]
    lines.append('全局(跨所有档) a 物理口径: 均值%+.1f%% 中位%+.1f%% std%.1f%% 负号%.0f%%  (n=%d)' %
                 (all_a.mean(), np.median(all_a), all_a.std(), (all_a < 0).mean() * 100, len(all_a)))
    lines.append('全局(跨所有档) b 物理口径: 均值%+.1f%% 中位%+.1f%% std%.1f%% 负号%.0f%%  (n=%d)' %
                 (all_b.mean(), np.median(all_b), all_b.std(), (all_b < 0).mean() * 100, len(all_b)))

    txt = '\n'.join(lines)
    open(os.path.join(HERE, out_prefix + '_summary.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    print('\n明细 CSV: %s' % csv, flush=True)
    return summary


if __name__ == '__main__':
    mags = MAGS
    n_per = 100
    out = 'bus20_a_diag'
    if len(sys.argv) > 1:
        mags = [int(m) for m in sys.argv[1].split(',')]
    if len(sys.argv) > 2:
        n_per = int(sys.argv[2])
    if len(sys.argv) > 3:
        out = sys.argv[3]
    analyze(mags, n_per, out)

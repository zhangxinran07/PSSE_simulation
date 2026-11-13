# -*- coding: utf-8 -*-
"""
A2.21 复核：Δf 投影到底是「物理 Qfrq=-1 吸收」还是「病态回归伪影」。

背景（2026-11-07 复核）：
  verify_frq_proj.py 存在两处问题：
  (1) 口径 bug：truth 的 a/H2 未除 0.75，与 obj_mix_iden 口径不一致，
      导致残差量级虚低（Q 残差 RMS 28.76 而非 37.87）。
  (2) 病态疑点：相位主导激励下 V≈1（|u| std 仅 ~1.3%），设计矩阵
      [Δf, V·Δf, V²·Δf] 三列几乎完全共线（都 ≈ Δf），全矩阵最小二乘 beta
      巨大（数千）且符号任意，与物理 Qfrq=-1 预期 dQ/dΔf_pu ≈ -Q0 完全不符。

本脚本用「正确口径」重算，并做：
  1) 设计矩阵共线性诊断：相关系数矩阵、条件数、VIF；
  2) 单列回归 resQ ~ Δf（这才是接近物理 Qfrq 的无偏估计）；
  3) 逐步/正交化净 RSS 贡献（先 [1,V,V²]，再逐列加 Δf / V·Δf / V²·Δf）；
  4) 随机列过拟合基准（同自由度）对照。

判据：
  若「单列 resQ~Δf 斜率 ≈ -Q0 ≈ -2.5」且「正交化后 Δf 的净贡献远大于随机列」，
  则 A2.21「Δf 是物理 Qfrq 吸收」方向成立，但证据脚本的全矩阵 beta 判据需作废。
"""
import numpy as np
import sys, pickle, math, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'package'))
from identification import _core_con, _core_var, _trim_k

W0 = 2 * math.pi * 60
DT = 0.01
PKL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'Sample_Generation', 'results', 'diag_root_g0.pkl')


def compute_df(u, dt):
    theta = np.unwrap(np.angle(u))
    return np.gradient(theta) / (2 * math.pi * 60 * dt)


def vif(X):
    """方差膨胀因子：对每列回归其余列，VIF = 1/(1-R²)。"""
    m, p = X.shape
    out = np.zeros(p)
    for j in range(p):
        y = X[:, j]
        Xr = np.delete(X, j, axis=1)
        b = np.linalg.lstsq(Xr, y, rcond=None)[0]
        resid = y - Xr @ b
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        out[j] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return out


def main():
    r = pickle.load(open(PKL, 'rb'))
    u, P, Q, truth = r['u'], r['P'], r['Q'], r['truth']
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n - k)

    # ---- 正确口径：truth 的 a/H2 除 0.75（对齐 obj_mix_iden）----
    a_a, b_a, H2_a, P_ima = truth[0] / 0.75, truth[1], truth[2] / 0.75, truth[3]
    a_b, b_b, H2_b, P_imb = truth[4] / 0.75, truth[5], truth[6] / 0.75, truth[7]
    ok_a, P_pa, Q_pa = _core_con(u, a_a, b_a, H2_a, P[0] * P_ima, DT)
    ok_b, P_pb, Q_pb = _core_var(u, a_b, b_b, H2_b, P[0] * P_imb, DT)
    assert ok_a and ok_b, '核发散'

    resP = (P_pa + P_pb - P)[sl]
    resQ = (Q_pa + Q_pb - Q)[sl]
    V = abs(u[sl])
    df = compute_df(u, DT)[sl]
    m = len(V)
    one = np.ones(m)

    print('=' * 72)
    print('【0】正确口径下残差统计（对齐 obj_mix_iden 的 a/H2 除 0.75）')
    print('  P 残差 RMS = %.4f MW     Q 残差 RMS = %.4f MVAr' % (np.sqrt(np.mean(resP**2)), np.sqrt(np.mean(resQ**2))))
    print('  V   = mean %.5f  std %.5f (%.3f%%)' % (V.mean(), V.std(), 100 * V.std() / V.mean()))
    print('  Δf  = std %.6f pu (%.5f Hz)   mean %.2e' % (df.std(), df.std() * 60.0, df.mean()))
    print('  Q0 = %.4f（残差中 Q 的初始值，用于物理预期 dQ/dΔf_pu ≈ -Q0）' % Q[0])
    print('  物理 Qfrq=-1 预期单列斜率 dQ/dΔf_pu ≈ %.3f' % (-Q[0]))

    # ---- 1) 共线性诊断 ----
    X4 = np.column_stack((one, V, V ** 2, df, V * df, V ** 2 * df))
    colnames = ['1', 'V', 'V²', 'Δf', 'V·Δf', 'V²·Δf']
    print('=' * 72)
    print('【1】设计矩阵 X4=[1,V,V²,Δf,V·Δf,V²·Δf] 共线性诊断')
    X4n = X4[:, 1:]  # 去掉常数列再算相关性
    corr = np.corrcoef(X4n, rowvar=False)
    print('  相关系数矩阵（去掉常数列）:')
    print('        ' + ' '.join('%8s' % colnames[i] for i in range(1, 6)))
    for i in range(1, 6):
        print('  %-6s ' % colnames[i] + ' '.join('%8.4f' % corr[i - 1, j - 1] for j in range(1, 6)))
    # 条件数（对标准化后的非常数列）
    Xs = (X4n - X4n.mean(axis=0)) / (X4n.std(axis=0) + 1e-30)
    s = np.linalg.svd(Xs, compute_uv=False)
    print('  标准化设计矩阵奇异值: %s' % np.array2string(s, precision=4))
    print('  条件数 = %.3g' % (s[0] / s[-1]))
    print('  VIF = %s' % np.array2string(vif(X4), precision=2))

    # ---- 2) 单列回归（物理 Qfrq 的无偏估计）----
    print('=' * 72)
    print('【2】单列回归 resQ ~ Δf（判据：斜率 ≈ -Q0 = %.3f）' % (-Q[0]))
    bdf_Q = np.sum(df * resQ) / np.sum(df * df)
    bdf_P = np.sum(df * resP) / np.sum(df * df)
    print('  resP ~ Δf  斜率 = %+.4f MW/pu' % bdf_P)
    print('  resQ ~ Δf  斜率 = %+.4f MVAr/pu   (物理预期 %.3f)' % (bdf_Q, -Q[0]))
    # 单个 Δf 列能解释的方差
    r2q = 1 - np.sum((resQ - bdf_Q * df) ** 2) / np.sum(resQ ** 2)
    r2p = 1 - np.sum((resP - bdf_P * df) ** 2) / np.sum(resP ** 2)
    print('  单列 Δf 对残差的 R²：P=%.3f%%  Q=%.3f%%' % (100 * r2p, 100 * r2q))

    # ---- 3) 逐步 / 正交化净 RSS 贡献 ----
    print('=' * 72)
    print('【3】逐步回归净 RSS 贡献（残差先投 [1,V,V²]，再看 Δf 各列的净贡献）')
    Xbase = np.column_stack((one, V, V ** 2))
    Qb = np.linalg.solve(Xbase.T @ Xbase, Xbase.T)
    Hbase = Xbase @ Qb
    resP_base = resP - Hbase @ resP
    resQ_base = resQ - Hbase @ resQ
    rss_base = np.sum(resP_base ** 2) + np.sum(resQ_base ** 2)
    print('  baseline [1,V,V²] 投影后 RSS = %.2f' % rss_base)
    # 对 df 三列做 Gram-Schmidt 正交化，得到每个正交分量的净贡献
    cols = {'Δf': df, 'V·Δf': V * df, 'V²·Δf': V ** 2 * df}
    # 逐列加入看累计净贡献（顺序敏感，故也做增量贡献）
    order = ['Δf', 'V·Δf', 'V²·Δf']
    remaining = dict(cols)
    used = []
    print('  %-10s %14s %14s %14s' % ('正交分量', '净RSS降幅', '累计RSS', '占baseline%'))
    rss_cur = rss_base
    # Gram-Schmidt 按顺序正交化
    orth_basis = []
    for nm in order:
        c = cols[nm]
        v = c.copy()
        for (ob, onm) in orth_basis:
            v = v - (np.dot(ob, v) / np.dot(ob, ob)) * ob
        nv = np.linalg.norm(v)
        if nv < 1e-30:
            orth_basis.append((v, nm))
            print('  %-10s %14s %14s %14s' % (nm, '~0(共线)', '%.2f' % rss_cur, ''))
            continue
        orth_basis.append((v, nm))
        # 该正交分量的净贡献 = 对残差的解释
        bP = np.dot(v, resP_base) / np.dot(v, v)
        bQ = np.dot(v, resQ_base) / np.dot(v, v)
        drop = np.dot(v, v) * (bP ** 2 + bQ ** 2)
        rss_cur -= drop
        print('  %-10s %14.2f %14.2f %13.1f%%' % (nm, drop, rss_cur, 100 * drop / rss_base))

    # ---- 4) 随机列基准 ----
    print('=' * 72)
    print('【4】随机列过拟合基准（同自由度 3 列，重复 20 次取 RSS 降幅分布）')
    rng = np.random.default_rng(0)
    drops = []
    for t in range(20):
        R = rng.normal(size=(m, 3))
        Xr = np.column_stack((one, V, V ** 2, R))
        Qr = np.linalg.solve(Xr.T @ Xr, Xr.T)
        Hr = Xr @ Qr
        rp = resP - Hr @ resP
        rq = resQ - Hr @ resQ
        drops.append(rss_base - (np.sum(rp ** 2) + np.sum(rq ** 2)))
    drops = np.array(drops)
    print('  3 随机列净降幅: mean=%.2f  std=%.2f  max=%.2f (baseline %.2f)' %
          (drops.mean(), drops.std(), drops.max(), rss_base))
    print('  对照: Δf 三列(见上)的累计净降幅应显著超过随机列 max，才算物理吸收。')

    print('=' * 72)
    print('结论要点：')
    print('  · 若【2】单列斜率 ≈ -Q0 且【3】Δf 净贡献 >> 随机列 max → A2.21 方向成立，')
    print('    但 verify_frq_proj.py 的全矩阵 beta 判据（V·Δf 列 ≈ -2.5）必须作废，')
    print('    因其设计矩阵共线、beta 不可解释。')
    print('  · 若【2】单列斜率远偏离 -Q0 或【3】净贡献 ≈ 随机列 → A2.21 被削弱。')


if __name__ == '__main__':
    main()

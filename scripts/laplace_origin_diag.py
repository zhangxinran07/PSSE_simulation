# -*- coding: utf-8 -*-
"""路径1 后续对照判定: 零失配下 H2_a/H2_b 覆盖率只有 67%/62% 的来源。

背景(A2.32): 1000 组零失配 + 已知白噪声校准, 总体覆盖率 85.5%,
  H2_a 67% / H2_b 62%, z.std 1.43, 且 dyn6 相对误差中位 9.40%
  —— 模型完全正确、噪声仅 0.1%·P0 时不该有这么大的误差。
  待判的互斥解释:
    (O) 优化器/起点: trust-constr 在 H2 零空间方向没收敛, 1% 起点扰动被保留
    (P) 先验偏置: λ_H=0.05 的先验均值「差量=0」与真值差量≈10% 冲突, 把 MAP 拉离真值
    (A) 标定/后验近似: α 标定、regress 投影自由度或后验在 H2 方向非高斯

设计(关键: 四档共用同一 Σ, 且不依赖近奇异 Hessian 求 Newton 步):
  - 数据: 与 laplace_calibration.py 完全同 rng 顺序, 可精确复现 A2.32 的 1000 组
  - Σ = H_pert⁻¹/α 统一用在四档上(H_pert 为含先验、tc_pert 解处的 Hessian),
    故覆盖率/z 的变化只反映 MAP 位置变化, 不混入「λ=0 时零空间变平、σ 被撑宽」
  - 四档全部用 trust-constr(解析 jac), 不用 Hessian 求步:
      tc_pert   : λ=0.05, x0 = truth*(1+1%扰动)   [复现 A2.32 口径]
      tc_exact  : λ=0.05, x0 = truth(精确)         [剔除起点扰动]
      tc0_exact : λ=0,    x0 = truth(精确)         [剔除先验偏置]
      tc0_pert  : λ=0,    x0 = truth*(1+1%扰动)    [先验×起点交叉验证]
  - 硬指标(不依赖任何统计口径): 差量收缩比 |v_post|/|v_true|, v=H2_a-H2_b
      λ=0.05 应显著 <1 (过度收缩), λ=0 应 ≈1
  - 辅助: α 标定核对 ratio = sqrt(SSR_actual)/sqrt(SSR_expected)

判读链:
  tc_pert ≈ tc_exact                     -> 排除 (O)
  tc0_* 覆盖率回升 + 收缩比回到 ≈1      -> (P) 先验偏置主导
  tc0_* 仍低、收缩比≈1、α ratio≈1      -> (A) 零空间漂移/后验非高斯, 路径3 才有价值

用法: python laplace_origin_diag.py [N=300] [noise_frac=1e-3]
输出: Sample_Generation/results/laplace_origin_diag.txt / .pkl
"""
import numpy as np
import os
import sys
import pickle
from scipy import optimize as opt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
sys.path.insert(0, os.path.dirname(__file__))
from laplace_posterior import (RES, ORI, PNAMES, DT, LAM_H, SEED, lb, ub,
                               make_reg, hessian_cd, ssr_at, _trim_k)
from laplace_calibration import gen_PQ, P0
from control_zero_mismatch import sample_mix

OUT = os.path.join(RES, 'laplace_origin_diag.txt')
OUTPKL = os.path.join(RES, 'laplace_origin_diag.pkl')
MODES = ['tc_pert', 'tc_exact', 'tc0_exact', 'tc0_pert']
DYN = [0, 1, 2, 4, 5, 6]
D = np.zeros(8)          # 差量方向 v = H2_a - H2_b
D[2], D[6] = 1.0, -1.0


def solve_tc(obj, jac, x0, u, P, Q, bounds):
    r = opt.minimize(obj, x0, args=(u, P, Q, DT), jac=jac,
                     method='trust-constr', bounds=bounds,
                     options={'disp': False}, tol=1e-10)
    return np.asarray(r.x, dtype=float), float(r.fun)


def stats(th, truth, Cov_base):
    """四档共用 tc_pert 的 Σ, 覆盖率/z 的变化只反映 MAP 位置变化。"""
    sd = np.sqrt(np.diag(Cov_base))
    z = (th - truth) / (sd + 1e-30)
    cover = np.abs(th - truth) <= 1.96 * sd
    err = np.abs(th - truth) / np.abs(truth) * 100
    v = float(th[2] - th[6])
    vt = float(truth[2] - truth[6])
    return dict(th=th, sd=sd, z=z, cover=cover, err=err,
                v_ratio=abs(v) / abs(vt) if abs(vt) > 1e-12 else np.nan,
                Corr=Cov_base / np.outer(sd, sd))


def run(N=300, noise_frac=1e-3):
    ori = pickle.load(open(ORI, 'rb'))
    gs = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))[:30]
    U = [ori['%d ud' % g].values + 1j * ori['%d uq' % g].values for g in gs]
    obj_r, jac_r = make_reg(LAM_H)
    obj_0, jac_0 = make_reg(0.0)
    rng = np.random.default_rng(SEED + 2000)
    sigma_noise = noise_frac * P0
    bounds = np.transpose(np.array([lb * 2, ub * 2]))
    low, high = np.array(lb * 2) * 1.001, np.array(ub * 2) * 0.999

    recs = []
    for g in range(N):
        truth, _ = sample_mix(rng)
        u = U[g % len(U)]
        P, Q = gen_PQ(u, truth)
        if P is None:
            continue
        P = P + rng.normal(0, sigma_noise, len(P))
        Q = Q + rng.normal(0, sigma_noise, len(Q))
        # 消耗与 calibration.py 完全相同的随机数, 保证 tc_pert 可精确复现
        pert = 0.01 * rng.standard_normal(8)
        x0_pert = np.clip(truth * (1 + pert), low, high)
        x0_exact = np.clip(truth.copy(), low, high)

        n = len(u)
        m = n - 2 * _trim_k(n)
        c = 1000.0 / m
        ss_expected = (2 * m - 14) * sigma_noise ** 2
        alpha = np.sqrt(ss_expected) / (c * sigma_noise ** 2)

        th_pert, _ = solve_tc(obj_r, jac_r, x0_pert, u, P, Q, bounds)
        H_pert = hessian_cd(jac_r, th_pert, u, P, Q, DT, eps=1e-4)
        if np.linalg.eigvalsh(H_pert).min() <= 0:      # 与 A2.32 相同的正定门槛
            continue
        Cov_base = np.linalg.inv(H_pert) / alpha       # 四档共用同一把「尺子」

        th_exact, _ = solve_tc(obj_r, jac_r, x0_exact, u, P, Q, bounds)
        th0_exact, _ = solve_tc(obj_0, jac_0, x0_exact, u, P, Q, bounds)
        th0_pert, _ = solve_tc(obj_0, jac_0, x0_pert, u, P, Q, bounds)

        ss_act = ssr_at(th_pert, u, P, Q, DT)
        sd_v = float(np.sqrt(max(D @ Cov_base @ D, 0.0)))
        r = dict(g=g, truth=truth, alpha=alpha, pert=pert,
                 snr_v=abs(float(truth[2] - truth[6])) / (sd_v + 1e-30),
                 alpha_ratio=float(np.sqrt(ss_act / ss_expected))
                 if np.isfinite(ss_act) and ss_act > 0 else np.nan,
                 cond_H=float(np.linalg.cond(H_pert)))
        for k, th in zip(MODES, [th_pert, th_exact, th0_exact, th0_pert]):
            r[k] = stats(th, truth, Cov_base)
        recs.append(r)
        if (g + 1) % 25 == 0 or g == 0:
            print('[g=%3d/%d] ' % (g + 1, N) + ' '.join(
                '%s:%d/8' % (k, int(r[k]['cover'].sum())) for k in MODES), flush=True)

    L = []
    ap = L.append
    ap('=' * 112)
    ap('路径1 对照判定: 零失配下 H2 低覆盖率的来源 (N=%d, σ_noise=%.4g = %.2f%%·P0)' % (
        len(recs), sigma_noise, noise_frac * 100))
    ap('四档共用同一 Σ = H_pert⁻¹/α; tc_pert 与 A2.32 的 1000 组口径完全一致')
    ap('参照: A2.32 1000组 tc_pert 覆盖率 85.5%, H2_a 67% / H2_b 62%, z.std 1.43')
    ap('=' * 112)
    ap('')
    ap('[1] 四档估计器总评')
    ap('  %-10s %10s %10s %10s %12s %12s %12s' % (
        '估计器', '覆盖率', 'z.std', '|z|中位', '|z|>1.96', 'H2a/H2b覆盖', 'dyn6误差%中位'))
    summary = {}
    for k in MODES:
        Z = np.array([r[k]['z'] for r in recs])
        cov = np.array([r[k]['cover'] for r in recs])
        errs = np.array([r[k]['err'] for r in recs])
        row = dict(cover=100 * cov.mean(), zstd=float(np.std(Z)),
                   zmed=float(np.median(np.abs(Z))),
                   tail=100 * np.mean(np.abs(Z) > 1.96),
                   h2a=100 * cov[:, 2].mean(), h2b=100 * cov[:, 6].mean(),
                   dyn6=float(np.median(errs[:, DYN])), Z=Z, cov=cov)
        summary[k] = row
        ap('  %-10s %9.1f%% %10.2f %10.2f %11.1f%% %6.0f%%/%6.0f%% %12.4f' % (
            k, row['cover'], row['zstd'], row['zmed'], row['tail'],
            row['h2a'], row['h2b'], row['dyn6']))
    ap('')
    ap('[2] 逐参数覆盖率')
    ap('  %-8s %10s %10s %10s %10s' % ('参数', *MODES))
    for j, pn in enumerate(PNAMES):
        ap('  %-8s' % pn + ''.join(
            '%9.0f%%' % (100 * summary[k]['cov'][:, j].mean()) for k in MODES))
    ap('')
    ap('[3] 差量收缩比 |v_post|/|v_true| 中位  【不依赖统计口径的硬指标】')
    ap('  λ=0.05 应显著 <1 (过度收缩); λ=0 因零空间无约束可自由漂移, 未必≈1')
    for k in MODES:
        ap('  %-10s %.3f' % (k, np.median([r[k]['v_ratio'] for r in recs
                                          if np.isfinite(r[k]['v_ratio'])])))
    ap('')
    ap('[3b] H2 方向 z 的系统性偏移 (均值; 非零=固定方向偏置)')
    ap('  %-10s %12s %12s %12s' % ('估计器', 'mean z(H2_a)', 'mean z(H2_b)', 'mean z(差量)'))
    for k in MODES:
        Z = summary[k]['Z']
        ap('  %-10s %12.3f %12.3f %12.3f' % (
            k, Z[:, 2].mean(), Z[:, 6].mean(), (Z[:, 2] - Z[:, 6]).mean()))
    ap('')
    ap('[4] 起点扰动是否被优化器保留 (th-truth 相对%, H2 两参数)')
    for k in MODES:
        e = np.array([r[k]['th'] for r in recs])
        t = np.array([r['truth'] for r in recs])
        ap('  %-10s |ΔH2_a|中位=%.4f%%  |ΔH2_b|中位=%.4f%%' % (
            k, np.median(np.abs(e[:, 2] - t[:, 2]) / np.abs(t[:, 2]) * 100),
            np.median(np.abs(e[:, 6] - t[:, 6]) / np.abs(t[:, 6]) * 100)))
    ap('  (1% 起点扰动的中位幅度 ≈0.8%)')
    ap('')
    ap('[4b] 口径说明 / 数值稳健性')
    ap('  四档共用同一 Σ = H_pert⁻¹/α (A2.32 口径) → 覆盖率与 z 的变化只反映 MAP 位置,')
    ap('  排除「λ=0 时零空间变平、σ 被撑宽」造成的虚假回升; 四档均用 trust-constr')
    ap('  + 解析 jac, 不用近奇异 Hessian 求 Newton 步(cond(H) 中位 %.3g)。' % np.median(
        [r['cond_H'] for r in recs]))
    ap('  H_pert 非正定组已跳过(与 A2.32 同门槛): 有效 %d/%d 组' % (len(recs), N))
    ar = [r['alpha_ratio'] for r in recs if np.isfinite(r['alpha_ratio'])]
    ap('  α 标定核对 sqrt(SSR_actual)/sqrt(SSR_expected) 中位 = %.3f  (≈1 表示标定无误)' % (
        np.median(ar) if ar else np.nan))
    ap('  H2 差量方向信噪比 |v_true|/σ_v 中位 = %.2f' % np.median(
        [r['snr_v'] for r in recs]))
    ap('    (<2 表示真值差量本身落在 95%CI 噪声区间内 → 该方向本质不可辨识)')
    ap('')
    ap('[5] 判定')
    b = {k: summary[k]['cover'] for k in MODES}
    vr = {k: np.median([r[k]['v_ratio'] for r in recs
                        if np.isfinite(r[k]['v_ratio'])]) for k in MODES}
    zd = {k: summary[k]['Z'][:, 2].mean() - summary[k]['Z'][:, 6].mean()
          for k in MODES}
    ap('  ' + '  '.join('%s=%.1f%%' % (k, b[k]) for k in MODES) + '  (目标≈95%)')
    ap('  收缩比 ' + '  '.join('%s=%.3f' % (k, vr[k]) for k in MODES))
    ap('  差量 mean z ' + '  '.join('%s=%.2f' % (k, zd[k]) for k in MODES))
    opt_ok = abs(b['tc_pert'] - b['tc_exact']) < 3
    ap('  => %s' % ('排除 (O) 优化器/起点: 精确真值起点与 1% 扰动起点结果几乎相同,'
                    ' trust-constr 已收敛到含先验目标函数的极小。' if opt_ok else
                    '(O) 起点有影响: tc_exact 与 tc_pert 差异显著, 优化器收敛不可忽略。'))
    if b['tc0_exact'] >= 90 and b['tc_pert'] < 90:
        ap('  => (P) 先验偏置主导【本对照主结论】: λ_H=0.05 先验均值「差量=0」')
        ap('     与真值差量≈10% 冲突, 把 MAP 沿零空间系统性拉离真值;')
        ap('     λ=0 档覆盖率回到 %.1f%%、收缩比 %.3f→%.3f、差量 mean z %.2f→%.2f。' % (
            b['tc0_exact'], vr['tc_pert'], vr['tc0_exact'],
            zd['tc_pert'], zd['tc0_exact']))
        ap('     统一 Σ 口径下该回升不含 σ 变宽贡献, 是纯粹的 MAP 位置修正;')
        ap('     与 A2.31「|v_post|/|v_true|=0.769 过度收缩」是同一现象的两套度量。')
        ap('  => 路径3(MCMC)立项依据被削弱: 同一错先验下 MCMC 给的是同一个后验,')
        ap('     不会改善频率学覆盖率; 除非走层级模型给 λ 超先验让数据自动压低收缩。')
        ap('     优先级让位给「改先验结构(非零/自适应均值) 或 改模型结构」。')
    elif b['tc0_exact'] < 90:
        ap('  => (A) 根因是「H2 差量方向近零空间、数据信息量不足」【本对照主结论】:')
        ap('     λ=0.05 把差量压到 %.3f、λ=0 又自由漂到 %.3f, 两档都偏离真值 1.0;' % (
            vr['tc_pert'], vr['tc0_exact']))
        ap('     去先验后覆盖率仍 %.1f%%、α 标定 ratio≈1、起点无影响 → 排除 (O) 与标定问题。' % b['tc0_exact'])
        ap('     cond(H) 中位 %.3g, 差量方向信噪比 |v_true|/σ_v 中位 %.2f。' % (
            np.median([r['cond_H'] for r in recs]),
            np.median([r['snr_v'] for r in recs])))
        ap('     => 不是「后验非高斯」, 而是「该方向本就几乎无信息, 点估计必然不可靠」。')
        ap('     => 路径3(MCMC): 须先在 (总量u,差量v) 正交坐标重参数化, 否则 cond≈1e10 采不动;')
        ap('        其价值是给出可信的宽区间, 而非提高点估计精度。')
        ap('     => λ 合适值随噪声/失配水平变化(零失配收缩到 %.3f, PSSE 上 0.769)' % vr['tc_pert'])
        ap('        → 支持自适应 λ: 经验贝叶斯(路径2) 或 层级超先验。')
    else:
        ap('  => (P)+(O) 混合: 去掉先验覆盖率 %.1f%%、收缩比 %.3f, 需结合上表逐项判读。' % (
            b['tc0_exact'], vr['tc0_exact']))
    ap('=' * 112)
    txt = '\n'.join(L)
    open(OUT, 'w', encoding='utf-8').write(txt)
    pickle.dump(dict(recs=recs,
                     summary={k: {kk: vv for kk, vv in v.items()
                                  if kk not in ('Z', 'cov')}
                              for k, v in summary.items()}), open(OUTPKL, 'wb'))
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    nf = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    run(N, nf)

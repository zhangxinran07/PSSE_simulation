# -*- coding: utf-8 -*-
"""路径1: 拉普拉斯近似 —— 8 参数后验不确定性 (30 组 PSSE, MAP 处 Hessian)。

在 A2.26 的 MAP 口径(单一总功率观测 + K=12 多起点 + H2 差量正则 λ_H=0.05)上,
把「目标函数 J(θ) = c*sqrt(SSR) + λ_H*v²」当作 -log posterior 的代理:
  1) identify_multistart 求 MAP 解 θ_hat
  2) 对解析 jac 做自适应中心差分 -> 8x8 Hessian H = ∇²J(θ_hat)
  3) 用 MAP 处残差估计 σ², 标定温度 α = sqrt(SSR)/(c*σ²),
     使 ∇²(-log post) ≈ α*H, 后验协方差 Σ = H⁻¹/α
  4) 导出: 逐参数后验标准差/95%CI、后验相关矩阵、特征值谱(零空间方向)、
     真值落入 95%CI 的覆盖率(校准检验)、差量 v=H2_a-H2_b 的先验收缩诊断

尺度无关结论(相关矩阵/特征向量/条件数/相对排序)不依赖 α, 是主结论;
绝对标准差依赖高斯噪声假设, 单独标注。

用法: python laplace_posterior.py [n_groups] [K] [eps]
"""
import numpy as np
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import (obj_mix_iden, obj_mix_jac, identify_multistart,
                            _core_con, _core_var, regress, _trim_k)

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
ORI = os.path.join(RES, 'ori_data_mix_20261106_033919.pkl')
RRES = os.path.join(RES, 'result_all_mix_20261106_033919.pkl')
OUT = os.path.join(RES, 'laplace_posterior.txt')
OUTPKL = os.path.join(RES, 'laplace_posterior.pkl')

lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
DT = 0.01
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
SEED = 20261107
LAM_H = 0.05
# 差量方向单位向量 v = H2_a - H2_b = d·θ,  d = e2 - e6 (未归一化)
D_VEC = np.zeros(8)
D_VEC[2] = 1.0
D_VEC[6] = -1.0
D_UNIT = D_VEC / np.sqrt(2.0)
S_VEC = np.zeros(8)          # 总量方向 u = H2_a + H2_b
S_VEC[2] = 1.0
S_VEC[6] = 1.0


def make_reg(lam_H):
    """与 regularize_h2_diff.py / fine.py 完全一致的含正则目标与解析梯度。"""
    def obj(para, u, P, Q, dt):
        return obj_mix_iden(para, u, P, Q, dt) + lam_H * (para[2] - para[6]) ** 2

    def jac(para, u, P, Q, dt):
        g = np.array(obj_mix_jac(para, u, P, Q, dt), dtype=float)
        g[2] += 2 * lam_H * (para[2] - para[6])
        g[6] -= 2 * lam_H * (para[2] - para[6])
        return g
    return obj, jac


def ssr_at(para, u, P, Q, dt=DT):
    """MAP 处的去趋势残差平方和 SSR (P/Q 两段拼接), 用于估计噪声方差 σ²。"""
    a_a, b_a, H2_a, mva_a = para[0] / 0.75, para[1], para[2] / 0.75, P[0] * para[3]
    a_b, b_b, H2_b, mva_b = para[4] / 0.75, para[5], para[6] / 0.75, P[0] * para[7]
    ok_a, P_pa, Q_pa = _core_con(u, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb = _core_var(u, a_b, b_b, H2_b, mva_b, dt)
    if not (ok_a and ok_b):
        return np.nan
    n = len(u)
    k = _trim_k(n)
    sl = slice(k, n - k)
    _, ss = regress(np.abs(u[sl]), (P_pa + P_pb - P)[sl], (Q_pa + Q_pb - Q)[sl])
    return ss


def hessian_cd(jac, para, u, P, Q, dt=DT, eps=1e-4):
    """对解析梯度做自适应相对步长中心差分, 返回对称化 8x8 Hessian。"""
    p = np.asarray(para, dtype=float)
    m = len(p)
    H = np.zeros((m, m))
    h = eps * np.maximum(np.abs(p), 1e-3)
    for j in range(m):
        pp = p.copy(); pp[j] += h[j]
        pm = p.copy(); pm[j] -= h[j]
        gp = np.asarray(jac(pp, u, P, Q, dt), dtype=float)
        gm = np.asarray(jac(pm, u, P, Q, dt), dtype=float)
        H[:, j] = (gp - gm) / (2 * h[j])
    return 0.5 * (H + H.T)


def one_group(g, u, P, Q, truth, K, eps, obj, jac):
    """单组: MAP -> Hessian -> 后验量。返回 dict。"""
    best, _ = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj,
                                  tol=1e-8, n_restarts=K, jac_func=jac, seed=SEED + g)
    th = np.asarray(best.x, dtype=float)
    H = hessian_cd(jac, th, u, P, Q, DT, eps=eps)

    # --- 数值健康检查 ---
    gmap = np.asarray(jac(th, u, P, Q, DT), dtype=float)
    sym = np.linalg.norm(H - H.T) / (np.linalg.norm(H) + 1e-30)
    ev_H, evec_H = np.linalg.eigh(H)
    posdef = bool(ev_H.min() > 0)
    finite = bool(np.all(np.isfinite(H)) and np.all(np.isfinite(gmap)))

    # --- 噪声标定 α ---
    n = len(u); k = _trim_k(n); m = n - 2 * k
    ss = ssr_at(th, u, P, Q)
    n_eff = 2 * m                    # P/Q 两段残差点
    n_par = 8 + 6                     # 8 个参数 + 3x2 回归系数
    sigma2 = ss / max(n_eff - n_par, 1)
    c = 1000.0 / m
    alpha = np.sqrt(ss) / (c * sigma2) if (ss > 0 and sigma2 > 0) else np.nan

    # 尺度无关量: H⁻¹ 的相关结构 / 特征谱(与 α 无关)
    # 用伪逆 + 特征截断保证数值稳定(若近奇异)
    H_reg = H.copy()
    if ev_H.min() <= 0:
        H_reg = evec_H @ np.diag(np.maximum(ev_H, ev_H.max() * 1e-10)) @ evec_H.T
    Cov_shape = np.linalg.inv(H_reg)
    sd_shape = np.sqrt(np.diag(Cov_shape))
    Corr = Cov_shape / np.outer(sd_shape, sd_shape)

    # 绝对尺度后验协方差 Σ = H⁻¹/α
    Cov = Cov_shape / alpha if np.isfinite(alpha) and alpha > 0 else Cov_shape * np.nan
    sd = np.sqrt(np.diag(Cov))
    ci_half = 1.96 * sd
    rel_sd = sd / np.abs(th) * 100           # 相对后验标准差 %
    rel_ci = ci_half / np.abs(th) * 100      # 95%CI 半宽(相对 %)
    cover = np.abs(truth - th) <= ci_half    # 真值是否落入 95%CI

    # --- 标准化空间(白化)的零空间诊断 ---
    # 原始 H 受参数量纲支配(a~1e3 vs H2~1e-1), 最小特征向量不代表物理零空间。
    # 后验相关矩阵 Corr 已量纲无关; 其最大特征向量 = 最不可辨识方向。
    ev_C, evec_C = np.linalg.eigh(Corr)
    null_dir = evec_C[:, -1]
    # 标准化坐标 η = θ/sd_shape, 故 θ 空间方向 d 映射到 η 空间为 d/sd_shape
    d_std = D_UNIT / sd_shape
    d_std = d_std / (np.linalg.norm(d_std) + 1e-30)
    s_std = S_VEC / sd_shape
    s_std = s_std / (np.linalg.norm(s_std) + 1e-30)
    cos_null = float(abs(null_dir @ d_std))      # 最不确定方向 vs 差量方向
    cos_sum = float(abs(null_dir @ s_std))       # 最不确定方向 vs 总量方向
    lam_C_max = float(ev_C[-1])                  # 最大特征值(8=各向同性, 越大越零空间)
    load_H2a_H2b = (float(null_dir[2]), float(null_dir[6]))

    # --- 差量收缩诊断 ---
    v_post = th[2] - th[6]
    v_true = truth[2] - truth[6]
    v_ratio = abs(v_post) / abs(v_true) if abs(v_true) > 1e-12 else np.nan
    v_sign = np.sign(v_post) == np.sign(v_true)
    v_true_small = bool(abs(v_true) / abs(truth[2] + truth[6]) < 0.02)
    # 后验校准: 实际 |MAP-真值| 是后验 σ 的几倍 (>>1 = 结构失配/模型误设主导)
    misfit_ratio = np.abs(truth - th) / (sd + 1e-30)

    return dict(g=g, th=th, truth=truth, fun=float(best.fun), H=H, sym=sym,
                posdef=posdef, finite=finite, ev_H=ev_H, evec_H=evec_H,
                ss=ss, sigma2=sigma2, alpha=alpha, Corr=Corr, Cov=Cov, sd=sd,
                rel_sd=rel_sd, rel_ci=rel_ci, cover=cover, misfit_ratio=misfit_ratio,
                v_post=v_post, v_true=v_true, v_ratio=v_ratio, v_sign=v_sign,
                v_true_small=v_true_small, null_dir=null_dir,
                cos_null=cos_null, cos_sum=cos_sum, lam_C_max=lam_C_max,
                load_H2a_H2b=load_H2a_H2b, grad_norm=float(np.linalg.norm(gmap)))


def run(n_groups=30, K=12, eps=1e-4):
    ori = pickle.load(open(ORI, 'rb'))
    truth_df = pickle.load(open(RRES, 'rb'))
    groups = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))[:n_groups]
    obj, jac = make_reg(LAM_H)

    recs = []
    for g in groups:
        u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)
        r = one_group(g, u, P, Q, truth, K, eps, obj, jac)
        recs.append(r)
        print('[g=%2d] fun=%.5f grad=%.3g sym=%.2e PD=%s eig_min=%.3g '
              'cond(H)=%.3g cos_null=%.3f lamC=%.2f v_ratio=%.3f cover=%d/8 misfit=%.1f' % (
                  g, r['fun'], r['grad_norm'], r['sym'], r['posdef'],
                  r['ev_H'].min(), r['ev_H'].max() / max(abs(r['ev_H'].min()), 1e-30),
                  r['cos_null'], r['lam_C_max'], r['v_ratio'], int(r['cover'].sum()),
                  np.median(r['misfit_ratio'])), flush=True)

    # ---------------- 汇总 ----------------
    L = []
    ap = L.append
    ap('=' * 108)
    ap('路径1 拉普拉斯近似: 8 参数后验不确定性 (%d 组, K=%d 起点, λ_H=%.2f, eps=%.0e)' % (len(recs), K, LAM_H, eps))
    ap('Σ = H⁻¹/α ; H=∇²J(MAP) [解析 jac 中心差分] ; α=sqrt(SSR)/(c·σ²), σ²=SSR/(2m-14)')
    ap('=' * 108)

    ap('')
    ap('[A] 数值健康检查')
    ap('  正定组数: %d/%d ; Hessian 有限: %d/%d ; 对称残差中位=%.2e ; MAP 梯度范数中位=%.3g' % (
        sum(r['posdef'] for r in recs), len(recs), sum(r['finite'] for r in recs), len(recs),
        np.median([r['sym'] for r in recs]), np.median([r['grad_norm'] for r in recs])))
    ap('  cond(H) 中位 = %.3g  (条件数越大 = 零空间越显著)' % np.median(
        [r['ev_H'].max() / max(abs(r['ev_H'].min()), 1e-30) for r in recs]))

    ap('')
    ap('[B] 逐参数后验标准差 / 95%CI / 真值覆盖率 / 失配倍数')
    ap('  %-8s %11s %11s %10s %11s %11s' % (
        '参数', '后验σ(相对%)', '95%CI半宽%', '覆盖率', 'MAP误差%中位', '误差/σ倍数'))
    err = np.array([np.abs(r['th'] - r['truth']) / np.abs(r['truth']) * 100 for r in recs])
    mis = np.array([r['misfit_ratio'] for r in recs])
    for j, pn in enumerate(PNAMES):
        ap('  %-8s %11.2f %11.2f %9.0f%% %11.2f %11.1f' % (
            pn,
            np.median([r['rel_sd'][j] for r in recs]),
            np.median([r['rel_ci'][j] for r in recs]),
            100 * np.mean([r['cover'][j] for r in recs]),
            np.median(err[:, j]), np.median(mis[:, j])))
    ap('  ※ 校准良好时覆盖率应≈95%, 误差/σ≈1~2')
    ap('  ※ 若误差/σ >> 1: 后验只刻画了「噪声引起的随机不确定性」, 而实际误差由')
    ap('    **结构失配(模型误设)的系统性偏差**主导 —— 加数据量/降噪声救不了, 须改模型')

    ap('')
    ap('[C] 后验相关矩阵 (%d 组中位, 尺度无关)' % len(recs))
    Cm = np.median(np.array([r['Corr'] for r in recs]), axis=0)
    ap('  %-8s' % '' + ' '.join('%7s' % p for p in PNAMES))
    for i, pn in enumerate(PNAMES):
        ap('  %-8s' % pn + ' '.join('%7.3f' % Cm[i, j] for j in range(8)))
    ap('  H2_a↔H2_b 后验相关中位 = %.3f' % np.median([r['Corr'][2, 6] for r in recs]))

    ap('')
    ap('[D] 标准化空间的零空间方向检验 (量纲无关)')
    ap('  后验相关阵最大特征值 λmax 中位 = %.2f  (8=各向同性, 越大=零空间越显著)' % np.median(
        [r['lam_C_max'] for r in recs]))
    ap('  最不确定方向 与「差量方向」|cos| 中位 = %.3f ; 与「总量方向」|cos| 中位 = %.3f' % (
        np.median([r['cos_null'] for r in recs]), np.median([r['cos_sum'] for r in recs])))
    ap('  最不确定方向在 (H2_a, H2_b) 上的载荷中位 = (%.3f, %.3f)  (反号≈差量方向)' % (
        np.median([r['load_H2a_H2b'][0] for r in recs]),
        np.median([r['load_H2a_H2b'][1] for r in recs])))
    ap('  cond(H) 中位 = %.3g (原始参数空间, 受量纲影响, 仅作数值健康参考)' % np.median(
        [r['ev_H'].max() / max(abs(r['ev_H'].min()), 1e-30) for r in recs]))

    ap('')
    ap('[E] 差量 v=H2_a-H2_b 的收缩/偏置诊断')
    vr = [r['v_ratio'] for r in recs if np.isfinite(r['v_ratio']) and not r['v_true_small']]
    ap('  |v_post|/|v_true| 中位 = %.3f  (<1 = 被压向 0, 即先验收缩)' % (
        np.median(vr) if vr else np.nan))
    ap('  v 符号与真值一致的组数 = %d/%d ; 真值差量近零(|v|/总量<2%%)的组数 = %d' % (
        sum(r['v_sign'] for r in recs), len(recs), sum(r['v_true_small'] for r in recs)))
    ap('  真值 |v|/总量 中位 = %.2f%% ; MAP |v|/总量 中位 = %.2f%%' % (
        100 * np.median([abs(r['v_true']) / abs(r['truth'][2] + r['truth'][6]) for r in recs]),
        100 * np.median([abs(r['v_post']) / abs(r['th'][2] + r['th'][6]) for r in recs])))

    ap('')
    ap('[F] 噪声标定与后验校准总评')
    ap('  σ² 中位 = %.4g ; α 中位 = %.4g ; SSR 中位 = %.4g' % (
        np.median([r['sigma2'] for r in recs]), np.median([r['alpha'] for r in recs]),
        np.median([r['ss'] for r in recs])))
    all_cov = 100 * np.mean([r['cover'].mean() for r in recs])
    ap('  全体 8 参数×%d 组 的 95%%CI 覆盖率 = %.1f%%' % (len(recs), all_cov))
    ap('  全参数「误差/σ」中位 = %.1f 倍' % np.median(mis))
    ap('  ※【口径警告】α 把 MAP 处残差当作「白噪声」来标定 σ², 因此 Σ 只反映')
    ap('    「给定模型正确」时的随机不确定性; 任何结构失配都表现为系统性偏置, 不在 Σ 内。')
    ap('  ※ 目标函数含 regress 二次多项式去趋势投影 (I-H), 会把结构失配的低频成分')
    ap('    一并吸收, 使残差 σ² 进一步偏小 → 后验 CI 虚假变窄。')
    ap('  ※ 因此 [B] 的低覆盖率不是拉普拉斯近似失效, 而是**模型误设的定量证据**;')
    ap('    尺度无关结论([C][D])仍成立, 绝对 σ 需按下行因子膨胀才可用。')
    infl = np.median(mis) / 1.96
    ap('  → 若要把 95%%CI 校准到实际误差, σ 需放大约 %.0f 倍 (等价于承认' % infl)
    ap('     残差中约 %.0f 倍方差来自未建模结构, 而非白噪声)' % (infl ** 2))
    ap('=' * 108)

    txt = '\n'.join(L)
    open(OUT, 'w', encoding='utf-8').write(txt)
    pickle.dump(recs, open(OUTPKL, 'wb'))
    print('\n' + txt, flush=True)
    print('\n[saved] %s\n[saved] %s' % (OUT, OUTPKL), flush=True)


if __name__ == '__main__':
    n_groups = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
    run(n_groups, K, eps)

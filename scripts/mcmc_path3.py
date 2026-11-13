# -*- coding: utf-8 -*-
"""路径3：完全贝叶斯 / MCMC（按 A2.33 收口的定位）。

A2.33 已把立项依据收口：H2 差量方向本质信息不足（cond(H)≈7e9、SNR 1.73<2），
MCMC **不会提高点估计精度、也不会把 95% 覆盖率拉回 95%**。本脚本的用途只有三个：
  ① 给出 H2 差量方向「可信的宽区间」（拉普拉斯高斯近似可能给窄了 / 形状不对）；
  ② 检验后验形状（非高斯 / 偏斜 / 截断 / 多峰）；
  ③ 用真实后验复核 A2.31 的「低覆盖率是 bias 而非区间太窄」。

⚠️ A2.33 的前置硬要求：必须先在 (总量 u, 差量 v) 正交坐标重参数化，否则
   cond(H)≈7e9 原始坐标 HMC/RWM 采不动。本脚本做两层变换：
   1) 物理正交变换 T: θ = T·φ,  φ = [s_a,v_a,s_b,v_b,s_H,v_H,s_P,v_P],
      s_x=(x_a+x_b)/√2, v_x=(x_a−x_b)/√2, T⁻¹=Tᵀ（严格正交）。
   2) Hessian 白化: H_φ = TᵀHT = UΛUᵀ,  z = √α·Λ^{1/2}Uᵀ(φ−φ̂),
      即 φ = φ̂ + A z,  A = U·diag(1/(√α·√λ))。高斯近似下 z~N(0,I)，
      各向同性，cond 不再是问题；小特征值按 λmax·1e-10 截断（与 A2.31 一致）。
   后验: log p(θ|y) = −α·J(θ) + const, J = obj_mix_iden + λ_H(H2_a−H2_b)²，
   α 与 A2.31/A2.33 **完全同口径**（PSSE 用残差标定；零失配用已知注入噪声标定）。
   采样器: 自适应随机游走 Metropolis（burn-in 期间调步长到接受率≈0.234，之后固定）。

用法：
    python scripts/mcmc_path3.py <psse|zero> <n_groups> <n_steps> <n_chains> [tag]
输出：Sample_Generation/results/mcmc_path3_<tag>.txt / .pkl
"""
import os
import sys
import time
import pickle
import numpy as np
import scipy.optimize as opt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'package'))
from laplace_posterior import (RES, ORI, RRES, PNAMES, DT, LAM_H, SEED, lb, ub,
                               make_reg, hessian_cd, one_group, _trim_k)
from laplace_calibration import gen_PQ, P0
from control_zero_mismatch import sample_mix

DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
NOISE_FRAC = 1e-3
LOW = np.array(lb * 2, dtype=float)
HIGH = np.array(ub * 2, dtype=float)


# ---------------- 正交 (总量, 差量) 变换 ----------------
def build_T():
    """θ = T φ; φ 顺序 [s_a,v_a,s_b,v_b,s_H,v_H,s_P,v_P]。T 严格正交。"""
    T = np.zeros((8, 8))
    r = 1.0 / np.sqrt(2.0)
    pairs = [(0, 4, 0), (1, 5, 2), (2, 6, 4), (3, 7, 6)]  # (a 侧, b 侧, φ 起始列)
    for ia, ib, j in pairs:
        T[ia, j] = r
        T[ia, j + 1] = r
        T[ib, j] = r
        T[ib, j + 1] = -r
    return T


T = build_T()


def whiten(H_theta, alpha, phi_hat):
    """返回 A（φ = φ̂ + A z）与特征值谱。"""
    H_phi = T.T @ H_theta @ T
    ev, U = np.linalg.eigh(H_phi)
    ev = np.maximum(ev, max(ev.max(), 1e-30) * 1e-10)
    A = U @ np.diag(1.0 / (np.sqrt(alpha) * np.sqrt(ev)))
    return A, ev


def make_logp(obj, u, P, Q, alpha, phi_hat, A):
    """z 空间的 log posterior（含参数边界，越界 = 硬拒绝）。"""
    def logp(z):
        th = T @ (phi_hat + A @ z)
        if np.any(th < LOW) or np.any(th > HIGH) or not np.all(np.isfinite(th)):
            return -np.inf
        try:
            J = float(obj(th, u, P, Q, DT))
        except Exception:
            return -np.inf
        if not np.isfinite(J):
            return -np.inf
        return -alpha * J
    return logp


# ---------------- 采样器 ----------------
def sample_chain(logp, z0, n_steps, burn, thin, seed, s0=0.8):
    """自适应 RWM；burn-in 内每 100 步调一次步长，之后固定（保证平稳分布）。"""
    rng = np.random.default_rng(seed)
    z = np.asarray(z0, dtype=float).copy()
    lp = logp(z)
    s = s0
    out, nacc, win_acc, rates = [], 0, 0, []
    for t in range(n_steps):
        zp = z + s * rng.standard_normal(8)
        lpp = logp(zp)
        accept = bool(np.isfinite(lpp) and np.log(rng.random() + 1e-300) < lpp - lp)
        if accept:
            z, lp = zp, lpp
            nacc += 1
            win_acc += 1
        if t < burn and (t + 1) % 100 == 0:
            rate = win_acc / 100.0
            rates.append(rate)
            s *= float(np.exp(np.clip(1.5 * (rate - 0.234), -0.4, 0.4)))
            s = float(np.clip(s, 1e-3, 20.0))
            win_acc = 0
        if t >= burn and (t - burn) % thin == 0:
            out.append(z.copy())
    return np.asarray(out), dict(acc=nacc / n_steps, step=s, rates=rates)


def rhat(chains):
    """Gelman-Rubin R-hat（chains: [n_chain, n_draw, 8]）。"""
    ch = np.asarray(chains, dtype=float)
    m, n = ch.shape[0], ch.shape[1]
    if m < 2 or n < 4:
        return np.full(8, np.nan)
    means = ch.mean(axis=1)
    vars_ = ch.var(axis=1, ddof=1)
    W = vars_.mean(axis=0)
    B = n * means.var(axis=0, ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / np.maximum(W, 1e-300))


def ess(chains, max_lag=200):
    """每链 ESS 求和（初正自相关序列截断）。"""
    ch = np.asarray(chains, dtype=float)
    m, n, d = ch.shape
    tot = np.zeros(d)
    for c in range(m):
        x = ch[c] - ch[c].mean(axis=0)
        var = x.var(axis=0, ddof=1) + 1e-300
        rho_sum = np.zeros(d)
        for lag in range(1, min(max_lag, n // 4)):
            r = (x[:-lag] * x[lag:]).mean(axis=0) / var
            if np.all(r < 0.05):
                break
            rho_sum += np.where(r > 0, r, 0)
        tot += n / (1 + 2 * rho_sum)
    return tot


# ---------------- 数据加载 ----------------
def load_psse(n_groups):
    ori = pickle.load(open(ORI, 'rb'))
    truth_df = pickle.load(open(RRES, 'rb'))
    gs = sorted(set(int(c.split()[0]) for c in ori.columns
                    if c.split()[0].isdigit()))[:n_groups]
    out = []
    for g in gs:
        u = ori['%d ud' % g].values + 1j * ori['%d uq' % g].values
        P, Q = ori['%d P' % g].values, ori['%d Q' % g].values
        truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)
        out.append((g, u, P, Q, truth))
    return out


def load_zero(n_groups):
    """与 A2.32/A2.33 完全同口径的零失配 + 已知白噪声数据。"""
    ori = pickle.load(open(ORI, 'rb'))
    gs = sorted(set(int(c.split()[0]) for c in ori.columns
                    if c.split()[0].isdigit()))[:30]
    U = [ori['%d ud' % g].values + 1j * ori['%d uq' % g].values for g in gs]
    rng = np.random.default_rng(SEED + 2000)
    sigma = NOISE_FRAC * P0
    out = []
    for g in range(n_groups):
        truth, _ = sample_mix(rng)
        u = U[g % len(U)]
        P, Q = gen_PQ(u, truth)
        if P is None:
            continue
        P = P + rng.normal(0, sigma, len(P))
        Q = Q + rng.normal(0, sigma, len(Q))
        # 消耗与 A2.33 tc_exact 相同的随机数（保持样本序列一致）
        _ = 0.01 * rng.standard_normal(8)
        out.append((g, u, P, Q, np.asarray(truth, dtype=float)))
    return out


def map_group(kind, g, u, P, Q, truth, obj, jac):
    """返回 (th_MAP, H, alpha, sd_laplace, cover_laplace)。"""
    if kind == 'psse':
        r = one_group(g, u, P, Q, truth, 12, 1e-4, obj, jac)
        th, H, alpha = r['th'], r['H'], r['alpha']
        sd = r['sd']
        cover = r['cover']
    else:
        n = len(u)
        m = n - 2 * _trim_k(n)
        c = 1000.0 / m
        sigma = NOISE_FRAC * P0
        ss_expected = (2 * m - 14) * sigma ** 2
        alpha = np.sqrt(ss_expected) / (c * sigma ** 2)
        bounds = np.transpose(np.array([lb * 2, ub * 2]))
        x0 = np.clip(truth.copy(), LOW * 1.001, HIGH * 0.999)
        res = opt.minimize(obj, x0, args=(u, P, Q, DT), jac=jac,
                           method='trust-constr', bounds=bounds,
                           options={'disp': False}, tol=1e-10)
        th = np.asarray(res.x, dtype=float)
        H = hessian_cd(jac, th, u, P, Q, DT, eps=1e-4)
        ev = np.linalg.eigvalsh(H)
        if ev.min() <= 0:
            H = H + np.eye(8) * abs(ev.min()) * 1.5
        Cov = np.linalg.inv(H) / alpha
        sd = np.sqrt(np.diag(Cov))
        cover = np.abs(th - truth) <= 1.96 * sd
    return th, H, alpha, sd, cover


# ---------------- 主流程 ----------------
def run(kind, n_groups, n_steps, n_chains, tag):
    obj, jac = make_reg(LAM_H)
    data = load_psse(n_groups) if kind == 'psse' else load_zero(n_groups)
    burn = n_steps // 3
    thin = 2
    outp = os.path.join(RES, 'mcmc_path3_%s.txt' % tag)
    outpk = os.path.join(RES, 'mcmc_path3_%s.pkl' % tag)
    recs, t0 = [], time.time()

    for g, u, P, Q, truth in data:
        tg = time.time()
        th, H, alpha, sd_lap, cov_lap = map_group(kind, g, u, P, Q, truth, obj, jac)
        if not np.isfinite(alpha) or alpha <= 0:
            print('[%s g=%d] α 无效, 跳过' % (kind, g), flush=True)
            continue
        phi_hat = T.T @ th
        A, ev = whiten(H, alpha, phi_hat)
        logp = make_logp(obj, u, P, Q, alpha, phi_hat, A)

        chains, diags = [], []
        for c in range(n_chains):
            z0 = np.zeros(8) if c == 0 else np.random.default_rng(SEED + g * 97 + c).standard_normal(8) * 0.7
            z_draws, d = sample_chain(logp, z0, n_steps, burn, thin,
                                      SEED + g * 997 + c * 31)
            chains.append(z_draws)
            diags.append(d)
        Zc = np.asarray(chains)                              # (n_chain, n_draw, 8)
        # θ = T(φ̂ + A z) → 逐链变换（行向量形式 θᵀ = φᵀTᵀ）
        theta = np.stack([(phi_hat[None, :] + Zc[c] @ A.T) @ T.T for c in range(n_chains)])
        post_mean = theta.reshape(-1, 8).mean(axis=0)
        post_sd = theta.reshape(-1, 8).std(axis=0, ddof=1)
        q025 = np.percentile(theta.reshape(-1, 8), 2.5, axis=0)
        q975 = np.percentile(theta.reshape(-1, 8), 97.5, axis=0)
        cov_mc = (truth >= q025) & (truth <= q975)   # 等尾 95% 区间覆盖率

        # 差量方向（H2_a−H2_b）/√2 的边际后验形状
        vh = (theta[:, :, 2] - theta[:, :, 6]).reshape(-1) / np.sqrt(2.0)
        vs = (vh - vh.mean()) / (vh.std() + 1e-30)
        skew = float(np.mean(vs ** 3))
        kurt = float(np.mean(vs ** 4) - 3.0)
        vh_q = np.percentile(vh, [2.5, 50, 97.5])

        R = rhat(theta)          # 在 θ 空间算收敛诊断（物理参数才是关心的量）
        E = ess(theta)
        err_map = np.median(np.abs(th[DYN] - truth[DYN]) / np.abs(truth[DYN]) * 100)
        err_pm = np.median(np.abs(post_mean[DYN] - truth[DYN]) / np.abs(truth[DYN]) * 100)
        z_bias = np.median(np.abs(truth - post_mean) / (post_sd + 1e-30))

        rec = dict(g=g, kind=kind, truth=truth, th=th, alpha=alpha,
                   sd_lap=sd_lap, cov_lap=cov_lap,
                   post_mean=post_mean, post_sd=post_sd, q025=q025, q975=q975,
                   cov_mc=cov_mc, ratio_sd=post_sd / (sd_lap + 1e-30),
                   skew_vh=skew, kurt_vh=kurt, vh_q=vh_q,
                   rhat=R, ess=E, acc=[d['acc'] for d in diags],
                   step=[d['step'] for d in diags], z_bias=z_bias,
                   width_ratio=(q975 - q025) / (2 * 1.96 * sd_lap + 1e-30),
                   err_map=err_map, err_pm=err_pm,
                   ev_phi=ev, n_draw=int(Zc.shape[1]))
        recs.append(rec)
        print('[%s %d/%d g=%d] %.1fs | R̂max=%.3f ESSmed=%.0f acc=%s | '
              'cov LAP=%d/8 MC=%d/8 | sd比中位=%.2f | MAPerr=%.2f%% PMerr=%.2f%%' % (
                  kind, len(recs), len(data), g, time.time() - tg,
                  np.nanmax(R), np.median(E),
                  ','.join('%.2f' % a for a in rec['acc']),
                  int(cov_lap.sum()), int(cov_mc.sum()),
                  np.median(rec['ratio_sd']), err_map, err_pm), flush=True)

    pickle.dump(recs, open(outpk, 'wb'))
    summarize(recs, kind, n_steps, n_chains, time.time() - t0, outp)
    print('DONE -> %s (%.1f min)' % (outp, (time.time() - t0) / 60), flush=True)


def summarize(recs, kind, n_steps, n_chains, wall, outp):
    L = []
    ap = L.append
    ap('=' * 116)
    ap('路径3 完全贝叶斯 / MCMC —— %s 数据, %d 组, %d 步 × %d 链, 用时 %.1f min' % (
        '真实 PSSE' if kind == 'psse' else '零失配+已知噪声', len(recs),
        n_steps, n_chains, wall / 60))
    ap('重参数化: θ=Tφ (总量/差量严格正交) + Hessian 白化 z=√α·Λ^½Uᵀ(φ−φ̂); '
       'logp = −α·J, α 与 A2.31/A2.33 同口径; 边界 lb*2/ub*2 硬截断')
    ap('=' * 116)
    ap('')
    if not recs:
        ap('无有效样本')
        open(outp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
        return
    R = np.array([r['rhat'] for r in recs])
    E = np.array([r['ess'] for r in recs])
    acc = np.array([r['acc'] for r in recs])
    ap('[1] 采样健康度')
    ap('  R̂ 中位/最大 = %.3f / %.3f ; R̂<1.1 的组数 = %d/%d' % (
        np.nanmedian(R), np.nanmax(R), int((np.nanmax(R, axis=1) < 1.1).sum()), len(recs)))
    ap('  ESS 中位 = %.0f (每链 %d 抽样) ; 接受率中位 = %.3f' % (
        np.median(E), recs[0]['n_draw'], np.median(acc)))
    ap('  H_φ 条件数中位 = %.3g' % np.median([r['ev_phi'].max() / r['ev_phi'].min() for r in recs]))
    ap('')
    ap('[2] 后验形状：MCMC 后验标准差 / 拉普拉斯 σ（>1 = 高斯近似把区间给窄了）')
    RAT = np.array([r['ratio_sd'] for r in recs])
    ap('  %-10s %10s %10s %10s' % ('参数', 'P25', '中位', 'P75'))
    for j, nm in enumerate(PNAMES):
        ap('  %-10s %10.3f %10.3f %10.3f' % (
            nm, np.percentile(RAT[:, j], 25), np.median(RAT[:, j]), np.percentile(RAT[:, j], 75)))
    ap('  差量 v_H=(H2_a−H2_b)/√2 边际: 偏度中位 %+.3f, 超额峰度中位 %+.3f' % (
        np.median([r['skew_vh'] for r in recs]), np.median([r['kurt_vh'] for r in recs])))
    ap('')
    ap('[3] 95% 区间覆盖率（真值落入等尾区间）与点估计精度')
    cl = np.array([r['cov_lap'] for r in recs])
    cm = np.array([r['cov_mc'] for r in recs])
    ap('  拉普拉斯 95%%CI 覆盖率 = %.1f%% (%d/%d)' % (
        100 * cl.mean(), int(cl.sum()), cl.size))
    ap('  MCMC 等尾 95%% 区间覆盖率 = %.1f%% (%d/%d)' % (
        100 * cm.mean(), int(cm.sum()), cm.size))
    ap('  H2_a/H2_b 单独: LAP %.1f%% / MC %.1f%%' % (
        100 * cl[:, H2].mean(), 100 * cm[:, H2].mean()))
    ap('  其余 6 参数:     LAP %.1f%% / MC %.1f%%' % (
        100 * cl[:, DYN].mean(), 100 * cm[:, DYN].mean()))
    WR = np.array([r['width_ratio'] for r in recs])
    ap('  MCMC 等尾区间宽度 / 拉普拉斯 95%%CI 宽度 中位 = %.3f' % np.median(WR))
    ap('  真值偏离后验均值 / 后验 σ (z_bias) 中位 = %.1f 倍' % (
        np.median([r['z_bias'] for r in recs])))
    ap('  dyn6 逐组中位相对误差: MAP %.2f%% → 后验均值 %.2f%%' % (
        np.median([r['err_map'] for r in recs]), np.median([r['err_pm'] for r in recs])))
    ap('')
    ap('[4] H2 差量方向的可信区间')
    w = np.array([r['vh_q'][2] - r['vh_q'][0] for r in recs])
    vt = np.array([abs(r['truth'][2] - r['truth'][6]) / np.sqrt(2) for r in recs])
    ap('  v_H 95%% 区间宽度中位 = %.4g ; |v_true| 中位 = %.4g ; 宽度/真值 = %.2f' % (
        np.median(w), np.median(vt), np.median(w / (vt + 1e-30))))
    ap('  真值落入 v_H 区间的组数 = %d/%d' % (
        sum(1 for r in recs if r['vh_q'][0] <= (r['truth'][2] - r['truth'][6]) / np.sqrt(2) <= r['vh_q'][2]),
        len(recs)))
    ap('')
    ap('[5] 判定（对应 A2.33 收口）')
    ap('  · 若 MCMC 区间与拉普拉斯接近、覆盖率仍远低于 95% → 低覆盖率是系统性 bias，')
    ap('    不是「高斯近似把区间给窄了」；A2.33「差量方向信息不足」得到独立复核。')
    ap('  · 若后验均值与 MAP 精度相当 → MCMC 不提高点估计精度，A2.33 结论成立。')
    ap('=' * 116)
    txt = '\n'.join(L)
    open(outp, 'w', encoding='utf-8').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'psse'
    ng = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    nc = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    tag = sys.argv[5] if len(sys.argv) > 5 else '%s_N%d_S%d' % (kind, ng, ns)
    run(kind, ng, ns, nc, tag)

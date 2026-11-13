# -*- coding: utf-8 -*-
"""路径1 决定性校准检验: 零结构失配 + 已知白噪声下的拉普拉斯后验覆盖率。

目的: 判定 PSSE 上「95%CI 覆盖率仅 9.6%」的归因
  (A) 拉普拉斯/Hessian/α 口径本身有错?  (B) 结构失配的系统性偏置不在 Σ 内?

设计(关键: 与 PSSE 口径只差「失配」, 其余全对齐):
  - u: 复用 PSSE 30 组真实激励
  - 真值: sample_mix 独立采样
  - P,Q: _core_con/_core_var 自身生成 => 零结构失配
  - **注入已知强度高斯白噪声 σ_noise**(相对 P0=6.0 的比例), 使 σ² 有确定值
  - α 用「已知 σ_noise」标定(不再用残差反推), 消除零失配下 α→∞ 的退化
  - MAP: 从 truth 附近启动的 trust-constr(零失配下真值即全局最优, 无需多起点)
  - 检验 |MAP-truth|/σ_post 是否 ~N(0,1), 覆盖率是否≈95%

用法: python laplace_calibration.py [N] [noise_frac]
"""
import numpy as np
import os
import sys
import pickle
from scipy import optimize as opt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import _core_con, _core_var
from control_zero_mismatch import sample_mix
from laplace_posterior import (RES, ORI, PNAMES, DT, LAM_H, SEED, lb, ub,
                               make_reg, hessian_cd, _trim_k)

OUT = os.path.join(RES, 'laplace_calibration.txt')
OUTPKL = os.path.join(RES, 'laplace_calibration.pkl')
P0 = 6.0


def gen_PQ(u, para):
    a_a, b_a, H2_a = para[0] / 0.75, para[1], para[2] / 0.75
    a_b, b_b, H2_b = para[4] / 0.75, para[5], para[6] / 0.75
    P_ima, P_imb = para[3], para[7]
    Ma, Mb = P0 * P_ima, P0 * P_imb
    Ps = P0 * (1 - P_ima - P_imb)
    ok_a, Pe_a, Qe_a = _core_con(u, a_a, b_a, H2_a, 1.0, DT)
    ok_b, Pe_b, Qe_b = _core_var(u, a_b, b_b, H2_b, 1.0, DT)
    if not (ok_a and ok_b):
        return None, None
    return Ps + Pe_a * Ma + Pe_b * Mb, 0.3 * Ps + Qe_a * Ma + Qe_b * Mb


def run(N=30, noise_frac=1e-3):
    ori = pickle.load(open(ORI, 'rb'))
    gs = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))[:30]
    U = [ori['%d ud' % g].values + 1j * ori['%d uq' % g].values for g in gs]
    obj, jac = make_reg(LAM_H)
    rng = np.random.default_rng(SEED + 2000)
    sigma_noise = noise_frac * P0
    bounds = np.transpose(np.array([lb * 2, ub * 2]))

    recs = []
    for g in range(N):
        truth, _ = sample_mix(rng)
        u = U[g % len(U)]
        P, Q = gen_PQ(u, truth)
        if P is None:
            continue
        P = P + rng.normal(0, sigma_noise, len(P))
        Q = Q + rng.normal(0, sigma_noise, len(Q))
        # 从真值附近启动(零失配下真值即全局最优), 避免多起点伪解干扰校准判断
        x0 = np.clip(truth * (1 + 0.01 * rng.standard_normal(8)),
                     np.array(lb * 2) * 1.001, np.array(ub * 2) * 0.999)
        res = opt.minimize(obj, x0, args=(u, P, Q, DT), jac=jac,
                           method='trust-constr', bounds=bounds,
                           options={'disp': False}, tol=1e-10)
        th = np.asarray(res.x, dtype=float)
        H = hessian_cd(jac, th, u, P, Q, DT, eps=1e-4)
        if np.linalg.eigh(H)[0].min() <= 0:
            continue
        C = np.linalg.inv(H)
        n = len(u); m = n - 2 * _trim_k(n)
        c = 1000.0 / m
        # α 用已知噪声标定: H ≈ (c/sqrt(SSR))·JᵀJ, ∇²(-logpost) ≈ JᵀJ/σ²
        #   => α = sqrt(SSR)/(c·σ²). 零失配+已知噪声下 SSR ≈ (2m-14)·σ²
        ss_expected = (2 * m - 14) * sigma_noise ** 2
        alpha = np.sqrt(ss_expected) / (c * sigma_noise ** 2)
        Cov = C / alpha
        sd = np.sqrt(np.diag(Cov))
        cover = np.abs(truth - th) <= 1.96 * sd
        z = (th - truth) / (sd + 1e-30)
        recs.append(dict(g=g, th=th, truth=truth, fun=float(res.fun),
                         Corr=Cov / np.outer(sd, sd), cover=cover, sd=sd, z=z,
                         err=np.abs(th - truth) / np.abs(truth) * 100,
                         sigma_noise=sigma_noise, alpha=alpha))
        if (g + 1) % 10 == 0 or g == 0:
            print('[g=%3d/%d] fun=%.3e cover=%d/8 |z|med=%.2f' % (
                g + 1, N, res.fun, int(cover.sum()), np.median(np.abs(z))), flush=True)

    L = []
    ap = L.append
    ap('=' * 108)
    ap('路径1 决定性校准检验: 零失配 + 已知白噪声 (N=%d, σ_noise=%.4g = %.2f%%·P0)' % (
        len(recs), sigma_noise, noise_frac * 100))
    ap('u 复用 PSSE 真实激励; P/Q 由 _core 自身生成; 注入已知白噪声; α 用已知 σ 标定')
    ap('对照: PSSE 30 组(结构失配) 覆盖率 9.6%%, 误差/σ 中位 11.6 倍')
    ap('=' * 108)
    ap('')
    ap('[1] 后验校准总评')
    ap('  95%%CI 覆盖率 = %.1f%%   (校准良好应≈95%%)' % (
        100 * np.mean([r['cover'].mean() for r in recs])))
    Z = np.array([r['z'] for r in recs])
    ap('  标准化偏差 z=(MAP-truth)/σ_post: |z| 中位 = %.2f  (应≈0.67)' % np.median(np.abs(Z)))
    ap('  z 的 std = %.2f  (应≈1.00)' % np.std(Z))
    ap('  |z|>1.96 的比例 = %.1f%%  (应≈5%%)' % (100 * np.mean(np.abs(Z) > 1.96)))
    ap('  MAP 相对误差中位(dyn6) = %.4f%%' % np.median(
        np.array([r['err'] for r in recs])[:, [0, 1, 2, 4, 5, 6]]))
    ap('')
    ap('[2] 逐参数覆盖率 / |z|中位')
    ap('  %-8s %10s %12s' % ('参数', '覆盖率', '|z|中位'))
    for j, pn in enumerate(PNAMES):
        ap('  %-8s %9.0f%% %12.2f' % (
            pn, 100 * np.mean([r['cover'][j] for r in recs]), np.median(np.abs(Z[:, j]))))
    ap('')
    ap('[3] 后验相关矩阵 (尺度无关)')
    Cm = np.median(np.array([r['Corr'] for r in recs]), axis=0)
    ap('  %-8s' % '' + ' '.join('%7s' % p for p in PNAMES))
    for i, pn in enumerate(PNAMES):
        ap('  %-8s' % pn + ' '.join('%7.3f' % Cm[i, j] for j in range(8)))
    ap('  H2_a↔H2_b 相关中位 = %.3f  (PSSE 口径 -0.955)' % np.median(
        [r['Corr'][2, 6] for r in recs]))
    ap('')
    ap('[4] 判读')
    ap('  覆盖率≈95%% 且 z.std≈1 → 路径1 的 Hessian/α 口径正确;')
    ap('    PSSE 上 9.6%% 的低覆盖率 = 结构失配的系统性偏置, 与近似方法无关。')
    ap('  H2_a↔H2_b 仍强负相关 → 零空间源于母线总功率观测降维, 与失配无关。')
    ap('=' * 108)
    txt = '\n'.join(L)
    open(OUT, 'w', encoding='utf-8').write(txt)
    pickle.dump(recs, open(OUTPKL, 'wb'))
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    nf = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    # 保护 A2.31 的 30 组证据: N!=30 时另存文件, 避免大样本跑完覆盖
    if N != 30:
        OUT = os.path.join(RES, 'laplace_calibration_N%d.txt' % N)
        OUTPKL = os.path.join(RES, 'laplace_calibration_N%d.pkl' % N)
        print('[output] %s / %s' % (OUT, OUTPKL), flush=True)
    run(N, nf)

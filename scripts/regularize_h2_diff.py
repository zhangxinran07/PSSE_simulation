# -*- coding: utf-8 -*-
"""H2/P_im 差量方向 Tikhonov 正则验证: 定住「此消彼长」零空间, 提升 H2 辨识精度。

回答张老师「把 P_im 和 H2 辨识准确」(2026-11-07) 的第三步 —— 先验/正则方案:

诊断结论(diagnose_pim_h2_psse.txt):
  - P_im 总量 4.58% / 差量 4.82% -> 已够准, 无需正则(λ_P=0)。
  - H2  总量 7.27% / 差量 20.80% / 比值 63.70% -> 差量在零空间(此消彼长 r=-0.61), 是唯一瓶颈。

正则设计(只碰零空间「差量」方向, 不碰可辨识的「总量」方向):
  obj = obj_mix_iden + λ_H·(H2_a - H2_b)² + λ_P·(P_ima - P_imb)²
  (H2_a-H2_b)² 与 (H2_a+H2_b) 正交 -> 收缩差量、保留总量。
  真值 H2_a,H2_b ~ iid U(0.1,0.2) -> 差量真值 |H2_a-H2_b|/总量 中位≈10%,
  故最优 λ 应把 H2 差量误差从 20.8% 压到 ~10%(纯偏置下限), 而不是压到 0。

本脚本在 PSSE 真实数据(30 组)上做 λ 网格扫描, 输出 H2/P_im 的总量/差量/比值误差随 λ 变化,
据此给出最优 λ 与净收益。λ_P 默认 0(先只验 H2), 可单独验证。
"""
import numpy as np
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import obj_mix_iden, obj_mix_jac, identify_multistart

PROJ = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(PROJ, 'Sample_Generation', 'results')
ORI = os.path.join(RES, 'ori_data_mix_20261106_033919.pkl')
RRES = os.path.join(RES, 'result_all_mix_20261106_033919.pkl')
OUT = os.path.join(RES, 'regularize_h2_diff.txt')

lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
DT = 0.01
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
SEED = 20261107


def make_reg(lam_P, lam_H):
    """构造带差量正则的目标函数与解析梯度。"""
    def obj(para, u, P, Q, dt):
        base = obj_mix_iden(para, u, P, Q, dt)
        return base + lam_H * (para[2] - para[6]) ** 2 + lam_P * (para[3] - para[7]) ** 2

    def jac(para, u, P, Q, dt):
        g = np.array(obj_mix_jac(para, u, P, Q, dt), dtype=float)
        g[2] += 2 * lam_H * (para[2] - para[6])
        g[6] -= 2 * lam_H * (para[2] - para[6])
        g[3] += 2 * lam_P * (para[3] - para[7])
        g[7] -= 2 * lam_P * (para[3] - para[7])
        return g
    return obj, jac


def decom(truths, xs):
    """总量/差量/比值误差 (P_im 与 H2 各 3 项)。"""
    SP, DP, RP, SH, DH, RH = [], [], [], [], [], []
    for t, x in zip(truths, xs):
        SP.append(abs((x[3] + x[7]) - (t[3] + t[7])) / abs(t[3] + t[7]))
        DP.append(abs((x[3] - x[7]) - (t[3] - t[7])) / abs(t[3] + t[7]))
        RP.append(abs(x[3] / x[7] - t[3] / t[7]) / abs(t[3] / t[7]))
        SH.append(abs((x[2] + x[6]) - (t[2] + t[6])) / abs(t[2] + t[6]))
        DH.append(abs((x[2] - x[6]) - (t[2] - t[6])) / abs(t[2] + t[6]))
        RH.append(abs(x[2] / x[6] - t[2] / t[6]) / abs(t[2] / t[6]))
    return (np.array(SP), np.array(DP), np.array(RP),
            np.array(SH), np.array(DH), np.array(RH))


def run(LAM_H, K=8, n_groups=30, lam_P=0.0):
    ori = pickle.load(open(ORI, 'rb'))
    truth_df = pickle.load(open(RRES, 'rb'))
    groups = sorted(set(int(c.split()[0]) for c in ori.columns if c.split()[0].isdigit()))[:n_groups]

    # 预载全部数据与真值
    data = []
    for g in groups:
        ud = ori['%d ud' % g].values
        uq = ori['%d uq' % g].values
        P = ori['%d P' % g].values
        Q = ori['%d Q' % g].values
        u = ud + 1j * uq
        truth = truth_df.iloc[g][[p + '_real' for p in PNAMES]].values.astype(float)
        data.append((g, u, P, Q, truth))

    lines = []
    lines.append('=' * 100)
    lines.append('H2 差量 Tikhonov 正则扫描 (PSSE 真实数据, %d 组, K=%d 起点)' % (len(data), K))
    lines.append('obj = obj_mix_iden + λ_H·(H2_a-H2_b)²   (λ_P=%.1g 固定)' % lam_P)
    lines.append('=' * 100)
    lines.append('%-10s %-14s %-14s %-14s | %-14s %-14s %-14s | %-12s %-12s' % (
        'λ_H', 'H2总量%', 'H2差量%', 'H2比值%', 'P_im总量%', 'P_im差量%', 'P_im比值%', 'dyn6中位%', 'P_im中位%'))
    lines.append('-' * 100)

    results = {}
    for lam in LAM_H:
        obj, jac = make_reg(lam_P, lam)
        xs, funs = [], []
        for (g, u, P, Q, truth) in data:
            best, _ = identify_multistart(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj,
                                          tol=1e-8, n_restarts=K, jac_func=jac, seed=SEED + g)
            xs.append(np.asarray(best.x))
            funs.append(best.fun)
        xs = np.array(xs)
        truths = [d[4] for d in data]
        SP, DP, RP, SH, DH, RH = decom(truths, xs)
        E = np.array([np.abs(xs[i] - truths[i]) / np.abs(truths[i]) for i in range(len(data))])
        line = '%-10.3g %-14.2f %-14.2f %-14.2f | %-14.2f %-14.2f %-14.2f | %-12.2f %-12.2f' % (
            lam, np.median(SH) * 100, np.median(DH) * 100, np.median(RH) * 100,
            np.median(SP) * 100, np.median(DP) * 100, np.median(RP) * 100,
            np.median(E[:, DYN]) * 100, np.median(E[:, PIM]) * 100)
        lines.append(line)
        results[lam] = (np.median(SH) * 100, np.median(DH) * 100, np.median(RH) * 100,
                        np.median(SP) * 100, np.median(DP) * 100, np.median(RP) * 100)
        print(line, flush=True)

    lines.append('-' * 100)
    lines.append('※ H2 差量下限 = 真值 |H2_a-H2_b|/总量 中位 ≈ 10%(纯偏置), 故最优 λ 应使差量≈10~12%, 而非压到 0。')
    lines.append('※ P_im 各项应随 λ_H 基本不变(λ_P=0), 验证正则不伤 P_im。')
    lines.append('=' * 100)
    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    # λ_H 网格(对数扫): 覆盖「无正则」到「过度收缩」; 冒烟显示最优在 0.03~0.1, 故收窄到有效区间
    LAM_H = [0.0, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(LAM_H, K, n_groups)

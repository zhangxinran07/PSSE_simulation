# -*- coding: utf-8 -*-
"""零结构失配对照实验：用辨识模型自身生成数据，再辨识，验证辨识器能否精确复现真值。

目的(2026-11-06 "零结构失配对照"):
  PSSE 实验中辨识误差的系统性偏差, 根源疑为「结构失配」——数据生成用 PSSE 完整
  CMLDBLU2 电机模型(含次暂态/堵转/Qfrq 频率依赖), 而辨识用简化 _core_con/_core_var。
  本脚本做决定性对照: 用 _core_con + _core_var **自身**生成 (u,P,Q), 再跑同一套
  identify_mix。若误差 → 0, 坐实结构失配; 若误差仍系统性存在, 则是辨识器内部 bug
  (如 0.75 缩放 / mva=P[0]*P_im 口径)。

设计与 mix_freq_sweep.py 严格对齐:
  - 真值采样复用 sample_mix(同一 seed 空间), 8 参数 = [a_a,b_a,H2_a,P_ima,a_b,b_b,H2_b,P_imb]
  - 生成口径与 obj_mix_iden 完全一致: a=para/0.75, H2=para/0.75, mva=P[0]*para
  - P0=6.0(对齐 PSSE 母线3 总负荷), P_static=P0*(1-P_ima-P_imb) 作为被回归吸收的静态负荷
  - 激励: probing 0.5~8Hz 8 分量(与 sweep 同频段), |u|≈0.99, du≈0.08
  - 辨识: trust-constr + obj_mix_jac, 边界/容差/初值复现与 mix_freq_sweep 一致

输出:
  Sample_Generation/results/control_zero_mismatch.txt  逐组相对误差汇总
"""
import numpy as np
import sys, os, time, pickle, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
from identification import _core_con, _core_var, obj_mix_iden, obj_mix_jac, identify as idf_id

DT = 0.01
W0 = 2 * math.pi * 60
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
SEED = 20261107
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'Sample_Generation', 'results')
os.makedirs(OUTDIR, exist_ok=True)
TXT = os.path.join(OUTDIR, 'control_zero_mismatch.txt')


# ==================== 真值采样(与 mix_freq_sweep.sample_mix 一致) ====================
def sample_mix(rng):
    while True:
        cc = {
            'P_ma': rng.uniform(0.2, 0.6),
            'P_mb': 0,
            'R_a': 0, 'X_a': rng.uniform(1.35, 2.05), 'X1_a': rng.uniform(0.12, 0.22),
            'T0_a': rng.uniform(0.08, 0.2), 'H2_a': rng.uniform(0.1, 0.2),
            'R_b': 0, 'X_b': rng.uniform(1.35, 2.05), 'X1_b': rng.uniform(0.12, 0.22),
            'T0_b': rng.uniform(0.08, 0.2), 'H2_b': rng.uniform(0.1, 0.2),
        }
        cc['P_mb'] = rng.uniform(0.2, 0.8 - cc['P_ma'])
        pima, xa, x1a, t0a, h2a = cc['P_ma'], cc['X_a'], cc['X1_a'], cc['T0_a'], cc['H2_a']
        pimb, xb, x1b, t0b, h2b = cc['P_mb'], cc['X_b'], cc['X1_b'], cc['T0_b'], cc['H2_b']
        para_real = [(xa / x1a - 1) / t0a / x1a, xa / x1a / t0a, h2a, pima,
                     (xb / x1b - 1) / t0b / x1b, xb / x1b / t0b, h2b, pimb]
        pr6 = para_real[0:3] + para_real[4:7]
        ok = (np.array(pr6) > np.array(lb[0:3] * 2)).all() and \
             (np.array(pr6) < np.array(ub[0:3] * 2)).all()
        if ok:
            return np.array(para_real), cc


# ==================== 激励生成(probing 0.5~8Hz, 8 分量) ====================
def gen_u(rng, n=3001):
    t = np.arange(n) * DT
    freqs = np.geomspace(0.5, 8.0, 8)
    mag = np.zeros(n)
    ang = np.zeros(n)
    for fi in freqs:
        ph_m = rng.uniform(0, 2 * np.pi)
        ph_a = rng.uniform(0, 2 * np.pi)
        mag += np.cos(2 * np.pi * fi * t + ph_m)
        ang += np.cos(2 * np.pi * fi * t + ph_a)
    mag = mag / np.abs(mag).max() * 0.04   # 幅值扰动峰值 0.04 → du≈0.08
    ang = ang / np.abs(ang).max() * 0.03   # 相位扰动峰值 0.03 rad
    u = (0.99 + mag) * np.exp(1j * ang)
    return u


# ==================== 生成 (P,Q): 口径与 obj_mix_iden 完全一致 ====================
def gen_PQ(u, para):
    """para 为辨识空间(与 obj_mix_iden 入参一致)。返回 P,Q(含静态负荷偏移)。"""
    a_a, b_a, H2_a = para[0] / 0.75, para[1], para[2] / 0.75
    a_b, b_b, H2_b = para[4] / 0.75, para[5], para[6] / 0.75
    P_ima, P_imb = para[3], para[7]
    P0 = 6.0
    MVA_a, MVA_b = P0 * P_ima, P0 * P_imb
    P_static = P0 * (1 - P_ima - P_imb)
    Q_static = 0.3 * P_static

    ok_a, Pe_a, Qe_a = _core_con(u, a_a, b_a, H2_a, 1.0, DT)   # 每单位 MVA 预测
    ok_b, Pe_b, Qe_b = _core_var(u, a_b, b_b, H2_b, 1.0, DT)
    assert ok_a and ok_b, 'truth 处生成失败'

    P = P_static + Pe_a * MVA_a + Pe_b * MVA_b
    Q = Q_static + Qe_a * MVA_a + Qe_b * MVA_b
    return P, Q


def relerr(x, xt):
    return np.abs(np.asarray(x) - np.asarray(xt)) / np.abs(np.asarray(xt))


def run(N):
    rng = np.random.default_rng(SEED)
    RESTARTS = 6
    errs_single = []      # 单次辨识(首个通过距离校验的结果)
    errs_best = []        # 多起点最优
    funs_best = []
    rows = []
    print('=' * 100, flush=True)
    print('零结构失配对照: _core 模型自身生成 -> identify_mix 复现真值', flush=True)
    print('N=%d 组 x %d 起点; P0=6.0; probing 0.5~8Hz; trust-constr + obj_mix_jac' % (N, RESTARTS), flush=True)
    print('=' * 100, flush=True)
    for g in range(N):
        truth, cc = sample_mix(rng)
        u = gen_u(np.random.default_rng(SEED + g))
        P, Q = gen_PQ(u, truth)
        obj_t = float(obj_mix_iden(truth, u, P, Q, DT))

        best_x, best_fun, best_e = None, np.inf, None
        first_e = None
        t_total = 0.0
        for r in range(RESTARTS):
            np.random.seed(SEED * 10 + g * RESTARTS + r)   # 初值可复现
            result, tiden = idf_id(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj_mix_iden,
                                   tol=1e-8, max_trials=5, jac_func=obj_mix_jac)
            t_total += tiden
            x = np.asarray(result.x)
            e = relerr(x, truth)
            if first_e is None:
                first_e = e
            if result.fun < best_fun:
                best_fun, best_x, best_e = result.fun, x, e

        errs_single.append(first_e)
        errs_best.append(best_e)
        funs_best.append(best_fun)
        rows.append((g, truth, best_x, obj_t, best_fun, t_total))
        dyn = [0, 1, 2, 4, 5, 6]
        print('  g=%02d obj_t=%.1e 单次dyn=%.2f%% 最优dyn=%.2f%% 最优fun=%.2e  (%.1fs)'
              % (g, obj_t, np.median(first_e[dyn]) * 100, np.median(best_e[dyn]) * 100,
                 best_fun, t_total), flush=True)

    def _summ(tag, E):
        med = np.median(E, axis=0) * 100
        p90 = np.percentile(E, axis=0, q=90) * 100
        dyn = [0, 1, 2, 4, 5, 6]
        return med, p90, np.median(E[:, dyn]) * 100

    med1, p901, dyn1 = _summ('单次', np.array(errs_single))
    medb, p90b, dynb = _summ('最优', np.array(errs_best))
    objts = np.array([r[3] for r in rows])
    funs = np.array(funs_best)
    n_global = int(np.sum(funs < 1e-6))

    lines = []
    lines.append('=' * 100)
    lines.append('零结构失配对照汇总: 辨识器对自身模型生成数据的复现误差')
    lines.append('N=%d; 每组 %d 次随机初值; P0=6.0; probing 0.5~8Hz; trust-constr + obj_mix_jac; tol=1e-8' % (N, RESTARTS))
    lines.append('=' * 100)
    lines.append('%-9s ' % '参数' + ' '.join('%-9s' % p for p in PNAMES))
    lines.append('-' * 100)
    lines.append('%-9s ' % '单次中位%' + ' '.join('%-9.3f' % m for m in med1))
    lines.append('%-9s ' % '最优中位%' + ' '.join('%-9.3f' % m for m in medb))
    lines.append('%-9s ' % '最优P90% ' + ' '.join('%-9.3f' % m for m in p90b))
    lines.append('-' * 100)
    lines.append('truth 处代价 obj_mix_iden(truth): 中位=%.2e  最大=%.2e' % (np.median(objts), objts.max()))
    lines.append('最优 fun: 中位=%.2e; 全局收敛(fun<1e-6)组数 = %d/%d' % (np.median(funs), n_global, N))
    lines.append('')
    lines.append('结论判据:')
    lines.append('  ① truth 处代价 ~1e-10 => 模型对真值零失配(生成与辨识同构, 无结构失配)')
    lines.append('  ② 若「最优中位误差 → 0」且全局收敛率高 => 辨识器公式/缩放/mva 口径无 bug,')
    lines.append('     单次辨识的偏差来自目标函数非凸(局部极小), 需多起点或更好初值')
    lines.append('  ③ 若「最优中位误差仍系统性 > 5%」=> 辨识器内部确有 bug(0.75 缩放 / mva 口径)')
    lines.append('=' * 100)
    txt = '\n'.join(lines)
    open(TXT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run(N)

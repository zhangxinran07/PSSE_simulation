# -*- coding: utf-8 -*-
"""闭环对照实验：ambient(低频白噪声) vs probing(6~13Hz multi-sine) 的 mix 8 参数辨识方差。

目的(2026-11-06 张老师"把闭环测试做了吧"):
  在真实 PSSE 闭环下, 对同一组 mix 真值(A 组恒转矩 con + B 组变转矩 var, 共 8 参数)
  分别施加两种激励, 各自辨识 8 参数, 对比辨识误差(相对误差 relerr), 验证可辨识性诊断
  得到的理论结论 —— 6~13Hz 探测激励比真实低频 ambient 激励能把 a/b/H2 辨识方差改善
  1~2 个数量级 —— 能否在闭环复现。

对照口径(公平性设计):
  - 同一组真值(同 seed 采样): ambient 与 probing 各自一条仿真, "苹果对苹果"。
  - 后处理 filter_cutoff_frequency 两者都 = 30Hz(ambient 信号本身 <2Hz, 10/30Hz 对低频
    无差别; probing 必须 >=2*f_hi=26Hz 才能保留 6~13Hz 成分, 故统一 30Hz 隔离出纯激励效果)。
  - 幅值: ambient=500MW(原口径, 闭环 |u|max-min≈0.059); probing=800MW(闭环实测 |u|max-min
    ≈0.076, 0.08 约束内最大)。两者都满足小干扰约束 <=0.08。
  - 辨识: 同一 trust-constr 口径(idf_id + obj_mix_iden + obj_mix_jac, lb*2/ub*2, tol=1e-8)。

用法:
  python mix_ambient_vs_probing.py [N] [mode]
    N     : 组数(默认 30)
    mode  : run(默认, 边跑边辨识) / report(只汇总已跑完的 pkl)
断点续跑: 每组 2 条仿真+辨识后立即 pickle 落盘, 已完成的 (g, excitation) 自动跳过。

输出:
  Sample_Generation/results/mix_ambient_vs_probing.pkl  逐组原始记录
  Sample_Generation/results/mix_ambient_vs_probing.txt  汇总对比表
"""
import os, sys, time, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.append(os.path.join(ROOT, 'package'))

# 必须最先 import psse_dynamic(PSSE 与 MKL DLL 冲突), 再 import numpy/matplotlib 相关
import psse_dynamic as ps_d
from identification import obj_mix_iden, obj_mix_jac, identify as idf_id

os.environ['PSSE_KEEP_OUTFILE'] = '1'   # 跳过 .out 删除, 避免批量 safe-delete 确认挂起后台

OUTDIR = os.path.join(ROOT, 'Sample_Generation', 'results')
os.makedirs(OUTDIR, exist_ok=True)
PKL = os.path.join(OUTDIR, 'mix_ambient_vs_probing.pkl')
TXT = os.path.join(OUTDIR, 'mix_ambient_vs_probing.txt')

BUSID = 3
BUSID_SET = [3, 4, 15, 20, 21, 24, 25, 27, 28]
SEED = 20261106
DT = 0.01
lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']

# 两种激励的配置
EXC = {
    'ambient': dict(disturbance_type='ambient_a', disturbance_range=500.0,
                    ambient_cutoff_frequency=2),
    'probing': dict(disturbance_type='probing', disturbance_range=800.0,
                    probing_f_lo=6.0, probing_f_hi=13.0, probing_n_comp=8),
}


# ==================== mix 真值采样(独立实现, 与 main_simu.sample_params 的 mix 分支一致) ====================
def sample_mix(rng):
    while True:
        cc = {
            'P_ma': rng.uniform(0.2, 0.6),
            'P_mb': 0,                       # 下面按约束采样
            'R_a': 0,
            'X_a': rng.uniform(1.35, 2.05),
            'X1_a': rng.uniform(0.12, 0.22),
            'T0_a': rng.uniform(0.08, 0.2),
            'H2_a': rng.uniform(0.1, 0.2),
            'R_b': 0,
            'X_b': rng.uniform(1.35, 2.05),
            'X1_b': rng.uniform(0.12, 0.22),
            'T0_b': rng.uniform(0.08, 0.2),
            'H2_b': rng.uniform(0.1, 0.2),
        }
        cc['P_mb'] = rng.uniform(0.2, 0.8 - cc['P_ma'])   # >=0.2 且 P_ma+P_mb <=0.8
        pima, xa, x1a, t0a, h2a = cc['P_ma'], cc['X_a'], cc['X1_a'], cc['T0_a'], cc['H2_a']
        pimb, xb, x1b, t0b, h2b = cc['P_mb'], cc['X_b'], cc['X1_b'], cc['T0_b'], cc['H2_b']
        para_real = [(xa / x1a - 1) / t0a / x1a, xa / x1a / t0a, h2a, pima,
                     (xb / x1b - 1) / t0b / x1b, xb / x1b / t0b, h2b, pimb]
        pr6 = para_real[0:3] + para_real[4:7]
        ok = (np.array(pr6) > np.array(lb[0:3] * 2)).all() and (np.array(pr6) < np.array(ub[0:3] * 2)).all()
        if ok:
            return np.array(para_real), cc


# ==================== 仿真 + 提取 ====================
def extract(df, busid):
    ud, uq, P, Q = [df.values[:, [i for i in range(len(df.columns)) if (key % busid) in df.columns[i]]].reshape(len(df))
                    for key in ["BUS%d UD", "BUS%d UQ", "PLOD %d[", "QLOD %d["]]
    return ud + 1j * uq, P, Q


def run_sim(cc, excitation, seed):
    kwargs = dict(
        study='IEEE39_RE', total_time=30,
        load_bus_monitor=[BUSID],
        ld_con_chng_buses=list(BUSID_SET),
        ld_con_chng_values=[ps_d.read_bus_params(b) for b in BUSID_SET],
        ambient_seed=seed,
        filter_cutoff_frequency=30,
    )
    kwargs.update(EXC[excitation])
    index = BUSID_SET.index(BUSID)
    for name, val in cc.items():
        kwargs['ld_con_chng_values'][index][ps_d.para_names.index(name)] = val
    df, d_para, t = ps_d.one_time_simulation(kwargs, max_retries=8)
    if df is None:
        return None, None, None, t
    u, P, Q = extract(df, BUSID)
    return u, P, Q, t


# ==================== 辨识 ====================
def identify_mix(u, P, Q):
    result, t = idf_id(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj_mix_iden,
                       tol=1e-8, max_trials=5, jac_func=obj_mix_jac)
    return np.asarray(result.x), float(result.fun), t


def relerr(x, xt):
    return np.abs(np.asarray(x) - np.asarray(xt)) / np.abs(np.asarray(xt))


# ==================== 主流程 ====================
def run(N):
    res = pickle.load(open(PKL, 'rb')) if os.path.exists(PKL) else {}
    total = N * 2
    done = 0
    print('=' * 92, flush=True)
    print('闭环对照: ambient(500MW 低通白噪声) vs probing(800MW 6~13Hz) 的 mix 8 参数辨识方差', flush=True)
    print('N=%d 组 x 2 激励 = %d 条 (30s/条仿真); seed=%d+g; filter_cutoff 统一 30Hz' % (N, total, SEED), flush=True)
    print('=' * 92, flush=True)

    for g in range(N):
        rec = res.setdefault(g, {})
        # 采样真值(同一组真值跑两种激励)
        if 'truth' not in rec:
            rng = np.random.default_rng(SEED + g)
            truth, cc = sample_mix(rng)
            rec['truth'] = truth.tolist()
            rec['cc'] = cc
        truth = np.array(rec['truth'])
        cc = rec['cc']

        for exc in ('ambient', 'probing'):
            if exc in rec and rec[exc].get('x') is not None:
                done += 1
                continue
            np.random.seed(SEED * 10 + g * 2 + (0 if exc == 'ambient' else 1))   # 辨识初值可复现
            print('[%3d/%d] g=%d %s 仿真中... %s' % (done + 1, total, g, exc, time.strftime('%H:%M:%S')), flush=True)
            u, P, Q, tsim = run_sim(cc, exc, SEED + g)
            if u is None:
                rec[exc] = {'error': 'sim_fail'}
                pickle.dump(res, open(PKL, 'wb'))
                print('    g=%d %s 仿真失败' % (g, exc), flush=True)
                continue
            x, fun, tiden = identify_mix(u, P, Q)
            v = float(abs(u).mean())
            rec[exc] = {'v': v, 'x': x.tolist(), 'fun': fun,
                        'tsim': tsim, 'tiden': tiden,
                        'err': relerr(x, truth).tolist()}
            pickle.dump(res, open(PKL, 'wb'))
            done += 1
            e = rec[exc]['err']
            print('    g=%d %s: |u|=%.4f  err[a_a,b_a,H2_a,a_b,b_b,H2_b]=[%s]  (sim %.0fs iden %.0fs)'
                  % (g, exc, v, ', '.join('%5.1f%%' % (e[i] * 100) for i in [0, 1, 2, 4, 5, 6]),
                     tsim, tiden), flush=True)

    print('全部完成 -> %s' % PKL, flush=True)
    report(res)


def report(res=None):
    if res is None:
        res = pickle.load(open(PKL, 'rb'))
    gs = sorted(k for k in res if isinstance(k, int))
    lines = []
    lines.append('=' * 96)
    lines.append('闭环对照汇总: ambient vs probing 的 mix 8 参数辨识误差(相对误差 relerr, %)')
    lines.append('口径: %d 组真值 x 2 激励; trust-constr + obj_mix_jac; filter_cutoff 统一 30Hz' % len(gs))
    lines.append('ambient=500MW 低通白噪声(cutoff=2)   probing=800MW 6~13Hz multi-sine')
    lines.append('=' * 96)

    # 逐参数统计(ambient 与 probing 各列的中位/均值), 只统计 8 参数各自的 relerr
    rows = []
    for exc in ('ambient', 'probing'):
        E = []
        for g in gs:
            rec = res[g]
            if exc in rec and 'err' in rec[exc]:
                E.append(rec[exc]['err'])
        E = np.array(E) if E else np.zeros((0, 8))
        rows.append((exc, E))

    lines.append('')
    lines.append('%-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s' %
                 ('激励', 'a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb'))
    lines.append('-' * 96)
    for exc, E in rows:
        if E.size:
            med = np.median(E, axis=0) * 100
            lines.append('%-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s' %
                         (exc + '(中位)', *['%.2f' % m for m in med]))
            mean = E.mean(axis=0) * 100
            lines.append('%-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s' %
                         (exc + '(均值)', *['%.2f' % m for m in mean]))
        lines.append('')

    # 改善倍数 = ambient_err / probing_err(>1 表示 probing 更准)
    Ea = rows[0][1]
    Ep = rows[1][1]
    if Ea.size and Ep.size:
        lines.append('-' * 96)
        lines.append('改善倍数(ambient误差 / probing误差, >1 表示 probing 更准):')
        lines.append('%-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s' %
                     ('', 'a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb'))
        ratio_med = np.median(Ea, axis=0) / np.maximum(np.median(Ep, axis=0), 1e-12)
        lines.append('%-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s' %
                     ('中位比', *['%.1fx' % r for r in ratio_med]))
        ratio_mean = Ea.mean(axis=0) / np.maximum(Ep.mean(axis=0), 1e-12)
        lines.append('%-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s' %
                     ('均值比', *['%.1fx' % r for r in ratio_mean]))
        # 核心判据: 动态参数(去 P_im)的整体误差
        dyn = [0, 1, 2, 4, 5, 6]
        lines.append('')
        lines.append('核心判据(动态参数 a/b/H2 六个参数误差均值):')
        lines.append('  ambient: 中位=%.2f%%  均值=%.2f%%' %
                     (np.median(Ea[:, dyn]) * 100, Ea[:, dyn].mean() * 100))
        lines.append('  probing: 中位=%.2f%%  均值=%.2f%%' %
                     (np.median(Ep[:, dyn]) * 100, Ep[:, dyn].mean() * 100))
        lines.append('  => 整体改善 %.1fx(均值) / %.1fx(中位)' %
                     (Ea[:, dyn].mean() / max(Ep[:, dyn].mean(), 1e-12),
                      np.median(Ea[:, dyn]) / max(np.median(Ep[:, dyn]), 1e-12)))

    lines.append('')
    lines.append('=' * 96)
    txt = '\n'.join(lines)
    open(TXT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    mode = sys.argv[2] if len(sys.argv) > 2 else 'run'
    if mode == 'report':
        report()
    else:
        run(N)

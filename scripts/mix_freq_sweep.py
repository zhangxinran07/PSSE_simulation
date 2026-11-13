# -*- coding: utf-8 -*-
"""闭环频段扫描：probing 频段对 mix 8 参数辨识误差的影响曲线。

目的(2026-11-06 张老师"闭环频段扫描"):
  probing 6~13Hz 闭环失效根因 = 结构失配(电机C次暂态 / 堵转D / Qfrq=-1 频率依赖),
  且随频率升高而放大。本脚本系统扫描 probing 频段(0.5~8Hz 分 4 档) + ambient 基准,
  画「辨识误差 vs 频段」曲线, 验证「2 电机聚合激励设计适用频段上限约 5~6Hz」,
  并找真实 CMLDBLU2 模型下的最优激励频段。

设计:
  - 5 个激励: ambient(0-2Hz 低通白噪声, 基准) + probing 4 档 [0.5-2, 2-4, 4-6, 6-8]Hz。
    (8~13Hz 档已移除: PSSE 在高频下反复 save 失败 ierr=3 卡死, 且 6~8Hz 已证明高频失效、无增量)
  - 同一组真值(同 seed) 跑全部 5 个激励, "苹果对苹果"。
  - 幅值按 du=|u|max-min 标定, 使各 probing 档 du 落在 ~0.08 以内(小干扰约束):
    ambient=500MW, prb_0.5-2=560MW, prb_2-4/prb_4-6=720MW, prb_6-8=800MW(冒烟实测反推);
    脚本额外记录每档实际 du, 供幅值约束核查。
  - 后处理 filter_cutoff 统一 30Hz(>=2*8Hz 保留最高频成分)。
  - 每条额外记录 truth 处代价 obj_mix_iden(truth) —— 直接量化「结构失配 vs 频段」。

用法:
  python mix_freq_sweep.py [N] [mode]
    N     : 组数(默认 30)
    mode  : run(默认, 边跑边辨识) / report(只汇总已跑完的 pkl)
断点续跑: 每组 5 条仿真+辨识后立即 pickle 落盘, 已完成的 (g, excitation) 自动跳过。

输出:
  Sample_Generation/results/mix_freq_sweep.pkl  逐组原始记录
  Sample_Generation/results/mix_freq_sweep.txt  汇总对比表(误差/代价 vs 频段)
  Sample_Generation/results/mix_freq_sweep.png  误差 vs 频段曲线
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
PKL = os.path.join(OUTDIR, 'mix_freq_sweep.pkl')
TXT = os.path.join(OUTDIR, 'mix_freq_sweep.txt')
PNG = os.path.join(OUTDIR, 'mix_freq_sweep.png')

BUSID = 3
BUSID_SET = [3, 4, 15, 20, 21, 24, 25, 27, 28]
SEED = 20261107
DT = 0.01
lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']

# 5 个激励: ambient 基准 + probing 4 档频段 (8~13Hz 档因 PSSE 高频 save 失败反复卡死, 已移除;
#  6~8Hz 已充分证明高频失效, 8~13Hz 只会更差, 无增量)
# 各 probing 档 disturbance_range 按冒烟实测 du 反推标定, 使 du 落在 0.08 附近:
#   prb_0.5-2: 800MW du=0.112 -> 560MW (~0.078)
#   prb_2-4 / prb_4-6: 800MW du=0.089 -> 720MW (~0.080)
#   prb_6-8: 800MW du=0.077 已达标, 保持
EXC = {
    'amb_0-2':   dict(disturbance_type='ambient_a', disturbance_range=500.0,
                      ambient_cutoff_frequency=2),
    'prb_0.5-2': dict(disturbance_type='probing', disturbance_range=560.0,
                      probing_f_lo=0.5, probing_f_hi=2.0, probing_n_comp=8),
    'prb_2-4':   dict(disturbance_type='probing', disturbance_range=720.0,
                      probing_f_lo=2.0, probing_f_hi=4.0, probing_n_comp=8),
    'prb_4-6':   dict(disturbance_type='probing', disturbance_range=720.0,
                      probing_f_lo=4.0, probing_f_hi=6.0, probing_n_comp=8),
    'prb_6-8':   dict(disturbance_type='probing', disturbance_range=800.0,
                      probing_f_lo=6.0, probing_f_hi=8.0, probing_n_comp=8),
}

# 各激励的中心频率(对数中心, 绘图 x 轴)
CENTER = {
    'amb_0-2': 1.00,
    'prb_0.5-2': 1.00,
    'prb_2-4': 2.83,
    'prb_4-6': 4.90,
    'prb_6-8': 6.93,
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
    exc_names = list(EXC.keys())
    total = N * len(exc_names)
    done = 0
    print('=' * 92, flush=True)
    print('闭环频段扫描: probing 频段对 mix 8 参数辨识误差的影响曲线', flush=True)
    print('N=%d 组 x %d 激励 = %d 条 (30s/条仿真); filter_cutoff 统一 30Hz' % (N, len(exc_names), total), flush=True)
    print('激励: %s' % ' | '.join('%s(%gHz)' % (e, CENTER[e]) for e in exc_names), flush=True)
    print('=' * 92, flush=True)

    for g in range(N):
        rec = res.setdefault(g, {})
        # 采样真值(同一组真值跑全部激励)
        if 'truth' not in rec:
            rng = np.random.default_rng(SEED + g)
            truth, cc = sample_mix(rng)
            rec['truth'] = truth.tolist()
            rec['cc'] = cc
        truth = np.array(rec['truth'])
        cc = rec['cc']

        for j, exc in enumerate(exc_names):
            if exc in rec and rec[exc].get('x') is not None:
                done += 1
                continue
            np.random.seed(SEED * 10 + g * len(exc_names) + j)   # 辨识初值可复现
            print('[%3d/%d] g=%d %s 仿真中... %s' % (done + 1, total, g, exc, time.strftime('%H:%M:%S')), flush=True)
            u, P, Q, tsim = run_sim(cc, exc, SEED + g)
            if u is None:
                rec[exc] = {'error': 'sim_fail'}
                pickle.dump(res, open(PKL, 'wb'))
                print('    g=%d %s 仿真失败' % (g, exc), flush=True)
                continue
            # truth 处代价(结构失配量化) + 辨识
            try:
                obj_truth = float(obj_mix_iden(truth, u, P, Q, DT))
            except Exception:
                obj_truth = float('nan')
            x, fun, tiden = identify_mix(u, P, Q)
            v = float(abs(u).mean())
            du = float(abs(u).max() - abs(u).min())
            rec[exc] = {'v': v, 'du': du, 'x': x.tolist(), 'fun': fun,
                        'obj_truth': obj_truth,
                        'tsim': tsim, 'tiden': tiden,
                        'err': relerr(x, truth).tolist()}
            pickle.dump(res, open(PKL, 'wb'))
            done += 1
            e = rec[exc]['err']
            print('    g=%d %s: |u|=%.4f du=%.4f obj_t=%.3f err[a_a,b_a,H2_a,a_b,b_b,H2_b]=[%s] (sim %.0fs iden %.0fs)'
                  % (g, exc, v, du, obj_truth, ', '.join('%5.1f%%' % (e[i] * 100) for i in [0, 1, 2, 4, 5, 6]),
                     tsim, tiden), flush=True)

    print('全部完成 -> %s' % PKL, flush=True)
    report(res)


def report(res=None):
    if res is None:
        res = pickle.load(open(PKL, 'rb'))
    gs = sorted(k for k in res if isinstance(k, int))
    exc_names = list(EXC.keys())
    dyn = [0, 1, 2, 4, 5, 6]   # 动态参数(去 P_im)
    lines = []
    lines.append('=' * 112)
    lines.append('闭环频段扫描汇总: probing 频段对 mix 8 参数辨识误差的影响')
    lines.append('口径: %d 组真值 x %d 激励; trust-constr + obj_mix_jac; filter_cutoff 统一 30Hz' % (len(gs), len(exc_names)))
    lines.append('=' * 112)

    # 收集每激励的误差矩阵 / obj_truth / du / fun
    rows = []
    for exc in exc_names:
        E, OT, DU, FN = [], [], [], []
        for g in gs:
            rec = res[g]
            if exc in rec and 'err' in rec[exc]:
                E.append(rec[exc]['err'])
                OT.append(rec[exc].get('obj_truth', float('nan')))
                DU.append(rec[exc].get('du', float('nan')))
                FN.append(rec[exc].get('fun', float('nan')))
        E = np.array(E) if E else np.zeros((0, 8))
        rows.append((exc, E, np.array(OT), np.array(DU), np.array(FN)))

    lines.append('')
    lines.append('【1】逐参数相对误差中位 (relerr, %)')
    lines.append('%-10s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s' % ('激励', *PNAMES))
    lines.append('-' * 112)
    for exc, E, OT, DU, FN in rows:
        if E.size:
            med = np.median(E, axis=0) * 100
            lines.append('%-10s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s' % (exc, *['%.2f' % m for m in med]))
        else:
            lines.append('%-10s (无数据)' % exc)

    lines.append('')
    lines.append('【2】动态参数误差 / 结构失配 / 幅值 vs 频段 (中位)')
    lines.append('%-10s %-9s %-12s %-12s %-10s %-10s' % ('激励', 'freq(Hz)', 'dyn_err%', 'obj_truth', 'du|u|', 'fun'))
    lines.append('-' * 112)
    for exc, E, OT, DU, FN in rows:
        if E.size:
            dyn_med = np.median(E[:, dyn]) * 100
            ot_med = np.median(OT[np.isfinite(OT)]) if np.isfinite(OT).any() else float('nan')
            du_med = np.median(DU[np.isfinite(DU)]) if np.isfinite(DU).any() else float('nan')
            fn_med = np.median(FN[np.isfinite(FN)]) if np.isfinite(FN).any() else float('nan')
            lines.append('%-10s %-9.2f %-12.2f %-12.4f %-10.4f %-10.4f' %
                         (exc, CENTER[exc], dyn_med, ot_med, du_med, fn_med))

    lines.append('')
    lines.append('=' * 112)
    txt = '\n'.join(lines)
    open(TXT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)

    # 画图: 动态参数误差 / 结构失配 vs 频段(对数 x 轴)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        freq = [CENTER[e] for e in exc_names]
        dyn_meds, ot_meds, du_meds = [], [], []
        for exc, E, OT, DU, FN in rows:
            dyn_meds.append(np.median(E[:, dyn]) * 100 if E.size else np.nan)
            ot_meds.append(np.median(OT[np.isfinite(OT)]) if np.isfinite(OT).any() else np.nan)
            du_meds.append(np.median(DU[np.isfinite(DU)]) if np.isfinite(DU).any() else np.nan)
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.plot(freq, dyn_meds, 'o-', color='#c0392b', label='动态参数误差中位(%)')
        ax1.set_xscale('log')
        ax1.set_xlabel('激励频段 (Hz, 对数)')
        ax1.set_ylabel('动态参数 a/b/H2 误差中位 (%)', color='#c0392b')
        ax1.tick_params(axis='y', labelcolor='#c0392b')
        ax1.grid(True, which='both', alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(freq, ot_meds, 's--', color='#2471a3', label='truth处代价 obj_truth')
        ax2.set_ylabel('truth 处代价(结构失配)', color='#2471a3')
        ax2.tick_params(axis='y', labelcolor='#2471a3')
        for i, e in enumerate(exc_names):
            ax1.annotate(e, (freq[i], dyn_meds[i]), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, loc='upper left', fontsize=9)
        plt.title('闭环频段扫描: 辨识误差与结构失配 vs 激励频段 (mix 8 参数, %d 组)' % len(gs))
        plt.tight_layout()
        plt.savefig(PNG, dpi=130)
        plt.close()
        print('图已保存 -> %s' % PNG, flush=True)
    except Exception as e:
        print('画图失败(不影响文字报告): %r' % e, flush=True)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    mode = sys.argv[2] if len(sys.argv) > 2 else 'run'
    if mode == 'report':
        report()
    else:
        run(N)

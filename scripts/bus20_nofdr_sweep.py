# -*- coding: utf-8 -*-
"""Bus20 馈线清零(noFdr, Rfdr=Xfdr=Xxf=0 对齐 Bus3/4) 的 ambient 幅值全扫描。

张老师 2026-10-21 需求:
  - 把 Bus20 的 10 档幅值(100~1000 MW)在 noFdr 下重跑, con(恒转矩 MA) 与 var(变转矩 MB) 两种都跑;
  - 与 Bus3/Bus4 的既有 mag_sweep 数据对比(同一真值 seed、同一辨识口径)。

口径(与 mag_sweep_bus20 / bus20_var_nofdr 完全对齐):
  - 真值采样: con→L.sample_con, var→V.sample_var, seed=20260927+g
  - 仿真: total_time=30, disturbance_range=mag, ambient_cutoff_frequency=2, ambient_seed=20260927+g
  - 辨识: con→L.identify_best_lsq, var→V.identify_best_var_lsq
  - 偏差: sgn=L.relerr_signed(x, truth)*100 (x[0] 即物理 a, 直接比 truth)
  - core = (H2_err + P_im_err)/2, err=L.relerr(x, truth)*100 (绝对值)

用法:
  python bus20_nofdr_sweep.py con [g0]    # 只跑 con
  python bus20_nofdr_sweep.py var [g0]    # 只跑 var
  python bus20_nofdr_sweep.py both [g0]   # con + var 都跑(默认)
  python bus20_nofdr_sweep.py report      # 只汇总
断点续跑: 每条仿真+辨识后立即 pickle 落盘, 已完成的 (mag,g) 自动跳过。
"""
import os, sys, time, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../PSSE_Simulation/scripts
ROOT = os.path.dirname(_HERE)                          # .../PSSE_Simulation (跨机器自动适配, 无关盘符)
OUTDIR = os.path.join(ROOT, 'sweep_output')
os.makedirs(OUTDIR, exist_ok=True)
os.chdir(ROOT)
sys.path.append(os.path.join(ROOT, 'package'))
sys.path.insert(0, _HERE)                              # 让 import psse_datalen_lsq/var 命中本目录

import psse_dynamic as ps_d
import psse_datalen_var as V
import psse_datalen_lsq as L

BUS = 20
SEED = 20260927
MAGS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
N_PER = 30
DYR = './Cases/IEEE39/ieee39_RE_ieee39_RE_motorD_Stall_enabled_noFdr20.dyr'

PKL = {m: os.path.join(OUTDIR, 'bus20_nofdr_sweep_%s.pkl' % m) for m in ('con', 'var')}


def extract(df, busid):
    ud, uq, P, Q = [df.values[:, [i for i in range(len(df.columns)) if (key % busid) in df.columns[i]]].reshape(len(df))
                    for key in ["BUS%d UD", "BUS%d UQ", "PLOD %d[", "QLOD %d["]]
    return ud + 1j * uq, P, Q


def build_params(cc):
    base = ps_d.read_bus_params(busid=BUS)
    for name, val in cc.items():
        if name in ps_d.para_names:
            base[ps_d.para_names.index(name)] = val
    return base


def run_sim(cc, mag, seed):
    kwargs = dict(
        disturbance_type='ambient_a', disturbance_range=mag, study='IEEE39_RE', type_option=500,
        total_time=30, ambient_cutoff_frequency=2,
        load_bus_monitor=[BUS],
        ld_con_chng_buses=[BUS], ld_con_chng_values=[build_params(cc)],
        ambient_seed=seed,
        dyradd=DYR,
    )
    df, d_para, t = ps_d.one_time_simulation(kwargs, max_retries=8)
    if df is None:
        return None, None
    u, P, Q = extract(df, BUS)
    return (u, P, Q), t


def sample(model, g):
    rng = np.random.default_rng(SEED + g)
    if model == 'con':
        return L.sample_con(rng)
    return V.sample_var(rng)


def identify(model, u, P, Q):
    if model == 'con':
        return L.identify_best_lsq(u, P, Q)
    return V.identify_best_var_lsq(u, P, Q)


def sweep(model):
    pkl = PKL[model]
    res = pickle.load(open(pkl, 'rb')) if os.path.exists(pkl) else {}
    total = len(MAGS) * N_PER
    done = 0
    print('=' * 90, flush=True)
    print('Bus20 noFdr %s 幅值全扫描: %d 档 x %d 组 = %d 条 (30s/条)' % (model, len(MAGS), N_PER, total), flush=True)
    print('dyr: %s   seed=20260927+g' % DYR, flush=True)
    print('=' * 90, flush=True)

    for mag in MAGS:
        for g in range(N_PER):
            key = (mag, g)
            if key in res and res[key].get('sgn') is not None:
                done += 1
                continue
            truth, cc = sample(model, g)
            (u, P, Q), t = run_sim(cc, mag, SEED + g)
            rec = res.setdefault(key, {'truth': truth, 'cc': cc})
            rec['truth'] = truth
            rec['cc'] = cc
            if u is None:
                rec['error'] = 'sim_fail'
                pickle.dump(res, open(pkl, 'wb'))
                print('  mag=%d g=%d 仿真失败' % (mag, g), flush=True)
                continue
            v = float(abs(u).mean())
            x, cost = identify(model, u, P, Q)
            if x is None:
                rec.update({'v': v, 'u': u, 'P': P, 'Q': Q, 'sim_time': t, 'error': 'iden_fail'})
                pickle.dump(res, open(pkl, 'wb'))
                print('  mag=%d g=%d 辨识失败' % (mag, g), flush=True)
                continue
            sgn = L.relerr_signed(x, truth) * 100
            err = L.relerr(x, truth) * 100
            rec.update({'u': u, 'P': P, 'Q': Q, 'sim_time': t,
                        'x': x.tolist(), 'sgn': sgn.tolist(), 'err': err.tolist(),
                        'cost': float(cost), 'v': v})
            pickle.dump(res, open(pkl, 'wb'))
            done += 1
            print('  [%3d/%d] mag=%d g=%d  |u|=%.4f  a_sgn=%+6.2f%%  b_sgn=%+6.2f%%  H2=%5.2f%%  Pim=%5.2f%%  core=%5.2f%%  (%.1fs)' %
                  (done, total, mag, g, v, sgn[0], sgn[1], err[2], err[3], 0.5 * (err[2] + err[3]), t), flush=True)
    print('%s 扫描完成 -> %s' % (model, pkl), flush=True)
    report(model, res)


def report(model, res=None):
    if res is None:
        res = pickle.load(open(PKL[model], 'rb'))
    lines = []
    lines.append('=' * 100)
    lines.append('Bus20 noFdr(馈线清零 Rfdr=Xfdr=Xxf=0) %s ambient 幅值扫描: 辨识偏差 vs 幅值' % model)
    lines.append('口径: %d 档 x %d 组, least_squares(trf)+头尾裁5%%+5初值取cost最小; 30s/条' % (len(MAGS), N_PER))
    lines.append('唯一变量=disturbance_range; seed=20260927+g; a_sgn=(x0-truth_a)/truth_a*100 (x0 即物理 a)')
    lines.append('=' * 100)
    lines.append('')
    lines.append('%-6s %-9s %-9s %-9s %-9s %-9s %-9s %-6s' %
                 ('幅值', 'a_sgn中位', 'a_sgn均值', 'b_sgn中位', 'H2中位', 'Pim中位', 'core中位', 'n'))
    lines.append('-' * 100)
    agg = {}
    for mag in MAGS:
        gs = [g for g in range(N_PER) if (mag, g) in res and 'sgn' in res[(mag, g)]]
        if not gs:
            continue
        asgn = np.array([res[(mag, g)]['sgn'][0] for g in gs])
        bsgn = np.array([res[(mag, g)]['sgn'][1] for g in gs])
        h2 = np.array([res[(mag, g)]['err'][2] for g in gs])
        pim = np.array([res[(mag, g)]['err'][3] for g in gs])
        core = 0.5 * (h2 + pim)
        agg[mag] = dict(a_med=float(np.median(asgn)), a_mean=float(asgn.mean()),
                        b_med=float(np.median(bsgn)), h2=float(np.median(h2)),
                        pim=float(np.median(pim)), core=float(np.median(core)), n=len(gs))
        lines.append('%-6d %+9.2f %+9.2f %+9.2f %9.2f %9.2f %9.2f %-6d' %
                     (mag, agg[mag]['a_med'], agg[mag]['a_mean'], agg[mag]['b_med'],
                      agg[mag]['h2'], agg[mag]['pim'], agg[mag]['core'], len(gs)))
    txt = '\n'.join(lines)
    open(os.path.join(OUTDIR, 'bus20_nofdr_sweep_%s_result.txt' % model), 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if model == 'report':
        report('con'); report('var')
    elif model == 'both':
        sweep('con'); sweep('var')
    else:
        sweep(model)

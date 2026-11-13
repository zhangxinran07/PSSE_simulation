# -*- coding: utf-8 -*-
"""probing 高频失配的最终根因定位(决定性实验)。

上一轮 diagnose_hf_misfit.py 已确认: probing 真值处代价 9.71e-1 >> ambient 2.44e-3,
是"模型/数值失配"(根因A), 不是辨识器收敛问题。

本脚本进一步区分两种子根因:
  S1. 数值步长: dt=0.01 的 RK4 积分在 6~13Hz 精度不足 -> 改 dt=0.002 应显著改善。
  S2. 模型结构: PSSE CMLDBLU2 负荷有 3 台电机(A=恒转矩, B/C=变转矩, 其中 C 带
      次暂态 Lpp/Tppo), 而辨识模型只聚合为 2 台(con+var), 高频下 3->2 聚合失效;
      或静态 ZIP 负荷的频率依赖(Pfrq/Qfrq)在 6~13Hz 无法被 [1,V,V^2] 回归吸收。

实验口径(一次性跑 probing 仿真, 采样到原生 0.002s, 滤波放宽到 100Hz 以保留 6~13Hz):
  (a) dt 扫描: 真值处 obj_mix_iden(truth) @ dt=0.002 / 0.005 / 0.01
      -> 若 dt 越小代价越低 => S1(数值); 若都高 => S2(结构)。
  (b) 电机功率对照: 2 电机模型预测(P_pa+P_pb) vs PSSE 各电机功率(MA+MB+MC)
      -> 若 2 电机模型能复现 MA+MB+MC 而总负荷仍失配 => 失配在静态/频率依赖部分;
         若 MA+MB+MC 本身复现不了 => 电机 3->2 聚合失效。

用法: python diagnose_hf_root.py [g]
"""
import os, sys, pickle
import numpy as np
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.append(os.path.join(ROOT, 'package'))
sys.path.insert(0, _HERE)

import psse_dynamic as ps_d
import mix_ambient_vs_probing as M
from identification import obj_mix_iden, _core_con, _core_var, _trim_k

os.environ['PSSE_KEEP_OUTFILE'] = '1'


def extract_full(df, busid):
    """提取 总电压/总负荷功率 + 各电机(MA/MB/MC)功率。"""
    cols = list(df.columns)
    def grab(*keys):
        idx = [i for i in range(len(cols)) if any(cols[i] == k for k in keys)]
        if not idx:
            # 退化为包含匹配
            idx = [i for i in range(len(cols)) if any(k in cols[i] for k in keys)]
        if not idx:
            return None
        return df.values[:, idx].reshape(len(df))
    ud = grab('BUS%d UD' % busid)
    uq = grab('BUS%d UQ' % busid)
    P = grab('PLOD %d[' % busid)
    Q = grab('QLOD %d[' % busid)
    maP, maQ = grab('Bus%d MA P' % busid), grab('Bus%d MA Q' % busid)
    mbP, mbQ = grab('Bus%d MB P' % busid), grab('Bus%d MB Q' % busid)
    mcP, mcQ = grab('Bus%d MC P' % busid), grab('Bus%d MC Q' % busid)
    print('  [列名抽样] UD列=%r  P总=%r  MA P=%r  MB P=%r  MC P=%r' % (
        ('BUS%d UD' % busid) in cols, ('PLOD %d[' % busid) in cols,
        ('Bus%d MA P' % busid) in cols, ('Bus%d MB P' % busid) in cols,
        ('Bus%d MC P' % busid) in cols))
    if ud is None or P is None:
        print('  [全部列名]', cols)
        raise RuntimeError('关键通道未找到')
    return ud + 1j * uq, P, Q, maP, maQ, mbP, mbQ, mcP, mcQ


def downsample(u, P, Q, dt_orig, dt_new):
    n = len(u)
    k = int(round(dt_new / dt_orig))
    sl = slice(0, n, k)
    return u[sl], P[sl], Q[sl]


def obj_at_dt(truth, u, P, Q, dt):
    try:
        return obj_mix_iden(truth, u, P, Q, dt)
    except Exception as e:
        return float('nan')


def main():
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    res = pickle.load(open(M.PKL, 'rb'))
    rec = res[g]
    truth = np.array(rec['truth'])
    cc = rec['cc']
    print('=' * 90)
    print('决定性根因定位 g=%d: truth=%s' % (g, np.round(truth, 3)))
    print('=' * 90)

    # ---- 跑 probing 仿真到原生 0.002s, 滤波 100Hz ----
    kwargs = dict(
        study='IEEE39_RE', total_time=30,
        load_bus_monitor=[M.BUSID],
        ld_con_chng_buses=list(M.BUSID_SET),
        ld_con_chng_values=[ps_d.read_bus_params(b) for b in M.BUSID_SET],
        ambient_seed=M.SEED + g,
        filter_cutoff_frequency=100,
        sample_time_step=0.002,
        simu_time_step=0.002,
        disturbance_type='probing', disturbance_range=800.0,
        probing_f_lo=6.0, probing_f_hi=13.0, probing_n_comp=8,
    )
    index = M.BUSID_SET.index(M.BUSID)
    for name, val in cc.items():
        kwargs['ld_con_chng_values'][index][ps_d.para_names.index(name)] = val
    print('跑 probing 仿真(原生 0.002s, 滤波 100Hz)...', flush=True)
    df, d_para, t = ps_d.one_time_simulation(kwargs, max_retries=8)
    if df is None:
        print('仿真失败'); return
    u, P, Q, maP, maQ, mbP, mbQ, mcP, mcQ = extract_full(df, M.BUSID)
    print('  数据点数=%d  总负荷 P 均值=%.3f MW' % (len(u), P.mean()))
    # 打印全部列名, 并把 df 落盘供后续复用
    print('  [全部列名]:')
    for i, c in enumerate(list(df.columns)):
        print('    %3d %r' % (i, c))
    import pickle as _pk
    _pk.dump({'df': df, 'u': u, 'P': P, 'Q': Q,
              'maP': maP, 'maQ': maQ, 'mbP': mbP, 'mbQ': mbQ, 'mcP': mcP, 'mcQ': mcQ,
              'truth': truth, 'cc': cc},
             open(os.path.join(M.OUTDIR, 'diag_root_g%d.pkl' % g), 'wb'))
    print('  [已落盘] diag_root_g%d.pkl' % g)

    # ---- (a) dt 扫描 ----
    print('\n(a) dt 扫描(真值处 obj_mix_iden, 越小越准):')
    a_a, b_a, H2_a, mva_a = truth[0] / 0.75, truth[1], truth[2] / 0.75, P[0] * truth[3]
    a_b, b_b, H2_b, mva_b = truth[4] / 0.75, truth[5], truth[6] / 0.75, P[0] * truth[7]
    for dt in (0.01, 0.005, 0.002):
        uu, PP, QQ = downsample(u, P, Q, 0.002, dt)
        c = obj_at_dt(truth, uu, PP, QQ, dt)
        print('  dt=%.3f  obj_mix_iden(truth)=%.4e  (n=%d)' % (dt, c, len(uu)))

    # ---- (b) 电机功率对照(在 0.01 网格上, 与辨识口径一致) ----
    print('\n(b) 电机功率对照(2电机模型 vs PSSE 3电机, dt=0.01):')
    uu, PP, QQ = downsample(u, P, Q, 0.002, 0.01)
    dt = 0.01
    ok_a, P_pa, Q_pa = _core_con(uu, a_a, b_a, H2_a, mva_a, dt)
    ok_b, P_pb, Q_pb = _core_var(uu, a_b, b_b, H2_b, mva_b, dt)
    # PSSE 电机总功率(MA+MB+MC), 同样下采样到 0.01
    def ds(x):
        return None if x is None else x[::5]
    maP1, mbP1, mcP1 = ds(maP), ds(mbP), ds(mcP)
    maQ1, mbQ1, mcQ1 = ds(maQ), ds(mbQ), ds(mcQ)
    P_mot = None if (maP1 is None or mbP1 is None or mcP1 is None) else maP1 + mbP1 + mcP1
    Q_mot = None if (maQ1 is None or mbQ1 is None or mcQ1 is None) else maQ1 + mbQ1 + mcQ1
    P_pred = P_pa + P_pb
    Q_pred = Q_pa + Q_pb
    k = _trim_k(len(uu)); sl = slice(k, len(uu) - k)
    def rms_pct(a, b, ref):
        return float(np.sqrt(np.mean((a[sl] - b[sl]) ** 2)) / abs(ref[sl].mean()) * 100)
    if P_mot is not None:
        print('  2电机模型 P_pred  vs  PSSE电机总功率(MA+MB+MC):')
        print('    P: RMS=%.3f%%   Q: RMS=%.3f%%' % (
            rms_pct(P_pred, P_mot, P_mot), rms_pct(Q_pred, Q_mot, Q_mot)))
    else:
        print('  [警告] 电机通道(MA/MB/MC P,Q)未提取到, 跳过电机对照')
    print('  2电机模型 P_pred  vs  总负荷 PLOD:')
    print('    P: RMS=%.3f%%   Q: RMS=%.3f%%' % (
        rms_pct(P_pred, PP, PP), rms_pct(Q_pred, QQ, QQ)))
    if P_mot is not None:
        print('  PSSE 静态/其他部分(PLOD - MA-MB-MC): 均值 P=%.3f MW' % (PP[sl].mean() - P_mot[sl].mean()))
    print('  => 若第一组 RMS 小(电机聚合OK)而第二组 RMS 大 => 失配在静态/频率依赖;')
    print('     若第一组 RMS 就大 => 电机 3->2 聚合失效(次暂态/双 var 电机)。')


if __name__ == '__main__':
    main()

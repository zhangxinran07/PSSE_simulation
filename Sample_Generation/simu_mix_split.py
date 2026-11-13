# -*- coding: utf-8 -*-
"""30 组分电机功率通道重跑 —— 供「分电机观测 vs 母线总功率观测」真实数据对照用。

背景(张老师 2026-11-08「多电机功率量测找找规律」):
  零失配合成数据(split_vs_total_power.py)已证: 分电机观测可消除 A/B 混淆、条件数降约
  2 个数量级、单起点无局部极小。但工程上只有母线总功率单一观测, 分电机观测只用于
  研究内找规律; 本脚本重跑 30 组真实数据并**保留分电机功率通道**, 以验证结构失配下
  「分电机观测改善 P_im/H2 精度」是否仍成立。

关键事实(已核实):
  run_psse_simulation 默认 var_channels 已含 'MA P','MA Q','MB P','MB Q','MC P','MC Q',
  'S P','S Q'; postprocess_outfile 返回的 df 已带全部分电机列(如 'BUS3 MA P')。
  因此本脚本只做一件事: 每组建模仿真后把**完整 df**(含分电机列) + 8 参数真值落盘,
  无需改 PSSE_KEEP_OUTFILE、也无需保留 90MB/轮的 .out 中间文件。

输出: Sample_Generation/results/ori_data_mix_split_<时间戳>.pkl
  结构: {'groups': [ {'df': df, 'truth': np.array(8), 'busid': 3, 'simulation_time': s}, ... ]}
  df 列为 ['PLOD 3[...]1','QLOD 3[...]1','BUS3 UD','BUS3 UQ','BUS3 MA P','BUS3 MA Q',
           'BUS3 MB P','BUS3 MB Q','BUS3 MC P','BUS3 MC Q','BUS3 S P','BUS3 S Q', ...]
  truth 顺序 = [a_a,b_a,H2_a,P_ima, a_b,b_b,H2_b,P_imb] (A 组恒转矩 4 + B 组变转矩 4)

用法(在 E 盘 PSSE 机, PSSE_Simulation 目录下):
  python Sample_Generation/simu_mix_split.py [n_groups] [busid]
默认 n_groups=30, busid=3。
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'package'))
import main_simu as M   # 复用 sample_params / kwargs / lb / ub / busid_set / dt / ps_d


def run(n_groups=30, busid_iden=3):
    M.kwargs['load_bus_monitor'] = [busid_iden]
    # 与 simu_iden_mix 一致: 填充 ld_con_chng_buses/values
    for busid in M.busid_set:
        M.kwargs['ld_con_chng_buses'].append(busid)
        M.kwargs['ld_con_chng_values'].append(M.ps_d.read_bus_params(busid=busid))

    out_path = './Sample_Generation/results/ori_data_mix_split_%s.pkl' % time.strftime('%Y%m%d_%H%M%S')
    groups = []

    for k in range(n_groups):
        print('%d/%d' % (k, n_groups), flush=True)
        para_real = M.sample_params(busid_iden, 'mix')   # 采样 A+B 两组参数(8 个真值)

        df, d_para, simulation_time = M.ps_d.one_time_simulation(M.kwargs)
        if df is None:
            print('  group %d: 仿真失败, 跳过' % k, flush=True)
            continue

        groups.append({
            'df': df,                                   # 完整 df(含分电机列)
            'truth': np.asarray(para_real, dtype=float),  # 8 参数真值
            'busid': busid_iden,
            'simulation_time': simulation_time,
        })
        pd.to_pickle({'groups': groups}, out_path)      # 逐组落盘, 防中断丢失
        print('  ok, 已存 %d 组' % len(groups), flush=True)

    print('DONE -> %s (%d/%d 组成功)' % (out_path, len(groups), n_groups), flush=True)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run(n, b)

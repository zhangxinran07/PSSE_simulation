# dynrun.py - 4 PSSe: 恒转矩/变转矩两种辨识的统一入口
from __future__ import with_statement
import os
import sys#<- it must be first declaration
sys.path.append(os.path.join(os.path.dirname(__file__), r'..\package'))  # 将 PSSE_Simulation 文件夹添加到系统路径中
import psse_dynamic as ps_d  # 必须在 numpy/matplotlib 之前 import(PSSE 与 MKL DLL 冲突)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from identification import obj_con_iden, obj_var_iden, obj_mix_iden, obj_mix_jac
from identification import identify as idf_id
import warnings
import time

np.set_printoptions(precision=3, suppress=True)  # 设置 numpy 打印选项
# 强行忽略来自 scipy 优化器的这个特定 UserWarning
warnings.filterwarnings('ignore', message='.*delta_grad == 0.0.*')
ori_data_path='./Sample_Generation/results/ori_data_%s.pkl' % time.strftime('%Y%m%d_%H%M%S')
result_path='./Sample_Generation/results/result_all_%s.pkl' % time.strftime('%Y%m%d_%H%M%S')
ori_data_path_var='./Sample_Generation/results/ori_data_var_%s.pkl' % time.strftime('%Y%m%d_%H%M%S')
result_path_var='./Sample_Generation/results/result_all_var_%s.pkl' % time.strftime('%Y%m%d_%H%M%S')
ori_data_path_mix='./Sample_Generation/results/ori_data_mix_%s.pkl' % time.strftime('%Y%m%d_%H%M%S')
result_path_mix='./Sample_Generation/results/result_all_mix_%s.pkl' % time.strftime('%Y%m%d_%H%M%S')
os.makedirs(os.path.dirname(ori_data_path), exist_ok=True)  # 自动创建 results 目录

kwargs={           # 恒转矩辨识(simu_iden)用
            'total_time': 30,
            'ambient_cutoff_frequency': 2,
            'disturbance_type': 'ambient_a',     # 'ambient_a'=MW幅值 / 'ambient_u'=电压范围 / 'ambient_p'=功率范围 / 'load_change' / 'fault'
            'disturbance_range': 500,    # 目标值, 单位由 disturbance_type 决定
            'ld_con_chng_buses': [],
            'ld_con_chng_values': [],
            'load_bus_monitor': []
        }

lb=[100,30,0.08,0.2]
ub=[1200,240,0.22,0.8]
busid_set=[3,4,15,20,21,24,25,27,28]
dt=0.01

def plot(df,k,title='Dynamic Simulation Results', xlabel='Time (s)'):
    plt.rcParams.update({
    'font.size': 8,  # 设置字体大小
    'axes.labelsize': 8,  # 设置坐标轴标签字体大小
    'font.family': 'Times New Roman',  # 设置字体为 Times New Roman
    'axes.titlesize': 10,  # 设置标题字体大小
    'xtick.labelsize': 7,  # 设置x轴刻度标签字体大小
    'ytick.labelsize': 7,  # 设置y轴刻度标签字体
    'axes.unicode_minus': False,  # 解决负号显示为方块的问题
    'lines.linewidth': 1.5,  # 设置线条宽度
    'figure.subplot.top': 0.9,  # 设置子图上边距
    'figure.subplot.bottom': 0.15,  # 设置子图下边距
    'figure.subplot.left': 0.15,  # 设置子图左边距
    'figure.subplot.right': 0.95,  # 设置子图右边距
})
    plt.rcParams['font.size'] = 8  # 设置字体大小
    fig=plt.figure(figsize=(9/2.54, 6/2.54))
    plt.plot(df.values[:,0], df.values[:,k])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(df.columns[k])
    plt.show()


def sample_params(busid_iden, mode):
    """采样一组电机参数(按 mode 选择参数集), 范围检查通过后写入 kwargs 并返回 para_real。

    mode='con': 仅 A 组(恒转矩), 返回 4 参数 (a, b, H2, P_im)
    mode='var': 仅 B 组(变转矩), 返回 4 参数
    mode='mix': A+B 两组(比例约束: P_ma/P_mb 各>=0.2 且和<=0.8), 返回 8 参数(A 组 4 + B 组 4)
    """
    index = busid_set.index(busid_iden)
    while True:

        if mode == 'mix':
            cc = {
                'P_ma': np.random.uniform(0.2, 0.6),   # A 组(恒转矩)占比
                'P_mb': 0,                      # B 组(变转矩)占比, 下面按约束采样
                'R_a':  0,
                'X_a':  np.random.uniform(1.35, 2.05),
                'X1_a': np.random.uniform(0.12, 0.22),
                'T0_a': np.random.uniform(0.08, 0.2),
                'H2_a': np.random.uniform(0.1, 0.2),
                'R_b':  0,
                'X_b':  np.random.uniform(1.35, 2.05),
                'X1_b': np.random.uniform(0.12, 0.22),
                'T0_b': np.random.uniform(0.08, 0.2),
                'H2_b': np.random.uniform(0.1, 0.2),
            }
            # P_ma ~ U(0.2,0.6);  P_mb ~ U(0.2, 0.8-P_ma)
            # => 两组占比各自落在 [0.2, 0.6], 且 P_ma+P_mb <= 0.8; 两台电机必定同时在场
            cc['P_mb'] = np.random.uniform(0.2, 0.8 - cc['P_ma'])   # 保证 >=0.2 且两者之和 <=0.8
            assert cc['P_ma'] >= 0.2 and cc['P_mb'] >= 0.2, '两组电机占比都必须 >0.2, 否则该组容量为 0、参数不可辨识'
            assert cc['P_ma'] + cc['P_mb'] <= 0.8 + 1e-9, '两组占比之和不得超过 0.8'

            pima, xa, x1a, t0a, h2a = cc['P_ma'], cc['X_a'], cc['X1_a'], cc['T0_a'], cc['H2_a']
            pimb, xb, x1b, t0b, h2b = cc['P_mb'], cc['X_b'], cc['X1_b'], cc['T0_b'], cc['H2_b']
            para_real=[(xa/x1a-1)/t0a/x1a, xa/x1a/t0a, h2a, pima,
                       (xb/x1b-1)/t0b/x1b, xb/x1b/t0b, h2b, pimb]   # A 组 4 个 + B 组 4 个

            # 只对两组的 a,b,H2 做范围检查(6 个): P_im(P_ma/P_mb)由比例约束单独保证, 不参与比较
            pr6 = para_real[0:3]+para_real[4:7]
            ok = (np.array(pr6)>np.array(lb[0:3]*2)).all() and (np.array(pr6)<np.array(ub[0:3]*2)).all()
            fmt = "%.3f %.3f %.3f %.3f  %.3f %.3f %.3f %.3f %s"
        else:
            if mode == 'con':
                cc = {
                    'P_ma': np.random.uniform(0.3, 0.7),
                    'P_mb': 0,
                    'R_a':  0,
                    'R_b':  0,
                    'X_a':  np.random.uniform(1.35, 2.05),
                    'X1_a': np.random.uniform(0.12, 0.22),
                    'T0_a': np.random.uniform(0.08, 0.2),
                    'H2_a': np.random.uniform(0.1, 0.2),
                }
                pim, x0, x1, t0, h2 = cc['P_ma'], cc['X_a'], cc['X1_a'], cc['T0_a'], cc['H2_a']
                para_real=[(x0/x1-1)/t0/x1,x0/x1/t0,h2,pim]
            else:  # 'var': 仅 B 组
                cc = {
                    'P_ma': 0,                        # A 组电机有功置 0(辨识 B 组)
                    'P_mb': np.random.uniform(0.3, 0.7),
                    'R_a':  0,
                    'R_b':  0,
                    'X_b':  np.random.uniform(1.35, 2.05),
                    'X1_b': np.random.uniform(0.12, 0.22),
                    'T0_b': np.random.uniform(0.08, 0.2),
                    'H2_b': np.random.uniform(0.1, 0.2),
                }
                pmb, xb, x1b, t0b, h2b = cc['P_mb'], cc['X_b'], cc['X1_b'], cc['T0_b'], cc['H2_b']
                para_real=[(xb/x1b-1)/t0b/x1b, xb/x1b/t0b, h2b, pmb]
            ok = (np.array(para_real)>np.array(lb)).all() and (np.array(para_real)<np.array(ub)).all()
            fmt = "%.3f %.3f %.3f %.3f %s"

        if ok:
            for name, val in cc.items():
                kwargs['ld_con_chng_values'][index][ps_d.para_names.index(name)] = val
            print(fmt % (*para_real, time.strftime('%H:%M:%S')))
            return para_real


def simu_iden_con(busid_iden=3):
    """恒转矩辨识(obj_con_iden): 每组随机采样 A 组电机参数并辨识"""

    kwargs['load_bus_monitor']=[busid_iden]
    for busid in busid_set:
        kwargs['ld_con_chng_buses'].append(busid)
        kwargs['ld_con_chng_values'].append(ps_d.read_bus_params(busid=busid))

    obj = obj_con_iden

    ori_data=pd.DataFrame()
    result_all=pd.DataFrame()

    for k in range(1000):
        print(k)

        para_real = sample_params(busid_iden, 'con')   # 采样 A 组参数(4 个)

        # 单次仿真 + 单次辨识(结论: 重复仿真取 cost 最小反而有害)
        df, d_para, simulation_time = ps_d.one_time_simulation(kwargs)
        if df is None:
            print('group %d: 仿真失败(重试/自适应修正仍失败), 跳过' % k)
            continue

        ud, uq, P, Q = [df.values[:, [i for i in range(len(df.columns)) if (key % busid_iden) in df.columns[i]]].reshape(len(df))
            for key in ["BUS%d UD", "BUS%d UQ", "PLOD %d[", "QLOD %d["]]
        u=ud+1j*uq

        ori_data=pd.concat([ori_data,pd.DataFrame(np.transpose([ud,uq,P,Q]),columns=['%d ud' % k,'%d uq' % k,'%d P' % k,'%d Q' % k])],axis=1)
        ori_data.to_pickle(ori_data_path)

        print("%.2f %.2f %.2f %.2f %.2f %.2e" % (abs(u).max(), abs(u).min(), P.max(), P.min(), simulation_time, obj(para_real, u, P, Q, dt)), end=' ')

        # 采样初值→优化→距离校验与耗时统计已封装在 identification.identify
        result, computation_time = idf_id(u, P, Q, dt, lb, ub, obj_func=obj, tol=1e-12)

        print(result.x, "%.2e %.1f" % (result.fun, computation_time))

        result_all=pd.concat([result_all,pd.DataFrame(list(result.x)+list(para_real)+[computation_time,result.fun]).T],axis=0)
        result_all.to_pickle(result_path)

    result_all.columns=['a','b','H2','P_im','a_real','b_real','H2_real','P_im_real','Time','fun']
    result_all.to_pickle(result_path)
    a=1


def simu_iden_var(busid_iden=3):
    """变转矩辨识(obj_var_iden): 每组随机采样 B 组电机参数(A 组有功置 0)并辨识"""

    kwargs['load_bus_monitor']=[busid_iden]
    for busid in busid_set:
        kwargs['ld_con_chng_buses'].append(busid)
        kwargs['ld_con_chng_values'].append(ps_d.read_bus_params(busid=busid))

    obj = obj_var_iden

    ori_data=pd.DataFrame()
    result_all=pd.DataFrame()

    for k in range(1000):
        print(k)

        para_real = sample_params(busid_iden, 'var')   # 采样 B 组参数(4 个)

        # 单次仿真 + 单次辨识(结论: 重复仿真取 cost 最小反而有害)
        df, d_para, simulation_time = ps_d.one_time_simulation(kwargs)

        ud, uq, P, Q = [df.values[:, [i for i in range(len(df.columns)) if (key % busid_iden) in df.columns[i]]].reshape(len(df))
            for key in ["BUS%d UD", "BUS%d UQ", "PLOD %d[", "QLOD %d["]]
        u=ud+1j*uq

        ori_data=pd.concat([ori_data,pd.DataFrame(np.transpose([ud,uq,P,Q]),columns=['%d ud' % k,'%d uq' % k,'%d P' % k,'%d Q' % k])],axis=1)
        ori_data.to_pickle(ori_data_path_var)

        print("%.2f %.2f %.2f %.2f %.2f %.2e" % (abs(u).max(), abs(u).min(), P.max(), P.min(), simulation_time, obj(para_real, u, P, Q, dt)), end=' ')

        # 变转矩辨识: obj_var_iden 与 obj_con_iden 同签名(para,u,P,Q,dt), identify 直接使用
        result, computation_time = idf_id(u, P, Q, dt, lb, ub, obj_func=obj, tol=1e-12)

        print(result.x, "%.2e %.1f" % (result.fun, computation_time))

        result_all=pd.concat([result_all,pd.DataFrame(list(result.x)+list(para_real)+[computation_time,result.fun]).T],axis=0)
        result_all.to_pickle(result_path_var)

    result_all.columns=['a','b','H2','P_im','a_real','b_real','H2_real','P_im_real','Time','fun']
    result_all.to_pickle(result_path_var)
    a=1


def simu_iden_mix(busid_iden=3, n_groups=1000):
    """混合辨识: 每组同时随机采样恒转矩(A 组)与变转矩(B 组)电机参数。

    比例约束(电机占比): P_ma, P_mb 各自 >= 0.2, 且 P_ma + P_mb <= 0.8。
    采样方法: P_ma ~ U(0.2, 0.6), 再 P_mb ~ U(0.2, 0.8 - P_ma), 两条件天然满足。
    para_real 输出 8 个参数: 前 4 个为 A 组(a,b,H2,P_im), 后 4 个为 B 组。
    辨识用混合目标函数 obj_mix_iden(8 参数: A 组恒转矩 + B 组变转矩)。
    """

    kwargs['load_bus_monitor']=[busid_iden]
    for busid in busid_set:
        kwargs['ld_con_chng_buses'].append(busid)
        kwargs['ld_con_chng_values'].append(ps_d.read_bus_params(busid=busid))

    obj = obj_mix_iden    # 混合目标函数(A 组 con + B 组 var, 共 8 参数)

    ori_data=pd.DataFrame()
    result_all=pd.DataFrame()

    for k in range(n_groups):
        print(k)

        para_real = sample_params(busid_iden, 'mix')   # 采样 A+B 两组参数(8 个)

        # 单次仿真 + 单次辨识(结论: 重复仿真取 cost 最小反而有害)
        df, d_para, simulation_time = ps_d.one_time_simulation(kwargs)
        if df is None:
            print('group %d: 仿真失败(重试/自适应修正仍失败), 跳过' % k)
            continue

        ud, uq, P, Q = [df.values[:, [i for i in range(len(df.columns)) if (key % busid_iden) in df.columns[i]]].reshape(len(df))
            for key in ["BUS%d UD", "BUS%d UQ", "PLOD %d[", "QLOD %d["]]
        u=ud+1j*uq

        ori_data=pd.concat([ori_data,pd.DataFrame(np.transpose([ud,uq,P,Q]),columns=['%d ud' % k,'%d uq' % k,'%d P' % k,'%d Q' % k])],axis=1)
        ori_data.to_pickle(ori_data_path_mix)

        print("%.2f %.2f %.2f %.2f %.2f %.2e" % (abs(u).max(), abs(u).min(), P.max(), P.min(), simulation_time, obj(para_real, u, P, Q, dt)), end=' ')

        # 采样初值→优化→距离校验与耗时统计已封装在 identification.identify
        # 混合辨识 8 参数: 上下界 lb/ub 各叠两组; tol 放松到 1e-8(8 维+噪声下过严会不收敛)
        # jac_func 用解析梯度(比数值差分快 ~2.6x, 精度高两个数量级)
        result, computation_time = idf_id(u, P, Q, dt, lb*2, ub*2, obj_func=obj,
                                          tol=1e-8, max_trials=5, jac_func=obj_mix_jac)

        print(result.x, "%.2e %.1f" % (result.fun, computation_time))

        result_all=pd.concat([result_all,pd.DataFrame(list(result.x)+list(para_real)+[computation_time,result.fun]).T],axis=0)
        result_all.to_pickle(result_path_mix)

    # 列名: 辨识结果 8(con 组 4 + var 组 4) + 真值 8 + 耗时 + fun
    result_all.columns=['a_a','b_a','H2_a','P_ima','a_b','b_b','H2_b','P_imb',
                        'a_a_real','b_a_real','H2_a_real','P_ima_real',
                        'a_b_real','b_b_real','H2_b_real','P_imb_real','Time','fun']
    result_all.to_pickle(result_path_mix)
    a=1


if __name__ == "__main__":
    #simu_iden_con()          # 恒转矩辨识(A 组参数, obj_con_iden)
    #simu_iden_var()          # 变转矩辨识(B 组参数, obj_var_iden)
    simu_iden_mix()          # 混合辨识(A+B 组参数, 比例约束 0.2/0.2/0.8)

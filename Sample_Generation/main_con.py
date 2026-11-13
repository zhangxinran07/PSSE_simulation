# dynrun.py - 4 PSSe
from __future__ import with_statement
import os
import sys#<- it must be first declaration
sys.path.append(os.path.join(os.path.dirname(__file__), '..\package'))  # 将 PSSE_Simulation 文件夹添加到系统路径中
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psse_dynamic as ps_d
from identification import obj_con_iden as obj2
from identification import obj_con_iden2 as obj
from scipy import optimize as opt
import warnings
import time

np.set_printoptions(precision=3, suppress=True)  # 设置 numpy 打印选项
# 强行忽略来自 scipy 优化器的这个特定 UserWarning
warnings.filterwarnings('ignore', message='.*delta_grad == 0.0.*')
lb=[200,60,0.08,0.2]
ub=[1000,120,0.22,0.8]
ld_para_names=['P_ma','P_mb','P_mc','P_md','P_e','R_a','X_a','X1_a','T0_a','H2_a','R_b','X_b','X1_b','T0_b','H2_b']
ld_para_indexes=[19,20,21,22,23,40, 41, 42, 44, 46,60, 61, 62, 64, 66]
pos = {name: i for i, name in enumerate(ld_para_names)}
kwargs={
            'disturbance_type': 'ambient_a',
            'study': 'IEEE39_RE',              # 这里的 study 变量会直接传递给子进程
            'total_time': 60,
            'case_path': './Cases/IEEE39/',
            'ambient_cutoff_frequency': 2,
            'filter_cutoff_frequency': 10,
            'sample_time_step':0.01,
            'disturbance_range': 800,
            'ld_con_chng_buses': [],
            'ld_con_chng_values': []
        }
dt=kwargs['sample_time_step']

ori_data_path='./Sample_Generation/results/ori_data_%s.pkl' % time.strftime('%Y%m%d_%H%M%S')
result_path='./Sample_Generation/results/result_all_%s.pkl' % time.strftime('%Y%m%d_%H%M%S')
os.makedirs(os.path.dirname(ori_data_path), exist_ok=True)  # 自动创建 results 目录



def worker(q, *args, **kwargs):
    """子进程包装器"""
    # ====== 新增：在系统层面把子进程的控制台输出彻底倒向黑洞 ======
    f_null = open(os.devnull, 'w')
    # 备份原子进程的 stdout 文件描述符（其实在子进程里不备份也行，因为运行完就死了）
    old_stdout_fileno = os.dup(sys.stdout.fileno())

    # 将系统的标准输出（fileno 1）强行重定向到 devnull
    os.dup2(f_null.fileno(), sys.stdout.fileno())
    # ==========================================================

    try:
        # 此时运行 run_psse_simulation，所有底层 Fortran 漏出来的打印都会被黑洞吞掉
        df, d_para = ps_d.run_psse_simulation(*args, **kwargs)
        q.put((df, d_para))
    except Exception as e:
        q.put(e)
    finally:
        # ====== 新增：恢复输出（好习惯，虽然子进程快死了） ======
        os.dup2(old_stdout_fileno, sys.stdout.fileno())
        f_null.close()

def one_time_simulation(kwargs):
    q = Queue()
    p = Process(target=worker, args=(q,), kwargs=kwargs)
    p.start()
    result = q.get()
    p.join()
    if isinstance(result, Exception):
        print(f"仿真发生错误: {result}")
        return None, None
    else:
        df, d_para = result
        #print(f"仿真成功完成，数据行数：{len(df)}")
        return df, d_para

def plot(df,k,title='Dynamic Simulation Results', xlabel='Time (s)'):
    plt.rcParams.update({
    'font.size': 8,  # 设置字体大小
    'axes.labelsize': 8,  # 设置坐标轴标签字体大小
    'font.family': 'Times New Roman',  # 设置字体为 Times New Roman
    'axes.titlesize': 10,  # 设置标题字体大小
    'xtick.labelsize': 7,  # 设置x轴刻度标签字体大小
    'ytick.labelsize': 7,  # 设置y轴刻度标签字体
    'axes.unicode_minus': False,  # 解决负号显示为方块的问题
    #'figure.figsize': (10, 6),  # 设置图形大小
    'lines.linewidth': 1.5,  # 设置线条宽度
    'figure.subplot.top': 0.9,  # 设置子图上边距
    'figure.subplot.bottom': 0.15,  # 设置子图下边距
    'figure.subplot.left': 0.15,  # 设置子图左边距
    'figure.subplot.right': 0.95,  # 设置子图右边距
})
    plt.rcParams['font.size'] = 8  # 设置字体大小
    fig=plt.figure(figsize=(9/2.54, 6/2.54))#
    plt.plot(df.values[:,0], df.values[:,k])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(df.columns[k])
    plt.show()

def read_bus_params(busid_set, path='./Cases/IEEE39/ieee39_RE_ieee39_RE_motorD_Stall_enabled.dyr'):
    for busid in busid_set:
        kwargs['ld_con_chng_buses'].append(busid)
        kwargs['ld_con_chng_values'].append(ps_d.read_bus_params(busid=busid))
    
def main(busid=4):

    read_bus_params([busid])
    ori_data=pd.DataFrame()

    for k in range(1000):
        print(k)
        
        while True:
            
            cc = {
                'P_ma': np.random.uniform(0.3, 0.7),
                'X_a':  np.random.uniform(1.35, 2.05),
                'X1_a': np.random.uniform(0.12, 0.22),
                'T0_a': np.random.uniform(0.08, 0.2),
                'H2_a': np.random.uniform(0.1, 0.2),
            }
            
            pim, x0, x1, t0, h2 = cc['P_ma'], cc['X_a'], cc['X1_a'], cc['T0_a'], cc['H2_a']            
            para_real=[(x0/x1-1)/t0/x1,x0/x1/t0,h2,pim]
            
            if (np.array(para_real)>np.array(lb)).all() and (np.array(para_real)<np.array(ub)).all():
                for name, val in cc.items():
                    kwargs['ld_con_chng_values'][0][pos[name]] = val
                kwargs['ld_con_chng_values'][0][pos['R_a']] = 0
                print("%.3f %.3f %.3f %.3f %s" % (*para_real, time.strftime('%H:%M:%S')))
                break
            
        for i in range(3):
            
            start=time.time()

            while True:
                df, d_para = one_time_simulation(kwargs)
                if df is None:
                    continue
                else:
                    break
            
            ud, uq, P, Q = [df.values[:, [i for i in range(len(df.columns)) if (key % busid) in df.columns[i]]].reshape(len(df))
                for key in ["BUS%d UD", "BUS%d UQ", "PLOD %d[", "QLOD %d["]]          
            u=ud+1j*uq

            ori_data=pd.concat([ori_data,pd.DataFrame(np.transpose([ud,uq,P,Q]),columns=['%d ud' % k,'%d uq' % k,'%d P' % k,'%d Q' % k])],axis=1)
            ori_data.to_pickle(ori_data_path)

            print("%.2f %.2f %.2f %.2f %.2f %.2e" % (abs(u).max(), abs(u).min(), P.max(), P.min(), time.time() - start, obj(para_real, u, P, Q, dt)))

        
            start=time.time()
            while True:
                while True:
                    x0=np.random.uniform(lb,ub)
                    if obj(x0,u,P,Q,dt)<0.5:
                        break
                result=opt.minimize(obj,x0, args=(u,P,Q,dt), method='trust-constr', bounds=np.transpose(np.array([lb,ub])), options={'disp': False},tol=1e-8)
                if np.sum(np.square(result.x-x0))>1000:
                    break

            computation_time=time.time() - start
            if k==0:
                result_all=pd.DataFrame(list(result.x)+list(para_real)+[computation_time,result.fun]).T
            else:
                result_all=pd.concat([result_all,pd.DataFrame(list(result.x)+list(para_real)+[computation_time,result.fun]).T],axis=0)

                result_all.to_pickle(result_path)

            print(result.x,end=' ')
            print("%.2e %.1f" % (result.fun, computation_time))

    result_all.columns=['a','b','H2','P_im','a_real','b_real','H2_real','P_im_real','Time','fun']
    result_all.to_pickle(result_path)
    a=1

if __name__ == "__main__":
    main()
    # P=6
    # para_real=[]
    # para_ori=[]
    # for i in range(10000):
    #     para=[np.random.uniform(0.3,0.7),np.random.uniform(1.35,2.25), np.random.uniform(0.12,0.22), np.random.uniform(0.08,0.2), np.random.uniform(0.1,0.2)]
    #     lf=0.75
    #     pimp=para[0]
    #     x0=para[1]
    #     x1=para[2]
    #     t0=para[3]
    #     h2=para[4]
    #     mva=P/lf*pimp
    #     a=(x0/x1-1)/t0/x1*mva
    #     b=x0/x1/t0
    #     h2=h2*mva
    #     para_real.append([a/mva,b,h2/mva,pimp])
    #     para_ori.append([a,b,h2,pimp])
    # r=np.array(para_real)
    # o=np.array(para_ori)
    # a=1
        

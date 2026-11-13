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
from identification import obj_con_iden as obj
from scipy import optimize as opt
import warnings
import time

np.set_printoptions(precision=3, suppress=True)  # 设置 numpy 打印选项
# 强行忽略来自 scipy 优化器的这个特定 UserWarning
warnings.filterwarnings('ignore', message='.*delta_grad == 0.0.*')


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

if __name__ == '__main__':

    busid=3
    lb=[500,60,0.2,0.2]
    ub=[2800,150,1,0.8]
    #result_all=[]
    ori_data=pd.DataFrame()

    kwargs={
                'disturbance_type': 'ambient_a',
                'study': 'IEEE39_RE',              # 这里的 study 变量会直接传递给子进程
                'case_path': './Cases/IEEE39/',
                'total_time': 60,
                'ambient_cutoff_frequency': 2,
                'filter_cutoff_frequency': 10,
                'disturbance_range': 1000,
                'ld_con_chng_buses': [3],
                'ld_con_chng_values': [[0.5, 0, -1, -1, -1, 0, 1.8, 0.22, 0.08, 0.15, 0, 1.8, 0.19, 0.2, 0.5]]
            }

    #df,d_para=ps_d.run_psse_simulation(**kwargs)

    for k in range(1000):
        print(k)

        start=time.time()

        while True:
            df, d_para = one_time_simulation(kwargs)
            if df is None:
                continue
            else:
                print(len(df),end=' ')
                if (len(df)>=kwargs['total_time']*100):
                    break

        u=df.values[:,[i for i in range(len(df.columns)) if "VOLT %d " % busid in df.columns[i]]].reshape(len(df))
        P=df.values[:,[i for i in range(len(df.columns)) if "PLOD %d[" % busid in df.columns[i]]].reshape(len(df))
        Q=df.values[:,[i for i in range(len(df.columns)) if "QLOD %d[" % busid in df.columns[i]]].reshape(len(df))
        ang=df.values[:,[i for i in range(len(df.columns)) if "ANGL %d " % busid in df.columns[i]]].reshape(len(df))
        #c=abs(u[0])**2*a_c*w0/P_p[0]
        #delta=c**2-4*b_c**2*w0**2

        ori_data=pd.concat([ori_data,pd.DataFrame(np.transpose([u,P,Q,ang]),columns=['%d u' % k,'%d P' % k,'%d Q' % k,'%d ang' % k])],axis=1)
        ori_data.to_pickle('./ori_data_60_4.pkl')

        con=lambda x: x[0]*u[0]**2-2*x[1]*x[3]*P[0]
        #nlc=opt.NonlinearConstraint(con,0,np.inf)

        print("\n%.2f %.2f %.2f %.2f %d %.2f" % (u.max(),u.min(),P.max(),P.min(),len(df), time.time() - start))


        for i in range(1):
            start=time.time()
            result_temp=[]
            fun_temp=[]
            for j in range(2):
                while True:
                    x0=np.random.uniform(lb,ub)
                    if obj(x0,df,busid,0.01)<1 and con(x0)>0:
                        break
                #print(x0)
                result=opt.minimize(obj,x0, args=(df,busid,0.01), method='trust-constr', bounds=np.transpose(np.array([lb,ub])), options={'disp': False},tol=1e-10)
                result_temp.append(result.x)
                fun_temp.append(result.fun)
                print(result.x,end=' ')
                print("%.2e %.1f" % (result.fun, time.time() - start))
            min_index=np.argmin(fun_temp)
            computation_time=time.time() - start
            if k==0:
                result_all=pd.DataFrame(list(result_temp[min_index])+[computation_time,fun_temp[min_index]]).T
            else:
                result_all=pd.concat([result_all,pd.DataFrame(list(result_temp[min_index])+[computation_time,fun_temp[min_index]]).T],axis=0)

                result_all.to_pickle('./result_all_60_4.pkl')

            print(result_temp[min_index],end=' ')
            print("%.2e %.1f" % (fun_temp[min_index], computation_time))

    #a=b2b.obj_fun(df,para=[0.02,1.8,0.22,0.08,0.1])

    #ps_s.run_psse_power_flow(study=study)  # 运行 PSSE 静态潮流计算
    result_all.columns=['a','b','H2','P_im','Time','fun']
    result_all.to_pickle('./result_all_60_4.pkl')
    a=1

# dynrun.py - 4 PSSe
from __future__ import with_statement
import os
import sys#<- it must be first declaration
sys.path.append(os.path.join(os.path.dirname(__file__), 'package'))  # 将 package 文件夹添加到系统路径中
import psse_dynamic as ps_d  # 必须在 numpy/matplotlib 之前 import(PSSE 与 MKL DLL 冲突)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import identification as idf
import b2b_simu as b2b
import psse_static as ps_s

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

def plot_figures(df,k,title='Dynamic Simulation Results', xlabel='Time (s)'):
    plt.rcParams['font.size'] = 8  # 设置字体大小
    fig=plt.figure(figsize=(9/2.54, 6/2.54))#
    plt.plot(df.values[:,0], df.values[:,k])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(df.columns[k])
    plt.show()

if __name__ == '__main__':

    study='IEEE39_RE'
    #df=ps_d.run_psse_simulation(disturbance_type='ambient_a', disturbance_range=500, type_option=800,study=study,total_time=50)  # 运行 PSSE 动态仿真

    df,d_para=ps_d.run_psse_simulation(disturbance_type='ambient_a', disturbance_range=500, type_option=500,study=study,total_time=10,ld_con_chng_buses=[3],ld_con_chng_values=[[-1,-1,-1,-1,-1,0.0,1.8,0.22,0.08,0.15,-1,-1,-1,-1,-1]])  # 运行 PSSE 动态仿真

    #df=ps_s.run_psse_opf(simulation_type='ambient', type_option=800,study=study)

    #df=ps_d.run_psse_simulation(disturbance_type='load_change', type_option=[12,500],study=study,total_time=30)  # 运行 PSSE 动态仿真
    df.to_pickle('data_ma.pkl')
    d_para.to_pickle('para.pkl')

    #a=b2b.obj_fun(df,para=[0.02,1.8,0.22,0.08,0.1])

    #ps_s.run_psse_power_flow(study=study)  # 运行 PSSE 静态潮流计算
    a=1

    #df=pd.read_pickle('data.pkl')  # 从文件中读取数据

    #plot_figures(df=df,k=7)  # 绘制图形

    #df.to_excel('%s_dynamic_results.xlsx'%study,sheet_name='Sheet1',float_format='%.3f')# 导出结果到Excel    

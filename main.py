# dynrun.py - 4 PSSe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psse_dynamic as ps_d
import identification as idf

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
    df=ps_d.run_psse_simulation(simulation_type='ambient', type_option=500,study=study,total_time=100)  # 运行 PSSE 动态仿真
    df.to_pickle('data_ma.pkl')

    #df=pd.read_pickle('data.pkl')  # 从文件中读取数据

    #plot_figures(df=df,k=7)  # 绘制图形

    #df.to_excel('%s_dynamic_results.xlsx'%study,sheet_name='Sheet1',float_format='%.3f')# 导出结果到Excel    
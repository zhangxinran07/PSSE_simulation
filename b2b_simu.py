import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from scipy.optimize import minimize
import statsmodels.api as sm
import math

channels=[(0, 'Time(s)'), (1, 'VOLT 3 [LOAD3 345.00]'), (2, 'ANGL 3 [LOAD3 345.00]'), (3, 'VOLT 15 [LOAD15 345.00]'), (4, 'ANGL 15 [LOAD15 345.00]'), (5, 'VOLT 24 [LOAD24 345.00]'), (6, 'ANGL 24 [LOAD24 345.00]'), (7, 'PLOD 3[LOAD3 345.00]1'), (8, 'PLOD 15[LOAD15 345.00]1'), (9, 'PLOD 24[LOAD24 345.00]1'), (10, 'QLOD 3[LOAD3 345.00]1'), (11, 'QLOD 15[LOAD15 345.00]1'), (12, 'QLOD 24[LOAD24 345.00]1'), (13, 'BUS3 LOAD MVA'), (14, 'BUS3 UD'), (15, 'BUS3 UQ'), (16, 'BUS3 LOAD BUS V'), (17, 'BUS3 LOW SIDE BUS V'), (18, 'BUS3 MA P'), (19, 'BUS3 MA Q'), (20, 'BUS3 TELE'), (21, 'BUS3 SPEED DEVIATION'), (22, 'BUS3 INIT LOAD TORQUE'), (23, 'BUS3 ID'), (24, 'BUS3 IQ'), (25, 'BUS3 MA I'), (26, 'BUS3 MA MVA'), (27, 'BUS3 TL'), (28, 'BUS3 EQ1'), (29, 'BUS3 ED1'), (30, 'BUS3 EQ2'), (31, 'BUS3 ED2'), (32, 'BUS3 SPEED DEVIATION'), (33, 'BUS3 ANGLE DEVIATION')]

def deri_cal(df):
    mva_amp=df.values[0,26]/100#容量基值放大倍数
    r,x1,x0,t0,H2=0.02/mva_amp,0.12/mva_amp,1.8/mva_amp,0.08,0.1
    u=(df.values[:,14]+1j*df.values[:,15])*np.exp(1j*((df.values[0,2]-df.values[:,2])/180*math.pi))
    #u=(df.values[:,14]+1j*df.values[:,15])
    e=df.values[:,29]+1j*df.values[:,28]
    i=(df.values[:,23]+1j*df.values[:,24])*np.exp(1j*((df.values[0,2]-df.values[:,2])/180*math.pi))
    s=-df.values[:,32]
    tm=df.values[0,22]
    P=df.values[:,18]
    Q=df.values[:,19]
    uei_diff=(u-e)+i*(r+1j*x1)
    de_diff=(-1-1j*120*math.pi*s*t0)*e-1j*(x0-x1)*i
    tele=-(e.real*i.real+e.imag*i.imag)/mva_amp
    P_p=-(u.real*i.real+u.imag*i.imag)
    Q_p=-(u.imag*i.real-u.real*i.imag)
    return u,e,tm,s,P,Q,mva_amp

def b2b_cal(u,e,tm,s,P,Q,mva_amp,dt=0.002):
    r,x1,x0,t0,H2=0.02/mva_amp,0.12/mva_amp,1.8/mva_amp,0.08,0.1
    n=len(u)
    de_p=np.zeros(n,dtype=complex)
    de=np.zeros(n,dtype=complex)
    ds=np.zeros(n)
    i=np.zeros(n,dtype=complex)
    e_p=np.zeros(n,dtype=complex)
    e_p[0]=e[0]
    s_p=np.zeros(n)
    s_p[0]=s[0]
    P_p=np.zeros(n)
    Q_p=np.zeros(n)
    P_p[0]=P[0]
    Q_p[0]=Q[0]
    for k in range(n-1):
        i[k]=(e_p[k]-u[k])/(r+1j*x1)
        de_p[k]=((-1-1j*120*math.pi*s_p[k]*t0)*e_p[k]-1j*(x0-x1)*i[k])/t0
        de[k]=(e[k+1]-e[k])/dt
        ds[k]=1/H2/2*(tm+(e_p[k].real*i[k].real+e_p[k].imag*i[k].imag)/mva_amp)
        e_p[k+1]=e_p[k]+de_p[k]*dt
        s_p[k+1]=s_p[k]+ds[k]*dt
        P_p[k+1]=-(u[k].real*i[k].real+u[k].imag*i[k].imag)
        Q_p[k+1]=-(u[k].imag*i[k].real-u[k].real*i[k].imag)
    a=1



if __name__ == '__main__':
    df=pd.read_pickle('data_ma.pkl')  # 从文件中读取数据
    u,e,tm,s,P,Q,mva_amp=deri_cal(df)
    b2b_cal(u,e,tm,s,P,Q,mva_amp)
    a=1
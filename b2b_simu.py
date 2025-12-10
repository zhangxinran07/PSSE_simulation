import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from scipy.optimize import minimize
import statsmodels.api as sm
import math

channels=[(0, 'Time(s)'), (1, 'VOLT 3 [LOAD3 345.00]'), (2, 'ANGL 3 [LOAD3 345.00]'), (3, 'VOLT 15 [LOAD15 345.00]'), (4, 'ANGL 15 [LOAD15 345.00]'), (5, 'VOLT 24 [LOAD24 345.00]'), (6, 'ANGL 24 [LOAD24 345.00]'), (7, 'VOLT 39 [GEN39 345.00]'), (8, 'ANGL 39 [GEN39 345.00]'), (9, 'PLOD 3[LOAD3 345.00]1'), (10, 'PLOD 15[LOAD15 345.00]1'), (11, 'PLOD 24[LOAD24 345.00]1'), (12, 'PLOD 39[GEN39 345.00]1'), (13, 'QLOD 3[LOAD3 345.00]1'), (14, 'QLOD 15[LOAD15 345.00]1'), (15, 'QLOD 24[LOAD24 345.00]1'), (16, 'QLOD 39[GEN39 345.00]1'), (17, 'BUS3 LOAD MVA'), (18, 'BUS3 UD'), (19, 'BUS3 UQ'), (20, 'BUS3 LOAD BUS V'), (21, 'BUS3 LOW SIDE BUS V'), (22, 'BUS3 MA P'), (23, 'BUS3 MA Q'), (24, 'BUS3 TELE'), (25, 'BUS3 SPEED DEVIATION'), (26, 'BUS3 INIT LOAD TORQUE'), (27, 'BUS3 ID'), (28, 'BUS3 IQ'), (29, 'BUS3 MA I'), (30, 'BUS3 MA MVA'), (31, 'BUS3 TL'), (32, 'BUS3 EQ1'), (33, 'BUS3 ED1'), (34, 'BUS3 EQ2'), (35, 'BUS3 ED2'), (36, 'BUS3 SPEED DEVIATION'), (37, 'BUS3 ANGLE DEVIATION')]

def deri_cal(df):
    index=[18,19,2,27,28,22,23,30,33,32,26]#Bus3 UD,UQ,angle,ID,IQ,MA P,MA Q,MA MVA,ED1,EQ1,init load torque
    mva_amp=df.values[0,30]/100#容量基值放大倍数
    r,x1,x0,t0,H2=0.02/mva_amp,0.12/mva_amp,1.8/mva_amp,0.08,0.1
    ang=(df.values[0,2]-df.values[:,2])/180*math.pi

    #u=(df.values[:,18]+1j*df.values[:,19])*np.exp(1j*((df.values[0,2]-df.values[:,2])/180*math.pi))
    u=(df.values[:,18]+1j*df.values[:,19])*np.exp(1j*ang)
    e=df.values[:,33]+1j*df.values[:,32]
    i=(df.values[:,27]+1j*df.values[:,28])*np.exp(1j*ang)
    s=-df.values[:,36]
    tm=df.values[0,26]
    P=df.values[:,22]
    Q=df.values[:,23]
    uei_diff=(u-e)+i*(r+1j*x1)
    de_diff=((-1-1j*120*math.pi*s*t0)*e-1j*(x0-x1)*i)/t0
    de=np.zeros(len(e)-1,dtype=complex)
    #tele=-(e.real*i.real+e.imag*i.imag)/mva_amp
    tele=df.values[:,24]
    P_p=-(u.real*i.real+u.imag*i.imag)
    Q_p=-(u.imag*i.real-u.real*i.imag)
    return u,e,tm,s,P,Q,mva_amp,i

def b2b_cal(u,P,Q,mva_amp,para,dt=0.002):
    r,x1,x0,t0,H2=para[0]/mva_amp,para[1]/mva_amp,para[2]/mva_amp,para[3],para[4]
    n=len(u)
    de_p=np.zeros(n,dtype=complex)
    ds=np.zeros(n)
    tele=np.zeros(n)
    i_p=np.zeros(n,dtype=complex)
    e_p=np.zeros(n,dtype=complex)
    e_p[0]=u[0]+i[0]*(r+1j*x1)
    s_p=np.zeros(n)
    s_p[0]=abs(((x1-x0)*i[0]+1j*e[0])/120/math.pi/t0/e_p[0])
    P_p=np.zeros(n)
    Q_p=np.zeros(n)
    P_p[0]=P[0]
    Q_p[0]=Q[0]
    i_p[0]=(e_p[0]-u[0])/(r+1j*x1)
    tm=-(e_p[0].real*i_p[0].real+e_p[0].imag*i_p[0].imag)/mva_amp
    #tm=P[0]/mva_amp
    for k in range(n-1):
        i_p[k]=(e_p[k]-u[k])/(r+1j*x1)
        de_p[k]=((-1-1j*120*math.pi*s_p[k]*t0)*e_p[k]-1j*(x0-x1)*i_p[k])/t0
        tele[k]=-(e_p[k].real*i_p[k].real+e_p[k].imag*i_p[k].imag)/mva_amp
        ds[k]=1/H2/2*(tm-tele[k])
        e_p[k+1]=e_p[k]+de_p[k]*dt
        s_p[k+1]=s_p[k]+ds[k]*dt
        P_p[k+1]=-(u[k].real*i_p[k].real+u[k].imag*i_p[k].imag)
        Q_p[k+1]=-(u[k].imag*i_p[k].real-u[k].real*i_p[k].imag)
    return sum((P_p-P)**2)+sum((Q_p-Q)**2)



if __name__ == '__main__':
    df=pd.read_pickle('data_ma.pkl')  # 从文件中读取数据
    u,e,tm,s,P,Q,mva_amp,i=deri_cal(df)
    para=[0.02,0.22,2.8,0.08,0.1]
    a=b2b_cal(u,P,Q,mva_amp,para)
    a=1
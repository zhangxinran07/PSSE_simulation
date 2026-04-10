import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from scipy.optimize import minimize
import statsmodels.api as sm
import math

busid=[3,4,15,20,21,24,25,27,28]

channels=[('time', 'Time(s)'), (1, 'VOLT 3 [LOAD3 345.00]'), (2, 'ANGL 3 [LOAD3 345.00]'), (3, 'VOLT 4 [LOAD4 345.00]'), (4, 'ANGL 4 [LOAD4 345.00]'), (5, 'VOLT 15 [LOAD15 345.00]'), (6, 'ANGL 15 [LOAD15 345.00]'), (7, 'VOLT 20 [LOAD20 138.00]'), (8, 'ANGL 20 [LOAD20 138.00]'), (9, 'VOLT 21 [LOAD21 345.00]'), (10, 'ANGL 21 [LOAD21 345.00]'), (11, 'VOLT 24 [LOAD24 345.00]'), (12, 'ANGL 24 [LOAD24 345.00]'), (13, 'VOLT 25 [LOAD25 345.00]'), (14, 'ANGL 25 [LOAD25 345.00]'), (15, 'VOLT 27 [LOAD27 345.00]'), (16, 'ANGL 27 [LOAD27 345.00]'), (17, 'VOLT 28 [LOAD28 345.00]'), (18, 'ANGL 28 [LOAD28 345.00]'), (19, 'VOLT 39 [GEN39 345.00]'), (20, 'ANGL 39 [GEN39 345.00]'), (21, 'PLOD 3[LOAD3 345.00]1'), (22, 'PLOD 4[LOAD4 345.00]1'), (23, 'PLOD 15[LOAD15 345.00]1'), (24, 'PLOD 20[LOAD20 138.00]1'), (25, 'PLOD 21[LOAD21 345.00]1'), (26, 'PLOD 24[LOAD24 345.00]1'), (27, 'PLOD 25[LOAD25 345.00]1'), (28, 'PLOD 27[LOAD27 345.00]1'), (29, 'PLOD 28[LOAD28 345.00]1'), (30, 'PLOD 39[GEN39 345.00]1'), (31, 'QLOD 3[LOAD3 345.00]1'), (32, 'QLOD 4[LOAD4 345.00]1'), (33, 'QLOD 15[LOAD15 345.00]1'), (34, 'QLOD 20[LOAD20 138.00]1'), (35, 'QLOD 21[LOAD21 345.00]1'), (36, 'QLOD 24[LOAD24 345.00]1'), (37, 'QLOD 25[LOAD25 345.00]1'), (38, 'QLOD 27[LOAD27 345.00]1'), (39, 'QLOD 28[LOAD28 345.00]1'), (40, 'QLOD 39[GEN39 345.00]1'), (41, 'BUS3 LOAD MVA'), (42, 'BUS3 UD'), (43, 'BUS3 UQ'), (44, 'BUS3 MA P'), (45, 'BUS3 MA Q'), (46, 'BUS3 MB P'), (47, 'BUS3 MB Q'), (48, 'BUS3 MA INIT LOAD TORQUE'), (49, 'BUS3 MA ID'), (50, 'BUS3 MA IQ'), (51, 'BUS3 MA MVA'), (52, 'BUS3 MB INIT LOAD TORQUE'), (53, 'BUS3 MB ID'), (54, 'BUS3 MB IQ'), (55, 'BUS3 MB MVA'), (56, 'BUS3 MA EQ1'), (57, 'BUS3 MA ED1'), (58, 'BUS3 MA SPEED DEVIATION'), (59, 'BUS3 MB EQ1'), (60, 'BUS3 MB ED1'), (61, 'BUS3 MB SPEED DEVIATION'), (62, 'BUS4 LOAD MVA'), (63, 'BUS4 UD'), (64, 'BUS4 UQ'), (65, 'BUS4 MA P'), (66, 'BUS4 MA Q'), (67, 'BUS4 MB P'), (68, 'BUS4 MB Q'), (69, 'BUS4 MA INIT LOAD TORQUE'), (70, 'BUS4 MA ID'), (71, 'BUS4 MA IQ'), (72, 'BUS4 MA MVA'), (73, 'BUS4 MB INIT LOAD TORQUE'), (74, 'BUS4 MB ID'), (75, 'BUS4 MB IQ'), (76, 'BUS4 MB MVA'), (77, 'BUS4 MA EQ1'), (78, 'BUS4 MA ED1'), (79, 'BUS4 MA SPEED DEVIATION'), (80, 'BUS4 MB EQ1'), (81, 'BUS4 MB ED1'), (82, 'BUS4 MB SPEED DEVIATION'), (83, 'BUS15 LOAD MVA'), (84, 'BUS15 UD'), (85, 'BUS15 UQ'), (86, 'BUS15 MA P'), (87, 'BUS15 MA Q'), (88, 'BUS15 MB P'), (89, 'BUS15 MB Q'), (90, 'BUS15 MA INIT LOAD TORQUE'), (91, 'BUS15 MA ID'), (92, 'BUS15 MA IQ'), (93, 'BUS15 MA MVA'), (94, 'BUS15 MB INIT LOAD TORQUE'), (95, 'BUS15 MB ID'), (96, 'BUS15 MB IQ'), (97, 'BUS15 MB MVA'), (98, 'BUS15 MA EQ1'), (99, 'BUS15 MA ED1'), (100, 'BUS15 MA SPEED DEVIATION'), (101, 'BUS15 MB EQ1'), (102, 'BUS15 MB ED1'), (103, 'BUS15 MB SPEED DEVIATION'), (104, 'BUS20 LOAD MVA'), (105, 'BUS20 UD'), (106, 'BUS20 UQ'), (107, 'BUS20 MA P'), (108, 'BUS20 MA Q'), (109, 'BUS20 MB P'), (110, 'BUS20 MB Q'), (111, 'BUS20 MA INIT LOAD TORQUE'), (112, 'BUS20 MA ID'), (113, 'BUS20 MA IQ'), (114, 'BUS20 MA MVA'), (115, 'BUS20 MB INIT LOAD TORQUE'), (116, 'BUS20 MB ID'), (117, 'BUS20 MB IQ'), (118, 'BUS20 MB MVA'), (119, 'BUS20 MA EQ1'), (120, 'BUS20 MA ED1'), (121, 'BUS20 MA SPEED DEVIATION'), (122, 'BUS20 MB EQ1'), (123, 'BUS20 MB ED1'), (124, 'BUS20 MB SPEED DEVIATION'), (125, 'BUS21 LOAD MVA'), (126, 'BUS21 UD'), (127, 'BUS21 UQ'), (128, 'BUS21 MA P'), (129, 'BUS21 MA Q'), (130, 'BUS21 MB P'), (131, 'BUS21 MB Q'), (132, 'BUS21 MA INIT LOAD TORQUE'), (133, 'BUS21 MA ID'), (134, 'BUS21 MA IQ'), (135, 'BUS21 MA MVA'), (136, 'BUS21 MB INIT LOAD TORQUE'), (137, 'BUS21 MB ID'), (138, 'BUS21 MB IQ'), (139, 'BUS21 MB MVA'), (140, 'BUS21 MA EQ1'), (141, 'BUS21 MA ED1'), (142, 'BUS21 MA SPEED DEVIATION'), (143, 'BUS21 MB EQ1'), (144, 'BUS21 MB ED1'), (145, 'BUS21 MB SPEED DEVIATION'), (146, 'BUS24 LOAD MVA'), (147, 'BUS24 UD'), (148, 'BUS24 UQ'), (149, 'BUS24 MA P'), (150, 'BUS24 MA Q'), (151, 'BUS24 MB P'), (152, 'BUS24 MB Q'), (153, 'BUS24 MA INIT LOAD TORQUE'), (154, 'BUS24 MA ID'), (155, 'BUS24 MA IQ'), (156, 'BUS24 MA MVA'), (157, 'BUS24 MB INIT LOAD TORQUE'), (158, 'BUS24 MB ID'), (159, 'BUS24 MB IQ'), (160, 'BUS24 MB MVA'), (161, 'BUS24 MA EQ1'), (162, 'BUS24 MA ED1'), (163, 'BUS24 MA SPEED DEVIATION'), (164, 'BUS24 MB EQ1'), (165, 'BUS24 MB ED1'), (166, 'BUS24 MB SPEED DEVIATION'), (167, 'BUS25 LOAD MVA'), (168, 'BUS25 UD'), (169, 'BUS25 UQ'), (170, 'BUS25 MA P'), (171, 'BUS25 MA Q'), (172, 'BUS25 MB P'), (173, 'BUS25 MB Q'), (174, 'BUS25 MA INIT LOAD TORQUE'), (175, 'BUS25 MA ID'), (176, 'BUS25 MA IQ'), (177, 'BUS25 MA MVA'), (178, 'BUS25 MB INIT LOAD TORQUE'), (179, 'BUS25 MB ID'), (180, 'BUS25 MB IQ'), (181, 'BUS25 MB MVA'), (182, 'BUS25 MA EQ1'), (183, 'BUS25 MA ED1'), (184, 'BUS25 MA SPEED DEVIATION'), (185, 'BUS25 MB EQ1'), (186, 'BUS25 MB ED1'), (187, 'BUS25 MB SPEED DEVIATION'), (188, 'BUS27 LOAD MVA'), (189, 'BUS27 UD'), (190, 'BUS27 UQ'), (191, 'BUS27 MA P'), (192, 'BUS27 MA Q'), (193, 'BUS27 MB P'), (194, 'BUS27 MB Q'), (195, 'BUS27 MA INIT LOAD TORQUE'), (196, 'BUS27 MA ID'), (197, 'BUS27 MA IQ'), (198, 'BUS27 MA MVA'), (199, 'BUS27 MB INIT LOAD TORQUE'), (200, 'BUS27 MB ID'), (201, 'BUS27 MB IQ'), (202, 'BUS27 MB MVA'), (203, 'BUS27 MA EQ1'), (204, 'BUS27 MA ED1'), (205, 'BUS27 MA SPEED DEVIATION'), (206, 'BUS27 MB EQ1'), (207, 'BUS27 MB ED1'), (208, 'BUS27 MB SPEED DEVIATION'), (209, 'BUS28 LOAD MVA'), (210, 'BUS28 UD'), (211, 'BUS28 UQ'), (212, 'BUS28 MA P'), (213, 'BUS28 MA Q'), (214, 'BUS28 MB P'), (215, 'BUS28 MB Q'), (216, 'BUS28 MA INIT LOAD TORQUE'), (217, 'BUS28 MA ID'), (218, 'BUS28 MA IQ'), (219, 'BUS28 MA MVA'), (220, 'BUS28 MB INIT LOAD TORQUE'), (221, 'BUS28 MB ID'), (222, 'BUS28 MB IQ'), (223, 'BUS28 MB MVA'), (224, 'BUS28 MA EQ1'), (225, 'BUS28 MA ED1'), (226, 'BUS28 MA SPEED DEVIATION'), (227, 'BUS28 MB EQ1'), (228, 'BUS28 MB ED1'), (229, 'BUS28 MB SPEED DEVIATION')]

def deri_cal_con(df,busid,para):
    key=["UD","UQ","MA EQ1","MA ED1","MA ID","MA IQ","MA INIT LOAD TORQUE","MA SPEED DEVIATION","MA P","MA Q","MA MVA"]
    index=[]
    for k in key:
        index+=[i for i in range(185) if "BUS%d" % busid in channels[i][1] and k in channels[i][1]]
    index+=[i for i in range(185) if " %d " % busid in channels[i][1]]
    key+=["VOLT","ANGL"]    
    i_dic=dict(zip(key,index))

    mva_amp=df.values[0,i_dic['MB MVA']]/100#容量基值放大倍数
    r,x0,x1,t0,H2=para[0]/mva_amp,para[1]/mva_amp,para[2]/mva_amp,para[3],para[4]
    ang=(df.values[0,i_dic['ANGL']]-df.values[:,i_dic['ANGL']])/180*math.pi

    #u=(df.values[:,18]+1j*df.values[:,19])*np.exp(1j*((df.values[0,2]-df.values[:,2])/180*math.pi))
    u=(df.values[:,i_dic['UD']]+1j*df.values[:,i_dic['UQ']])*np.exp(1j*ang)
    e=df.values[:,i_dic['MA ED1']]+1j*df.values[:,i_dic['MA EQ1']]
    i=(df.values[:,i_dic['MA ID']]+1j*df.values[:,i_dic['MA IQ']])*np.exp(1j*ang)
    s=-df.values[:,i_dic['MA SPEED DEVIATION']]
    tm=df.values[0,i_dic['MA INIT LOAD TORQUE']]
    P=df.values[:,i_dic['MA P']]
    Q=df.values[:,i_dic['MA Q']]
    uei_diff=(u-e)+i*(r+1j*x1)
    de_diff=((-1-1j*120*math.pi*s*t0)*e-1j*(x0-x1)*i)/t0
    de=np.zeros(len(e)-1,dtype=complex)
    #tele=-(e.real*i.real+e.imag*i.imag)/mva_amp
    tele=df.values[:,24]
    P_p=-(u.real*i.real+u.imag*i.imag)
    Q_p=-(u.imag*i.real-u.real*i.imag)
    return u,e,tm,s,P,Q,mva_amp,i

def deri_cal_var(df,busid,para):
    #index=[18,19,2,27,28,22,23,30,33,32,26]#Bus3 UD,UQ,angle,ID,IQ,MA P,MA Q,MA MVA,ED1,EQ1,init load torque
    key=["UD","UQ","MB EQ1","MB ED1","MB ID","MB IQ","MB INIT LOAD TORQUE","MB SPEED DEVIATION","MB P","MB Q","MB MVA"]
    index=[]
    for k in key:
        index+=[i for i in range(185) if "BUS%d" % busid in channels[i][1] and k in channels[i][1]]
    index+=[i for i in range(185) if " %d " % busid in channels[i][1]]
    key+=["VOLT","ANGL"]    
    i_dic=dict(zip(key,index))

    mva_amp=df.values[0,i_dic['MB MVA']]/100#容量基值放大倍数
    r,x0,x1,t0,H2=para[0]/mva_amp,para[1]/mva_amp,para[2]/mva_amp,para[3],para[4]
    ang=(df.values[0,i_dic['ANGL']]-df.values[:,i_dic['ANGL']])/180*math.pi

    #u=(df.values[:,18]+1j*df.values[:,19])*np.exp(1j*((df.values[0,2]-df.values[:,2])/180*math.pi))
    u=(df.values[:,i_dic['UD']]+1j*df.values[:,i_dic['UQ']])*np.exp(1j*ang)
    e=df.values[:,i_dic['MB ED1']]+1j*df.values[:,i_dic['MB EQ1']]
    i=(df.values[:,i_dic['MB ID']]+1j*df.values[:,i_dic['MB IQ']])*np.exp(1j*ang)
    s=-df.values[:,i_dic['MB SPEED DEVIATION']]
    tm=df.values[0,i_dic['MB INIT LOAD TORQUE']]
    P=df.values[:,i_dic['MB P']]
    Q=df.values[:,i_dic['MB Q']]
    uei_diff=(u-e)+i*(r+1j*x1)
    de_diff=((-1-1j*120*math.pi*s*t0)*e-1j*(x0-x1)*i)/t0
    de=np.zeros(len(e)-1,dtype=complex)
    #tele=-(e.real*i.real+e.imag*i.imag)/mva_amp
    tele=df.values[:,38]
    P_p=-(u.real*i.real+u.imag*i.imag)
    Q_p=-(u.imag*i.real-u.real*i.imag)
    return u,e,tm,s,P,Q,mva_amp,i

def obj_fun_con_ori(df,para,busid,dt=0.002):
    key=["UD","UQ","MA EQ1","MA ED1","MA ID","MA IQ","MA INIT LOAD TORQUE","MA SPEED DEVIATION","MA P","MA Q","MA MVA"]
    index=[]
    for k in key:
        index+=[i for i in range(185) if "BUS%d" % busid in channels[i][1] and k in channels[i][1]]
    index+=[i for i in range(185) if " %d " % busid in channels[i][1]]
    key+=["VOLT","ANGL"]    
    i_dic=dict(zip(key,index))

    mva_amp=df.values[0,i_dic['MA MVA']]/100#容量基值放大倍数
    r,x0,x1,t0,H2=para[0]/mva_amp,para[1]/mva_amp,para[2]/mva_amp,para[3],para[4]
    ang=(df.values[0,i_dic['ANGL']]-df.values[:,i_dic['ANGL']])/180*math.pi
    u=(df.values[:,i_dic['UD']]+1j*df.values[:,i_dic['UQ']])*np.exp(1j*ang)
    P=df.values[:,i_dic['MA P']]
    Q=df.values[:,i_dic['MA Q']]

    n=len(u)
    de_p=np.zeros(n,dtype=complex)
    ds=np.zeros(n)
    tele=np.zeros(n)
    i_p=np.zeros(n,dtype=complex)
    e_p=np.zeros(n,dtype=complex)
    e_p[0]=u[0]-((P[0]+1j*Q[0])/(u[0])).conj()*(r+1j*x1)
    i_p[0]=(e_p[0]-u[0])/(r+1j*x1)
    s_p=np.zeros(n)
    s_p[0]=abs(((x1-x0)*i_p[0]+1j*e_p[0])/120/math.pi/t0/e_p[0])
    P_p=np.zeros(n)
    Q_p=np.zeros(n)
    P_p[0]=P[0]
    Q_p[0]=Q[0]
    tm=-(e_p[0].real*i_p[0].real+e_p[0].imag*i_p[0].imag)/mva_amp
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

def obj_fun_var_ori(df,para,busid,dt=0.002):
    key=["UD","UQ","MB EQ1","MB ED1","MB ID","MB IQ","MB INIT LOAD TORQUE","MB SPEED DEVIATION","MB P","MB Q","MB MVA"]
    index=[]
    for k in key:
        index+=[i for i in range(185) if "BUS%d" % busid in channels[i][1] and k in channels[i][1]]
    index+=[i for i in range(185) if " %d " % busid in channels[i][1]]
    key+=["VOLT","ANGL"]    
    i_dic=dict(zip(key,index))

    mva_amp=df.values[0,i_dic['MB MVA']]/100#容量基值放大倍数
    r,x0,x1,t0,H2=para[0]/mva_amp,para[1]/mva_amp,para[2]/mva_amp,para[3],para[4]
    ang=(df.values[0,i_dic['ANGL']]-df.values[:,i_dic['ANGL']])/180*math.pi
    u=(df.values[:,i_dic['UD']]+1j*df.values[:,i_dic['UQ']])*np.exp(1j*ang)
    P=df.values[:,i_dic['MB P']]
    Q=df.values[:,i_dic['MB Q']]

    n=len(u)
    de_p=np.zeros(n,dtype=complex)
    ds=np.zeros(n)
    tele=np.zeros(n)
    i_p=np.zeros(n,dtype=complex)
    e_p=np.zeros(n,dtype=complex)
    e_p[0]=u[0]-((P[0]+1j*Q[0])/(u[0])).conj()*(r+1j*x1)
    i_p[0]=(e_p[0]-u[0])/(r+1j*x1)
    s_p=np.zeros(n)
    s_p[0]=abs(((x1-x0)*i_p[0]+1j*e_p[0])/120/math.pi/t0/e_p[0])
    P_p=np.zeros(n)
    Q_p=np.zeros(n)
    P_p[0]=P[0]
    Q_p[0]=Q[0]
    tm=-(e_p[0].real*i_p[0].real+e_p[0].imag*i_p[0].imag)/mva_amp
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
    d_para=pd.read_pickle('para.pkl')
    busid=3
    para=[d_para.loc[busid,'ra'],d_para.loc[busid,'xa'],d_para.loc[busid,'x1a'],d_para.loc[busid,'tda'],d_para.loc[busid,'ha']]
    #u,e,tm,s,P,Q,mva_amp,i=deri_cal_con(df,3,para)
    y=obj_fun_con_ori(df,para,busid)
    a=1
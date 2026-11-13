import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import differential_evolution as de
import statsmodels.api as sm
import math
from scipy.signal import butter, filtfilt

busid=[3,4,15,20,21,24,25,27,28]

def regress(V,P,Q):#多元回归，返回偏差值和回归结果,用ZIP和指数模型比较，选更好的
    X=np.column_stack((np.ones((len(V),1)),V,V**2))
    modelP=sm.OLS(P,X).fit()
    X=np.column_stack((np.ones((len(V),1)),V,V**2))
    modelQ=sm.OLS(Q,X).fit()
    #return [modelP.params,modelQ.params],np.sum(np.square(modelP.resid))+10*np.sum(np.square(modelQ.resid))
    return [modelP.params,modelQ.params],np.sum(np.square(modelP.resid))+np.sum(np.square(modelQ.resid))

def deri_cal_con(df,para,busid):
    key=["UD","UQ","MA EQ1","MA ED1","MA ID","MA IQ","MA INIT LOAD TORQUE","MA SPEED DEVIATION","MA P","MA Q","MA MVA"]
    index=[]
    for k in key:
        index+=[i for i in range(len(df.columns)) if "BUS%d" % busid in df.columns[i] and k in df.columns[i]]
    index+=[i for i in range(len(df.columns)) if " %d " % busid in df.columns[i]]
    key+=["VOLT","ANGL"]    
    i_dic=dict(zip(key,index))

    mva_amp=df.values[0,i_dic['MA MVA']]/100#容量基值放大倍数
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

def deri_cal_var(df,para,busid):
    #index=[18,19,2,27,28,22,23,30,33,32,26]#Bus3 UD,UQ,angle,ID,IQ,MA P,MA Q,MA MVA,ED1,EQ1,init load torque
    key=["UD","UQ","MB EQ1","MB ED1","MB ID","MB IQ","MB INIT LOAD TORQUE","MB TL","MB TELE","MB SPEED DEVIATION","MB P","MB Q","MB MVA"]
    index=[]
    for k in key:
        index+=[i for i in range(len(df.columns)) if "BUS%d" % busid in df.columns[i] and k in df.columns[i]]
    index+=[i for i in range(len(df.columns)) if " %d " % busid in df.columns[i]]
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
    t1=df.values[:,i_dic['MB TL']]
    t2=df.values[:,i_dic['MB TELE']]
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

def obj_con_iden(para,df,busid,dt=0.01):
    key=["UD","UQ","MA P","MA Q","S P","S Q"]
    index=[]
    for k in key:
        index+=[i for i in range(len(df.columns)) if "BUS%d" % busid in df.columns[i] and k in df.columns[i]]
    index+=[i for i in range(len(df.columns)) if " %d " % busid in df.columns[i]]
    index+=[i for i in range(len(df.columns)) if "PLOD %d[" % busid in df.columns[i]]
    index+=[i for i in range(len(df.columns)) if "QLOD %d[" % busid in df.columns[i]]
    key+=["VOLT","ANGL","P","Q"]    
    i_dic=dict(zip(key,index))

    a_c,b_c,H2_c,pp_c=para[0],para[1],para[2],para[3]
        
    u=(df.values[:,i_dic['UD']]+1j*df.values[:,i_dic['UQ']])
    P=df.values[:,i_dic['P']]
    Q=df.values[:,i_dic['Q']]

    n=len(u)
    f_p=np.zeros(n,dtype=complex)
    s_p=np.zeros(n)
    w0=2*math.pi*60
    P_p=np.zeros(n)
    Q_p=np.zeros(n)
    P_p[0]=P[0]*pp_c
    c=abs(u[0])**2*a_c*w0/P_p[0]
    delta=c**2-4*b_c**2*w0**2
    if delta<0:
        return 1e16*abs(delta)
    else:
        s_p[0]=(c-math.sqrt(delta))/2/w0**2
    f_p[0]=a_c*u[0]/(b_c+1j*s_p[0]*w0)
    
    Q_p[0]=-(f_p[0].real*u[0].real+f_p[0].imag*u[0].imag)
    tm=P_p[0]
    
    for k in range(n-1):
        try:
            np.seterr(over='ignore', invalid='ignore')
            uk = u[k]
            uk1 = u[k + 1]

            if k >= 1 and k + 2 < n:
                umid = (-u[k - 1] + 9 * u[k] + 9 * u[k + 1] - u[k + 2]) / 16
            else:
                umid = 0.5 * (u[k] + u[k + 1])  #测试4点插值

            # k1
            k1_f = (-b_c - 1j * s_p[k] * w0) * f_p[k] + a_c * uk
            Pe1 = f_p[k].real * uk.imag - f_p[k].imag * uk.real
            k1_s = (tm - Pe1) / (2 * H2_c)

            # k2
            f2 = f_p[k] + 0.5 * dt * k1_f
            s2 = s_p[k] + 0.5 * dt * k1_s
            k2_f = (-b_c - 1j * s2 * w0) * f2 + a_c * umid
            Pe2 = f2.real * umid.imag - f2.imag * umid.real
            k2_s = (tm - Pe2) / (2 * H2_c)

            # k3
            f3 = f_p[k] + 0.5 * dt * k2_f
            s3 = s_p[k] + 0.5 * dt * k2_s
            k3_f = (-b_c - 1j * s3 * w0) * f3 + a_c * umid
            Pe3 = f3.real * umid.imag - f3.imag * umid.real
            k3_s = (tm - Pe3) / (2 * H2_c)

            # k4
            f4 = f_p[k] + dt * k3_f
            s4 = s_p[k] + dt * k3_s
            k4_f = (-b_c - 1j * s4 * w0) * f4 + a_c * uk1
            Pe4 = f4.real * uk1.imag - f4.imag * uk1.real
            k4_s = (tm - Pe4) / (2 * H2_c)

            f_p[k + 1] = f_p[k] + dt / 6 * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)
            s_p[k + 1] = s_p[k] + dt / 6 * (k1_s + 2 * k2_s + 2 * k3_s + k4_s)

        except RuntimeWarning:
            print("Error in iteration %d" % k)
            return 1e16*abs(delta)
    P_p=(f_p.real*u.imag-f_p.imag*u.real)
    Q_p=-(f_p.real*u.real+f_p.imag*u.imag)
    if not (s_p[n-1]>0 and s_p[n-1]<0.2):
        return 1e16*abs(delta)

    return math.sqrt(regress(abs(u),P_p-P,Q_p-Q)[1])/n*1000

if __name__ == '__main__':
    df=pd.read_pickle('data_ma.pkl')  # 从文件中读取数据
    d_para=pd.read_pickle('para.pkl')
    #busid=[3,4,15,20,21,24,25,27,28]
    busid=3
    type='a'
    if type=='a':
        mva_amp=d_para.loc[busid,'mva']
        
        # r,x0,x1,t0,h2=d_para.loc[busid,'ra'],d_para.loc[busid,'xa'],d_para.loc[busid,'x1a'],d_para.loc[busid,'tda'],d_para.loc[busid,'ha']
        # obj_fun_con_ori(df,[r,x0,x1,t0,h2],busid)
        # obj_con_sim(df,[r,x0,x1,t0,h2],busid)
        [r,x0,x1,t0,h2,mva_amp,pp]=d_para.loc[busid,['ra','xa','x1a','tda','ha','mva','pa']]
        a=(x0/x1-1)/t0/x1*mva_amp
        b=x0/x1/t0
        h2=h2*mva_amp

        
        obj_con_iden([a,b,h2,pp],df,busid)
        
        result_all=[]
        for i in range(100):
            i
            result=de(obj_con_iden, bounds=[(2000,3500),(90,180),(0.3,1),(0.2,0.8)], args=(df,busid,0.01),disp=False,popsize=30)
            result_all.append(result.x)
            print(result.x)
        a=1
        #y=obj_con_iden([a,b,h2,pp],df,busid)
    elif type=='b':
        mva_amp=d_para.loc[busid,'mvb']
        r,x0,x1,t0,h2=d_para.loc[busid,'rb']/mva_amp,d_para.loc[busid,'xb']/mva_amp,d_para.loc[busid,'x1b']/mva_amp,d_para.loc[busid,'tdb'],d_para.loc[busid,'hb']*mva_amp
        a=(x0/x1-1)/t0/x1
        b=x0/x1/t0
        pp=d_para.loc[busid,'pb']
        #y=obj_var_sim(df,[a,b,h2,pp],busid)
    elif type=='c':
        mva_amp=d_para.loc[busid,'mvc']
        r,x0,x1,t0,h2=d_para.loc[busid,'rc']/mva_amp,d_para.loc[busid,'xc']/mva_amp,d_para.loc[busid,'x1c']/mva_amp,d_para.loc[busid,'tdc'],d_para.loc[busid,'hc']*mva_amp
        a=(x0/x1-1)/t0/x1
        b=x0/x1/t0
        pp=d_para.loc[busid,'pc']
        #y=obj_var_sim(df,[a,b,h2,pp],busid)
    #u,e,tm,s,P,Q,mva_amp,i=deri_cal_var(df,para,busid)
    a=1
    
def obj_fun_con_ori(df,para,busid,dt=0.01):
    key=["UD","UQ","MA EQ1","MA ED1","MA ID","MA IQ","MA INIT LOAD TORQUE","MA SPEED DEVIATION","MA P","MA Q","MA MVA"]
    index=[]
    for k in key:
        index+=[i for i in range(len(df.columns)) if "BUS%d" % busid in df.columns[i] and k in df.columns[i]]
    index+=[i for i in range(len(df.columns)) if " %d " % busid in df.columns[i]]
    key+=["VOLT","ANGL"]    
    i_dic=dict(zip(key,index))

    mva_amp=df.values[0,i_dic['MA MVA']]/100#容量基值放大倍数
    r,x0,x1,t0,H2=para[0]/mva_amp,para[1]/mva_amp,para[2]/mva_amp,para[3],para[4]
    ang=(df.values[0,i_dic['ANGL']]-df.values[:,i_dic['ANGL']])/180*math.pi
    u=(df.values[:,i_dic['UD']]+1j*df.values[:,i_dic['UQ']])
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
    
    nyq=1/dt/2#奈奎斯特频率
    normal_cutoff=10/nyq#归一化截止频率
    b, a = butter(5, normal_cutoff, btype='low', analog=False)  # 5阶低通滤波器
    P_p=filtfilt(b, a, P_p)  # 滤波
    Q_p=filtfilt(b, a, Q_p)  # 滤波
    
    return (sum((P_p-P)**2)+sum((Q_p-Q)**2))/n

def obj_con_sim(df,para,busid,dt=0.01):
    key=["UD","UQ","MA EQ1","MA ED1","MA ID","MA IQ","MA INIT LOAD TORQUE","MA SPEED DEVIATION","MA P","MA Q","MA MVA"]
    index=[]
    for k in key:
        index+=[i for i in range(len(df.columns)) if "BUS%d" % busid in df.columns[i] and k in df.columns[i]]
    index+=[i for i in range(len(df.columns)) if " %d " % busid in df.columns[i]]
    key+=["VOLT","ANGL"]    
    i_dic=dict(zip(key,index))

    mva_amp=df.values[0,i_dic['MA MVA']]/100#容量基值放大倍数
    if len(para)>4:
        r,x0,x1,t0,H2_c=para[0]/mva_amp,para[1]/mva_amp,para[2]/mva_amp,para[3],para[4]*mva_amp
        a_c=(x0/x1-1)/t0/x1
        b_c=x0/x1/t0
    else:
        a_c,b_c,H2_c,x1=para[0],para[1],para[2],para[3]
    u=(df.values[:,i_dic['UD']]+1j*df.values[:,i_dic['UQ']])
    P=df.values[:,i_dic['MA P']]
    Q=df.values[:,i_dic['MA Q']]

    n=len(u)
    f_p=np.zeros(n,dtype=complex)
    f_p2=np.zeros(n,dtype=complex)
    s_p=np.zeros(n)
    w0=2*math.pi*60
    c=abs(u[0])**2*a_c*w0/P[0]
    s_p[0]=(c-math.sqrt(c**2-4*b_c**2*w0**2))/2/w0**2
    f_p[0]=a_c*u[0]/(b_c+1j*s_p[0]*w0)
    f_p2[0]=a_c*u[0]/(b_c+1j*s_p[0]*w0)
    P_p=np.zeros(n)
    Q_p=np.zeros(n)
    P_p[0]=P[0]
    Q_p[0]=-(f_p[0].real*u[0].real+f_p[0].imag*u[0].imag)
    tm=P[0]
    
    for k in range(n-1):
        df1=(-b_c-1j*s_p[k]*w0)*f_p[k]+a_c*u[k]#f(xn,yn)
        ds1=1/H2_c/2*(tm-f_p[k].real*u[k].imag+f_p[k].imag*u[k].real)
        f1=f_p[k]+df1*dt#yn+1/2*k1*dt
        s1=s_p[k]+ds1*dt
        df2=(-b_c-1j*s1*w0)*f1+a_c*u[k+1]
        ds2=1/H2_c/2*(tm-f1.real*u[k+1].imag+f1.imag*u[k+1].real)
        f_p[k+1]=f_p[k]+(df1+df2)*dt/2
        s_p[k+1]=s_p[k]+(ds1+ds2)*dt/2
    
    P_p=(f_p.real*u.imag-f_p.imag*u.real)
    Q_p=-(f_p.real*u.real+f_p.imag*u.imag)
    # nyq=1/dt/2#奈奎斯特频率
    # normal_cutoff=10/nyq#归一化截止频率
    # b, a = butter(5, normal_cutoff, btype='low', analog=False)  # 5阶低通滤波器
    # P_p=filtfilt(b, a, P_p)  # 滤波
    # Q_p=filtfilt(b, a, Q_p)  # 滤波

    return (sum((P_p-P)**2)+sum((Q_p-Q+abs(u)**2/x1)**2))/n

def obj_fun_var_ori(df,para,busid,dt=0.01):
    key=["UD","UQ","MB EQ1","MB ED1","MB ID","MB IQ","MB INIT LOAD TORQUE","MB SPEED DEVIATION","MB P","MB Q","MB MVA"]
    index=[]
    for k in key:
        index+=[i for i in range(len(df.columns)) if "BUS%d" % busid in df.columns[i] and k in df.columns[i]]
    index+=[i for i in range(len(df.columns)) if " %d " % busid in df.columns[i]]
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
    tm=P[0]/mva_amp/(1-s_p[0])**2
    for k in range(n-1):
        i_p[k]=(e_p[k]-u[k])/(r+1j*x1)
        de_p[k]=((-1-1j*120*math.pi*s_p[k]*t0)*e_p[k]-1j*(x0-x1)*i_p[k])/t0
        tele[k]=-(e_p[k].real*i_p[k].real+e_p[k].imag*i_p[k].imag)/mva_amp
        ds[k]=1/H2/2*(tm*(1-s_p[k])**2-tele[k])
        e_p[k+1]=e_p[k]+de_p[k]*dt
        s_p[k+1]=s_p[k]+ds[k]*dt
        P_p[k+1]=-(u[k].real*i_p[k].real+u[k].imag*i_p[k].imag)
        Q_p[k+1]=-(u[k].imag*i_p[k].real-u[k].real*i_p[k].imag)
    return (sum((P_p-P)**2)+sum((Q_p-Q)**2))/n

def obj_var_sim(df,para,busid,dt=0.01):
    #key=["UD","UQ","MB EQ1","MB ED1","MB ID","MB IQ","MB INIT LOAD TORQUE","MB SPEED DEVIATION","MB P","MB Q","MB MVA"]
    key=["UD","UQ","MC EQ1","MC ED1","MC ID","MC IQ","MC INIT LOAD TORQUE","MC SPEED DEVIATION","MC P","MC Q","MC MVA"]
    index=[]
    for k in key:
        index+=[i for i in range(len(df.columns)) if "BUS%d" % busid in df.columns[i] and k in df.columns[i]]
    index+=[i for i in range(len(df.columns)) if " %d " % busid in df.columns[i]]
    key+=["VOLT","ANGL"]    
    i_dic=dict(zip(key,index))

    mva_amp=df.values[0,i_dic[key[10]]]/100#容量基值放大倍数
    if len(para)>3:
        r,x0,x1,t0,H2_v=para[0]/mva_amp,para[1]/mva_amp,para[2]/mva_amp,para[3],para[4]
        a_v=(x0/x1-1)/t0/x1
        b_v=x0/x1/t0
    else:
        a_v,b_v,H2_v=para[0],para[1],para[2]
    ang=(df.values[0,i_dic['ANGL']]-df.values[:,i_dic['ANGL']])/180*math.pi
    u=(df.values[:,i_dic['UD']]+1j*df.values[:,i_dic['UQ']])*np.exp(1j*ang)
    P=df.values[:,i_dic[key[8]]]#MB/MC P
    Q=df.values[:,i_dic[key[9]]]#MB/MC Q

    n=len(u)
    df_p=np.zeros(n,dtype=complex)
    ds_p=np.zeros(n)
    tele=np.zeros(n)
    f_p=np.zeros(n,dtype=complex)
    s_p=np.zeros(n)
    w0=2*math.pi*60
    c=abs(u[0])**2*a_v*w0/P[0]
    s_p=np.zeros(n)
    s_p[0]=(c-math.sqrt(c**2-4*b_v**2*w0**2))/2/w0**2
    f_p[0]=a_v*u[0]/(b_v+1j*s_p[0]*w0)
    P_p=np.zeros(n)
    Q_p=np.zeros(n)
    P_p[0]=P[0]
    Q_p[0]=Q[0]-abs(u[0])**2/x1
    tm=P[0]/mva_amp/(1-s_p[0])**2
    for k in range(n-1):
        df_p[k]=(-b_v-1j*s_p[k]*w0)*f_p[k]+a_v*u[k]
        f_p[k+1]=f_p[k]+df_p[k]*dt
        tele[k]=(f_p[k+1].real*u[k].imag-f_p[k+1].imag*u[k].real)/mva_amp
        ds_p[k]=1/H2_v/2*(tm*(1-s_p[k])**2-tele[k])
        s_p[k+1]=s_p[k]+ds_p[k]*dt
        P_p[k+1]=(f_p[k+1].real*u[k+1].imag-f_p[k+1].imag*u[k+1].real)
        Q_p[k+1]=-(f_p[k+1].real*u[k+1].real+f_p[k+1].imag*u[k+1].imag)
    return (sum((P_p-P)**2)+sum((Q_p-Q+abs(u)**2/x1)**2))/n

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
import math

def para_iden(data, dt=0.002,type='con'):
    if type=='con':
        target_fun=target_con
        # 恒转矩电动机参数辨识    
    mP=np.mean(data[:,3])#平均有功功率
    mV=np.mean(data[:,0])#平均电压幅值
    #通过优化方法进行参数辨识
    bnds=[(10,80),(3,30),(0.05,0.8),(0,1)]
    while True:
        x0=np.zeros(len(bnds))
        while True:
            for i in range(len(bnds)):
                x0[i]=np.random.uniform(bnds[i][0],bnds[i][1])  # 随机初始化参数
            if target_fun(x0,data,dt)<1e25 and mV**2*x0[0]-2*x0[1]>0:
                break
        result=minimize(target_fun, x0, args=(data,dt), method='Nelder-Mead', bounds=bnds)
        if result.success:
            break
    return result
    #返回辨识结果和验证曲线

def target_con(x,data,dt):
    # 目标函数
    t=data[:,0]
    V=data[:,1]
    theta=data[:,2]/180*math.pi
    P_m=data[:,3]
    Q_m=data[:,4]
    a=x[0]
    b=x[1]
    H2=x[2]
    Tm=x[3]*np.mean(P_m)
    n=len(t)
    omega=100*math.pi
    c=V[0]**2*a
    if c**2-4*b**2<0:
        return 1e30#s0不是实数，这一组结果无效，返回无穷大
    s=np.zeros(n+1)
    ds=np.zeros(n+1)
    E=np.zeros(n+1,dtype=complex)
    dE=np.zeros(n+1,dtype=complex)
    P_d=np.zeros(n)
    Q_d=np.zeros(n)
    V_ph=V*np.exp(1j*theta)
    s[0]=(c-math.sqrt(c**2-4*b**2))/2/omega#初始滑差
    E[0]=a*V_ph[0]/(b+1j*s[0]*omega)#初始电动机电压
    for i in range(n):
        dE[i]=a*V_ph[i]-(b+1j*s[i]*omega)*E[i]
        ds[i]=1/H2*(1-E[i].real*V_ph[i].imag+E[i].imag*V_ph[i].real)
        E[i+1]=E[i]+dE[i]*dt
        s[i+1]=s[i]+ds[i]*dt
        P_d[i]=(E[i+1].real*V_ph[i].imag-E[i+1].imag*V_ph[i].real)*Tm
        Q_d[i]=(-V_ph[i].real*E[i+1].real-V_ph[i].imag*E[i+1].imag)*Tm
    P_j=P_m-P_d
    Q_j=Q_m-Q_d
    if s[n]<1:
        #这里用线性回归计算静负荷参数，返回功率偏差
        return regress(V[100:],P_j[100:],Q_j[100:])[1]/n#返回偏差值
    else:
        return 1e30#如果滑差跑飞，返回无穷大

def regress(V,P,Q):#多元回归，返回偏差值和回归结果,用ZIP和指数模型比较，选更好的
    X=np.column_stack((np.ones((len(V),1)),V,V**2))
    modelP=sm.OLS(P,X).fit()
    modelQ=sm.OLS(Q,X).fit()
    return [modelP.params,modelQ.params],np.sum(np.square(modelP.resid))+np.sum(np.square(modelQ.resid))

if __name__ == '__main__':
    df=pd.read_pickle('data_ma.pkl')  # 从文件中读取数据
    k=5#步长间隔10个采样，0.01
    data=df.values[k::k,[0,1,2,7,10]]#节点3的辨识所需数据
    #c=target_con([40,10,1,0.8],data,0.002)  # 调用参数辨识函数
    c=para_iden(data,dt=0.002*k)
    b=1
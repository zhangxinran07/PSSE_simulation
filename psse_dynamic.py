from __future__ import with_statement               #<- it must be first declaration
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
version=35

if version==35:
    PSSPY_PATH = r"C:\Program Files\PTI\PSSE35\35.5\PSSPY39"
    sys.path.append(PSSPY_PATH)
    os.environ['PATH'] += ';' + PSSPY_PATH
    import psse35

elif version==36:
    PSSPY_PATH = r"C:\Program Files\PTI\PSSE36\36.2\PSSPY312"
    sys.path.append(PSSPY_PATH)
    os.environ['PATH'] += ';' + PSSPY_PATH
    import psse36


import psspy,redirect
import dyntools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Import PSS/E default
_i = psspy.getdefaultint()
_f = psspy.getdefaultreal()
_s = psspy.getdefaultchar()
redirect.psse2py()

def generate_ambient_signals(amplitude, frequency_sampling, length, size):
    while True:
        y=np.random.uniform(-amplitude,amplitude,[length*500,size])  # 随机生成一个发电机的扰动
        y_final = np.zeros((length, size))  # 初始化最终结果数组
        nyq=frequency_sampling/2#奈奎斯特频率
        normal_cutoff=1/nyq#归一化截止频率,1Hz
        b, a = butter(5, normal_cutoff, btype='low', analog=False)  # 5阶低通滤波器
        y_filt = filtfilt(b, a, y, axis=0)  # 对每一列进行滤波
        for i in range(size):
            flag=0
            for j in range(length):
                if abs(y_filt[j,i]) < 0.001:
                    flag=1
                    y_final[:,i] = y_filt[j:j+length,i]
                    break
        if flag==1:
            break
    return y_final

def run_psse_simulation(simulation_type, study, type_option=4, case_path='./Cases/', time_step=0.002, total_time=20):
    #type:'ambient', 'load_change', 'fault'，后面两个需要给节点号
    # 加载案例文件
    sav_file = case_path+'%s.sav'%study  # 替换为你的案例文件路径
    cnv_file = case_path+'%s_cnv.sav'%study
    dyr_file = case_path+'%s.dyr'%study  # 替换为你的动态模型文件
    dyradd  = case_path+'ieee39_RE_ieee39_RE_motorD_Stall_enabled.dyr'
    snp_file = case_path+'%s_%s.snp'%(study,study)
    log_file = case_path+'%s.log'%study  # 日志文件
    sys.stdout = open(log_file, 'w')  # 重定向输出到日志
    busid=[3,15,24]

    psspy.psseinit(20000)
    psspy.case(sav_file)
    psspy.solution_parameters_3([_i,100,_i],
                                [_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])  
    # -------------------------------------------------------------------------
    # 2: convert case and create snp file
    #    
    # -------------------------------------------------------------------------
    #** convert loads and create converted case
    psspy.conl(_i,_i,1,[0,_i],[_f,_f,_f,_f])
    psspy.bsys(sid=1,numbus=len(busid)+1,buses=busid+[39])#39节点是平衡节点
    psspy.conl(1,0,2,[_i,_i],[ 50.0, 50.0,0.0, 50.0])
    psspy.conl(1,1,2,[_i,_i],[ 50.0, 50.0,0.0, 50.0])
    psspy.conl(_i,_i,3,[_i,_i],[_f,_f,_f,_f])   
    psspy.fdns([1,0,1,1,1,0,99,0])
    #convert gens
    psspy.cong()
    psspy.ordr(0)
    psspy.fact()
    psspy.tysl(0)
    psspy.tysl(0)
    psspy.save(cnv_file)
    psspy.dyre_new([1,1,1,1],dyr_file,
                    r"""conec.for""",
                    r"""conet.for""","")  
    #** Save snapshot for dynamics
    psspy.snap([-1,-1,-1,-1,-1],snp_file)

    if dyradd:
    #** Read in-lib DYRE records
        psspy.dyre_add([_i,_i,_i,_i],dyradd,'','')


    # 加载动态模型文件  
        
    # 设置仿真参数
    
    ierr=psspy.dynamics_solution_param_2([99,_i,_i,_i,_i,_i,_i,_i],
                                  [1.0,_f,time_step, time_step*4,_f,_f,_f,_f])
    
    # 监控母线电压 (类型=1, 子类型=3)
    ierr = psspy.chsb(sid=1, all=0, status = [-1, -1, -1, 1, 14, 0])#14：子系统1母线电压和相角
    ierr = psspy.chsb(sid=1, all=0, status = [-1, -1, -1, 1, 25, 0])#14：子系统1有功负荷
    ierr = psspy.chsb(sid=1, all=0, status = [-1, -1, -1, 1, 26, 0])#14：子系统1无功负荷
    #监控负荷功率等信息，[3,15,24]的初始var：137,283,429
    start_var=[]
    start_state=[]
    for bus in busid:
        start_var.append(psspy.lmodind(bus, '1','CHARAC','VAR')[1])
        start_state.append(psspy.lmodind(bus, '1','CHARAC','STATE')[1])
    #idx=[15,16]+list(range(25,29))+list(range(31,39))
    #ident=['UD','UQ','S P','S Q','E P','E Q','MA P','MA Q','MB P','MB Q','MC P','MC Q','MD P','MD Q']
    idx_var=[0,15,16,29,30,31,32,71,72,73,74,75,76,79,80]
    ident_var=['load MVA','UD','UQ','load bus v','low side bus v','MA P','MA Q','Tele','Speed Deviation','init load torque','Id','Iq','MA I','MA MVA','TL']
    idx_state=list(range(0,6))
    ident_state=['Eq1','Ed1','Eq2','Ed2','speed deviation','angle deviation']
    #for i in range(len(busid)):
    for i in [0]:
        for j in range(len(idx_var)):
            ierr=psspy.var_channel(status=[-1,start_var[i]+idx_var[j]],ident='Bus%d %s'%(busid[i],ident_var[j]))
        for j in range(len(idx_state)):
            ierr=psspy.state_channel(status=[-1,start_state[i]+idx_state[j]],ident='Bus%d %s'%(busid[i],ident_state[j]))

    outfile = case_path+'%s.out'%study  # 输出文件名

    ierr = psspy.strt_2([0,1],outfile)  # 初始化动态模型

    #load buses: [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29]   
    
    if simulation_type=='ambient':

        ambient_time_step=0.1
        load_change_buses=[12,7,8,23,16,18,26,29]  # 需要更改的负荷总线

        am_sg=generate_ambient_signals(type_option, 1/ambient_time_step, round(total_time/ambient_time_step), len(load_change_buses))  # 随机生成一个发电机的扰动

        # 运行动态仿真
        for n in np.arange(1, round(total_time/ambient_time_step)+1, 1):
            for j in range(len(load_change_buses)):
                ierr = psspy.load_chng_6(load_change_buses[j], '1 ', [], [am_sg[n-1,j], _f, _f, _f, _f, _f, _f, _f])
            t=n*ambient_time_step
            ierr = psspy.run(0, t, 0, 0, 0)        
            

    elif simulation_type=='load_change':
        total_time = 10  # 总仿真时间 (秒)
        ierr = psspy.run(0, 1, 0, 0, 0)    
        ierr = psspy.load_chng_6(type_option, '1 ', [], [100, _f, _f, _f, _f, _f, _f, _f])   
        ierr = psspy.run(0, total_time, 0, 0, 0)    

    sys.stdout.close()                # ordinary file object

    chnfobj=dyntools.CHNF(outfile)
    short_title, chanid, chandata=chnfobj.get_data()    

    data_ori=np.zeros((len(chandata['time']), len(chanid)))  # 初始化数据数组
    data_no_filt=np.zeros((len(chandata['time']), len(chanid)))  # 初始化数据数组

    for i in list(chanid.keys()):#把结果数据转化为numpy数组
        if i=='time':
            data_ori[:,0] = chandata['time']
        else:
            data_ori[:, i] = chandata[i]

    sys.stdout = sys.__stdout__

    # 关闭 PSSE
    psspy.pssehalt_2()

    j=0
    n=0
    while j<len(chandata['time'])-1:  # 找到时间点重复的位置,取变化后的数值
        if (abs(data_ori[j,0]-data_ori[j+1,0])<1e-5):
            data_no_filt[n,:] = data_ori[j+1,:]
            j+=2
        else:
            data_no_filt[n,:] = data_ori[j,:]
            j+=1
        n+=1
    data_no_filt[n,:] = data_ori[j,:]
    data_no_filt=data_no_filt[0:n+1,:]
    data_filt=np.zeros(np.shape(data_no_filt))
    data_filt[0:2,:]=data_no_filt[0:2,:]

    nyq=1/time_step//2#奈奎斯特频率
    normal_cutoff=2/nyq#归一化截止频率
    b, a = butter(5, normal_cutoff, btype='low', analog=False)  # 5阶低通滤波器
    data_filt[2:n+1,:] = filtfilt(b, a, data_no_filt[2:n+1,:], axis=0)  # 对每一列进行滤波

    data_filt[:,0]=data_no_filt[:,0]
    df=pd.DataFrame(data_filt, columns=list(chanid.values()))    

    print(chanid)

    return df
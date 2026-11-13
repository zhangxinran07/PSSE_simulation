from __future__ import with_statement               #<- it must be first declaration
import os
import sys
import matpower as mp

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

def run_psse_opf(simulation_type, study, type_option=4, case_path='./Cases/', time_step=0.002, total_time=20,ld_con_chng_buses=[],ld_con_chng_indexes=[],ld_con_chng_values=[]):
    #ld_con_chng_buses:要修改的负荷模型所在节点号列表
    #ld_con_chng_indexes:要修改的负荷模型参数索引列表
    #ld_con_chng_values:要修改的负荷模型参数值列表
    #type:'ambient', 'load_change', 'fault'，后面两个需要给节点号
    # 加载案例文件
    sav_file = case_path+'%s.sav'%study  # 替换为你的案例文件路径
    cnv_file = case_path+'%s_cnv.sav'%study
    dyr_file = case_path+'%s.dyr'%study  # 替换为你的动态模型文件
    dyradd  = case_path+'ieee39_RE_ieee39_RE_motorD_Stall_enabled.dyr'
    snp_file = case_path+'%s_%s.snp'%(study,study)
    log_file = case_path+'%s.log'%study  # 日志文件
    sys.stdout = open(log_file, 'w')  # 重定向输出到日志
    
    busid=[3,4,15,20,21,24,25,27,28]

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

    # 修改负荷模型参数
    for i in range(len(ld_con_chng_buses)):
        for j in range(len(ld_con_chng_indexes[i])):
            if ld_con_chng_values[i][j]!=-1:
                ierr=psspy.change_ldmod_con(ld_con_chng_buses[i], '1', 'CMLDBLU2', ld_con_chng_indexes[i][j], ld_con_chng_values[i][j])

    # 加载动态模型文件  
        
    # 设置仿真参数
    
    ierr=psspy.dynamics_solution_param_2([99,_i,_i,_i,_i,_i,_i,_i],
                                  [1.0,_f,time_step, time_step*4,_f,_f,_f,_f])
    
    # 监控母线电压 (类型=1, 子类型=3)
    ierr = psspy.chsb(sid=1, all=0, status = [-1, -1, -1, 1, 14, 0])#14：子系统1母线电压和相角
    ierr = psspy.chsb(sid=1, all=0, status = [-1, -1, -1, 1, 25, 0])#14：子系统1有功负荷
    ierr = psspy.chsb(sid=1, all=0, status = [-1, -1, -1, 1, 26, 0])#14：子系统1无功负荷
    
    para_name=['pa','pb','ra','xa','x1a','tda','ha','rb','xb','x1b','tdb','hb']
    para_index=[18,19,39,40,41,43,45,59,60,61,63,65]
    bus_para=[]
    for bus in busid:
        ival=psspy.lmodind(bus,'1','CHARAC','CON')[1]
        para_value=[]
        for i in range(len(para_index)):
            para_value.append(psspy.dsrval('CON',ival+para_index[i])[1])    
        bus_para.append(para_value)
    d_para=pd.DataFrame(data=bus_para,columns=para_name,index=busid)

    #调用3号节点r参数：d_para.loc[3,'ra']

    ierr=psspy.load_chng_6(ibus=20,id='1',realar=[580,_f,_f,_f,_f,_f,_f,_f])#改母线的负荷，前面两个参数分别是P和Q
    ierr=psspy.fnsl()#求解潮流，牛顿拉夫逊法
    psspy.list(sid=0,all=1,opt=4,vrev=0)#opt 1:summary 2: bus 4:发电 18：负荷
    psspy.list(sid=0,all=1,opt=18,vrev=0)#opt 1:summary 2: bus 4:发电 18：负荷
    
    ierr, pqlod = psspy.loddt2(ibus=3, id='1',string1='TOTAL',string2='ACT')#pqlod:有功无功负荷
    ierr, cmpval = psspy.gendat(ibus=31)#cmpval:发电机有功无功出力
    a=1

def run_psse_power_flow(study, case_path='./Cases/'):
    #ld_con_chng_buses:要修改的负荷模型所在节点号列表
    #ld_con_chng_indexes:要修改的负荷模型参数索引列表
    #ld_con_chng_values:要修改的负荷模型参数值列表
    #type:'ambient', 'load_change', 'fault'，后面两个需要给节点号
    # 加载案例文件
    sav_file = case_path+'%s.sav'%study  # 替换为你的案例文件路径
    cnv_file = case_path+'%s_cnv.sav'%study
    snp_file = case_path+'%s_%s.snp'%(study,study)
    

    psspy.psseinit(20000)
    psspy.case(sav_file)
    psspy.solution_parameters_3([_i,100,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) 
    ierr=psspy.load_chng_6(ibus=20,id='1',realar=[580,_f,_f,_f,_f,_f,_f,_f])#改母线的负荷，前面两个参数分别是P和Q
    ierr=psspy.machine_chng_4(ibus=30,inode="1",realar=[250.0,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])#改发电机有功出力
    ierr=psspy.plant_data_4(ibus=32, inode=0, realar=[1.00,_f])#改发电机电压
    ierr=psspy.fnsl()#求解潮流，牛顿拉夫逊法
    psspy.list(sid=0,all=1,opt=4,vrev=0)#opt 1:summary 2: bus 4:发电 18：负荷
    psspy.list(sid=0,all=1,opt=18,vrev=0)#opt 1:summary 2: bus 4:发电 18：负荷
    
    ierr, pqlod = psspy.loddt2(ibus=3, id='1',string1='TOTAL',string2='ACT')#pqlod:有功无功负荷
    ierr, cmpval = psspy.gendat(ibus=31)#cmpval:发电机有功无功出力
    a=1
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

def run_psse_opf(study, case_path='./Cases/'):
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
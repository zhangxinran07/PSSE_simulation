import os
import sys

PSSPY_PATH = r"C:\Program Files\PTI\PSSE35\35.5\PSSPY39"
sys.path.append(PSSPY_PATH)
os.environ['PATH'] += ';' + PSSPY_PATH

import psse35

import psspy,redirect
import dyntools
# Import PSS/E default
_i = psspy.getdefaultint()
_f = psspy.getdefaultreal()
_s = psspy.getdefaultchar()
#redirect.psse2py()

#___________________________________________________________
def runscript(myscript):
    import psspy
    if os.path.isfile(myscript):
       root,ext = os.path.splitext(myscript)
       #print root,ext
       if   ext.upper()=='.PY':
            execfile(myscript)
       elif ext.upper()=='.IDV':  
            psspy.runrspnsfile(myscript)
       elif ext.upper()=='.IRF':  
            psspy.runiplanfile(myscript)
#___________________________________________________________
if __name__ == '__main__':
  psspy.psseinit(1200)
  #___________________________________________________________
  # study data
  keyword = 'IEEE39_RE'
  savkey  = 'IEEE39_RE'
  study   = 'IEEE39_RE'
  conl    = 'Conl.idv'
  cnvfile = '%s_cnv.sav'%study
  channels= 'channels.idv'
  snpfile = '%s'%keyword
  dyrfile = '%s.dyr'%study
  outfile = '%s.out'%study
  logfile = '%s.log'%study
  #fault   = False
  sys.stdout = open(logfile,'w') # redirect all prints to this log file

  # open .sav file and do corresponding preprocessing work 
  # before dynamic simulation
  psspy.case(savkey)
  psspy.fnsl((_i,0,_i,_i,_i,_i,_i,0))
  #runscript(conl)
  for i in [1,2,3]:
      ierr = psspy.conl(0, 1, i, [0, 0], [0, 100, 0, 100])[0]
  psspy.cong(0)
  psspy.ordr(0)
  psspy.fact()
  psspy.tysl(0)
  psspy.tysl(0)
  psspy.save(cnvfile)

  # open .dyr file for dynamic simulation
  psspy.dyre_new([1,1,1,1],dyrfile,"","","")
  psspy.snap([-1,-1,-1,-1,-1],snpfile)

  # set output channels
  # define subsystem for Area of interest
  #sid    = -1       # -1 = All buses, or enter 0-9 for a subsystem:
  #basekv = []
  #areas  = []
  #buses  = []
  #owners = []
  #zones  = []
  #if basekv: 
  #  usekv = 1
  #else:
  #  usekv = 0    
  #btflag  = 1       # default=2 all bus types, =1 online buses only
  #idfilter= ''      # to filter for gen types, enter macid to select only those units
  #if sid > -1:    
  #  ierr = psspy.bsys(sid, 
  #                    usekv      , basekv, 
  #                    len(areas) , areas, 
  #                    len(buses) , buses,
  #                    len(owners), owners, 
  #                    len(zones) , zones)
  # 设置输出通道（监控特定变量）
        # 监控总线电压 (类型=1, 子类型=1)
  ierr = psspy.chsb(sid=0, all=1, status = [-1, -1, -1, 1, 12, 0])

    # 监控总线频率 (类型=1, 子类型=3)
  ierr = psspy.chsb(sid=0, all=1, status = [-1, -1, -1, 1, 13, 0])
  #sid    = 2       # -1 = All buses, or enter 0-9 for a subsystem:
  #region = [1]
  #ierr = psspy.zsys(sid, len(region), region)
  #print 'psspy.asys - ',ierr
  #psspy.chsb(2,0,[-1,-1,-1,5,0,0])   # system totals by zone subsystem

  psspy.dynamics_solution_param_2([99,_i,_i,_i,_i,_i,_i,_i],
                                  [1.0,_f,0.004, 0.016,_f,_f,_f,_f])
  psspy.set_relang(1,38,r"""1""")
  # create .out file
  psspy.strt_2([0,1],outfile)

  # apply unbalanced-branch-fault between bus1 and bus2
  #if fault:
  #	psspy.run(0, 1.0,99,9,0)    
  # run to 1.0 seconds after initiate  
  #    psspy.dist_spcb_fault_2(1,2,r"""1""",
  #                            [3,0,3,1,0,0,1],
  #                            [ 0.425,0.0,0.0,0.0,0.0])
  #    psspy.run(0, 1.05,19,1,0)   # run to 1.05 seconds after applying fault  
  #    psspy.dist_clear_fault(1)   # clear fault

  psspy.run(0, 3.0,99,9,0)   # run to 11 seconds after clearing fault
  print('动态仿真完成')
  output_file = r"dynamic_results.csv"
  dyntools.CHNF.xlsout(dyntools.CHNF(outfile), 
                     channels = '', 
                     show = 'True',
                     xlsfile =  output_file,
                     sheet = '',
                     overwritesheet = True)

  sys.stdout.close()                # ordinary file object
  sys.stdout = sys.__stdout__

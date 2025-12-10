# rundyn.python
'''
This Python scripts shows how to interactively debug and develop
PSSE dynamics User Model using:
- PSSE Environment Manager
- dyntools

The scripts does following.
a) creates DLL [using psse_env_manager]
b) loads and unloads DLL in PSSE
c) runs PSSE dynamics
d) plots channels from dynamics output file [using dyntools]
'''
# -------------------------------------------------------------------------
import os, sys
import runpy

# -------------------------------------------------------------------------
# (1) ---------------------- Input data ----------------------------------
studyname = 'uexc'
studytype = 'fault'         #= 'flat', for a non-disturbance run; ='fault' for ...
savfile   = "savnw.sav"
cnvfile   = "savnw_cnv.sav"
conlfile  = 'conl.idv'
dyrfile   = "savnw_UEXC.dyr"
snpfile   = "savnw_UEXC.snp"
candl     = 'compile_and_link.py'
pltshow   = True
channelfile = "channels.idv"
# Modify these dynamics run steps as required.
tend = 5.0          #time to end simulation
print_steps = 99    #print to screen rate
save_steps  = 9     #post-ctg save to disk rate
#_________________________________________________________
try:
   psspy            #return True if running from PSSe GUI
   GUI = True
except:
   GUI = False  
   psseversion = 34     #default
   if 'PSSEVERSION' in os.environ:
      psseversion = int(os.environ['PSSEVERSION'])
   exec('import psse{}'.format(psseversion))
   import psspy
   psspy.psseinit()
#_________________________________________________________
dllname = r"uexc_%s.dll"%psseversion
study   = '%s_%s_v%s'%(studyname,studytype,psseversion)
outfile = r"%s.out"%study
pltfile = r"%s.png"%study
logfile = r"%s.log"%study
#_________________________________________________________
import psse_env_manager
import dyntools
import redirect

_i = psspy.getdefaultint()
_f = psspy.getdefaultreal()
_s = psspy.getdefaultchar()
import redirect
#_________________________________________________________
if GUI:
    psspy.progress_output(2,logfile,[0,0])
    try:
        redirect.py2psse()
    except:
        pass
else:
    sys.stdout = open(logfile, 'w') # redirect all prints to this log file
    try:
        redirect.psse2py()
    except:
        pass
#  ---------------------- CNV case, load DYR ------------------------------
if os.path.isfile(savfile):
   # -------------------------------------------------------------------------
   # 1: LOAD PSSE CASE
   #    'replace' gen at bus 3018 with new solar PV plant
   #   or just load case with solar PV model included
   # -------------------------------------------------------------------------
   psspy.case(savfile)
   psspy.solution_parameters_3([_i,100,_i],
                               [_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])   
   #** convert loads and create converted case
   psspy.runrspnsfile(conlfile)
   psspy.fdns([1,0,1,1,1,0,99,0])
   psspy.cong()
   psspy.ordr(0)
   psspy.fact()
   psspy.tysl(0)
   psspy.tysl(0)
   #psspy.save(cnvfile)
else:
   quit()
if os.path.isfile(dyrfile):  
   #** Read DYRE records -  system dynamics + solar PV dynamics
   psspy.dyre_new([1,1,1,1],dyrfile,
                   '','','')  
   #** Save snapshot for dynamics
   psspy.snap([-1,-1,-1,-1,-1],snpfile)
else:
   quit()
#  ---------------------- drop, create and load DLL --------------------
dllfile = os.path.join(os.getcwd(), dllname)
#unload model if recently loaded:    
if dllfile in sys.modules:
   ierr = psspy.dropmodellibrary(dllfile)

if not os.path.exists(dllfile):
   #compile
   #ierr = psse_env_manager.create_dll(psseversion, src1, 
   #       modsources=modsources, 
   #       objlibfiles= ['uexc.lib'],
   #       dllname=dllname,
   #       workdir=os.getcwd(), showprg=True, useivfvrsn='latest', shortname='USRV33',
   #       description='User Model', majorversion=1, minorversion=0, buildversion=0, companyname='',
   #       mypathlib=False)
   #runpy.run_path(candl)          #instead of execfile
   quit()

ierr = psspy.addmodellibrary(dllfile)
#  ---------------------- CHANNELS definition --------------------------
#** Set channels
if os.path.isfile(channelfile):  #assume to be an idv
   psspy.runrspnsfile(channelfile)
else:
    psspy.chsb(0,1,[-1,-1,-1,1,1,0])    #Angle
    psspy.chsb(0,1,[-1,-1,-1,1,7,0])    #Speed
    psspy.chsb(0,1,[-1,-1,-1,1,2,0])    #Pelec
    psspy.chsb(0,1,[-1,-1,-1,1,3,0])    #Qelec
    psspy.chsb(0,1,[-1,-1,-1,1,14,0])   #V & angle
    if studytype == 'fault':      #fault study
       psspy.chsb(0,1,[-1,-1,-1,1,12,0])   #Freq at bus
       psspy.chsb(0,1,[-1,-1,-1,1,25,0])   #Pload
       psspy.chsb(0,1,[-1,-1,-1,1,26,0])   #Qload
       psspy.chsb(0,1,[-1,-1,-1,1,16,0])   #Pflow & Qflow
       psspy.chsb(0,1,[-1,-1,-1,1,4,0])    #Eterm
       psspy.chsb(0,1,[-1,-1,-1,1,5,0])    #Efd main field V
       psspy.chsb(0,1,[-1,-1,-1,1,6,0])    #Pmech
# (6) ---------------------- run dynamics ---------------------------------
psspy.strt(0,outfile)
#run flat
psspy.run(0, 1.0,99,99,0)
#run fault
if studytype=='fault':
    psspy.dist_bus_fault(154,1,0.0,[0.0,-0.2E+10])
    psspy.run(0, 1.0833,print_steps,1,0)
    psspy.dist_clear_fault(1)
#run post-fault to end
psspy.run(0, tend,print_steps,save_steps,0)
# ------ exit activities --------------------
ierr = psspy.dropmodellibrary(dllfile)
print(' Simulation ended.')
if GUI:
   psspy.progress_output(1)
else:
   sys.stdout.close()                # ordinary file object
   sys.stdout = sys.__stdout__
print(' Simulation ended.')
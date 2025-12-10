#candl.py
'''Create a UDM DLL.
It adds all *.flx, *.f,...,*.obj, *.lib files in working directory
to create a dii file.

default vars:
    keyname = 'dsusr'
    psseversion = 34

Open a "PSSe enabled" CMD window at your working folder,
    - To use with default parameters (edit lines 26-29 for updates), run it as:
        c:\..>python candl.py

    - To enter a dll name mymodel, run it as:
        c:\..>python candl.py mymodel
'''
import os, sys
import psse_env_manager
#__________________________________________________________________
def find_files(pattern,search_path=''):
    import glob
    """Given a search path, yield all files matching the pattern"""
    return glob.glob(os.path.join(search_path,pattern))
#______________________________________________________________
# [vars]
psseversion = 34
keyname = 'dsusr'
mypathlib = False  # set PATH and LIB values using installed components
src_lst = []                

work_dir= os.getcwd()
if len(sys.argv)>1:
   keyname = sys.argv[1]
dllname= "%s_%s.dll"%(keyname,psseversion)
dlllib = dllname.replace('.dll','.lib')
try: os.remove(dlllib)
except: pass

for ext in ['*.flx','*.f','*.for','*.f90']:       #include conec & conet files
    src_lst += find_files(ext,work_dir)           #get source files
objlibfiles  = find_files('*.obj',work_dir)
objlibfiles += find_files('*.lib',work_dir)

ierr = psse_env_manager.create_dll(psseversion, 
                                   src_lst, 
                                   modsources =[], 
                                   objlibfiles=objlibfiles, 
                                   dllname=dllname,
                                   workdir=work_dir, 
                                   showprg=True, 
                                   useivfvrsn ='latest', 
                                   shortname  = keyname,
                                   description='User Model', 
                                   majorversion=1, 
                                   minorversion=0, 
                                   buildversion=0, 
                                   companyname='',
                                   mypathlib=mypathlib)

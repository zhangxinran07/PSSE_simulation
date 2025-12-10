# qplot.py
'''
This Python scripts plots channels from dynamics output file [using dyntools]

If an empty channels list 'chanlst' is pass to qplot, the list of all channels will be printed.
Run the qplot script in the Dos window as,
	C:>python qplot.py<enter>
'''
# -------------------------------------------------------------------------
import os, sys

def qplot(outfile,chanlst,pltfile,pltshow=True,nrows=2,ncols=2,
          title = '', ylabel= '', xlabel= '',
          xscale =None, yscale = None, xaxis  = None):
    # Modify these plot steps as required.
    import psspy
    import dyntools
    
    pyver = '%s%s'%(sys.version_info[0],sys.version_info[1])
    psseversion = int(psspy.psseversion()[1])
    print('psseversion=',psseversion,'  pyver=',pyver)
    if 'PSSEVERSION' in os.environ:
       psseversion = int(os.environ['PSSEVERSION'])
    if psseversion>33:
       psspy.set_fpcw_py()     # To use PSSE, numpy and matplotlib, this is needed.

    chnfobj = dyntools.CHNF(outfile)
    nplot = len(chanlst)
    if nplot < 1:
       sh_ttl, ch_id = chnfobj.get_id()
       print('Channels in %s\nchan,    description'%outfile)
       for key in ch_id:
           print('%6s = %s'%(key,ch_id[key]))
       return
    chnfobj.set_plot_page_options(size='letter', orientation='portrait')
    chnfobj.set_plot_markers('square', 'triangle_up', 'thin_diamond', 'plus', 'x', 'circle', 'star', 'hexagon1')
    chnfobj.set_plot_line_styles('solid', 'dashed', 'dashdot', 'dotted')
    chnfobj.set_plot_line_colors('blue', 'red', 'black', 'green', 'cyan', 'magenta', 'pink', 'purple')

    optnfmt  = {'rows':nrows,'columns':ncols,
                'dpi':300,
                'showttl':True, 'showoutfnam':True, 'showlogo':False,
                'legendtype':1, 
                'addmarker':True}
    # compose optnchn
    # optnchn = Is a dictionary specifying channels to plot and plotting options
    #         = { 1 : {'chns'  : ch# or 
    #                            [ch1, ch2, ..] or 
    #                            {fnam:ch#} or 
    #                            {fnam:[ch1, chn2, ..]} or
    #                            [ch#, 'vf'] or 
    #                            [ch1, 'vf1', ch2, 'vf2', ch3, ch4, 'vf4'] or
    #                            [ch1, ch2, 'vf2', ch3, ch4, 'vf4'] or
    #                            {fnam: [ch1, 'vf1', ch2, 'vf2', ch3, ch4, 'vf4', ...]}
    #                            where 'vf' is any valid Python expression (valid function)
    #                            that modifes immediate preceding channel data.
    #                            (See Note 2)
    #                  'title' : '',
    #                  'ylabel': '',
    #                  'xlabel': '',
    #                  'xscale': [min, max],
    #                  'yscale': [min, max],
    #                  'xaxis' : ch# or {fnam:ch#} or [ch#, 'vf'] or {fnam:[ch#, 'vf']}
    #                 },
      
    optnchn = {}
    for k in range(nplot):
        key = k+1
        #inner dict:
        idict = {'chns':chanlst[k]}
        idict['title'] = title
        optnchn[key] = idict

    figfiles = chnfobj.xyplots(optnchn,optnfmt,pltfile)

    if pltshow:
        chnfobj.plots_show()
    else:
        chnfobj.plots_close()
    if psseversion>33:
       psspy.set_fpcw_psse()   # To use PSSE, numpy and matplotlib, this is needed.
    return

# (1) ---------------------- Input data ----------------------------------
studyname = 'uexc'
studytype = 'fault'         #or = 'flat', for a non-disturbance run
pltshow = True
#_________________________________________________________
psseversion = 34     #default
if 'PSSEVERSION' in os.environ:
   psseversion = int(os.environ['PSSEVERSION'])
exec('import psse{}'.format(psseversion))
import psspy
psspy.psseinit()
#_________________________________________________________
study   = '%s_%s_v%s'%(studyname,studytype,psseversion)
outfile = r"%s.out"%study
pltfile = r"%s.png"%study
#_________________________________________________________
import dyntools

# ---------------------- do plots -------------------------------------
#chanlst  = [6,12,18,24]
chanlst  = []
#figfiles = qplot(outfile,chanlst,pltfile,pltshow=True,nrows=2,ncols=2)
figfiles = qplot(outfile,chanlst,pltfile)

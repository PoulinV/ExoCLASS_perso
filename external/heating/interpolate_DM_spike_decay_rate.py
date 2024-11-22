#!/usr/bin/env python
# coding: utf-8

# ## PBHs+WIMPs scenario: effective spike decay rate
# [Author: Julien Lavalle - Oct. 2024]
#
# This series of Jupyter routines [optimized for Python 3] ultimately provides an interpolating function that allows one to predict the decay rates of spikes of self-annihilating WIMP dark matter that formed in the radiation-domination epoch around PBHs.
#
# These routines interpolate results calculated from a private code, and collected in a Python npz data file.
#
# The only requirement is to define the path to the data file. Follow the instructions below.

# In[7]:


# System + write/read data files
import sys
import os
from pathlib import Path

# Scientific libraries
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import scipy as sc
from scipy import interpolate

# Figures (in case)
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator, MultipleLocator
#from matplotlib.lines import Line2D
#import matplotlib.patches as mpatches
# Color gradient
#import matplotlib.cm as cm
#get_ipython().run_line_magic('matplotlib', 'inline')

# Computation time
import time
# To day-stamp the data
from datetime import datetime


#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.size": 10,
#    "pdf.fonttype":42})


# #### 1st step: define absolute dir path to data + declare global variables (vectors and data matrix)


###
### Here, we set up the python environment to load the tabulated values of GammaBH,
### read from a data file, as a function of mbh, mchi, xkd, rhomax, assuming
### fbh = 0. A multidimensional interpolation function is defined that allows to
###

#LOCAL_DATA_DIR_NAME = '/Path/To/Your/Data/Directory/'
LOCAL_DATA_DIR_NAME = '/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/ExoCLASS_perso/external/heating/'
LOCAL_FILE_NAME = 'new_rhosquareV_GaussMeth_xkd_mchi_mbh_rhomax0_rhosquareV_log10.npz'

### Init vectors and matrices
### (Lxxx indicates that tabulated values are in the form log10(xxx))
N_INIT = 2
VAR_LXKD_V = np.zeros(N_INIT)
VAR_LMCHI_V = np.zeros(N_INIT)
VAR_LMBH_V = np.zeros(N_INIT)
VAR_LRHOMAX_V = np.zeros(N_INIT)
VAR_LRHOSQUAREV_MX = np.zeros((N_INIT,N_INIT,N_INIT,N_INIT,N_INIT))


# #### 2nd step: define the loading function


###
### Function that loads the data from an npz file, and initializes the data vectors
### and matrices from which the interpolation will be performed.
###
def load_GammaBH_data_table(fname='file_name'):
    global LOCAL_DATA_DIR_NAME
    global VAR_LXKD_V, VAR_LMCHI_V, VAR_LMBH_V, VAR_LRHOMAX_V, VAR_LRHOSQUAREV_MX
    fullname = LOCAL_DATA_DIR_NAME + fname
    yes_file = os.path.isfile(fullname)
    if yes_file:
        #print('LOADING DATA FILE ...')
        npzfile = np.load(fullname)
        header = npzfile['header']
        #print(header)
        VAR_LMBH_V = npzfile['l10mbh']
        VAR_LXKD_V = npzfile['l10xkd']
        VAR_LMCHI_V = npzfile['l10mchi']
        VAR_LRHOMAX_V = npzfile['l10rhomax']
        VAR_LRHOSQUAREV_MX = npzfile['l10rhosquareV']
        return 'DONE'
    else:
        return 'FILE {} NOT FOUND'.format(fullname)


# #### 3rd step: define the main interpolating function


##
## Interpolation function that determines the log10 of the effective spike decay rate
## in [1/s], as a function of the following parameters:
## mbh[Msun], fbh (DM fraction in BHs), mchi[GeV], xkd (kinetic decoupling),
## sigv [cm3/s], dt [s, since matter-radiation equality].
## The effective time is encoded in the data in terms of rhomax = mchi/(sigv*dt).
## Note that oDM is omega_dm = Omega_dm * h^2.
##
def interpol_l10GammaBH_from_data_table(mbh,fbh,mchi,xkd,sigv,dt,oDM=0.11933):

    global VAR_LXKD_V, VAR_LMCHI_V, VAR_LMBH_V, VAR_LRHOMAX_V, VAR_LRHOSQUAREV_MX
    grid_points = (VAR_LXKD_V,VAR_LMCHI_V,VAR_LMBH_V,VAR_LRHOMAX_V)

    #cosmo_fraction = 1.
    cosmo_fraction = oDM/0.11933 ## The original calculation was performed with Planck+18.

    epsilon_time = 1.e-5# minimal time [s] to avoid numerical crashes
    GeV_IN_g = 1.782661845e-24 # convert a GeV into g
    TimeEQ = 1.6110761e+12 # Time [s] spent between end of inflation and equality
    dteff = dt+epsilon_time
    mchig = mchi*GeV_IN_g # GeV -> g

    # We calculate the approximate saturation density, which is only used here as
    # an effective time.
    rhomax = mchig/(sigv*dteff)/cosmo_fraction # g/cm3

    # The required point coordinates in this parameter space.
    lmbh = np.log10(mbh)
    lmchi, lxkd, lrhomax = np.log10(mchi), np.log10(xkd), np.log10(rhomax)
    this_point = np.array([lxkd,lmchi,lmbh,lrhomax])

    # The corresponding J-factor value (log10)
    lGammaBH = interpolate.interpn(grid_points,VAR_LRHOSQUAREV_MX,this_point)

    # Now, we introduce an approximate correction by hand to account for the
    # DM fraction in BHs. Asympotically, the correction goes from (1-f)^2 for light
    # BHs to (1-f)^4/3 for heavy BHs.
    mbreak = 5.e-7*(1.-fbh)*(xkd*1.e-4)**(3./2.) * (3.e-26*TimeEQ*8./(sigv*dteff))**(1./3.)
    lmeff = np.log10(mbh/mbreak)
    x = np.tanh(lmeff)# -1 (1) if mbh << mbreak (>>mbreak)
    x = (x+1.)/2. # 0 (1) if mbh<<mbreak (>>mbreak)
    corrfbh = (1.-fbh)**2 * (1.-x) + (1.-fbh)**(4./3.)*x
    lGammaBH += np.log10(corrfbh)

    ## Finally, we multiply by the factor that translates the J factor into a decay rate.
    J_into_GammaBH = sigv/(2.*mchig**2)
    lGammaBH += np.log10(J_into_GammaBH) ## log10(Gamma/s)

    ## Extra-correction if one departs from Planck+18 cosmological parameters:
    lGammaBH += np.log10(cosmo_fraction**2) ## log10(Gamma/s)


    return lGammaBH[0]


# #### 4rd step: load data


###
### Here, we initialize the data that will feed the interpolating function.
###
load_GammaBH_data_table(LOCAL_FILE_NAME)


# #### 5th step: enjoy!


### Define your point in parameter space:
### for BHs (mbh in Msun, fraction <=1)
#mbh, fbh = 1.e-15, 1.e-3
### then for WIMPs (mchi in GeV, sigv in cm3/s)
#mchi, xkd, sigv = 100, 100, 3.e-26
### and finally pick up a time since equality in seconds:
#dt = 60.*60*24*365*1.e10

#l10GAmmaBH = interpol_l10GammaBH_from_data_table(mbh,fbh,mchi,xkd,sigv,dt)
#print('log10(GammaBH) = {:.3e}'.format(l10GAmmaBH))

### Time test
#niter = 100
#start_time = time.time()
#for i in range(niter):
#    interpol_l10GammaBH_from_data_table(mbh,fbh,mchi,xkd,sigv,dt)

#end_time = time.time()
#print('time for {:d} iterations: '.format(niter),end_time-start_time)


# #### 6th step: Call with arguments from line command

list_arg = sys.argv
#print('arguments:',list_arg[1:])
nargs = len(list_arg)

#print(list_arg)

if nargs != 8:
    print('Incorrect number of parameters! Check arguments!')
    print('Should be: nbflines,mbh,fbh,mchi,xkd,sigv,oDM')
    print('Current ones are: ', list_arg)
    exit()

params = np.array(list_arg[1:])
params = params.astype(np.float64)
nbflines,mbh,fbh,mchi,xkd,sigv,oDM = params
nbflines = int(nbflines)
#print(nbflines,mbh,fbh,mchi,xkd,sigv,oDM)

## log10(time/second)
L10_TIME = np.linspace(0.,19.,nbflines)
L10_GAMMA_BH = np.zeros(nbflines)

for i in range(int(nbflines)):
    dt = 10.**L10_TIME[i]
    L10_GAMMA_BH[i] = interpol_l10GammaBH_from_data_table(mbh,fbh,mchi,xkd,sigv,dt,oDM)

fname = LOCAL_DATA_DIR_NAME+'temp_wimps_pbhs.dat'
print("%g" % (nbflines))
for i in range(int(nbflines)):
    print("%g %g"% (L10_TIME[i],L10_GAMMA_BH[i]))

#np.savetxt(fname,np.c_[L10_TIME,L10_GAMMA_BH], fmt='%.7e', delimiter=' ',header='{:d}'.format(nbflines),comments='',newline='\n')

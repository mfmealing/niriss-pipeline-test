import numpy as np
import matplotlib.pyplot as plt
import matplotlib
 
matplotlib.style.use('classic')
 
import pylightcurve as plc
from astropy.io import fits

from lmfit import Model as lmfit_Model

def transit_model(t, rat, t0, gamma0, gamma1, per, ars, inc, w, ecc, a, b, ldc_type='quad'):

    lc = plc.transit([gamma0, gamma1], rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
    syst = (a*t) + b
 
    lc = lc * syst
    return lc

seg_list = ['001', '002', '003', '004']

plt.figure('segments')
for seg in seg_list: 

    f='/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/jw02722003001_04101_00001-seg%s_nis/jw02722003001_04101_00001-seg%s_nis_1Dspec_box_extract.fits'%(seg, seg)
    
    hdul = fits.open(f)
    int_times = hdul[1].data
    slc = hdul[2].data
    wav = hdul[3].data
    var = (hdul[4].data)**2
    
    bjd = int_times['int_mid_BJD_TDB']
    mjd = int_times['int_mid_MJD_UTC']
    
    wlc = np.nansum(slc, axis=1)
    
    if seg == seg_list[0]:
        slc_stack = slc
        var_stack = var
        bjd_stack = bjd
    else:
        slc_stack = np.vstack((slc_stack, slc))
        var_stack = np.vstack((var_stack, slc))
        bjd_stack = np.hstack((bjd_stack, bjd))

    
    plt.plot(bjd, wlc)

wlc = np.nansum(slc_stack, axis=1)
slc = slc_stack
var = var_stack
bjd = bjd_stack
plt.figure('wlc')
plt.plot(bjd, wlc, '.')

# 4. time-bin
# ============================================================================= 
plt.figure('wlc after time bin')
time_bin = 5
idx = np.arange(0, slc.shape[0], time_bin)
bjd = (np.add.reduceat(bjd, idx)/  time_bin) [:-1]
slc = np.add.reduceat(slc, idx, axis=0)[:-1]
var = np.add.reduceat(var, idx, axis=0)[:-1]
idx = np.argwhere((wav<1.0 ) | (wav>2.0)).T[0]
wlc = np.nansum(slc[:,idx],axis=1)
wlc_var = np.nansum(var[:,idx],axis=1)
print ('time_step (s): ', np.diff(bjd)[0]*24*60*60)   
plt.errorbar(bjd, wlc, wlc_var**0.5, fmt='ro')


# =============================================================================
# initial guess
# =============================================================================
t = bjd-bjd[0]
            
lm_rat = 0.05412
lm_t0 = (bjd[-1] - bjd[0]) / 2
lm_gamma0 = 0.2
lm_gamma1 = 0.2
lm_per = 32.940045
lm_ars = 79.9
lm_inc = 89.550
lm_w = 90
lm_ecc = 0
lm_a = (wlc[-1]-wlc[0])/(t[-1]-t[0])
lm_b = wlc[0]

initial_guess = transit_model(t, lm_rat, lm_t0, lm_gamma0 , lm_gamma1,
                                lm_per, lm_ars, lm_inc, 
                                lm_w, lm_ecc, lm_a, lm_b)

plt.figure('intial guess')
plt.plot(t, wlc, 'bo')
plt.plot(t, initial_guess, 'g-', linewidth=3)


# =============================================================================
# lm_fit to wlc
# =============================================================================

gmodel = lmfit_Model(transit_model)

lm_params = gmodel.make_params()
lm_params.add('rat', value=lm_rat,vary=True)
lm_params.add('t0', value=lm_t0,  vary=True)
lm_params.add('gamma0', value=lm_gamma0,  vary=True, min =0, max=1)
lm_params.add('gamma1', value=lm_gamma1,  vary=True, min=0, max=1)
lm_params.add('per', value=lm_per, vary=False)
lm_params.add('ars', value=lm_ars, vary=True)
lm_params.add('inc', value=lm_inc, vary=True)
lm_params.add('w', value=lm_w,  vary=False)
lm_params.add('ecc', value=lm_ecc, vary=False)
lm_params.add('a', value=lm_a, vary=True)
lm_params.add('b', value=lm_b, vary=True)

result = gmodel.fit(wlc, lm_params, t=t, ldc_type = 'quad', weights=(1/wlc_var**(1/2)))
print (result.fit_report())

model_fit = transit_model(t, result.params['rat'].value, 
                         result.params['t0'], result.params['gamma0'], result.params['gamma1'], lm_per, 
                         result.params['ars'], result.params['inc'], lm_w, lm_ecc,
                         result.params['a'].value, result.params['b'].value)

plt.figure('lm_fit')
plt.plot(t, wlc, 'bo')
plt.plot(t, model_fit, 'r-', linewidth = 3)


# =============================================================================
# residuals and best fit models
# =============================================================================

res = model_fit - wlc
print(np.std(res))

plt.figure('residuals')
plt.plot(t, res, '.')
plt.grid(True)

var = np.var(res)

chi = np.nansum(res**2/var)
print(chi)

reduce_chi = chi/len(wlc)
print(reduce_chi)
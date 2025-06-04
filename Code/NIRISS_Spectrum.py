import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from uncertainties import ufloat
import bin_spectrum
 
matplotlib.style.use('classic')

 
import pylightcurve as plc
from astropy.io import fits
import emcee
import corner
 

from lmfit import Model as lmfit_Model

def transit_model_lin(t, rat, t0, gamma0, gamma1, per, ars, inc, w, ecc, a, b, ldc_type='quad'):

    lc = plc.transit([gamma0, gamma1], rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
    syst = (a*t) + b
 
    lc = lc * syst
    return lc

def transit_model_quad(t, rat, t0, gamma0, gamma1, per, ars, inc, w, ecc, a, b, c, ldc_type='quad'):

    lc = plc.transit([gamma0, gamma1], rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
    syst = (a*t**2) + (b*t) + c
 
    lc = lc * syst
    return lc

def wavelength_bin(data_dic, bin_type='None', R=100, bin_size=10, wavgrid=None):
    slc = data_dic['slc']; var = data_dic['var']; bjd = data_dic['bjd']
    wav = data_dic['wav']
 
    if bin_type != 'None':
 
        var,_,_ = bin_spectrum.bin_spectrum(var, wav, R, bin_size, bin_type=bin_type, wavgrid = wavgrid)
        slc, wav, edges = bin_spectrum.bin_spectrum(slc, wav, R, bin_size, bin_type=bin_type, wavgrid = wavgrid)
 
        print ('number of wavelength bins',  slc.shape[1])
        data_dic['slc'] = slc; data_dic['var'] = var; data_dic['bjd'] = bjd
        data_dic['wav'] = wav
        data_dic['edges'] = edges
        
    else:
        print ('no wavelength binning...')
    return data_dic  

file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/nis_1Dspec_box_extract_combined.fits'

hdul = fits.open(file)
int_times = hdul[1].data
slc = hdul[2].data
wav = hdul[3].data
var = (hdul[4].data)**2
bjd = int_times['int_mid_BJD_TDB']
mjd = int_times['int_mid_MJD_UTC']
wlc = np.nansum(slc, axis=1)

# idx = np.argwhere((wav<2.0) | (wav>2.8)).T[0]
# Do i need to do all 3 or just wav?
# wav = wav[idx]
# slc = slc[:,idx]
# var = var[:,idx]


plt.figure('wlc')
plt.plot(bjd, wlc, '.')

lit_vals = np.loadtxt('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/literature_vals.csv', delimiter=',')
fixed_vals = np.loadtxt('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/lm_fit_fixed_vals.csv', delimiter=',')

t= bjd-bjd[0]
lm_rat = lit_vals[0]
lm_t0 = fixed_vals[0]
lm_gamma0 = fixed_vals[1]
lm_gamma1 = fixed_vals[2]
lm_per = lit_vals[1]
lm_ars = fixed_vals[3]
lm_inc = fixed_vals[4]
lm_w = 90
lm_ecc = 0
lm_a = (wlc[-1]-wlc[0])/(t[-1]-t[0])
lm_b = wlc[0]

gmodel = lmfit_Model(transit_model_lin)

lm_params = gmodel.make_params()
lm_params.add('rat', value=lm_rat,vary=True)
lm_params.add('t0', value=lm_t0,  vary=False)
lm_params.add('gamma0', value=lm_gamma0,  vary=False)
lm_params.add('gamma1', value=lm_gamma1,  vary=False)
lm_params.add('per', value=lm_per, vary=False)
lm_params.add('ars', value=lm_ars, vary=False)
lm_params.add('inc', value=lm_inc, vary=False)
lm_params.add('w', value=lm_w,  vary=False)
lm_params.add('ecc', value=lm_ecc, vary=False)
lm_params.add('a', value=lm_a, vary=True)
lm_params.add('b', value=lm_b, vary=True)

data_dic ={}
data_dic['slc'] = slc
data_dic['wav'] = wav
data_dic['var'] = var
data_dic['bjd'] = bjd

bin_size = 18
R = 1
bin_type = 'col'
wavgrid = None
data_dic = wavelength_bin(data_dic, bin_type=bin_type, R=R, bin_size=bin_size, wavgrid=wavgrid)

no_bins = int(len(slc[1])/bin_size)
slc_new = data_dic['slc']
var_new = data_dic['var']
wav_avg = data_dic['wav']
rat = []
rat_err = []

for i in range(no_bins):
    result = gmodel.fit(slc_new[:,i], lm_params, t=t, ldc_type = 'quad')
    #print(result.params['rat'].value)
    rat.append(result.params['rat'].value)
    rat_err.append(result.params['rat'].stderr)

# for j in range(len(slc_new)):
#     plt.figure('slc all')
#     plt.plot(bjd, slc_new[j], '.')

rprs2 = []
rprs2_err = []

for k in range(len(rat)):
    x = ufloat(rat[k],rat_err[k])
    x = x**2
    rprs2.append(x.nominal_value)
    rprs2_err.append(x.std_dev)

plt.figure('transmission spectrum')
plt.errorbar(wav_avg, rprs2, rprs2_err, fmt='bo')

# vals = [wav_avg, rprs2, rprs2_err]
# np.save('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/spectra_vals_all_wav.npy', vals)
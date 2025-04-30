import bin_spectrum
import numpy as np
from astropy.io import fits
import pandas as pd

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

file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/nis_1Dspec_box_extract_combined.fits'

hdul = fits.open(file)
int_times = hdul[1].data
slc = hdul[2].data
wav = hdul[3].data
var = (hdul[4].data)**2
bjd = int_times['int_mid_BJD_TDB']
mjd = int_times['int_mid_MJD_UTC']
wlc = np.nansum(slc, axis=1)

# note data_dic is a dictionary that contains various values.  
data_dic ={}
data_dic['slc'] = slc
data_dic['wav'] = wav
data_dic['var'] = var
data_dic['bjd'] = bjd

with open("/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/spectrum_vals_lit.txt", "r") as f:
    header = f.readline().strip().split(",")
    vals_true = pd.read_csv(f, sep=' ', names=header)

wav_true, depth_true, depth_err_true = vals_true['Wavelength (microns)'], vals_true[' transit depth'], vals_true[' uncertainty']


bin_size = 60
R = 1
bin_type = 'col'
wavgrid = wav_true
data_dic = wavelength_bin(data_dic, bin_type=bin_type, R=R, bin_size=bin_size, wavgrid=wavgrid)

slc = data_dic['slc']
wav = data_dic['wav']
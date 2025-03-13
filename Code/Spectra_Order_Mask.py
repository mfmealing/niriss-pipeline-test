import os
os.environ['CRDS_PATH'] ='/Users/c24050258/crds_cache'
os.environ['CRDS_SERVER_URL'] ='https://jwst-crds.stsci.edu'
 
import asdf
import copy
import shutil
import numpy as np
import requests
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from numpy.polynomial.polynomial import Polynomial

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



# jwst imports 
import jwst
print(jwst.__version__)

# Individual steps that make up calwebb_spec2 and datamodels
from jwst.assign_wcs.assign_wcs_step import AssignWcsStep

output_dir = './output'
if not os.path.exists(output_dir ): 
    os.makedirs(output_dir )

def interpolate_nans(array):
    nan_vals = np.isnan(array)
    if np.any(nan_vals):
        nan_ind = np.where(nan_vals)[0]
        non_nan_ind = np.where(~nan_vals)[0]
        non_nan_vals = array[non_nan_ind]
        interp = interpolate.interp1d(non_nan_ind, non_nan_vals, bounds_error=False, fill_value='extrapolate')
        array[nan_ind] = interp(nan_ind)
    return array

bkd_model = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/model_background256.npy')    

file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/jw02722003001_04101_00001-seg001_nis/jw02722003001_04101_00001-seg001_nis_rateints.fits'
hdul = fits.open(file)
sci = hdul[1].data
err = hdul[2].data
dq = hdul[3].data 
int_times = hdul[4].data  
varp = hdul[5].data   
varr = hdul[6].data  
     
step = AssignWcsStep()
step.output_dir = output_dir
result = step.run(file)
wav = result.wavelength

section = result.data[:,210:250,720:770]
bkd_section = bkd_model[210:250,720:770]
section_med = np.nanmedian(section, axis=0)
scale_arr = section_med / bkd_section
scale_val = np.nanmedian(scale_arr)
scaled_bkd = scale_val * bkd_model
result.data = result.data - scaled_bkd

first_pix = result.data[:,5:25,:]
last_pix = result.data[:,-30:-5,:]
bkd_stack = np.hstack((first_pix,last_pix))
bkd_med = np.nanmedian(bkd_stack, axis=1)
bkd_3d = np.expand_dims(bkd_med, axis=1)
bkd_final = np.repeat(bkd_3d, result.data.shape[1], axis=1)
result.data = result.data - bkd_final

nans = np.isnan(result.data)
nans_frac = np.sum(nans, axis=0) / result.data.shape[0]
low_nans = np.array(np.where((nans_frac>0) & (nans_frac<0.1)))
result.data[:,low_nans[0],low_nans[1]] = np.apply_along_axis(interpolate_nans, axis=0, arr=result.data[:,low_nans[0],low_nans[1]])
result.data = np.apply_along_axis(interpolate_nans, axis=2, arr=result.data)


integration = np.median(result.data, axis=0)

# plt.figure('fig1')
# plt.plot(integration[:,340])

# plt.figure('fig2')
# plt.plot(np.gradient(integration[:,340]))

# box = 5
# bbox = np.ones(box) / box
# a = np.convolve(integration[:,340], bbox, 'same')
# plt.figure('fig1')
# plt.plot(a)

# plt.figure('fig2')
# plt.plot(np.gradient(a))

order1_first = []
order1_last = []
order2_first = []
order2_last = []
order3_first = []
order3_last = []
box = 5
bbox = np.ones(box) / box

for i in range(4, integration.shape[1]-4):
    order1 = integration[:95, i]
    smooth1 = np.convolve(order1, bbox, 'same')
    der1 = np.gradient(smooth1)
    max_der1 = np.argmax(der1)
    min_der1 = np.argmin(der1)
    order1_first.append(max_der1)
    order1_last.append(min_der1)
    
x_1 = np.arange(4, integration.shape[1]-4)
plt.figure('spectra order mask')
plt.imshow(integration, aspect='auto', vmin=0, vmax=20)
# plt.plot(x_1, order1_first, '.', c='r')
# plt.plot(x_1, order1_last, '.', c='r')

deg = 5
poly_coeff = np.polyfit(x_1, order1_first, deg)
poly = np.poly1d(poly_coeff)
plt.plot(x_1, poly(x_1)-12, '.', c='b')
poly_coeff = np.polyfit(x_1, order1_last, deg)
poly = np.poly1d(poly_coeff)
plt.plot(x_1, poly(x_1)+10, '.', c='b')

for j in range(4, 1750):
    if j<700:
        order2 = integration[100:120, j]
        smooth2 = np.convolve(order2, bbox, 'same')
        der2 = np.gradient(smooth2)
        min_der2 = np.argmin(der2)
        order2_first.append(min_der2+75)
        order2_last.append(min_der2+100)
    else:
        order2 = integration[75:241, j]
        smooth2 = np.convolve(order2, bbox, 'same')
        der2 = np.gradient(smooth2)
        max_der2 = np.argmax(der2)
        min_der2 = np.argmin(der2)
        order2_first.append(max_der2+75)
        order2_last.append(min_der2+75)

x_2 = np.arange(4, 1750)
# plt.plot(x_2, order2_first, '.', c='r')
# plt.plot(x_2, order2_last, '.', c='r')

deg = 4
poly_coeff = np.polyfit(x_2, order2_first, deg)
poly = np.poly1d(poly_coeff)
plt.plot(x_2, poly(x_2)-4, '.', c='b')
poly_coeff = np.polyfit(x_2, order2_last, deg)
poly = np.poly1d(poly_coeff)
plt.plot(x_2, poly(x_2)+3, '.', c='b')

for k in range(4, 792):
    order3 = integration[125:200, k]
    smooth3 = np.convolve(order3, bbox, 'same')
    der3 = np.gradient(smooth3)
    max_der3 = np.argmax(der3)
    min_der3 = np.argmin(der3)
    order3_first.append(max_der3+125)
    order3_last.append(min_der3+125)

x_3 = np.arange(4, 792)
# plt.plot(x_3, order3_first, '.', c='r')
# plt.plot(x_3, order3_last, '.', c='r')

deg = 3
poly_coeff = np.polyfit(x_3, order3_first, deg)
poly = np.poly1d(poly_coeff)
plt.plot(x_3, poly(x_3)-3, '.', c='b')
poly_coeff = np.polyfit(x_3, order3_last, deg)
poly = np.poly1d(poly_coeff)
plt.plot(x_3, poly(x_3)+1, '.', c='b')
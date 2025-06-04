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
from jwst.flatfield.flat_field_step import FlatFieldStep

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

file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_17b_NIRISS/nis_rateints_combined.fits'
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

step = FlatFieldStep()
result = step.run(result)

section = result.data[:,210:250,720:770]
bkd_section = bkd_model[210:250,720:770]
section_med = np.nanmedian(section, axis=0)
scale_arr = section_med / bkd_section
scale_val = np.nanmedian(scale_arr)
scaled_bkd = scale_val * bkd_model
result.data = result.data - scaled_bkd

result.data[:,251:,:] = result.data[:,:,:5] = result.data[:,:,2043:] = 0
result.err[:,251:,:] = result.err[:,:,:5] = result.err[:,:,2043:] = 0
result.wavelength[251:,:] = result.wavelength[:,:5] = result.wavelength[:,2043:] = 0

nans = np.isnan(result.data)
nans_frac = np.sum(nans, axis=0) / result.data.shape[0]
low_nans = np.array(np.where((nans_frac>0) & (nans_frac<0.1)))
result.data[:,low_nans[0],low_nans[1]] = np.apply_along_axis(interpolate_nans, axis=0, arr=result.data[:,low_nans[0],low_nans[1]])
result.data = np.apply_along_axis(interpolate_nans, axis=2, arr=result.data)


integration = np.median(result.data, axis=0)

# plt.figure('fig1')
# plt.plot(integration[:,250])

# plt.figure('fig2')
# plt.plot(np.gradient(integration[:,250]))

# box = 5
# bbox = np.ones(box) / box
# a = np.convolve(integration[:,250], bbox, 'same')
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
plt.imshow(integration, aspect='auto', vmin=0, vmax=10)
# plt.plot(x_1, order1_first, '.', c='r')
# plt.plot(x_1, order1_last, '.', c='r')

deg = 4
poly_coeff_first = np.polyfit(x_1, order1_first, deg)
poly_first = np.poly1d(poly_coeff_first)
y_1_first = poly_first(x_1)
plt.plot(x_1, y_1_first-12, '.', c='b')
poly_coeff_last = np.polyfit(x_1, order1_last, deg)
poly_last = np.poly1d(poly_coeff_last)
y_1_last = poly_last(x_1)
plt.plot(x_1, y_1_last+10, '.', c='b')

for j in range(5, 1820):
    if j<700:
        order2 = integration[100:110, j]
        smooth2 = np.convolve(order2, bbox, 'same')
        der2 = np.gradient(smooth2)
        min_der2 = np.argmin(der2)
        order2_first.append(min_der2+75)
        order2_last.append(min_der2+100)
    elif j>1700:
        order2 = integration[180:250, j]
        smooth2 = np.convolve(order2, bbox, 'same')
        der2 = np.gradient(smooth2)
        max_der2 = np.argmax(der2)
        min_der2 = np.argmin(der2)
        order2_first.append(max_der2+180)
        order2_last.append(min_der2+180)
    else:
        order2 = integration[75:250, j]
        smooth2 = np.convolve(order2, bbox, 'same')
        der2 = np.gradient(smooth2)
        max_der2 = np.argmax(der2)
        min_der2 = np.argmin(der2)
        order2_first.append(max_der2+75)
        order2_last.append(min_der2+75)

x_2 = np.arange(5, 1820)
# plt.plot(x_2, order2_first, '.', c='r')
# plt.plot(x_2, order2_last, '.', c='r')

deg = 4
poly_coeff_first = np.polyfit(x_2, order2_first, deg)
poly_first = np.poly1d(poly_coeff_first)
y_2_first = poly_first(x_2)
plt.plot(x_2, y_2_first-4, '.', c='b')
poly_coeff_last = np.polyfit(x_2, order2_last, deg)
poly_last = np.poly1d(poly_coeff_last)
y_2_last = poly_last(x_2)
plt.plot(x_2, y_2_last+3, '.', c='b')

for k in range(5, 865):
    order3 = integration[130:220, k]
    smooth3 = np.convolve(order3, bbox, 'same')
    der3 = np.gradient(smooth3)
    max_der3 = np.argmax(der3)
    min_der3 = np.argmin(der3)
    order3_first.append(max_der3+130)
    order3_last.append(min_der3+130)

x_3 = np.arange(5, 865)
# plt.plot(x_3, order3_first, '.', c='r')
# plt.plot(x_3, order3_last, '.', c='r')

deg = 2
poly_coeff_first = np.polyfit(x_3, order3_first, deg)
poly_first = np.poly1d(poly_coeff_first)
y_3_first = poly_first(x_3)
plt.plot(x_3, y_3_first-3, '.', c='b')
poly_coeff_last = np.polyfit(x_3, order3_last, deg)
poly_last = np.poly1d(poly_coeff_last)
y_3_last = poly_last(x_3)
plt.plot(x_3, y_3_last+1, '.', c='b')

mask_points = []

for i in range (integration.shape[0]):
    row_data_1 = integration[i, 4:integration.shape[1]-4]
    y_coord_1 = np.full_like(row_data_1, i)
    row_mask_1 = (y_coord_1 >= y_1_first-12) & (y_coord_1 <= y_1_last+10)
    x_val_1 = x_1[row_mask_1]
    y_val_1 = y_coord_1[row_mask_1]
    mask_points.append(np.column_stack((x_val_1, y_val_1)))
    
    row_data_2 = integration[i, 5:1820]
    y_coord_2 = np.full_like(row_data_2, i)
    row_mask_2 = (y_coord_2 >= y_2_first-4) & (y_coord_2 <= y_2_last+3)
    x_val_2 = x_2[row_mask_2]
    y_val_2 = y_coord_2[row_mask_2]
    mask_points.append(np.column_stack((x_val_2, y_val_2)))
    
    row_data_3 = integration[i, 5:865]
    y_coord_3 = np.full_like(row_data_3, i)
    row_mask_3 = (y_coord_3 >= y_3_first-3) & (y_coord_3 <= y_3_last+1)
    x_val_3 = x_3[row_mask_3]
    y_val_3 = y_coord_3[row_mask_3]
    mask_points.append(np.column_stack((x_val_3, y_val_3)))

mask_points = np.vstack(mask_points)
plt.scatter(mask_points[:,0], mask_points[:,1], c='r')
xx
file_name = file.replace('nis_rateints_combined.fits', 'masked_spectra.npy')     
np.save(file_name, mask_points)
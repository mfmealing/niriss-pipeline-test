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

integration = result.data[0]

order_1 = integration[15:90]
high_snr = np.where(order_1>40)
plt.imshow(integration, aspect='auto', vmin=0, vmax=20)
x = high_snr[1]
y = high_snr[0]+15
# plt.plot(x, y, '.', c='y')

deg = 4
poly_coeff = np.polyfit(x, y, deg)
poly = np.poly1d(poly_coeff)
plt.plot(x, poly(x), '.', c='b')
plt.plot(x, poly(x)+12, '.', c='r')
plt.plot(x, poly(x)-10, '.', c='r')


mask = np.ones_like(integration, dtype=bool)
mask[120:, :900] = False
mask[:90] = False
mask[:, 1750:] = False
mask[240:] = False
mask[80:95, 1300:1750] = False

order_2 = integration[80:250, 650:1800]
high_snr_2 = np.where((integration>5) & mask)
x_2 = high_snr_2[1]
y_2 = high_snr_2[0]
# plt.plot(x_2, y_2, '.', c='y')

# deg = 5
# poly_coeff = np.polyfit(x_2, y_2, deg)
# poly = np.poly1d(poly_coeff)
# plt.plot(x_2, poly(x_2), '.', c='b')
# plt.plot(x_2, poly(x_2)+14, '.', c='r')
# plt.plot(x_2, poly(x_2)-14, '.', c='r')


gaps = np.diff(x_2)
gap_min = 10
gap_start_idx = np.where(gaps > gap_min)[0]
gap_start_idx = gap_start_idx[0]

x_before_gap = x_2[gap_start_idx]
x_after_gap = x_2[gap_start_idx + 1]
y_before_gap = y_2[gap_start_idx]
y_after_gap = y_2[gap_start_idx + 1]

poly = Polynomial.fit([x_before_gap, x_after_gap], [y_before_gap, y_after_gap], deg=1)
x_gap = np.linspace(x_before_gap, x_after_gap, 2000)

x_comb = np.concatenate([x_2[:gap_start_idx + 1], x_gap, x_2[gap_start_idx + 1:]])
y_comb = np.concatenate([y_2[:gap_start_idx + 1], poly(x_gap), y_2[gap_start_idx + 1:]])

deg = 4
poly_coeff = np.polyfit(x_comb, y_comb, deg)
poly = np.poly1d(poly_coeff)

x_dense = np.linspace(min(x_comb), max(x_comb), 2000)
plt.plot(x_dense, poly(x_dense), '.', c='b')
plt.plot(x_dense, poly(x_dense)+14, '.', c='r')
plt.plot(x_dense, poly(x_dense)-14, '.', c='r')


# x_comb = x_comb.reshape(-1, 1)
# x_train, x_test, y_train, y_test = train_test_split(x_comb, y_comb, test_size=0.2, random_state=42)

# degrees = [1,2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# train_errors = []
# test_errors = []

# for degree in degrees:
#     poly = PolynomialFeatures(degree)
#     x_poly_train = poly.fit_transform(x_train)
#     x_poly_test = poly.transform(x_test)
    
#     model = LinearRegression()
#     model.fit(x_poly_train, y_train)
    
#     y_train_pred = model.predict(x_poly_train)
#     y_test_pred = model.predict(x_poly_test)
    
#     train_errors.append(mean_squared_error(y_train, y_train_pred))
#     test_errors.append(mean_squared_error(y_test, y_test_pred))

# best_degree = degrees[np.argmin(test_errors)]
# print(f"The best degree based on the test error is: {best_degree}")
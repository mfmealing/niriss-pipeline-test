import random
import tqdm

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
import pastasoss

# jwst imports 
import jwst
print(jwst.__version__)

 
# The entire calwebb_spec2 pipeline
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline

# Individual steps that make up calwebb_spec2 and datamodels
from jwst.assign_wcs.assign_wcs_step import AssignWcsStep
from jwst.background.background_step import BackgroundStep
from jwst.imprint.imprint_step import ImprintStep
from jwst.msaflagopen.msaflagopen_step import MSAFlagOpenStep
from jwst.extract_2d.extract_2d_step import Extract2dStep
from jwst.srctype.srctype_step import SourceTypeStep
from jwst.master_background.master_background_step import MasterBackgroundStep
from jwst.wavecorr.wavecorr_step import WavecorrStep
from jwst.flatfield.flat_field_step import FlatFieldStep
from jwst.straylight.straylight_step import StraylightStep
from jwst.fringe.fringe_step import FringeStep
from jwst.pathloss.pathloss_step import PathLossStep
from jwst.barshadow.barshadow_step import BarShadowStep
from jwst.photom.photom_step import PhotomStep
from jwst.resample import ResampleSpecStep
from jwst.cube_build.cube_build_step import CubeBuildStep
from jwst.extract_1d.extract_1d_step import Extract1dStep

from jwst import datamodels

from jwst.datamodels import dqflags

# miri specific steps
from jwst.rscd import RscdStep
from jwst.firstframe import FirstFrameStep
from jwst.lastframe import LastFrameStep
from jwst.reset import ResetStep
from jwst.gain_scale import GainScaleStep
from jwst.group_scale import GroupScaleStep

# import pipeline_lib

output_dir = './output'
if not os.path.exists(output_dir ): 
    os.makedirs(output_dir )
    

from jwst.stpipe import Step 

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
spectra_mask = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/masked_spectra.npy')
spectra_mask = spectra_mask.astype(int)
spectra_mask_order1 = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/masked_spectra_order_1.npy')
spectra_mask_order1 = spectra_mask_order1.astype(int)
field_mask = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/field_star_mask.npy')
field_mask = field_mask.astype(int)

file_list = ['/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/jw02734002001_04101_00001-seg001_nis/jw02734002001_04101_00001-seg001_nis_rateints.fits',
             '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/jw02734002001_04101_00001-seg002_nis/jw02734002001_04101_00001-seg002_nis_rateints.fits',
             '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/jw02734002001_04101_00001-seg003_nis/jw02734002001_04101_00001-seg003_nis_rateints.fits']
field_file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/jw02734002001_04102_00001-seg001_nis/jw02734002001_04102_00001-seg001_nis_rateints.fits'


for file in file_list:
  
    rateints_file = file
  
    #do stacking
 
    hdul = hdul = fits.open(rateints_file)
    sci = hdul[1].data
    err = hdul[2].data
    dq = hdul[3].data 
    int_times = hdul[4].data  
    varp = hdul[5].data   
    varr = hdul[6].data  
    
    if file == file_list[0]:
        sci_stack = sci
        err_stack = err
        dq_stack = dq
        int_times_stack = int_times
        varp_stack = varp
        varr_stack = varr
 
    else:
        sci_stack = np.vstack((sci_stack, sci))
        err_stack = np.vstack((err_stack, err))
        dq_stack = np.vstack((dq_stack, dq))
        varp_stack = np.vstack((varp_stack, varp))
        varr_stack = np.vstack((varr_stack, varr))
        int_times_stack = np.hstack((int_times_stack, int_times))
   
    hdul.close()
    
# pick first file as the template 
hdul = hdul = fits.open(rateints_file)
  
header = hdul[0].header
header['EXSEGNUM'] = 1
header['EXSEGTOT'] = 1
header['INTSTART'] =  1
header['INTEND'] = header['NINTS'] # assuming all the files have the same total nints in the header
hdul[0].header = header
  
hdul[1].data =  sci_stack
hdul[2].data =  err_stack
hdul[3].data =  dq_stack
hdul[4].data =  int_times_stack
hdul[5].data =  varp_stack
hdul[6].data =  varr_stack
 
outfile0  = rateints_file  
outfile0 = outfile0.replace('jw02734002001_04101_00001-seg003_nis/jw02734002001_04101_00001-seg003_', '')
outfile0 = outfile0.replace('.fits', '_combined.fits')
 
# # # Write the new HDU structure to outfile
hdul.writeto(outfile0, overwrite=True)

hdul.close()

file = outfile0
    
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

mask = np.ones((256,2048), dtype=bool)
mask[spectra_mask[:,1], spectra_mask[:,0]] = False
mask[field_mask[:,1], field_mask[:,0]] = False
mask_3d = np.expand_dims(mask, axis=0)
mask_3d = np.tile(mask_3d, (result.data.shape[0], 1, 1))
bkd_mask = np.where(mask_3d, result.data, np.nan)

bkd_med = np.nanmedian(bkd_mask, axis=1)
bkd_3d = np.expand_dims(bkd_med, axis=1)
bkd_final = np.repeat(bkd_3d, result.data.shape[1], axis=1)
result.data[:,:,:700] = result.data[:,:,:700] - bkd_final[:,:,:700]

result.data[:,251:,:] = result.data[:,:,:5] = result.data[:,:,2043:] = 0
result.err[:,251:,:] = result.err[:,:,:5] = result.err[:,:,2043:] = 0
result.wavelength[251:,:] = result.wavelength[:,:5] = result.wavelength[:,2043:] = 0

nans = np.isnan(result.data)
nans_frac = np.sum(nans, axis=0) / result.data.shape[0]
low_nans = np.array(np.where((nans_frac>0) & (nans_frac<0.1)))
if low_nans.shape[1] > 0:
    result.data[:,low_nans[0],low_nans[1]] = np.apply_along_axis(interpolate_nans, axis=0, arr=result.data[:, low_nans[0], low_nans[1]])
result.data = np.apply_along_axis(interpolate_nans, axis=2, arr=result.data)

# step = Extract1dStep()
# result = step.run(result)

plt.figure('science data')
plt.imshow(result.data[5], aspect='auto', vmin=0, vmax=10)

hdul = fits.open(field_file)
sci = hdul[1].data
err = hdul[2].data
dq = hdul[3].data 
int_times = hdul[4].data  
varp = hdul[5].data   
varr = hdul[6].data  

step = AssignWcsStep()
step.output_dir = output_dir
result_field = step.run(field_file)
wav_field = result_field.wavelength

step = FlatFieldStep()
result_field = step.run(result_field)

section = result_field.data[:,210:250,720:770]
bkd_section = bkd_model[210:250,720:770]
section_med = np.nanmedian(section, axis=0)
scale_arr = section_med / bkd_section
scale_val = np.nanmedian(scale_arr)
scaled_bkd = scale_val * bkd_model
result_field.data = result_field.data - scaled_bkd

mask = np.ones((256,2048), dtype=bool)
mask[spectra_mask[:,1], spectra_mask[:,0]] = False
# mask[field_mask[:,1], field_mask[:,0]] = False
mask_3d = np.expand_dims(mask, axis=0)
mask_3d = np.tile(mask_3d, (result_field.data.shape[0], 1, 1))
bkd_mask = np.where(mask_3d, result_field.data, np.nan)
bkd_med = np.nanmedian(bkd_mask, axis=1)
bkd_3d = np.expand_dims(bkd_med, axis=1)
bkd_final = np.repeat(bkd_3d, result_field.data.shape[1], axis=1)
result_field.data[:,:,:700] = result_field.data[:,:,:700] - bkd_final[:,:,:700]

result_field.data[:,251:,:] = result_field.data[:,:,:5] = result_field.data[:,:,2043:] = 0

nans = np.isnan(result_field.data)
nans_frac = np.sum(nans, axis=0) / result_field.data.shape[0]
low_nans = np.array(np.where((nans_frac>0) & (nans_frac<0.1)))
if low_nans.shape[1] > 0:
    result_field.data[:,low_nans[0],low_nans[1]] = np.apply_along_axis(interpolate_nans, axis=0, arr=result_field.data[:,low_nans[0],low_nans[1]])
result_field.data = np.apply_along_axis(interpolate_nans, axis=2, arr=result_field.data)

med_sci = np.median(result.data[:11], axis=0)

med_field = np.median(result_field.data, axis=0)
med_field[:,:500] = 0
# plt.figure('field star data')
# plt.imshow(med_field, aspect='auto', vmin=0, vmax=1)
med_field_3d = np.expand_dims(med_field, axis=0)
field_star = np.repeat(med_field_3d, result.data.shape[0], axis=0)
scale = med_field / np.max(med_field)

plt.figure('scaled field star data')
plt.imshow(scale, aspect='auto', vmin=0, vmax=0.5)

plt.figure()
plt.imshow(scale, aspect='auto', vmin=0, vmax=0.5)

sd_scale1 = 3
sd_scale2 = 6

cutoff = 1700
bkd1 = np.array(np.where(scale[:,:cutoff]>(sd_scale1*np.std(scale))))
bkd2 = np.array(np.where(scale[:,cutoff:]>(sd_scale2*np.std(scale))))
bkd2[1] += cutoff
plt.plot(bkd1[1], bkd1[0], '.', c='r')
plt.plot(bkd2[1], bkd2[0], '.', c='r')
bkd_all = np.hstack((bkd1,bkd2))
y,x = bkd_all

# result.data = result.data - field_star

for i,j in zip(y,x):
    alpha = np.dot(result.data[:,i,j],field_star[:,i,j]) / np.dot(field_star[:,i,j],field_star[:,i,j])
    alpha = np.clip(alpha, 0, 5)
    
    result.data[:,i,j] = result.data[:,i,j] - field_star[:,i,j]*alpha

# med_scale = med_sci / med_field

# plt.figure('scale')
# plt.imshow(med_scale, aspect='auto', vmin=0, vmax=10)

# result.data = result.data - field_star*med_scale

plt.figure('after field star removal')
plt.imshow(result.data[5], aspect='auto', vmin=0, vmax=10)
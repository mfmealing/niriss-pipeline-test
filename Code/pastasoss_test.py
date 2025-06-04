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

stage_1_file_list = ['/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/jw02734002001_04101_00001-seg001_nis/jw02734002001_04101_00001-seg001_nis_rateints.fits',
             '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/jw02734002001_04101_00001-seg002_nis/jw02734002001_04101_00001-seg002_nis_rateints.fits',
             '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/jw02734002001_04101_00001-seg003_nis/jw02734002001_04101_00001-seg003_nis_rateints.fits']

for file in stage_1_file_list:
  
    rateints_file = file
  
    #do stacking
 
    hdul = hdul = fits.open(rateints_file)
    sci = hdul[1].data
    err = hdul[2].data
    dq = hdul[3].data 
    int_times = hdul[4].data  
    varp = hdul[5].data   
    varr = hdul[6].data  
    
    if file == stage_1_file_list[0]:
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
result.data[:,low_nans[0],low_nans[1]] = np.apply_along_axis(interpolate_nans, axis=0, arr=result.data[:,low_nans[0],low_nans[1]])
result.data = np.apply_along_axis(interpolate_nans, axis=2, arr=result.data)

plt.figure('after nan removal')
plt.imshow(result.data[50], aspect='auto', vmin=0, vmax=5)

wav = np.nanmean(result.wavelength, axis=0)

pwcpos = result.meta.instrument.pupil_position
trace1 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='1', interp=True) 
trace2 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='2', interp=True)

x1, y1, wav1 = trace1.x, trace1.y, trace1.wavelength
x2, y2, wav2 = trace2.x, trace2.y, trace2.wavelength

plt.plot(x1,y1, lw=1.5, color='cornflowerblue')
plt.plot(x2,y2, lw=1.5, color='orangered')

points1 = np.linspace(0, len(x1)-1, 5).astype(int)
points2 = np.linspace(0, len(x2)-1, 5).astype(int)

for i in points1:
    if i == points1[0]:
        plt.text(x1[i]+75, y1[i]-10, f"{wav1[i]:.1f}", ha="center", fontsize=12, color='cornflowerblue')
    elif i == points1[-1]:
        plt.text(x1[i]-75, y1[i]-25, f"{wav1[i]:.1f}", ha="center", fontsize=12, color='cornflowerblue')
    else:
        plt.text(x1[i], y1[i]-10, f"{wav1[i]:.1f}", ha="center", fontsize=12, color='cornflowerblue')

for j in points2:
    if j == points2[0]:
        plt.text(x2[j]+75, y2[j]-10, f"{wav2[j]:.1f}", ha="center", fontsize=12, color='orangered')
    elif j == points2[-1]:
        plt.text(x2[j]+75, y2[j]-10, f"{wav2[j]:.1f}", ha="center", fontsize=12, color='orangered')
    else:
        plt.text(x2[j], y2[j]-10, f"{wav2[j]:.1f}", ha="center", fontsize=12, color='orangered')

result_med = np.nanmedian(result.data, axis=0)
npix = 20

flux1 = [result_med[int(y)-npix:int(y)+npix, int(x)].sum() for x, y in zip(x1, y1)]
flux2 = [result_med[int(y)-npix:int(y)+npix, int(x)].sum() for x, y in zip(x2, y2)]
xx
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
fig.suptitle('Extracted SOSS Spectra')
ax1.set_title('Order 1')
ax1.plot(wav1, flux1, lw=1.5, color='cornflowerblue')
ax1.set_xlabel('wavelength [um]')
ax1.set_ylabel('DN/s')


ax2.set_title('Order 2')
ax2.plot(wav2, flux2, lw=1.5, color='orangered')
ax2.set_xlabel('wavelength [um]')
ax2.set_ylabel('DN/s')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4), dpi=187)
plt.plot(wav1, flux1, lw=1.5, color='cornflowerblue')
plt.xlabel('Wavelength [um]')
plt.ylabel('DN/s')
plt.text(1.25, 22000, 'Order 1', color='cornflowerblue')
plt.xticks(color="cornflowerblue")
plt.twiny()
plt.plot(wav2, flux2, lw=1.5, color='orangered')
plt.text(0.7, 12000, 'Order 2', color='orangered')
plt.xticks(color="orangered")
plt.ylabel('DN/s (order 2)')
plt.xlabel('Wavelength [um]')
# plt.legend()

plt.tight_layout()
plt.show()
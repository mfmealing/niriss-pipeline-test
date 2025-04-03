import os
os.environ['CRDS_PATH'] ='/Users/c24050258/crds_cache'
os.environ['CRDS_SERVER_URL'] ='https://jwst-crds.stsci.edu'
# #this must come BEFORE jwst is imported

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

# jwst imports 
import jwst
print(jwst.__version__)

from jwst.pipeline import calwebb_detector1

# Individual steps that make up calwebb_detector1
from jwst.dq_init import DQInitStep
from jwst.saturation import SaturationStep
from jwst.superbias import SuperBiasStep
from jwst.ipc import IPCStep                                                                                    
from jwst.refpix import RefPixStep                                                                
from jwst.linearity import LinearityStep
from jwst.persistence import PersistenceStep
from jwst.dark_current import DarkCurrentStep
from jwst.jump import JumpStep
from jwst.ramp_fitting import RampFitStep
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

output_dir = './output'
if not os.path.exists(output_dir ): 
    os.makedirs(output_dir )
    

from jwst.stpipe import Step 

bkd_model = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/model_background256.npy')
spectra_mask = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/Masked_Spectra.npy')
spectra_mask = spectra_mask.astype(int)
stage_1_file_list = ['/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/jw02722003001_04101_00001-seg001_nis/jw02722003001_04101_00001-seg001_nis_uncal.fits',
                     '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/jw02722003001_04101_00001-seg002_nis/jw02722003001_04101_00001-seg002_nis_uncal.fits',
                     '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/jw02722003001_04101_00001-seg003_nis/jw02722003001_04101_00001-seg003_nis_uncal.fits',
                     '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/jw02722003001_04101_00001-seg004_nis/jw02722003001_04101_00001-seg004_nis_uncal.fits']

for file in stage_1_file_list:
  
    rateints_file = file
  
    # in primary header
    #must change EXSEGNUM = 1
    # EXSEGTOT = 1
    # INTSTART = 1
    # INTEND = NINTS
    #other headers don't need changing
    # stack SCI, GROUP and INT_TIMES
    # asdf extension appers to be same in each segment (looking at the plot)
  
    #do stacking
 
    hdul = hdul = fits.open(rateints_file)
    sci = hdul[1].data
    # err = hdul[2].data
    # dq = hdul[3].data 
    # int_times = hdul[4].data  
    # varp = hdul[5].data   
    # varr = hdul[6].data  
    
    if file == stage_1_file_list[0]:
        sci_stack = sci
        # err_stack = err
        # dq_stack = dq
        # int_times_stack = int_times
        # varp_stack = varp
        # varr_stack = varr
 
    else:
        sci_stack = np.vstack((sci_stack, sci))
        # err_stack = np.vstack((err_stack, err))
        # dq_stack = np.vstack((dq_stack, dq))
        # varp_stack = np.vstack((varp_stack, varp))
        # varr_stack = np.vstack((varr_stack, varr))
        # int_times_stack = np.hstack((int_times_stack, int_times))
   
    hdul.close()
    
# sci_sum = np.nansum(sci_stack, axis=1)
# sci_sum_final = np.nansum(sci_sum, axis=1)
# plt.figure('img')
# plt.plot(sci_sum_final, '.')
# xx

# pick first file as the template 
# rateints_file = '/Users/user1/Downloads/JWST webinars/NIRSpec_grating/fits_files/%s_%s_run_1_rateints.fits'%(nrs,seg_list[0])
hdul = hdul = fits.open(rateints_file)
  
header = hdul[0].header
header['EXSEGNUM'] = 1
header['EXSEGTOT'] = 1
header['INTSTART'] =  1
header['INTEND'] = header['NINTS'] # assuming all the files have the same total nints in the header
hdul[0].header = header
  
hdul[1].data =  sci_stack
# hdul[2].data =  err_stack
# hdul[3].data =  dq_stack
# hdul[4].data =  int_times_stack
# hdul[5].data =  varp_stack
# hdul[6].data =  varr_stack


 
outfile0  = rateints_file  
idx = outfile0.find('.fits')
aa = outfile0[idx:]
tag = 'test'
outfile0  = outfile0.replace(aa, '%s.fits'%(tag))
 
  
# idx = outfile0.find('seg')
  
# aa = outfile0[idx:idx+6]
  
# outfile0  = outfile0.replace(aa, 'COMBINED')
 
# # # Write the new HDU structure to outfile
hdul.writeto(outfile0, overwrite=True)
  
hdul.close()
  
rateints_file = outfile0
    
hdul = fits.open(rateints_file)
sci = hdul[1].data
# err = hdul[2].data
# dq = hdul[3].data 
# int_times = hdul[4].data  
# varp = hdul[5].data   
# varr = hdul[6].data  
     

step = GroupScaleStep()
result = step.run(rateints_file)
 
step = DQInitStep()
result = step.run(result)
        
step = SaturationStep()
result = step.run(result)
 
step = SuperBiasStep()
result = step.run(result)

group_flux = []
for k in range(result.data.shape[1]):
    group = result.data[:,k,:,:]
    group_med = np.nanmedian(group, axis=0)
    group_3d = np.expand_dims(group_med, axis=0)
    flux = np.repeat(group_3d, result.data.shape[0], axis=0)
    group_flux.append(flux)

flux_final = np.stack(group_flux, axis=1)
f_noise = result.data - flux_final

# plt.figure('before 1/f subtraction')
# plt.imshow(result.data[100,3], aspect='auto', vmin=0, vmax=300)

mask = np.ones((256,2048), dtype=bool)
mask[spectra_mask[:,1], spectra_mask[:,0]] = False
mask_4d = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)
mask_4d = np.tile(mask_4d, (result.data.shape[0], result.data.shape[1], 1, 1))
f_noise_mask = np.where(mask_4d, f_noise, np.nan)
f_noise_med = np.nanmedian(f_noise_mask, axis=2)
f_noise_4d = np.expand_dims(f_noise_med, axis=2)
f_noise_final = np.repeat(f_noise_4d, result.data.shape[2], axis=2)
result.data = result.data - f_noise_final

step = LinearityStep()
result = step.run(result)
      
step = DarkCurrentStep()
result = step.run(result)
 
step = JumpStep()
step.rejection_threshold  = 5
print ("step.rejection_threshold", step.rejection_threshold)
result = step.run(result)           
        
step = RampFitStep()
result = step.run(result)[1]
     
step = GainScaleStep()
result = step.run(result)

flux_sum = np.nansum(result.data, axis=1)
flux_sum_2 = np.nansum(flux_sum, axis=1)

plt.figure('wlc')
plt.plot(flux_sum_2, '.')
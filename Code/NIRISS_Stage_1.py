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
    
from jwst.stpipe import Step 

bkd_model = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/model_background256.npy')
spectra_mask = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_17b_NIRISS/masked_spectra.npy')
spectra_mask = spectra_mask.astype(int)
# field_mask = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_17b_NIRISS/field_star_mask.npy')
# field_mask = field_mask.astype(int)

seg_list = ['001', '002', '003', '004', '005']

for seg in seg_list: 

    file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_17b_NIRISS/jw01353101001_04101_00001-seg%s_nis/jw01353101001_04101_00001-seg%s_nis_uncal.fits'%(seg, seg)
    result = file
    
    step = GroupScaleStep()
    result = step.run(result)
     
    step = DQInitStep()
    result = step.run(result)
            
    step = SaturationStep()
    result = step.run(result)
     
    step = SuperBiasStep()
    result = step.run(result)
    
    file_name = file.replace('uncal', 'pre_1f')
    result.save(file_name) 
        
    hdul = hdul = fits.open(file_name)
    sci = hdul[1].data
    
    if seg == seg_list[0]:
        sci_stack = sci
        
    else:
        sci_stack = np.vstack((sci_stack, sci))
    hdul.close()
   
group_flux = []
for k in range(sci_stack.shape[1]):
    group = sci_stack[:,k,:,:]
    group_med = np.nanmedian(group, axis=0)
    group_flux.append(group_med)

med_stack_final = np.stack(group_flux, axis=0)
np.save('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_17b_NIRISS/group_med_pre_1f.npy', med_stack_final)
group_med_stack = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_17b_NIRISS/group_med_pre_1f.npy')

for seg in seg_list: 

    file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_17b_NIRISS/jw01353101001_04101_00001-seg%s_nis/jw01353101001_04101_00001-seg%s_nis_uncal.fits'%(seg, seg)
    result = file
    
    step = GroupScaleStep()
    result = step.run(result)
     
    step = DQInitStep()
    result = step.run(result)
    
    step = SaturationStep()
    result = step.run(result)
    
    step = SuperBiasStep()
    result = step.run(result)
    
    group_med_4d = np.expand_dims(group_med_stack, axis=0)
    flux_final = np.repeat(group_med_4d, result.data.shape[0], axis=0)
    f_noise = result.data - flux_final
    
    mask = np.ones((256,2048), dtype=bool)
    mask[spectra_mask[:,1], spectra_mask[:,0]] = False
    # mask[field_mask[:,1], field_mask[:,0]] = False
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
    
    plt.figure('img 1')
    plt.imshow(result.data[0], aspect='auto')
    plt.show()
    
    file_name = file.replace('uncal', 'rateints')     
    result.save(file_name)
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
    
spectra_mask = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/Masked_Spectra.npy')
spectra_mask = spectra_mask.astype(int)
bkd_model = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/model_background256.npy')
wlc = []    

from jwst.stpipe import Step 

seg_list = ['001', '002', '003', '004']

for seg in seg_list: 

    file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/jw02722003001_04101_00001-seg%s_nis/jw02722003001_04101_00001-seg%s_nis_uncal.fits'%(seg, seg)
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

flux_final = np.stack(group_flux, axis=0)
np.save('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/group_med_pre_1f.npy', flux_final)
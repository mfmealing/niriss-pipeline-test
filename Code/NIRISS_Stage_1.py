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
          
    # step = RefPixStep()
    # result = step.run(result)
    
    # plt.figure('before partial background removal')
    # plt.imshow(result.data[0,7], aspect='auto', vmin=0, vmax=300)
    
    # data_med = np.nanmedian(result.data, axis=1)
    
    # section_700 = data_med[:,210:250,250:500]
    # bkd_section_700 = bkd_model[210:250,250:500]
    # section_med_700 = np.nanmedian(section_700, axis=0)
    # # if i == 1:
    # #     section_med_ = section_med
    # scale_arr_700 = section_med_700 / bkd_section_700
    # scale_val_700 = np.nanmedian(scale_arr_700)
    # scaled_bkd_700 = scale_val_700 * bkd_model
    # result.data[:,:,:,:700] = result.data[:,:,:,:700] - scaled_bkd_700[:,:700]
    
    # plt.figure('after partial background removal')
    # plt.imshow(result.data[0,7], aspect='auto', vmin=0, vmax=300)
    
    # section = data_med[:,210:250,750:850]
    # bkd_section = bkd_model[210:250,750:850]
    # section_med = np.nanmedian(section, axis=0)
    # # if i == 1:
    # #     section_med_ = section_med
    # scale_arr = section_med / bkd_section
    # scale_val = np.nanmedian(scale_arr)
    # scaled_bkd = scale_val * bkd_model
    # result.data[:,:,:,700:] = result.data[:,:,:,700:] - scaled_bkd[:,700:]
    
    # plt.figure('after full background removal')
    # plt.imshow(result.data[0,7], aspect='auto', vmin=0, vmax=300)
    
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
    
    # plt.figure('after 1/f subtraction')
    # plt.imshow(result.data[0,7], aspect='auto', vmin=0, vmax=300)

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
    
    
    # print(result.data.shape)
    plt.figure('img 1')
    plt.imshow(result.data[0], aspect='auto')
    plt.show()
    
    file_name = file.replace('uncal', 'rateints')     
    result.save(file_name) 
    
    flux_sum = np.nansum(result.data, axis=1)
    flux_sum_2 = np.nansum(flux_sum, axis=1)
    wlc.append(flux_sum_2)

start_index = 0
for j in wlc:
    x_values = np.arange(start_index, start_index + len(j))
    plt.figure('combined curve')
    plt.plot(x_values, j, '.')
    start_index += len(j)
# plt.figure('img 2')
# plt.plot(wlc, '.')
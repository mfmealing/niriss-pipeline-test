#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:28:19 2023

@author: user1

"""

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

for i in range(1,5):

    file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/jw02722003001_04101_00001-seg00'+str(i)+'_nis/jw02722003001_04101_00001-seg00'+str(i)+'_nis_uncal.fits'  
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
    
    group_flux = []
    for k in range(result.data.shape[1]):
        group = result.data[:,k,:,:]
        group_med = np.nanmedian(group, axis=0)
        group_3d = np.expand_dims(group_med, axis=0)
        flux = np.repeat(group_3d, result.shape[0], axis=0)
        group_flux.append(flux)
    
    flux_final = np.stack(group_flux, axis=1)
    f_noise = result.data - flux_final
    
    plt.imshow(f_noise[0,0,:,:], vmin=-40,  vmax=40)
    plt.show()
    xx
    
    for j in range(f_noise.shape[0]):
        first_pix = f_noise[j,:,0:5,:]
        last_pix = f_noise[j,:,-5:,:]
        f_noise_stack = np.hstack((first_pix,last_pix))
        f_noise_med = np.nanmedian(f_noise_stack, axis=1)
        f_noise_3d = np.expand_dims(f_noise_med, axis=1)
        f_noise_final = np.repeat(f_noise_3d, result.shape[2], axis=1)
        result.data[j] = result.data[j] - f_noise_final
        

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
    
    
    print(result.data.shape)
    plt.figure('img 1')
    plt.imshow(result.data[0], aspect='auto')
    plt.show()
    
    file_name = file.replace('uncal', 'rateints')     
    result.save(file_name) 
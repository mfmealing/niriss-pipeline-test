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
          
    step = RefPixStep()
    result = step.run(result)
    
    for j in range(result.data.shape[0]):
        first_pix = result.data[j,:,0:5,:]
        last_pix = result.data[j,:,-5:,:]
        bkg_stack = np.hstack((first_pix,last_pix))
        bkg_med = np.nanmedian(bkg_stack, axis=1)
        bkg_3d = np.expand_dims(bkg_med, axis=1)
        bkg = np.repeat(bkg_3d, result.shape[2], axis=1)
        result.data[j] = result.data[j] - bkg
            
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
    
    
    print (result.data.shape)
    plt.figure('img 1')
    plt.imshow(result.data[0], aspect='auto')
    plt.show()
    
    file_name = file.replace('uncal', 'rateints')     
    result.save(file_name) 
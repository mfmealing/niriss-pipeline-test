
"""
    Bit	Value	Name	Description
    0	1	DO_NOT_USE	Bad pixel. Do not use.
    1	2	SATURATED	Pixel saturated during exposure
    2	4	JUMP_DET	Jump detected during exposure
    3	8	DROPOUT	Data lost in transmission
    4	16	RESERVED	 
    5	32	RESERVED	 
    6	64	RESERVED	 
    7	128	RESERVED	 
    8	256	UNRELIABLE_ERROR	Uncertainty exceeds quoted error
    9	512	NON_SCIENCE	Pixel not on science portion of detector
    10	1024	DEAD	Dead pixel
    11	2048	HOT	Hot pixel
    12	4096	WARM	Warm pixel
    13	8192	LOW_QE	Low quantum efficiency
    14	16384	RC	RC pixel
    15	32768	TELEGRAPH	Telegraph pixel
    16	65536	NONLINEAR	Pixel highly nonlinear
    17	131072	BAD_REF_PIXEL	Reference pixel cannot be used
    18	262144	NO_FLAT_FIELD	Flat field cannot be measured
    19	524288	NO_GAIN_VALUE	Gain cannot be measured
    20	1048576	NO_LIN_CORR	Linearity correction not available
    21	2097152	NO_SAT_CHECK	Saturation check not available
    22	4194304	UNRELIABLE_BIAS	Bias variance large
    23	8388608	UNRELIABLE_DARK	Dark variance large
    24	16777216	 UNRELIABLE_SLOPE	Slope variance large (i.e., noisy pixel)
    25	33554432	 UNRELIABLE_FLAT	Flat variance large
    26	67108864 	OPEN	Open pixel (counts move to adjacent pixels)
    27	134217728	ADJ_OPEN	Adjacent to open pixel
    28	268435456	UNRELIABLE_RESET	Sensitive to reset anomaly
    29	536870912	MSA_FAILED_OPEN	Pixel sees light from failed-open shutter
    30	1073741824	OTHER_BAD_PIXEL	A catch-all flag   
    """
    

# from multiprocessing import Process, Queue

# from numba import jit, prange

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

spectra_mask = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/Masked_Spectra.npy')
spectra_mask = spectra_mask.astype(int)

combined_array = []

seg_list = ['001', '002', '003', '004']

for seg in seg_list: 

    file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/jw02722003001_04101_00001-seg%s_nis/jw02722003001_04101_00001-seg%s_nis_rateints.fits'%(seg, seg)
    
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
    if seg == seg_list[0]:
        section_med_ = section_med
    scale_arr = section_med_ / bkd_section
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

    result.data = result.data[:,:251,5:2043]
    result.err = result.err[:,:251,5:2043]
    result.wavelength = result.wavelength[:251,5:2043]

    nans = np.isnan(result.data)
    nans_frac = np.sum(nans, axis=0) / result.data.shape[0]
    low_nans = np.array(np.where((nans_frac>0) & (nans_frac<0.1)))
    result.data[:,low_nans[0],low_nans[1]] = np.apply_along_axis(interpolate_nans, axis=0, arr=result.data[:,low_nans[0],low_nans[1]])
    result.data = np.apply_along_axis(interpolate_nans, axis=2, arr=result.data)
    
    plt.figure('after nan removal')
    plt.imshow(result.data[50], aspect='auto', vmin=0, vmax=5)
    
    wav = np.nanmean(result.wavelength, axis=0)
    
    # # =============================================================================
    # #         box extraction
    # # =============================================================================
     
    print ('extracting 1D spectra with box extraction')
    # for intg in range(sci.shape[0]):
        
    
    seq = np.arange(result.data.shape[0])  
    from tqdm import tqdm
    flux_array = np.zeros((result.data.shape[0], result.data.shape[2]))
    flux_var_array = np.zeros((result.data.shape[0], result.data.shape[2]))
             
    for intg in tqdm(seq):
       
          # print ('extracting 1D spectrum from integration... %s'%(intg))
       
          img = result.data[intg]
          
          img_err = result.err[intg]
          img_var = img_err**2
          
          flux_simple  = np.sum(img, axis=0)
          
          flux_array[intg] = flux_simple
          flux_var_array[intg] = np.sum(img_var, axis=0)
          
    
    n = np.arange(100.0)
    hdu= fits.PrimaryHDU(n)
    hdul = fits.HDUList([hdu])
    table_hdu  = fits.BinTableHDU(data=int_times)
    hdul.append(table_hdu)
    hdul[1].header['EXTNAME']= 'INT_TIMES'
    
    hdul.append(fits.ImageHDU(np.ones(10)))
    hdul[2].header['EXTNAME']= 'SPEC'
    hdul[2].data= flux_array
    
    hdul.append(fits.ImageHDU(np.ones(10)))
    hdul[3].header['EXTNAME']= 'WAV'
    hdul[3].data= wav
    
    hdul.append(fits.ImageHDU(np.ones(10)))
    hdul[4].header['EXTNAME']= 'ERR'
    hdul[4].data=  flux_var_array**0.5
      
    filename = file.replace('rateints','1Dspec_box_extract')
    hdul.writeto(filename, overwrite=True)
    
    white_lc = np.nansum(flux_array, axis=1)
    combined_array.append(white_lc)
    
    plt.figure('seperate curves')
    plt.plot(white_lc, '.')
    plt.show()
    
    
start_index = 0

for j in combined_array:
    x_values = np.arange(start_index, start_index + len(j))
    plt.figure('combined curve')
    plt.plot(x_values, j, '.', c='b')
    start_index += len(j)

plt.show()    
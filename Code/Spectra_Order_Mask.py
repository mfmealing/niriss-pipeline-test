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

# Individual steps that make up calwebb_spec2 and datamodels
from jwst.assign_wcs.assign_wcs_step import AssignWcsStep

output_dir = './output'
if not os.path.exists(output_dir ): 
    os.makedirs(output_dir )
    

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

integration = result.data[0]

order_1 = integration[15:90]
high_snr = np.where(order_1>50)
plt.imshow(result.data[0], aspect='auto')
x = high_snr[1]
y = high_snr[0]+15
# plt.plot(x, y, '.', c='y')

deg = 4
poly_coeff = np.polyfit(x, y, deg)
poly = np.poly1d(poly_coeff)
plt.plot(x, poly(x), '.', c='b')
plt.plot(x, poly(x)+12, '.', c='r')
plt.plot(x, poly(x)-10, '.', c='r')
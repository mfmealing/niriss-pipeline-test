import numpy as np
import matplotlib.pyplot as plt

cc = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/nis_rateints_combined_med.npy'
med = np.load(cc)
# plt.figure('med data')
# plt.imshow(med, aspect='auto', vmin=0, vmax=10)

dd = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/jw02734002001_04102_00001-seg001_nis/jw02734002001_04102_00001-seg001_nis_F277W_med.npy'
f2 = np.load(dd)
# plt.figure('med F227W')
# plt.imshow(f2, aspect='auto', vmin=0, vmax=10)

spectra_mask_order1 = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/masked_spectra_order_1.npy')
spectra_mask_order1 = spectra_mask_order1.astype(int)

mask_order1 = np.zeros((256,2048), dtype=bool)
mask_order1[spectra_mask_order1[:,1], spectra_mask_order1[:,0]] = True
box_mask_order1 = np.where(mask_order1, med, np.nan)
f2_mask_order1 = np.where(mask_order1, f2, np.nan)

seq = np.arange(430)  
from tqdm import tqdm
flux_array = np.zeros(430)
f2_flux_array = np.zeros(430)

for col in tqdm(seq):
      img = box_mask_order1[:,col]
      flux  = np.nansum(img, axis=0)
      flux_array[col] = flux
      
      f2_img = f2_mask_order1[:,col]
      f2_flux = np.nansum(f2_img, axis=0)
      f2_flux_array[col] = f2_flux

plt.figure('spectra')
plt.plot(seq, flux_array, '.')
plt.plot(seq, f2_flux_array, '.')
plt.show()

sec = box_mask_order1[:,:430]
f2_sec = f2_mask_order1[:,:430]

scale = sec / f2_sec
scale_med = np.nanmedian(scale)

# plt.figure()
# plt.imshow(scale, aspect='auto', vmin=0, vmax=10)

med = med - (f2*scale_med)

plt.figure()
plt.imshow(med, aspect='auto', vmin=0, vmax=10)
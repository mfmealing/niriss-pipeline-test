import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

vals = np.load('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/spectra_vals_all_wav.npy')
wav, depth, depth_err = vals[0], vals[1], vals[2]

with open("/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/K2_18b_NIRISS/spectrum_vals_lit.txt", "r") as f:
    header = f.readline().strip().split(",")
    vals_true = pd.read_csv(f, sep=' ', names=header)

wav_true, depth_true, depth_err_true = vals_true['Wavelength (microns)'], vals_true[' transit depth'], vals_true[' uncertainty']

plt.figure('transmission spectrum')
plt.errorbar(wav, depth*100, depth_err*100, fmt='bo', label='Current Work')
plt.errorbar(wav_true, depth_true*100, depth_err_true*100, fmt='ro', label='Madhusudhan 2023')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Transit Depth (%)')
plt.xlim(0.7,3)
plt.legend(loc=0)
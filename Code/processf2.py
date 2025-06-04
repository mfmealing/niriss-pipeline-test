#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 16:36:34 2025

@author: c1341133

Produces the zeroth order image for subtraction

"""

import numpy as np
import matplotlib.pyplot as plt


# aa = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/jw02734002001_04102_00001-seg001_nis/jw02734002001_04102_00001-seg001_nis_F277W.npy'
# f2 = np.load(aa)
# plt.figure('F277W')
# plt.imshow(f2[0], aspect='auto', vmin=0, vmax=20)

# bb = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/nis_rateints_combined.npy'
# data = np.load(bb)
# plt.figure('data')
# plt.imshow(data[0], aspect='auto', vmin=0, vmax=20)

# med = np.median(data, axis=0)
# np.save('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/nis_rateints_combined_med.npy', med)

cc = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/nis_rateints_combined_med.npy'
med = np.load(cc)
# plt.figure('med')
# plt.imshow(med, aspect='auto', vmin=0, vmax=20)

dd = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/NIRISS_Pipeline_Test/Data/WASP_96b_NIRISS/jw02734002001_04102_00001-seg001_nis/jw02734002001_04102_00001-seg001_nis_F277W_med.npy'
f2 = np.load(dd)
# plt.figure('F277W med')
# plt.imshow(f2, aspect='auto', vmin=0, vmax=20)

xmin = 735
xmax = 775
ymin=113
ymax = 165

sec = med[ymin:ymax,xmin:xmax]
# plt.figure('sec')
# plt.imshow(sec, aspect='auto', vmin=0, vmax=5)
print(np.argmax(sec))


bkg = med[ymin-5:ymin, xmin:xmax]
bkg_med = np.nanmedian(bkg,axis=0)
sec = sec-bkg_med
# plt.figure('sec after bkg sub')
# plt.imshow(sec, aspect='auto', vmin=0, vmax=5)

# bkg = np.hstack((sec[:,0:5], sec[:,-5:]))
# print (bkg.shape)
# plt.figure('bkg')
# plt.imshow(bkg, aspect='auto', vmin=0, vmax=50)
# bkg = np.median(bkg, axis=1)
# sec= sec-bkg[:, np.newaxis] 


fsec = f2[ymin:ymax,xmin:xmax]
# plt.figure('fsec')
# plt.imshow(fsec, aspect='auto', vmin=0, vmax=5)
print(np.unravel_index(np.argmax(fsec), fsec.shape))
print(fsec.shape)


fsec_max_y = np.unravel_index(np.argmax(fsec), fsec.shape)[0]
fsec_max_x = np.unravel_index(np.argmax(fsec), fsec.shape)[1]
fsec_y = fsec.shape[0]
fsec_x = fsec.shape[1]

corr = sec/fsec
corr_scale = corr*fsec

plt.figure('correction scale check')
plt.imshow(corr_scale, aspect='auto', vmin=0, vmax=5)
 
f2[:,0:500]=0
f2[np.isnan(f2)]=0
plt.figure('f2')
plt.imshow(f2, aspect='auto', vmin=0, vmax=5)

 
 
buffer = np.zeros((50,f2.shape[1]))
f2 = np.vstack((buffer, f2, buffer))
buffer2 = np.zeros((50,f2.shape[0])).T
f2 = np.hstack((buffer2, f2, buffer2))

plt.figure('f2 w buffer')
plt.imshow(f2, aspect='auto', vmin=0, vmax=5)

f0 = np.zeros_like(f2)

for i in range(30):
    
    max_pos = np.unravel_index(np.argmax(f2), f2.shape)
    max_scale = np.max(f2)/np.max(fsec)
    
    # f0[max_[0]-26:  max_[0]-26+50,  max_[1]-19:max_[1]-19+40] = fsec*max_scale*corr/20

    # f2 = f0*1 
    # f2[max_[0]-26:  max_[0]-26+50,  max_[1]-19:max_[1]-19+40] = 0
    # max_ = np.unravel_index(np.argmax(f2), f2.shape)
    # f0[max_[0]-26:  max_[0]-26+50,  max_[1]-19:max_[1]-19+40] += fsec*max_scale*corr/20
    # plt.imshow(f0*20, aspect='auto', vmin=0, vmax=50)
    
    f2[max_pos[0] - fsec_max_y:  max_pos[0] - fsec_max_y + fsec_y,  max_pos[1] - fsec_max_x:max_pos[1] - fsec_max_x + fsec_x] = 0
    f0[max_pos[0] - fsec_max_y:  max_pos[0] - fsec_max_y + fsec_y,  max_pos[1] - fsec_max_x:max_pos[1] - fsec_max_x + fsec_x] += fsec * max_scale * corr


    
plt.figure('f0 w addition')
plt.imshow(f0, aspect='auto', vmin=0, vmax=f0.max())

 
f0 = f0[50:-50]
f0 = f0[:,50:-50]

plt.figure('f44 before')
plt.imshow(med, aspect='auto', vmin=0, vmax=50)   

med_final = med - f0

plt.figure('f44 after')
plt.imshow(med_final, aspect='auto', vmin=0, vmax=50)   

xxx 

# np.save('./zeroth_order_img.npy', f2)

# =============================================================================
#  
# =============================================================================
# plt.figure('f44med')
# plt.imshow(med, aspect='auto', vmin=0, vmax=50)   

# sssss 

# pp = f22 -f2

# plt.figure('pp')
# plt.imshow(pp*21, aspect='auto', vmin=0, vmax=50)   

# xxx


# ccol = 960
# ccol = 741
# # ccol = 1326
# # ccol = 1140


# plt.figure('ooo')
# plt.plot(med[:,ccol])
# plt.plot(f2[:,ccol]*21.5)

# np.save('./f2.npy', f2)
 

# xxx


# col = 738
# # col =1222
# plt.figure('qqq')
# data_col = data[0][:,col]
# f2_col = f2[:,col]
# plt.plot(data_col)
# plt.plot(f2_col)

# x = np.arange(data.shape[1])
# idx = np.argsort(f2_col)
# f2_col_sorted = f2_col[idx]
# x_sorted = x[idx]

# plt.figure('ppp')
# plt.plot(f2_col_sorted, 'o-')
# idx = x_sorted[::-1][0:15]
# plt.figure('qqq')
# plt.plot(x[idx], f2_col[idx], 'bo')
# plt.plot(x[idx], data_col[idx], 'ro')

# scale = np.mean(data_col[idx]/f2_col[idx])
# f2_col *=scale
# plt.plot(f2_col, 'r--')
# print (scale)







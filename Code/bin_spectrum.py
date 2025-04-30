#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:22:18 2022

@author: user1
"""

import emcee
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

# from ipywidgets import interact, interactive

import numpy as np
# from gp.modeling import Model
#from pytransit import QuadraticModel, GeneralModel
# import batman
import pylightcurve as plc
import matplotlib.pyplot as plt
from time import time as tm

from scipy import interpolate
from astropy.io import fits
 
global fit_params 
global non_fit_params 

def bin_spectrum(slc, wav, R, bin_size, bin_type = 'R-bin', wavgrid = None):
    # =============================================================================
    #  find minimum R-power
    # =============================================================================
    # R = wav/ abs(np.gradient(wav))
    # fig = plt.figure('ex_spec')
    # ax = fig.add_subplot(2,2,1)
    # ax.plot(R)
    # plt.figure('Native R')
    # plt.plot(R)
    
    spec = slc
    
    if bin_type == 'R-bin' or  bin_type == 'R' or bin_type == 'R-bin' or bin_type == 'R_min' or bin_type == 'R-min':
    
        # print ('R min and max:', R.min(), R.max())
        if bin_type == 'R_min' or bin_type == 'R-min':
            print ('picking min R')
            R = np.int(R.min())
            print ('binning to R =', R)
         
        
        # =============================================================================
        #  R-bin
        # # =============================================================================
        
        
        wav_range = [wav[0], wav[-1]]
        pixSize = 18 # any number will do , doesn't make a difference
     
        spec_stack = spec
     
        print('binning to R power of %s...' % (R))
        
        
        wl = wav
        x_wav = wav
        x_pix = np.arange(len(wl))
        
        # remove zeros from wl solution ends
        for i in range(len(wl)):
            if wl[i] > 0:
                idx0 = i
                break
        for i in range(len(wl)-1, 0, -1):
            if wl[i] > 0:
                idx1 = i+1
                break
        
        wl0 = wl[idx0:idx1]
        x_pix = x_pix[idx0:idx1]
        
        # b) find w0, the starting wavelength
        if wl0[-1] < wl0[0]:
            w0 = wl0[-1]
        else:
            w0 = wl0[0]
         
        # c) calculate the size of each bin in microns of wavelength
        dw = w0/(R-0.5)
        bin_sizes = [dw]
        for i in range(1000):
            dw2 = (1+1/(R-0.5))*dw
            bin_sizes.append(dw2)
            dw = dw2
            if np.sum(bin_sizes) > wav_range[1]-w0:
                break
        bin_sizes = np.array(bin_sizes)
        
        # d) find the edges of each bin in wavelength space
        wavcen = w0+np.cumsum(bin_sizes)  # the central wavelength of each bin
        wavedge1 = wavcen-bin_sizes/2.  # front edges
        wavedge2 = wavcen+bin_sizes/2.  # back edges
        # obtain an average value for each edge
        wavedge = np.hstack(
            (wavedge1[0], ((wavedge1[1:]+wavedge2[:-1])/2.), wavedge1[-1]))
        
        # e)  find the bin edges in spatial units, in microns where 0 is at the left edge and centre of pixel is pixsize/2
        # e1) translate wl to x (microns)
        wl_osr = x_wav
        x_osr = np.arange(pixSize/2., (pixSize)*(len(x_wav)), pixSize)
        # e2) convert wavedge to xedge (bin edges in spatial units)
        xedge = interpolate.interp1d(
            wl_osr, x_osr, kind='linear', bounds_error=False)(wavedge)
        # e3) invert depending on wavelength solution
        if wl0[-1] < wl0[0]:
            xedge = xedge[::-1]
            wavcen = wavcen[::-1]
        xedge = xedge[1:]  # make len(xedge) = len(wavcen)
        # e4) remove nans
        idx = np.argwhere(np.isnan(xedge))
        xedge0 = np.delete(xedge, idx)
        wavcen0 = np.delete(wavcen, idx)
        # So now we have A) edges of bins in wavelength units, B) edges of bins distance units : xedge0
        
        for intg in range(spec_stack.shape[0]):
        
            #print('binning intg %s' % (intg))
        
            spec = spec_stack[intg]  # pick a 1 D spectrum
            ct = 0
            count = []
            xedge1 = xedge0
            wavcen1 = wavcen0
            xedgepix = xedge1/pixSize
        
            for j in range(len(wavcen1)-1):
        
                # selects if next bin edge is NOT in the same pixel
                if int(xedgepix[ct+1]) > int(xedgepix[ct]):
        
                    # selects if next bin edge is in the NEXT pixel
                    if int(xedgepix[ct+1]) == int(xedgepix[ct]) + 1:
                        # signal from the left pixel
                        fracLeft = 1-(xedgepix[ct]-int(xedgepix[ct]))
                        SLeft = spec[int(xedgepix[ct])]*fracLeft
                        # signal from the right pixel
                        fracRight = xedgepix[ct+1]-int(xedgepix[ct+1])
                        SRight = spec[int(xedgepix[ct + 1])]*fracRight
                        # add these together
                        S = SLeft + SRight
                        count.append(S)
                        ct = ct+1
        
                    # selects if next bin edge is NOT in the NEXT pixel
                    else:
                        qq = int(xedgepix[ct])
                        temp = 0
                        # signal from the left pixel
                        fracLeft = 1 - (xedgepix[ct]-int(xedgepix[ct]))
                        SLeft = spec[qq]*fracLeft
                        # add this to a cumulative count
                        temp += SLeft
                        # move to the next pixel
                        for i in range(1000):
                            qq = qq+1
                            S = spec[qq]
                            # add whole pixel count to cumulative
                            temp += S
                            # check if next pixel has the bin edge
                            if xedgepix[ct+1] < qq+2:
                                # add the right pixel fraction to the count
                                fracRight = (xedgepix[ct+1]-int(xedgepix[ct+1]))
                                SRight = spec[qq+1]*fracRight
                                # final count for bin
                                temp += SRight
        
                                count.append(temp)
                                ct = ct+1
                                break
        
                else:
                    # selects if next bin edge is in SAME pixel
                    # find fraction of pixel in the bin
                    frac = xedgepix[ct+1]-xedgepix[ct]
                    # add count
                    S = frac*spec[int(xedgepix[ct])]
                    count.append(S)
                    ct = ct+1
        
            wavcen_list0 = wavcen0[1:]
        
            if intg == 0:
                count_array = count
            else:
                count_array = np.vstack((count_array, count))
        
            # plt.figure('comp spec')
            # plt.plot(wavcen_list0, count, 'ro-')
            # plt.plot(wav, spec, 'bo-')
            # plt.grid()
        
            # plt.figure('comp R')
            # plt.plot(wav, wav/np.gradient(wav), 'bo-')
            # plt.plot(wav, [R]*len(wav), 'ro-')
            # plt.grid()
           
        
        binned_lc = count_array
        binned_wav = wavcen_list0
        binned_edges = wavedge 
        
         
    
    
    elif bin_type == 'wavgrid':
     
        # print ('R min and max:', R.min(), R.max())
        if bin_type == 'R_min' or bin_type == 'R-min':
            print ('picking min R')
            R = np.int(R.min())
            print ('binning to R =', R)
         
        
        # =============================================================================
        #  R-bin
        # # =============================================================================
        
        
        wav_range = [wav[0], wav[-1]]
        pixSize = 18
        spec_stack = spec
     
        print('binning to fixed wavelength grid...')
        
        
        wl = wav
        x_wav = wav
        x_pix = np.arange(len(wl))
        
        # remove zeros from wl solution ends
        for i in range(len(wl)):
            if wl[i] > 0:
                idx0 = i
                break
        for i in range(len(wl)-1, 0, -1):
            if wl[i] > 0:
                idx1 = i+1
                break
        
        wl0 = wl[idx0:idx1]
        x_pix = x_pix[idx0:idx1]
        
        # b) find w0, the starting wavelength
        if wl0[-1] < wl0[0]:
            w0 = wl0[-1]
        else:
            w0 = wl0[0]
         
        # c) calculate the size of each bin in microns of wavelength
        dw = w0/(R-0.5)
        bin_sizes = [dw]
        for i in range(1000):
            dw2 = (1+1/(R-0.5))*dw
            bin_sizes.append(dw2)
            dw = dw2
            if np.sum(bin_sizes) > wav_range[1]-w0:
                break
        bin_sizes = np.array(bin_sizes)
        
        # d) find the edges of each bin in wavelength space
        wavcen = w0+np.cumsum(bin_sizes)  # the central wavelength of each bin
        wavedge1 = wavcen-bin_sizes/2.  # front edges
        wavedge2 = wavcen+bin_sizes/2.  # back edges
        # obtain an average value for each edge
        wavedge = np.hstack(
            (wavedge1[0], ((wavedge1[1:]+wavedge2[:-1])/2.), wavedge1[-1]))
        
        
        # =============================================================================
        #         
        # =============================================================================
        
        wavcen = wavgrid
        edges = []
        for i in range(len(wavcen)-1):
            edges.append(wavcen[i] + (wavcen[i+1]- wavcen[i]) / 2)
        edges = np.array(edges)
        plt.figure('test1')
        plt.plot(edges, 'o-')
        
        #deal with edge points
        diff_ =np.diff(edges[:4])
        diff0 = diff_[0]-np.gradient(diff_).mean()
        edges0 = edges[0]-diff0
        edges = np.array([edges0] + edges.tolist())
        # plt.plot(edges, 'o-')
        
        diff_ =np.diff(edges[-4:])
        diff1 = diff_[-1]+np.gradient(diff_).mean()
        edges1 = edges[-1]+diff1
        edges = np.array(edges.tolist()+[edges1])
        # plt.plot(edges, 'o-')
        wavedge = edges
      
        
        # e)  find the bin edges in spatial units, in microns where 0 is at the left edge and centre of pixel is pixsize/2
        # e1) translate wl to x (microns)
        wl_osr = x_wav
        x_osr = np.arange(pixSize/2., (pixSize)*(len(x_wav)), pixSize)
        # e2) convert wavedge to xedge (bin edges in spatial units)
        xedge = interpolate.interp1d(
            wl_osr, x_osr, kind='linear', bounds_error=False)(wavedge)
        # e3) invert depending on wavelength solution
        if wl0[-1] < wl0[0]:
            xedge = xedge[::-1]
            wavcen = wavcen[::-1]
        xedge = xedge[1:]  # make len(xedge) = len(wavcen)
        # e4) remove nans
        idx = np.argwhere(np.isnan(xedge))
        xedge0 = np.delete(xedge, idx)
        wavcen0 = np.delete(wavcen, idx)
        # So now we have A) edges of bins in wavelength units, B) edges of bins distance units : xedge0
        
        for intg in range(spec_stack.shape[0]):
        
            #print('binning intg %s' % (intg))
        
            spec = spec_stack[intg]  # pick a 1 D spectrum
            ct = 0
            count = []
            xedge1 = xedge0
            wavcen1 = wavcen0
            xedgepix = xedge1/pixSize
        
            for j in range(len(wavcen1)-1):
        
                # selects if next bin edge is NOT in the same pixel
                if int(xedgepix[ct+1]) > int(xedgepix[ct]):
        
                    # selects if next bin edge is in the NEXT pixel
                    if int(xedgepix[ct+1]) == int(xedgepix[ct]) + 1:
                        # signal from the left pixel
                        fracLeft = 1-(xedgepix[ct]-int(xedgepix[ct]))
                        SLeft = spec[int(xedgepix[ct])]*fracLeft
                        # signal from the right pixel
                        fracRight = xedgepix[ct+1]-int(xedgepix[ct+1])
                        SRight = spec[int(xedgepix[ct + 1])]*fracRight
                        # add these together
                        S = SLeft + SRight
                        count.append(S)
                        ct = ct+1
        
                    # selects if next bin edge is NOT in the NEXT pixel
                    else:
                        qq = int(xedgepix[ct])
                        temp = 0
                        # signal from the left pixel
                        fracLeft = 1 - (xedgepix[ct]-int(xedgepix[ct]))
                        SLeft = spec[qq]*fracLeft
                        # add this to a cumulative count
                        temp += SLeft
                        # move to the next pixel
                        for i in range(1000):
                            qq = qq+1
                            S = spec[qq]
                            # add whole pixel count to cumulative
                            temp += S
                            # check if next pixel has the bin edge
                            if xedgepix[ct+1] < qq+2:
                                # add the right pixel fraction to the count
                                fracRight = (xedgepix[ct+1]-int(xedgepix[ct+1]))
                                SRight = spec[qq+1]*fracRight
                                # final count for bin
                                temp += SRight
        
                                count.append(temp)
                                ct = ct+1
                                break
        
                else:
                    # selects if next bin edge is in SAME pixel
                    # find fraction of pixel in the bin
                    frac = xedgepix[ct+1]-xedgepix[ct]
                    # add count
                    S = frac*spec[int(xedgepix[ct])]
                    count.append(S)
                    ct = ct+1
        
            wavcen_list0 = wavcen0[1:]
        
            if intg == 0:
                count_array = count
            else:
                count_array = np.vstack((count_array, count))
        
            # plt.figure('comp spec')
            # plt.plot(wavcen_list0, count, 'ro-')
            # plt.plot(wav, spec, 'bo-')
            # plt.grid()
        
            # plt.figure('comp R')
            # plt.plot(wav, wav/np.gradient(wav), 'bo-')
            # plt.plot(wav, [R]*len(wav), 'ro-')
            # plt.grid()
           
        
        binned_lc = count_array
        binned_wav = wavcen_list0
        binned_edges = edges
        
     
      
    
    
    
    else:
    # =============================================================================
    # bins per pix columns
    # =============================================================================

        print ('binning by pixel columns')
        print ('binning to %s pixel columns'%(bin_size))
    
        offs =0
    
        
        # spec_stack = np.load('/Users/user1/Desktop/wasp32b_spec.npy')
        # wl =  np.load('/Users/user1/Desktop/wasp32b_wav.npy')
            
        spec = np.add.reduceat(spec, np.arange(int(offs), spec.shape[1])[::int(bin_size)], axis = 1)
        wl = np.add.reduceat(wav, np.arange(offs,len(wav))[::int(bin_size)])  / bin_size            
        
        if wl[-1] < wl [-2]:
            wl = wl[0:-1]
            spec = spec[:,0:-1]
            
        
                        
        binned_lc = spec
        binned_wav = wl
        # binned_edges = None
        
       
        half_bin = bin_size / 2
        bin_edges = np.zeros(len(wl) + 1)
        bin_edges[:-1] = wl - half_bin * (wav[1] - wav[0])
        bin_edges[-1] = wl[-1] + half_bin * (wav[1] - wav[0])   
        
        
        # print (bin_edges)
        # print (wl)
        
        # plt.figure('ppp')
        # plt.plot(wl, 'o')
        # plt.plot(bin_edges, 'o')
        
         
    slc = binned_lc
    wav = binned_wav
    edges = bin_edges
    
    return slc, wav, edges
    
    
 

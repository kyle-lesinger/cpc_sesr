#!/usr/bin/env python3

'''Functions for scripts'''

import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.signal import detrend
from sklearn.neighbors import KernelDensity
import datetime as dt
from glob import glob
import pyet
import math
import config.conversionUtils as cUtils
import config.STATIC as call
import config.gefDataUtils as gData
import config.dataUtils as dUtils
import config.detrendUtils as trendUtils

def gefsv12_ESR_mask_and_interpolate_and_running_mean(EVP,REFET,land_mask_vals,num_days_in_rolling_mean):

    # obs = obs.rename({'PEVPR':'pet','EVP':'evp','refET':'refet'})
    EVP['ESR_refet'] = EVP['data'].copy(deep=True)
    EVP['ESR_refet'][:,:,:,:,:] =  np.nan

    '''Create ESR, this leads to bad values such as inf, np.nan, and very high values which we will fix later'''
    ESR_refet = EVP['data'].values / REFET['data'].values
    ESR_refet_xr = EVP['data'] / REFET['data']
    ESR_refet_mask_inf = np.where(np.isinf(ESR_refet), np.nan, ESR_refet)


    '''Add these 2 masks becuase Jordan Christian did in his SESR calculations'''
    ESR_refet_final1 = np.where(ESR_refet_mask_inf < 0, 0,ESR_refet_mask_inf)
    ESR_refet_final1 = np.where(ESR_refet_final1 > 3, np.nan, ESR_refet_final1)
    ESR_refet_final = linear_interpolate_nans_GEFSv12(ESR_refet_final1, land_mask_vals)
    # ESR_refet_final1 = np.where(ESR_refet_final1 > 1, 1, ESR_refet_final1)

    '''replace data'''
    REFET.data[:,:,:,:,:] = ESR_refet_final
    REFET = REFET.clip(min=0)
    esr = REFET.rolling(L=num_days_in_rolling_mean, center=True).mean() #We want the centered rolling mean

    return(esr)

def create_ESR_GEFSv12_hindcast(initDates,recompute,cpc_analysis):

    
    land_mask_vals = dUtils.load_CONUS_mask()['CONUS_mask'][0,:,:].values #Values of 1 indicate a land area

    if recompute:
        if cpc_analysis == False:
            save_dir = f'{call.gefs_dir}/ESR_hindcast'
        else:
            save_dir = f'{call.gefs_dir}/ESR_hindcast_cpc_source'
            
        os.makedirs(save_dir, exist_ok=True)
        print('Recomputing calculation of ESR on GEFSv12 hindcast data')
        for _date in initDates:
            save_file = f'{save_dir}/esr_{_date}.nc'
    
            # break
            print(f'Making ESR for date {_date}.')
            EVP = xr.open_dataset(f'{call.gefs_dir}/GEFSv12_merged/lhtfl_sfc/lhtfl_sfc_{_date}.nc')
            EVP.close()
            if cpc_analysis == False:
                REFET = xr.open_dataset(f'{call.gefs_dir}/ETo_hindcast/refet_{_date}.nc')
            else:
                REFET = xr.open_dataset(f'{call.gefs_dir}/ETo_hindcast_cpc_source/merged_inits/refet_julian_{_date}.nc').rename({'ETo_Penman':'data'})
                REFET['L'] = np.arange(len(REFET.L.values))
            REFET.close()
                
            
            esr = gefsv12_ESR_mask_and_interpolate_and_running_mean(EVP,REFET,land_mask_vals,num_days_in_rolling_mean=call.mean_rolling_length)
            esr.to_netcdf(save_file)

def linear_interpolate_nans_GEFSv12(data, land_mask_vals):
    '''Only interpolate based on lead, there really should be missing values or inf values'''
    return_data = data.copy()
    for M in range(data.shape[1]):
        for Y in range(data.shape[3]):
            for X in range(data.shape[4]):
                if ~np.isnan(land_mask_vals[Y,X]):
                    arr = data[0,M,:,Y,X]
                    if (np.count_nonzero(np.isnan(arr)) !=0):
                        if (np.count_nonzero(np.isnan(arr)) == len(arr)):
                            pass
                        else:
                            """Interpolate NaN values in a 1D array using linear interpolation."""
                            nans, x = np.isnan(arr), lambda z: z.nonzero()[0]
                            arr[nans] = np.interp(x(nans), x(~nans), arr[~nans])
                            return_data[0,M,:,Y,X] = arr
    return return_data


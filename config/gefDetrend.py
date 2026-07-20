#!/usr/bin/env python3

'''Functions for scripts'''

import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.signal import detrend
from sklearn.neighbors import KernelDensity
import config.detrendUtils as trendUtils
import datetime as dt
from glob import glob
import pickle
from scipy import stats
import config.gefDataUtils as gData
from scipy.stats import linregress
import config.climpredUtils as clim
import config.FDtimeSeriesPlot as fdplot
import config.dataUtils as dUtils
import config.STATIC as call

def convert_int_leads_to_day_of_year(datasets_int):

    for idx,init in enumerate(datasets_int):
        julian_dates = gData.julian_date_HINDCAST(datasets_int[idx].S.values[0],datasets_int[idx].L.shape[0])
        init['L'] = julian_dates
        datasets_int[idx] = init
    return datasets_int

def find_slope_and_intercept_ESR_hindcast_to_detrend(gef_subset_by_year, land_mask, climatology, slope_arr, intercept_arr, idx):

    print(f'Working on doy {idx+1} and finding the slope and intercept for GEFSv12 hindcast ESR climatology_years_{climatology[0]}-{climatology[1]}.nc')

    for Y in range(gef_subset_by_year.Y.shape[0]):  # Latitude
        for X in range(gef_subset_by_year.X.shape[0]):  # Longitude
            if np.isnan(land_mask[Y,X]):
                pass
            else:
                # print(f'Working on Y {Y} and X {X}')
                ts = gef_subset_by_year['data'][:, Y, X]  # Extract time series as 1-d array
                ts_arr = ts.values

                #get the index values of non np.nan values
                index_vals = [idx for idx,i in enumerate(ts_arr) if ~np.isnan(i)]
                
                no_nan_values = ts[index_vals] #subset only non np.nan values

                if len(no_nan_values) !=0:
                    t = np.arange(len(no_nan_values))
                    mask = np.isfinite(no_nan_values)
                    '''Manually add the scipy.stats code here to save the slope and intercept'''

                    slope, intercept, r_value, p_value, std_err = stats.linregress(t[mask], no_nan_values)
                    
                    slope_arr[idx,Y,X] = slope
                    intercept_arr[idx,Y,X] = intercept

    return slope_arr, intercept_arr 



def save_ESR_detrend_vals_day_of_year_climatology(climatology=call.short_clim, cpc_source = True):
    '''Save the files this way so that we do not have to do manipulations later to re-create any data'''
    
    print(f'Saving the slope and intercept values for each day of year based on the ensemble mean for climatology {climatology}.')

    if cpc_source == True:
        save_dir = f'{call.gefs_dir}/climatology_ESR_detrend_hindcast_cpc_source'
    else:
        save_dir = f'{call.gefs_dir}/climatology_ESR_detrend_hindcast'
        
    os.makedirs(save_dir, exist_ok = True)

    save_file = f'{save_dir}/climatology_ESR_detrend_slope_intercept_hindcast_{climatology[0]}-{climatology[1]}.nc'

    try:
        land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    except NameError:
        land_mask = dUtils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    if os.path.exists(save_file):
        return xr.open_dataset(save_file),land_mask
    else:
        
        '''First load the data, taking the ensemble mean to reduce computation time.'''
        if cpc_source == False:
            esr_list = sorted(glob(f'{call.gefs_dir}/ESR_hindcast/*.nc'))
        else:
            esr_list = sorted(glob(f'{call.gefs_dir}/ESR_hindcast_cpc_source/*.nc'))
            
        datasets_int = [xr.open_dataset(f).mean(dim='M').load()  for f in esr_list]
    
        '''Convert to julian days for easier processing'''
        datasets_int = convert_int_leads_to_day_of_year(datasets_int)
    
        gef = xr.concat(datasets_int, dim='S')
        gef['S'] = [pd.to_datetime(i) for i in gef['S'].values]
        
        gef_trnd_OUT = gef.mean(dim='S').copy(deep=True).rename({'data':'slope'})
        gef_trnd_OUT['intercept'] = gef_trnd_OUT['slope'].copy(deep=True)
    
        slope_arr = np.empty_like(gef_trnd_OUT.slope)
        intercept_arr = np.empty_like(gef_trnd_OUT.slope)
            
        for idx,doy in enumerate(np.arange(1,367)):
            '''Very important. To remove any autocorrelation and non-independence of samples, we only want to keep
            a single sample from all of the data for each year. So we cannot include 5 day of year 40 values for each
            year.
            So to fix this, we are going to look at each year and find each day of year and then just take the mean.
            This will give us 1 sample
            '''
            # idx,doy= 0,1

            
            '''Get all values by day'''
            gef_subset = gef.sel(L=doy)
            gef_subset_by_year = gef_subset.resample(S='Y').mean(skipna=True)
            slope_arr, intercept_arr = find_slope_and_intercept_ESR_hindcast_to_detrend(gef_subset_by_year, land_mask, climatology, slope_arr, intercept_arr, idx)
            '''Now for each lat and lon grid cell, find the slope and intercept and save'''
            
        gef_trnd_OUT['slope'][:,:,:] = slope_arr[:,:,:]
        gef_trnd_OUT['intercept'][:,:,:] = intercept_arr[:,:,:]
        
        gef_trnd_OUT.to_netcdf(save_file)
        return gef_trnd_OUT,land_mask
    
    # gef = clim.rename_subx_for_climpred(gef)
    

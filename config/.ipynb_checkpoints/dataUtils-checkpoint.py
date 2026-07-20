#!/usr/bin/env python3

'''Functions for scripts'''

import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.signal import detrend
from sklearn.neighbors import KernelDensity
import config.detrendUtils as trendUtils
import config.STATIC as call

def load_CONUS_mask():
    return(xr.open_dataset(f'{call.mask_dir}/NLDAS_masks-veg-soil_0.50.nc4'))


def remove_duplicate_and_sortby_time(obs, refET):
    # Remove duplicate times
    obs,refET = obs.sel(time=~obs.indexes['time'].duplicated()), refET.sel(time=~refET.indexes['time'].duplicated())
    # Sort by time
    obs, refET = obs.sortby('time'), refET.sortby('time')

    return obs, refET

def linear_interpolate_nans(data, land_mask_vals):
    return_data = data.copy()
    for Y in range(data.shape[1]):
        for X in range(data.shape[2]):
            if ~np.isnan(land_mask_vals[Y,X]):
                arr = data[:,Y,X]
                if (np.count_nonzero(np.isnan(arr)) !=0):
                    if (np.count_nonzero(np.isnan(arr)) == len(arr)):
                        pass
                    else:
                        """Interpolate NaN values in a 1D array using linear interpolation."""
                        nans, x = np.isnan(arr), lambda z: z.nonzero()[0]
                        arr[nans] = np.interp(x(nans), x(~nans), arr[~nans])
                        return_data[:,Y,X] = arr
    return return_data


def load_ET_PET_RefET_and_compute_ESR_baseline_years(window_of_centered_mean, year_ranges_tuple_1, year_ranges_tuple_2,recompute=False):
    '''New steps to complete

    1.) Need to load data from different sources (PET and RefET)
    2.) Need to verify that they have the same days and are in the same format (lat/lon)
    3.) Need to perform both analysis simulataneously for both PET and RefET for ESR
    
    '''
    esr_dir_save = call.noah_dir
    os.makedirs(esr_dir_save, exist_ok=True)

    save_obs_mean=f'{esr_dir_save}/esr_rolling_mean_0.50_degrees.nc'

    if recompute==False:
        return('Completed part of pre-processing')
    else:

        #Values of 0,1 indicate land in US/Canada/Mexico. np.nan values indicate ocean primarily.
        land_mask = load_CONUS_mask()['CONUS_mask'][0,:,:] 
        land_mask_vals = load_CONUS_mask()['CONUS_mask'][0,:,:].values 
    
        print('Loading observation file for et, pet, and esr. Computing yearly sum, 7-day rolling centered mean, and leaving the data untouched as original.')
    
        '''For clarify, we already created ESR from a different server but I dont like the division results due to errors, so lets redo it'''
        obs = xr.open_dataset(f'{call.noah_dir}/et_0.50_degrees.nc') #contains ET, PEVP (potential ET)
        refET = xr.open_dataset(f'{call.noah_dir}/RefET_daily_halfdeg.19800101-20240729.nc') # contains only refet that is already in mm/day

        obs, refET = remove_duplicate_and_sortby_time(obs, refET)

        print('We have manually selected the range for EVP and PEVPR and REFET to be from 1980-01-01 through 2024-07-29')
        refET = refET.sel(time=slice(call.noah_start,call.noah_end))
        obs = obs.sel(time=slice(call.noah_start,call.noah_end))
        
        '''Conversions'''
        #Original UNITS for PEVPR = W/m2 so we convert to mm/day
        convert_PEVPR = (obs['PEVPR'].values * 0.0864)/ 2.45  #0.0864 MJ/kg    2.45 MJ/kg
        obs['PEVPR'][:,:,:] = convert_PEVPR

        #Original UNITS for PEVPR = kg/m2 so we convert to mm/day by multiplying by 24 (since we have already taken the daily average
        obs['EVP'] = (obs['EVP'] * 24) #multiply by 24 hours to get the day

        #The data was given to me in mm/day format
        obs['refET'] = obs['PEVPR'].copy(deep=True)
        obs['refET'][:,:,:] = refET.refet.values

        obs = obs.clip(min=0) #Make sure we do not have any negative fluxes
    
        print('Masking ocean and water body grid cells with np.nan')
        obs = xr.where(~np.isnan(land_mask.values),obs,np.nan).transpose('time','lat','lon') #mask ocean and other water bodies
        refET = xr.where(~np.isnan(land_mask.values),refET,np.nan).transpose('time','lat','lon')

        obs = make_ESR_mask_and_interpolate_and_running_mean(obs,land_mask_vals,window_of_centered_mean)
        obs = obs.where(~np.isnan(land_mask),np.nan)
        obs.to_netcdf(save_obs_mean)
        
        '''Create a yearly summation file for plotting. Also create the ESR and mask for np.nan, infinite values, and then
        do a 7-day centered rolling mean.'''
        for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
            # year_ranges_tuple=year_ranges_tuple_1
            '''First make the ESR'''
            obs_subset = obs.sel(time =slice(f'{year_ranges_tuple[0]-1}-01-01',f'{year_ranges_tuple[1]}-12-31')).copy(deep=True)
            yearly_sum = obs_subset.resample(time="YE").sum() #We want the yearly sum
            yearly_sum = yearly_sum.where(~np.isnan(land_mask),np.nan) #mask ocean and other water bodies
            
            # monthly_sum = obs_subset.resample(time='ME').sum()
            # monthly_sum = monthly_sum.where(~np.isnan(land_mask),np.nan)#mask ocean and other water bodies
            yearly_sum.to_netcdf(f'{esr_dir_save}/esr_annual_sum_average_0.50_degrees_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
        
    return('Completed part of pre-processing')



def make_ESR_mask_and_interpolate_and_running_mean(obs,land_mask_vals,window_of_centered_mean):

    # obs = obs.rename({'PEVPR':'pet','EVP':'evp','refET':'refet'})
    obs['ESR_pet'] = obs['EVP'].copy(deep=True)
    obs['ESR_pet'][:,:,:] =  np.nan
    obs['ESR_refet'] = obs['ESR_pet'].copy(deep=True)

    '''Create ESR, this leads to bad values such as inf, np.nan, and very high values which we will fix later'''
    ESR_pet = obs['EVP'].values / obs['PEVPR'].values
    ESR_pet_xr = obs['EVP'] / obs['PEVPR']
    ESR_pet_mask_inf = np.where(np.isinf(ESR_pet_xr.values), np.nan, ESR_pet_xr.values)

    ESR_refet = obs['EVP'].values / obs['refET'].values
    ESR_refet_xr = obs['EVP'] / obs['refET']
    ESR_refet_mask_inf = np.where(np.isinf(ESR_refet_xr.values), np.nan, ESR_refet_xr.values)


    '''Add these 2 masks becuase Jordan Christian did in his SESR calculations'''
    ESR_pet_final1 = np.where(ESR_pet_mask_inf < 0, 0,ESR_pet_mask_inf)
    ESR_pet_final1 = np.where(ESR_pet_final1 > 3, np.nan, ESR_pet_final1)
    ESR_pet_final = linear_interpolate_nans(ESR_pet_final1, land_mask_vals)
    # ESR_pet_final1 = np.where(ESR_pet_final1 > 1, 1, ESR_pet_final1)
    

    ESR_refet_final1 = np.where(ESR_refet_mask_inf < 0, 0,ESR_refet_mask_inf)
    ESR_refet_final1 = np.where(ESR_refet_final1 > 3, np.nan, ESR_refet_final1)
    ESR_refet_final = linear_interpolate_nans(ESR_refet_final1, land_mask_vals)
    # ESR_refet_final1 = np.where(ESR_refet_final1 > 1, 1, ESR_refet_final1)
    
    obs['ESR_pet'][:,:,:] = ESR_pet_final
    obs['ESR_refet'][:,:,:] = ESR_refet_final

    obs = obs.rolling(time=window_of_centered_mean, center=True).mean() #We want the centered rolling mean
    obs = obs.clip(min=0)
    return(obs)





def convert_RZSM_and_take_centered_mean_and_save(window_of_centered_mean, year_ranges_tuple_1, year_ranges_tuple_2, recompute=False):
    '''Take the 7-day mean, create an annual average'''
    print('Loading observation file for RZSM. Computing yearly sum, 7-day rolling centered mean, and saving data (also converting RZSM by dividing by 1000).')

    if recompute==True:
        land_mask = load_CONUS_mask()['CONUS_mask'][0,:,:] #Values of 1 indicate a land area
        obs = xr.open_dataset(f'{call.noah_dir}/rzsm_0.50_degrees.nc') #contains RZSM and SOILM
        obs = obs['RZSM'].isel(depth_2=0).to_dataset()
        obs = obs/1000 #convert to m3/m3
        print('Masking ocean and water body grid cells with np.nan')
        obs = xr.where(~np.isnan(land_mask.values),obs,np.nan).transpose('time','lat','lon') #mask ocean and other water bodies
        # Remove duplicate times
        obs = obs.sel(time=~obs.indexes['time'].duplicated())
        # Sort by time
        obs = obs.sortby('time')
        obs = obs.clip(min=0)
    
        save_obs_mean_full=f'{call.noah_dir}/RZSM_rolling_mean_0.50_degrees_full.nc'
        obs.to_netcdf(save_obs_mean_full)
        
        for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
            save_obs_mean=f'{call.noah_dir}/RZSM_rolling_mean_0.50_degrees_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'

            # break
            yearly_avg = obs.resample(time="YE").mean() #We want the yearly sum
            yearly_avg = xr.where(~np.isnan(land_mask.values),yearly_avg,np.nan).transpose('time','lat','lon') #mask ocean and other water bodies
            monthly_avg = obs.resample(time='ME').mean()
            monthly_avg = xr.where(~np.isnan(land_mask.values),monthly_avg,np.nan).transpose('time','lat','lon') #mask ocean and other water bodies
            # window_size = 7  # Weekly centered rolling mean number of values (this removes the first 3 values and elimanates the last 3 values of each grid cells time series)
            obs_mean = obs.rolling(time=window_of_centered_mean, center=True).mean() #We want the centered rolling mean
            obs_mean.to_netcdf(save_obs_mean)
    return('Completed part of pre-processing')

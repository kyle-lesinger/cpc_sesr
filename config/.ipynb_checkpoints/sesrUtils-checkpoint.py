#!/usr/bin/env python3

'''Functions for scripts'''

import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.stats import percentileofscore as pos
import bottleneck as bn
from numba import njit,prange
import config.dataUtils as dutils
import config.STATIC as call


def load_ESR_obs_to_compute_mean_and_std(time_period):
    return(xr.open_dataset(f'{call.noah_dir}/ESR_de-trend_years_{time_period[0]}-{time_period[1]}.nc').load())


def create_SESR_by_doy(window,year_ranges_tuple_1,year_ranges_tuple_2):
    save_dir = f'{call.sesr_dir}'
    os.makedirs(save_dir, exist_ok=True)

    pet_refet_name = ['pet','refet']
    #Open 
    for time_period in [year_ranges_tuple_1,year_ranges_tuple_2]:
            
        # window,time_period=0, (1981,2020)
        save_file=f'{save_dir}/SESR_window_size_{window}_years_{time_period[0]}-{time_period[1]}.nc'
        
        if os.path.exists(save_file):
            print(f'Already completed SESR for {save_file}.')
        else:
            print(f'Making the SESR for window size {window} and time period {time_period[0]}-{time_period[1]}.')
            obs = load_ESR_obs_to_compute_mean_and_std(time_period)
            obs['SESR_pet'] = obs['ESR_pet_detrend'].copy(deep=True)
            obs['SESR_pet'][:,:,:] = np.nan

            obs['SESR_refet'] = obs['SESR_pet'].copy(deep=True)

            for pet_refet_NAME_index,ESR_pet_or_refet_detrend_NAME in enumerate(['ESR_pet_detrend','ESR_refet_detrend']):
                # pet_refet_idx,pet_or_refet = 1,'ESR_refet_detrend'
            
                obs_mean_std = xr.open_dataset(f'{call.noah_dir}/climatology_ESR_detrend_mean_std/ESR_mean_std_window_size_{window}_years_{time_period[0]}-{time_period[1]}.nc')
            
                for idx, doy in enumerate(range(1, 367)):
                    # idx, doy = 0,1 #for testing
                    md = pd.to_datetime(obs.isel(time=idx).time.values)
                    month_day = f'2000-{md.month:02}-{md.day:02}'
                    
                    mean_, std_ = obs_mean_std[f'mean_{pet_refet_name[pet_refet_NAME_index]}'].sel(time=month_day).values, obs_mean_std[f'std_{pet_refet_name[pet_refet_NAME_index]}'].sel(time=month_day).values
                
                    # Create a mask for the target month and day. This selects only the same doy from all the other years
                    mask = (obs['time'].dt.month == md.month) & (obs['time'].dt.day == md.day)
                
                    # Get the indices and dates
                    indices = np.where(mask)[0]
                    dates = obs['time'].values[mask] #actual doy values of the days of the year
            
                    obs = standardize_ESR(obs,mask,mean_,std_,pet_refet_name,pet_refet_NAME_index,ESR_pet_or_refet_detrend_NAME)
        
            #This will fix the infinite values and the random few values that are very high or very low        
            # obs_mean['SESR'][:,:,:] = np.where(obs_mean['SESR'].values > 5, 5,obs_mean['SESR'].values)
            # obs_mean['SESR'][:,:,:] = np.where(obs_mean['SESR'].values < -5,-5,obs_mean['SESR'].values)

            obs = obs.interpolate_na(dim='time', method='linear')
            obs = obs.where(~np.isinf(obs), np.nan)
            '''We must add this because we did a 7-day rolling mean centered on the 4th day,
            and the interpolation will mess that up by giving it values.'''
            obs['SESR_refet'][:3,:,:] = np.nan
            obs['SESR_pet'][:3,:,:] = np.nan
    
            obs['SESR_refet'][-3:,:] = np.nan
            obs['SESR_pet'][-3:,:,:] = np.nan
            
            obs.to_netcdf(save_file)
    return(f'Completed {save_file}.')


def standardize_ESR(obs,mask,mean_,std_,pet_refet_name,pet_refet_NAME_index,ESR_pet_or_refet_detrend_NAME):
    # Subtract the value from the selected dates
    obs[f'SESR_{pet_refet_name[pet_refet_NAME_index]}'][mask, :, :] = obs[ESR_pet_or_refet_detrend_NAME][mask, :, :].values - mean_
    with np.errstate(divide='ignore', invalid='ignore'):
        obs[f'SESR_{pet_refet_name[pet_refet_NAME_index]}'][mask, :, :] = obs[f'SESR_{pet_refet_name[pet_refet_NAME_index]}'][mask, :, :].values / std_

    return obs
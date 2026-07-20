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


def return_obs_mean_out_file(obs_mean):
    obs_mean_out = obs_mean.sel(time=slice(call.doy_start,call.doy_end)).copy(deep=True)
    del obs_mean_out['EVP']
    del obs_mean_out['PEVPR']
    del obs_mean_out['refET']
    del obs_mean_out['ESR_pet']
    del obs_mean_out['ESR_pet_detrend']
    obs_mean_out = obs_mean_out.rename({'SESR_pet':'SESR_pet_mean'})
    obs_mean_out = obs_mean_out.rename({'SESR_refet':'SESR_pet_std'})
    obs_mean_out = obs_mean_out.rename({'ESR_refet_detrend':'SESR_refet_mean'})
    obs_mean_out = obs_mean_out.rename({'ESR_refet':'SESR_refet_std'})
    return obs_mean_out

def delta_SESR_return_std_and_mean_diff_across_years(window, year_ranges_tuple_1, year_ranges_tuple_2, ):
    '''Create the mean and standard deviation distribution. Select all days with the window for the distribution.
    Also compute the percentile of delta_SESR values for each grid cell.'''

    number_of_weeks = call.num_weeks_difference_SESR # this is for calculating the difference between periods. We are using a difference of 1 week as the standard
    #We keep this as a constant and not an argument to help with the multi-processing function.
    
    save_dir = call.dzSESR_dir
    os.makedirs(save_dir, exist_ok=True)

    mean_std_dir = call.SESR_clim_dir
    os.makedirs(mean_std_dir, exist_ok=True)

    total_days = number_of_weeks * 7

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
        # year_ranges_tuple=year_ranges_tuple_1
        # break
        save_dsesr = f'{save_dir}/dzSESR_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        save_mean_std = f'{mean_std_dir}/dzSESR_mean_std_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        
        if os.path.exists(save_dsesr):
            print(f'Already completed dzSESR_delta_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
            pass
        else:
            print(f'Working on creating dzSESR_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
            obs_mean = xr.open_dataset(f'{call.sesr_dir}/SESR_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').load()
        
            time_index_short = pd.to_datetime(obs_mean.sel(time=slice('2000-01-01','2000-12-31')).time.values)
            #Just stick with these days because its a leap year and shared between both data periods
            time_index_full = pd.to_datetime(obs_mean.time.values)

            obs_mean_out =return_obs_mean_out_file(obs_mean)

            '''Empty arrays to fill'''
            obs_mean['dzSESR_pet'] = obs_mean['SESR_pet'].copy(deep=True)
            obs_mean['dzSESR_pet'][:,:,:] = np.nan
            obs_mean['dzSESR_refet'] = obs_mean['dzSESR_pet'].copy(deep=True)
            
            replace_pet = np.empty_like(obs_mean['dzSESR_refet'].values)
            replace_refet = replace_pet.copy()

            dsesr_name, mean_std_name = ['dzSESR_refet','dzSESR_pet'], ['refet','pet']
            for idx_dsesr,dsesr in enumerate(dsesr_name):
                # break
                if dsesr == 'dzSESR_refet':
                    replace_refet, obs_mean_out = subtract_data_by_week_of_year_and_STANDARDIZE(time_index_short,time_index_full,total_days,obs_mean_out,replace_refet,obs_mean,dsesr_name, mean_std_name,idx_dsesr)
                else:
                    replace_pet, obs_mean_out = subtract_data_by_week_of_year_and_STANDARDIZE(time_index_short,time_index_full,total_days,obs_mean_out,replace_pet,obs_mean,dsesr_name, mean_std_name,idx_dsesr)



            #Now replace the values
            obs_mean['dzSESR_pet'][:,:,:] = replace_pet
            obs_mean['dzSESR_refet'][:,:,:] = replace_refet
    
            obs_mean.to_netcdf(save_dsesr)
            
            obs_mean_out.to_netcdf(save_mean_std)
    return(f'Completed making delta SESR for window_size_{window}_years.')


def subtract_data_by_week_of_year_and_STANDARDIZE(time_index_short,time_index_full,total_days,obs_mean_out,replace_standardized_array,obs_mean,dsesr_name, mean_std_name,idx_dsesr):
    for idx,date in enumerate(time_index_short):
        # break
        #Grab all the same month and day values across all years
        #Need to add this because the leap year dates don't have enough values
        if date == pd.to_datetime('2000-02-29'):
            new_date = pd.to_datetime('2000-02-28')
        else:
            new_date = date
        
        #Select all the days across all years with the same month and day
        mask_current_week = (time_index_full.month == new_date.month) & (time_index_full.day == new_date.day)
        true_indices_current_week = np.where(mask_current_week)[0]
        selected_data = obs_mean.isel(time=true_indices_current_week)
    
        previous_week_indices = np.array([i - total_days for i in true_indices_current_week if i-total_days >=0]) #Make no negative index values
        selected_data_previous = obs_mean.isel(time=previous_week_indices)
    
        # #Sometimes we have a mis-match between years (specifically the number of data points, they must be equal!), so this fixes it
        if len(selected_data_previous.time.values) > len(selected_data.time.values):
            selected_data_previous = selected_data_previous.isel(time = slice(0,len(selected_data.time.values)))
        elif len(selected_data_previous.time.values) < len(selected_data.time.values):
            selected_data = selected_data.isel(time = slice(1,len(selected_data.time.values)))
    
        #Now find the mean difference across all years and average
        weekly_difference = selected_data[f'SESR_{mean_std_name[idx_dsesr]}'].values - selected_data_previous[f'SESR_{mean_std_name[idx_dsesr]}'].values
        obs_mean_out[f'SESR_{mean_std_name[idx_dsesr]}_mean'][idx,:,:] = bn.nanmean(weekly_difference,axis=0)
        obs_mean_out[f'SESR_{mean_std_name[idx_dsesr]}_std'][idx,:,:] = bn.nanstd(weekly_difference,axis=0)
    
        #Standardize the values
        replace_standardized_array[true_indices_current_week,:,:]= (obs_mean[f'SESR_{mean_std_name[idx_dsesr]}'][true_indices_current_week,:,:].values - obs_mean_out[f'SESR_{mean_std_name[idx_dsesr]}_mean'][idx,:,:].values) /  obs_mean_out[f'SESR_{mean_std_name[idx_dsesr]}_std'][idx,:,:].values
        
    #After all are completed, change infinite values and very large positive or negative values
    replace_standardized_array = np.where(np.isfinite(replace_standardized_array),replace_standardized_array,np.nan)
    
    return replace_standardized_array, obs_mean_out

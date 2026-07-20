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



def get_data_within_45_day_window(obs, doy, window):
    # Function to get data within ±n days for a specific day of the year (DOY)
    
    start_doy = (doy - window) % 366
    end_doy = (doy + window) % 366

    # Extract the day of year from the time dimension
    time_doy = obs['time'].dt.dayofyear

    if (window==0) and (doy==366):
        mask = (time_doy >= 365) #just include this so that we can have values 
    elif start_doy > end_doy:
        mask = (time_doy >= start_doy) | (time_doy <= end_doy)
    elif doy ==1:
        mask = (time_doy >= 366-window) | (time_doy <= end_doy)
    else:
        mask = (time_doy >= start_doy) & (time_doy <= end_doy)
    
    return obs.sel(time=mask)

def subset_window_further(fill_file, all_data, window,idx_for_mean_std, doy):

    '''Now only select certain values based on the window size
    
    E.g., window size 0 = only those days of the year
    window size 1 = choose days seperated by 4 days (becuase the centered mean accounts for the other values partially)
    window size 2  = choose 4 and 8 days separated
    '''
    
    window_subset = window*4
    
    if window==0:
        add_dex = [0]
    elif window ==1:
        add_dex = [0,4]
    elif window == 2:
        add_dex = [0,4,8]
    elif window ==3:
        add_dex = [0,4,8,12]

    final_subset_indices = []
    for idxx,date in enumerate(all_data.time.values):
        # break
        pd_date = pd.to_datetime(date)
        if (pd_date.dayofyear == doy):
            if idxx < window_subset:
                # break
                #for this, only add values ahead of the index
                final_subset_indices = final_subset_indices + [idxx+j for j in add_dex]
            elif idxx > window_subset:
                final_subset_indices = final_subset_indices + [idxx+j for j in add_dex]
                final_subset_indices = final_subset_indices + [idxx-j for j in add_dex]
    
    #Now 
    final_subset_indices= sorted(final_subset_indices)
    final_subset_indices = [i for i in final_subset_indices if i < len(all_data.time.values)] #remove indices outside of range
    final_subset_indices = [i for i in final_subset_indices if i > 0]
    
    fill_file['mean'][idx_for_mean_std,:,:] = all_data['ESR_detrend'][final_subset_indices,:,:].mean(dim='time').values
    fill_file['std'][idx_for_mean_std,:,:] = all_data['ESR_detrend'][final_subset_indices,:,:].std(dim='time').values
    
    return(fill_file)


def subset_window_further_esr(fill_file, all_data, window,idx_for_mean_std, doy, pet_or_refet ,):

    '''Now only select certain values based on the window size
    
    E.g., window size 0 = only those days of the year
    window size 1 = choose days seperated by 4 days (becuase the centered mean accounts for the other values partially)
    window size 2  = choose 4 and 8 days separated
    '''
    
    window_subset = window*4
    
    if window==0:
        add_dex = [0]
    elif window ==1:
        add_dex = [0,4]
    elif window == 2:
        add_dex = [0,4,8]
    elif window ==3:
        add_dex = [0,4,8,12]

    final_subset_indices = []
    for idxx,date in enumerate(all_data.time.values):
        # break
        pd_date = pd.to_datetime(date)
        if (pd_date.dayofyear == doy):
            if idxx < window_subset:
                # break
                #for this, only add values ahead of the index
                final_subset_indices = final_subset_indices + [idxx+j for j in add_dex]
            elif idxx > window_subset:
                final_subset_indices = final_subset_indices + [idxx+j for j in add_dex]
                final_subset_indices = final_subset_indices + [idxx-j for j in add_dex]
    
    #Now 
    final_subset_indices= sorted(final_subset_indices)
    final_subset_indices = [i for i in final_subset_indices if i < len(all_data.time.values)] #remove indices outside of range
    final_subset_indices = [i for i in final_subset_indices if i > 0]
    if pet_or_refet == 'ESR_pet_detrend':
        fill_file['mean_pet'][idx_for_mean_std,:,:] = all_data[pet_or_refet][final_subset_indices,:,:].mean(dim='time').values
        fill_file['std_pet'][idx_for_mean_std,:,:] = all_data[pet_or_refet][final_subset_indices,:,:].std(dim='time').values
    else:
        fill_file['mean_refet'][idx_for_mean_std,:,:] = all_data[pet_or_refet][final_subset_indices,:,:].mean(dim='time').values
        fill_file['std_refet'][idx_for_mean_std,:,:] = all_data[pet_or_refet][final_subset_indices,:,:].std(dim='time').values
        
    return(fill_file)



def load_obs_to_compute_mean_and_std(time_period):
    return(xr.open_dataset(f'{call.noah_dir}/ESR_de-trend_years_{time_period[0]}-{time_period[1]}.nc').load())

def create_ESR_mean_std(window,year_ranges_tuple_1,year_ranges_tuple_2):

    save_dir = f'{call.noah_dir}/climatology_ESR_detrend_mean_std'
    os.makedirs(save_dir, exist_ok=True)
    year_ranges_tuple_1=call.long_clim
    year_ranges_tuple_2=call.short_clim
    # year_ranges_tuple=year_ranges_tuple_1
    
    for year_ranges_tuple in [year_ranges_tuple_1,year_ranges_tuple_2]:
        save_file=f'{save_dir}/ESR_mean_std_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        if os.path.exists(save_file):
            pass
        else:
            print(f'Making mean and standard deviation based on window and saving into {save_file}.')
            # break
            obs = load_obs_to_compute_mean_and_std(year_ranges_tuple) #load the previously created data
            fill_pet = obs.sel(time=slice('2000-01-01','2000-12-31')).copy(deep=True).rename({'EVP':'mean_pet'})['mean_pet'].to_dataset()
            
            fill_pet['mean_pet'][:,:,:] = np.nan
            fill_pet['std_pet'] = fill_pet['mean_pet'].copy(deep=True)
            fill_pet['mean_refet'] = fill_pet['mean_pet'].copy(deep=True)
            fill_pet['std_refet'] = fill_pet['mean_pet'].copy(deep=True)


            for pet_or_refet in ['ESR_pet_detrend','ESR_refet_detrend']:
                # pet_or_refet='ESR_pet_detrend'
                for idx_for_mean_std,doy in enumerate(range(1,367)):
                    # print(f'Working on idx {idx_for_mean_std} out of 366.')
                    #First get data within the disribution window range
                    all_data = get_data_within_45_day_window(obs, doy, window*5) #Just grab all within 45 days and subset from there
                    '''Now to not grab extra data that is already contained within the distribution, we need to further change the window size to
                        only account for weeks'''
                    
                    fill_pet = subset_window_further_esr(fill_pet, all_data, window, idx_for_mean_std,doy, pet_or_refet)

    
            #Save to netcdf
            fill_pet = fill_pet.clip(min=0)
            fill_pet.to_netcdf(save_file)
    return(f'Completed making the mean and standard deviation for window size {window}.')

    

def return_mean_and_std_by_doy_and_window(window, time_period, obs_mean):
    save_dir = call.SESR_clim_dir
    os.makedirs(save_dir, exist_ok=True)
    
    mean_and_std_file = f'{save_dir}/ESR_mean_std_window_size_{window}_years_{time_period[0]}-{time_period[1]}.nc'

    if os.path.exists(mean_and_std_file):
        return xr.open_dataset(mean_and_std_file).load()
    else:
        esr_mean_std_out = obs_mean['ESR'].sel(time = '2000').to_dataset().rename({'ESR':'mean'}).copy(deep=True)
        esr_mean_std_out['std'] = esr_mean_std_out['mean'].copy(deep=True)

        esr_mean_std_out['mean'][:,:,:] = np.nan
        esr_mean_std_out['std'][:,:,:] = np.nan
        
        for idx, doy in enumerate(range(1, 367)):
            # break
            subset = get_data_within_doy(obs_mean['ESR_detrend'], doy, window).sel(time=slice(f'{time_period[0]-1}-12-01',f'{time_period[1]}-12-31'))
            esr_mean_std_out['mean'][idx,:,:] = subset.mean(dim='time').values
            esr_mean_std_out['std'][idx,:,:] = subset.std(dim='time').values

        esr_mean_std_out['mean'][:,:,:] = np.where(esr_mean_std_out['mean'].values <0,0,esr_mean_std_out['mean'].values)
        esr_mean_std_out['std'][:,:,:] = np.where(esr_mean_std_out['std'].values <0,0,esr_mean_std_out['std'].values)
        esr_mean_std_out.to_netcdf(mean_and_std_file)
        
        return esr_mean_std_out.load()








def create_standardized_SESR_by_doy(obs_mean, esr_mean_std_out):
    for idx, doy in enumerate(range(1, 367)):
        # idx, doy = 0,1 #for testing
        md = pd.to_datetime(obs_mean.isel(time=idx).time.values)
        month_day = f'2000-{md.month:02}-{md.day:02}'
        
        mean_, std_ = esr_mean_std_out['mean'].sel(time=month_day).values, esr_mean_std_out['std'].sel(time=month_day).values
    
        # Create a mask for the target month and day. This selects only the same doy from all the other years
        mask = (obs_mean['time'].dt.month == md.month) & (obs_mean['time'].dt.day == md.day)
    
        # Get the indices and dates
        indices = np.where(mask)[0]
        dates = obs_mean['time'].values[mask] #actual doy values of the days of the year
    
        obs_mean = standardize_data(obs_mean,mask,mean_,std_)
    
    #This will fix the infinite values and the random few values that are very high or very low        
    obs_mean['SESR'][:,:,:] = np.where(obs_mean['SESR'].values > 5, 5,obs_mean['SESR'].values)
    obs_mean['SESR'][:,:,:] = np.where(obs_mean['SESR'].values < -5,-5,obs_mean['SESR'].values)
    return obs_mean

def get_time_indices_to_fill(obs_mean):
    # Convert the time coordinate to a pandas DatetimeIndex
    time_index1 = pd.DatetimeIndex(obs_mean['time'].values)
    
    # Extract the DOY from the time coordinate
    doy1 = time_index1.dayofyear

    time_index = pd.Index(obs_mean['time'].values)
    return(time_index1,doy1, time_index)

# Define a numba-compatible percentile of score function
@njit(parallel=True)
def pos_numba(data, value):
    if len(data) == 0:
        return np.nan
    else:
        return np.sum(data < value) / len(data) * 100

@njit(parallel=True)
def create_percentile_with_numba(small_obs, large_obs, percentile_fill_file, indices_to_fill, full_fill_file,land_mask):
    for Y in prange(small_obs.shape[1]):
        for X in range(small_obs.shape[2]):
            # Ensure there are no np.nan
            if np.isnan(land_mask[Y,X]):
                pass
            elif np.all(np.isnan(large_obs[:,Y,X])):
                #Just in case it didn't catch in the previous "if" statement
                pass
            elif np.any(np.isnan(large_obs[:,Y,X])):
                '''If there are some missing values, specifically np.nan values'''
                # Get non-nan values
                valid_large_indices = np.where(~np.isnan(large_obs[:,Y,X]))[0]
                no_nan_vals_obs_full = np.take(large_obs[:,Y,X], valid_large_indices)

                valid_small_indices = np.where(~np.isnan(small_obs[:,Y,X]))[0]
                no_nan_vals_obs_small = np.take(small_obs[:,Y,X], valid_small_indices)
                
                for idx in range(len(valid_small_indices)):
                    percentile_fill_file[valid_small_indices[idx],Y,X] = pos_numba(no_nan_vals_obs_full, no_nan_vals_obs_small[idx])
            
            else:
                '''If all values are present, no np.nan values'''
                for idx in range(small_obs.shape[0]):
                    percentile_fill_file[idx,Y,X] = pos_numba(large_obs[:,Y,X], small_obs[idx,Y,X])

    for idx in range(len(indices_to_fill)):
        full_fill_file[indices_to_fill[idx],:,:] = percentile_fill_file[indices_to_fill[idx],:,:]
        
    return full_fill_file


def percentile_of_score_without_numba(land_mask, all_vals, only_doy_small_subset, percentile_out_arr, fill_values, indices_to_fill):
    for Y,_ in enumerate(range(land_mask.shape[0])):
        for X,_ in enumerate(range(land_mask.shape[1])):
            # break
            #Make sure there are no np.nan
            if ~np.isnan(land_mask[Y,X]):
                if np.all(np.isnan(all_vals[:,Y,X])):
                    pass
                elif np.any(np.isnan(all_vals[:,Y,X])):
                    #get no nan values
                    no_nan_vals_obs_full = all_vals[:,Y,X][~np.isnan(all_vals[:,Y,X])]
                    no_nan_indices_obs_full = np.where(~np.isnan(all_vals[:,Y,X]))[0]

                    no_nan_vals_obs_small = only_doy_small_subset[:,Y,X][~np.isnan(only_doy_small_subset[:,Y,X])]
                    no_nan_indices_obs_small = np.where(~np.isnan(only_doy_small_subset[:,Y,X]))[0]
                    percentile_out_arr[no_nan_indices_obs_small,Y,X] = pos(all_vals[no_nan_indices_obs_full,Y,X],only_doy_small_subset[no_nan_indices_obs_small,Y,X])
                
                else:
                    percentile_out_arr[:,Y,X] = pos(all_vals[:,Y,X],only_doy_small_subset[:,Y,X])
                    
    fill_values[indices_to_fill,:,:] = percentile_out_arr
    
    return fill_values
                    




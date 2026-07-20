#!/usr/bin/env python3

'''Functions for scripts'''

import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.stats import percentileofscore as pos
import bottleneck as bn
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


def subset_window_further_SESR_percentile(obs_full, window, doy):

    '''Now only select certain values based on the window size
    
    E.g., window size 0 = only those days of the year
    window size 1 = choose days seperated by 4 days (becuase the centered mean accounts for the other values partially)
    window size 2  = choose 4 and 8 days separated
    '''
    
    window_subset = window*4

    #With current window*4 and window = 5, there will be about 40 days in the distribution with values
    
    if window==0:
        add_dex = [0]
    elif window ==1:
        add_dex = [0,4]
    elif window == 2:
        add_dex = [0,4,8]
    elif window ==3:
        add_dex = [0,4,8,12]

    final_subset_indices = []
    for idxx,date in enumerate(obs_full.time.values):
        # break
        pd_date = pd.to_datetime(date)
        if (pd_date.dayofyear == doy):
            if idxx <= window_subset:
                # break
                #for this, only add values ahead of the index
                final_subset_indices = final_subset_indices + [idxx+j for j in add_dex]
            elif idxx > window_subset:
                final_subset_indices = final_subset_indices + [idxx+j for j in add_dex]
                final_subset_indices = final_subset_indices + [idxx-j for j in add_dex]
                
    return np.unique(final_subset_indices)


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



def make_empty_variables(obs_full):
    '''Make empty variables to fill'''
    obs_full['dzSESR_pct_refet'] = obs_full['SESR_pet'].copy(deep=True)
    obs_full['dzSESR_pct_refet'][:,:,:] = np.nan

    obs_full['dzSESR_pct_pet'] = obs_full['dzSESR_pct_refet'].copy(deep=True)
    obs_full['SESR_pct_refet'] = obs_full['dzSESR_pct_refet'].copy(deep=True)
    obs_full['SESR_pct_pet'] = obs_full['dzSESR_pct_refet'].copy(deep=True)

    '''Just do this to ensure that none of the data is actually just a mirror to another object'''
    fill_sesr_pet = np.empty(obs_full['SESR_pct_pet'].shape)
    fill_sesr_pet[:,:,:] = np.nan
    fill_sesr_refet = np.empty(obs_full['SESR_pct_pet'].shape)
    fill_sesr_pet[:,:,:] = np.nan
    fill_dzsesr_pet = np.empty(obs_full['SESR_pct_pet'].shape)
    fill_sesr_pet[:,:,:] = np.nan
    fill_dzsesr_refet = np.empty(obs_full['SESR_pct_pet'].shape)
    fill_sesr_pet[:,:,:] = np.nan
    
    return obs_full, fill_sesr_pet, fill_sesr_refet, fill_dzsesr_pet, fill_dzsesr_refet


def make_percentile_from_SESR_and_dzSESR_by_day_of_year(window,year_ranges_tuple_1,year_ranges_tuple_2):
    '''This means that for each grid cell, each day of the year has a value ranked at 100% somewhere in the timeseries.
    
    The function make_percentile_from_SESR_and_dzSESR_from_all_dates() is the opposite. There is only 1 value per grid cell in the time series which has a ranking of 100%.'''
    
    save_dir = call.dzSESR_perc_dir
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
        
        save_dzsesr_percentile = f'{save_dir}/dzSESR_percentile_by_doy_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        
        if os.path.exists(save_dzsesr_percentile):
            print(f'Completed dzSESR_delta_percentile_by_doy_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
            pass
        else:
            obs_full = xr.open_dataset(f'{call.dzSESR_dir}/dzSESR_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').load()
            '''We now have the SESR_delta, now make percentiles based on the data.'''
            
            obs_full, fill_sesr_pet, fill_sesr_refet, fill_dzsesr_pet, fill_dzsesr_refet = make_empty_variables(obs_full)
            
            time_index = pd.Index(obs_full['time'].values)
            # Convert the time coordinate to a pandas DatetimeIndex
            time_index1 = pd.DatetimeIndex(obs_full['time'].values)
            
            # Extract the DOY from the time coordinate
            doy1 = time_index1.dayofyear

            vars_to_fill = ['dzSESR_pct_refet','dzSESR_pct_pet','SESR_pct_refet','SESR_pct_pet']
            vars_needed = ['dzSESR_refet','dzSESR_pet','SESR_refet','SESR_pet']

            for var_idx, var_fill in enumerate(vars_to_fill):
                print(f'Finding {var_fill} percentile of score by day of year for each grid cell window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')

                # break
                for idx, doy in enumerate(range(1, 367)):
                    # idx,doy=0,1
                    subset_45 = get_data_within_45_day_window(obs_full, doy, 5*window) #Get all values within the specified window from the doy
                    subset = subset_window_further_SESR_percentile(obs_full, window, doy) #grab based on window and doy
                    subset = [i for i in subset if i<len(obs_full.time.values)]
                    subset = [i for i in subset if i>=0]
                    
                    # Get the index time labels where the DOY is equal to the desired value
                    matching_times = time_index1[doy1 == doy]
                    indices_to_fill = [time_index.get_loc(date) for date in matching_times]
                    indices_to_fill = np.array(indices_to_fill)
    
                    all_vals = obs_full[vars_needed[var_idx]][subset,:,:].values # ~(window_distribution, 112, 233) (time, lat, lon)
                
                    small_subset = obs_full[vars_needed[var_idx]][indices_to_fill,:,:].values
                    
                    # Y,X=40,30
                    percentile_out = np.empty(shape = (len(indices_to_fill),obs_full.lat.shape[0],obs_full.lon.shape[0]))
                    percentile_out[:,:,:] = np.nan
                    
                    '''Percentile of score - without numba'''
                    if var_fill == 'dzSESR_pct_refet':
                        fill_dzsesr_refet = percentile_of_score_without_numba(land_mask, all_vals, small_subset, percentile_out, fill_dzsesr_refet, indices_to_fill)
                        obs_full[var_fill][:,:,:] = fill_dzsesr_refet
                    elif var_fill == 'dzSESR_pct_pet':
                        fill_dzsesr_pet = percentile_of_score_without_numba(land_mask, all_vals, small_subset, percentile_out, fill_dzsesr_pet, indices_to_fill)
                        obs_full[var_fill][:,:,:] = fill_dzsesr_pet
                    if var_fill == 'SESR_pct_refet':
                        fill_sesr_refet = percentile_of_score_without_numba(land_mask, all_vals, small_subset, percentile_out, fill_sesr_refet, indices_to_fill)
                        obs_full[var_fill][:,:,:] = fill_sesr_refet
                    if var_fill == 'SESR_pct_pet':
                        fill_sesr_pet = percentile_of_score_without_numba(land_mask, all_vals, small_subset, percentile_out, fill_sesr_pet, indices_to_fill)
                        obs_full[var_fill][:,:,:] = fill_sesr_pet


            obs_full.to_netcdf(save_dzsesr_percentile)
                        
    return(f'Completed making the percentiles for both time tuples with window size of {window} years.')



def make_percentile_from_SESR_and_dzSESR_from_all_dates(window,year_ranges_tuple_1,year_ranges_tuple_2):

    '''This means that for each grid cell, there is only 1 value which is ranked as 100% (highest percentile) across the time series.
    
    The function make_percentile_from_SESR_and_dzSESR_by_day_of_year() is the opposite. Each day of year has a percentile value of 100%. I'm still testing which ones work the best'''
    save_dir = call.dzSESR_percentile
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
        
        save_dzsesr_percentile = f'{save_dir}/dzSESR_percentile_all_dates_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        
        if os.path.exists(save_dzsesr_percentile):
            print(f'Completed dzSESR_delta_percentile_all_dates_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
            pass
        else:
            obs_full = xr.open_dataset(f'{call.dzSESR_dir}/dzSESR_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').load()
            '''We now have the SESR_delta, now make percentiles based on the data.'''
            
            obs_full, fill_sesr_pet, fill_sesr_refet, fill_dzsesr_pet, fill_dzsesr_refet = make_empty_variables(obs_full)
            
            vars_to_fill = ['dzSESR_pct_refet','dzSESR_pct_pet','SESR_pct_refet','SESR_pct_pet']
            vars_needed = ['dzSESR_refet','dzSESR_pet','SESR_refet','SESR_pet']

            for var_idx, var_fill in enumerate(vars_to_fill):
                print(f'Finding {var_fill} percentile of score from all dates for each grid cell window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')
                
                all_vals = obs_full[vars_needed[var_idx]].values # ~(window_distribution, 112, 233) (time, lat, lon)
                #get the index values of non np.nan values

                '''Percentile of score - without numba'''
                if var_fill == 'dzSESR_pct_refet':
                    fill_dzsesr_refet = percentile_of_score_without_numba_for_all_dates(land_mask, all_vals, obs_full,  fill_dzsesr_refet,)
                    obs_full[var_fill][:,:,:] = fill_dzsesr_refet
                elif var_fill == 'dzSESR_pct_pet':
                    fill_dzsesr_pet = percentile_of_score_without_numba_for_all_dates(land_mask, all_vals, obs_full,  fill_dzsesr_pet,)
                    obs_full[var_fill][:,:,:] = fill_dzsesr_pet
                if var_fill == 'SESR_pct_refet':
                    fill_sesr_refet = percentile_of_score_without_numba_for_all_dates(land_mask, all_vals, obs_full,  fill_sesr_refet,)
                    obs_full[var_fill][:,:,:] = fill_sesr_refet
                if var_fill == 'SESR_pct_pet':
                    fill_sesr_pet = percentile_of_score_without_numba_for_all_dates(land_mask, all_vals, obs_full,  fill_sesr_pet,)
                    obs_full[var_fill][:,:,:] = fill_sesr_pet

            obs_full.to_netcdf(save_dzsesr_percentile)
                        
    return(f'Completed making the percentiles for both time tuples with window size of {window} years.')


def percentile_of_score_without_numba_for_all_dates(land_mask, all_vals, obs_full, fill_values):

    fill_values[:,:,:] = np.nan
    # Y,X=40,30
    percentile_out = np.empty(shape=(obs_full[list(obs_full.keys())[0]].shape))
    percentile_out[:,:,:] = np.nan
                
    for Y,_ in enumerate(range(land_mask.shape[0])):
        for X,_ in enumerate(range(land_mask.shape[1])):
            # break
            # print(f'Working on Y {Y} and X {X}')
            #Make sure there are no np.nan
            if ~np.isnan(land_mask[Y,X]):
                if np.all(np.isnan(all_vals[:,Y,X])):
                    pass
                elif np.any(np.isnan(all_vals[:,Y,X])):
                    #get no nan values
                    no_nan_vals_obs_full = all_vals[:,Y,X][~np.isnan(all_vals[:,Y,X])]
                    no_nan_indices_obs_full = np.where(~np.isnan(all_vals[:,Y,X]))[0]

                    percentile_out[no_nan_indices_obs_full,Y,X] = pos(all_vals[no_nan_indices_obs_full,Y,X],all_vals[no_nan_indices_obs_full,Y,X])
                else:
                    percentile_out[:,Y,X] = pos(all_vals[:,Y,X],all_vals[:,Y,X])
                    
    fill_values[:,:,:] = percentile_out
    
    return fill_values



def make_percentile_from_RZSM_by_day_of_year(window,year_ranges_tuple_1,year_ranges_tuple_2):
    '''This means that for each grid cell, each day of the year has a value ranked at 100% somewhere in the timeseries.
    
    The function make_percentile_from_SESR_and_dzSESR_from_all_dates() is the opposite. There is only 1 value per grid cell in the time series which has a ranking of 100%.'''
    
    save_dir = call.rzsm_clim_perc
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
        
        save_percentile = f'{save_dir}/RZSM_percentile_by_doy_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        
        if os.path.exists(save_percentile):
            print(f'Completed RZSM_percentile_by_doy_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
            pass
        else:
            
            obs_full = xr.open_dataset(f'{call.noah_dir}RZSM_de-trend_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').load()
            '''We now have the make percentiles based on the data by the day of year'''
            obs_full['rzsm_dtrnd_pct'] = obs_full['rzsm_detrend'].copy(deep=True)
            obs_full['rzsm_dtrnd_pct'][:,:,:] = np.nan
            
            time_index = pd.Index(obs_full['time'].values)
            # Convert the time coordinate to a pandas DatetimeIndex
            time_index1 = pd.DatetimeIndex(obs_full['time'].values)
            
            # Extract the DOY from the time coordinate
            doy1 = time_index1.dayofyear

            vars_to_fill = ['rzsm_dtrnd_pct',]
            vars_needed = ['rzsm_detrend',]

            fill_rzsm = np.empty(obs_full['rzsm_dtrnd_pct'].shape)
            fill_rzsm[:,:,:] = np.nan
            
            for var_idx, var_fill in enumerate(vars_to_fill):
                print(f'Finding {var_fill} percentile of score by day of year for each grid cell window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')

                # break
                for idx, doy in enumerate(range(1, 367)):
                    # idx,doy=0,1
                    subset = get_data_within_45_day_window(obs_full, doy, window) #Get all values within the specified window from the doy
                    subset_index = subset_window_further_SESR_percentile(obs_full, window, doy) #grab based on window and doy
                    
                    subset = [i for i in subset_index if i<len(obs_full.time.values)]
                    subset = [i for i in subset if i>=0]
                    
                    # Get the index time labels where the DOY is equal to the desired value
                    matching_times = time_index1[doy1 == doy]
                    indices_to_fill = [time_index.get_loc(date) for date in matching_times]
                    indices_to_fill = np.array(indices_to_fill)
    
                    all_vals = obs_full[vars_needed[var_idx]][subset,:,:].values # ~(window_distribution, 112, 233) (time, lat, lon)
                
                    small_subset = obs_full[vars_needed[var_idx]][indices_to_fill,:,:].values
                    
                    # Y,X=40,30
                    percentile_out = np.empty(shape = (len(indices_to_fill),obs_full.lat.shape[0],obs_full.lon.shape[0]))
                    percentile_out[:,:,:] = np.nan
                    
                    fill_ = percentile_of_score_without_numba(land_mask, all_vals, small_subset, percentile_out, fill_rzsm, indices_to_fill)
                    obs_full[var_fill][:,:,:] = fill_


            obs_full.to_netcdf(save_percentile)
                        
    return(f'Completed making the percentiles for both time tuples with window size of {window} years.')

def make_percentile_from_RZSM_from_all_dates(window,year_ranges_tuple_1,year_ranges_tuple_2):

    '''This means that for each grid cell, there is only 1 value which is ranked as 100% (highest percentile) across the time series.
    
    The function make_percentile_from_SESR_and_dzSESR_by_day_of_year() is the opposite. Each day of year has a percentile value of 100%. I'm still testing which ones work the best'''
    save_dir = call.rzsm_clim_perc
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
        
        save_percentile = f'{save_dir}/RZSM_percentile_all_dates_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        
        if os.path.exists(save_percentile):
            print(f'Completed RZSM_percentile_all_dates_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
            pass
        else:
            obs_full = xr.open_dataset(f'{call.noah_dir}/RZSM_de-trend_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').load()
            '''We now have the SESR_delta, now make percentiles based on the data.'''
            
            obs_full = xr.open_dataset(f'{call.noah_dir}/RZSM_de-trend_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').load()
            '''We now have the make percentiles based on the data by the day of year'''
            obs_full['rzsm_dtrnd_pct'] = obs_full['rzsm_detrend'].copy(deep=True)
            obs_full['rzsm_dtrnd_pct'][:,:,:] = np.nan

            fill = np.empty_like(obs_full['rzsm_dtrnd_pct'].values)
            
            vars_to_fill = ['rzsm_dtrnd_pct',]
            vars_needed = ['rzsm_detrend',]

            for var_idx, var_fill in enumerate(vars_to_fill):
                print(f'Finding {var_fill} percentile of score from all dates for each grid cell window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')
                
                all_vals = obs_full[vars_needed[var_idx]].values # ~(window_distribution, 112, 233) (time, lat, lon)
                #get the index values of non np.nan values


                fill = percentile_of_score_without_numba_for_all_dates(land_mask, all_vals, obs_full,  fill,)
                obs_full[var_fill][:,:,:] = fill
                

            obs_full.to_netcdf(save_percentile)
                        
    return(f'Completed making the percentiles for both time tuples with window size of {window} years.')


#!/usr/bin/env python3

'''Functions for scripts for Flash drought classification.s'''

import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.stats import percentileofscore as pos
import bottleneck as bn
from numba import njit,prange
import config.dataUtils as dutils
import config.metricUtils as mutils



def delta_SESR_below_40th_and_SESR_below_20th_percentile_numpy_broadcasting(delta_sesr_percentile_vals,sesr_percentile_vals,idx,minimum_length):

    """
    Check if SESR and delta SESR values meet the specified percentile criteria.

    Args:
        delta_sesr_percentile_vals (ndarray): Delta SESR percentile values.
        sesr_percentile_vals (ndarray): SESR percentile values.
        idx (int): Current index.
        minimum_length (int): Minimum length to consider.

    Returns:
        ndarray: Boolean array indicating if conditions are met.
    """

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values #np.nan values indicate water values
    
    #Percentiles only (get every 7th for the previous weeks)
    delta_SESR_percentile = delta_sesr_percentile_vals[idx-minimum_length:idx,:,:][::7]
    SESR_percentile = sesr_percentile_vals[idx-minimum_length:idx,:,:][::7]
    
    '''Must be below 40th percentile for all 5 weeks. But 1 week can be above the 40th if it is surrounded
    by other weeks below the 40th percentile'''
    less_than_40 = delta_SESR_percentile <= 40

    '''Easier to manually loop through than the figure out a broadcasting solution.'''
    # Y,X=40,30
    for Y in range(less_than_40.shape[1]):
        for X in range(less_than_40.shape[2]):
            if np.isnan(land_mask[Y,X]):
                pass
            else:
                for idx,lead in enumerate(np.arange(less_than_40.shape[0])):
                    # break
                    if np.count_nonzero(less_than_40[:,Y,X]) < 4:
                        #If there are less than 4 total True (meaning less than 4 values which are less than the 40th percentile); then break
                        continue
                    elif (less_than_40[idx,Y,X] == False) and (np.count_nonzero(less_than_40[:,Y,X]) == 4):
                        #Must equal to 4, only 1 week can be above the 40th percentile.
                        if idx in [1,2,3]:
                            #If week is surrounded by weeks that are True (e.g., True False True), then we make it into (True True True)
                            if (less_than_40[idx-1,Y,X] == True) and (less_than_40[idx+1,Y,X]==True):
                                less_than_40[idx,Y,X] = True    

    '''First week is not bad, but last 4 weeks are bad. This actually fixes all of'''
    condition1 = np.logical_and(less_than_40[0,:,:] == False, np.all(less_than_40[1:,:,:],axis=0))
    # condition1.shape
    '''Check if all are bad values (less than 40th percentile)'''
    condition2 = np.all(less_than_40,axis=0) 
    # condition2.shape

    final_condition = np.logical_or(condition1,condition2)

    final_condition =  np.logical_and(SESR_percentile[-1,:,:]<=20,final_condition)
    return(final_condition)

def SESR_delta_step_1_flash_drought_classification(window,year_ranges_tuple,minimum_length_in_weeks):
    '''Find if the following criteria are met:
    1) A minimum length of five pentad changes in SESR, equivalent to a length of six pentads (30 days).
    2) A final SESR value below the 20th percentile of SESR values.
    3a) ΔSESR must be at or below the 40th percentile between individual pentads.
    3b) No more than one ΔSESR above the 40th percentile following a ΔSESR that meets criterion 3a.

    An additional script will be needed to get the last step (4)
    4) The mean change in SESR during the entire length of the flash drought must be less than the 25th
percentile.
    '''
    
    save_dir = 'Data/SESR_FD_step_1'
    save_file=f'{save_dir}/SESR_fd_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_file):
        return xr.open_dataset(save_file).load()
    else:
        #Load data
        delta_sesr = xr.open_dataset(f'Data/SESR_delta/SESR_delta_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').load()
        delta_sesr = delta_sesr.rename({'ESR':'fd_sesr'})
        delta_sesr['fd_sesr'][:,:,:] = 0
        
        sesr_percentile_vals = delta_sesr['SESR_percentile'].values
        delta_sesr_percentile_vals = delta_sesr['SESR_delta_percentile'].values
                                       
        #because it must be at least 28 days, we can start from the first 28 days
        minimum_length = (minimum_length_in_weeks*7) + 7 #must include the plus 7 days because we can account for a single week of above the 40th percentile
        
        add_data_to_fd_index_before = np.array(delta_sesr.fd_sesr.values)
        add_data_to_fd_index_after = np.array(delta_sesr.fd_sesr.values)
        # print(np.count_nonzero(add_data_to_fd_index_before))
        
        print(f'\nWindow size {window}. Checking if length >= {minimum_length_in_weeks} weeks, SESR less than 20th percentile, and delta_SESR less than 40th percentile for years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')
        '''Check if SESR is less than 20th and delta_SESR is less than 40th'''
        
        for idx,day in enumerate(delta_sesr.time.values):
            if (idx >= minimum_length):
                # break
                add_data_to_fd_index_before[idx,:,:] = delta_SESR_below_40th_and_SESR_below_20th_percentile_numpy_broadcasting(delta_sesr_percentile_vals,sesr_percentile_vals,idx,minimum_length)
       
        print('Adding weeks if they are surrounded by weeks considered in FD for step 1.')
        #Now add the criteria for the drought to continue
        for idx,day in enumerate(delta_sesr.time.values):
            # print(f'Working on idx {idx}')
            #make sure that we are within the index ranges
            if (idx >=minimum_length) and (idx < len(delta_sesr.time.values)-8):
                add_data_to_fd_index_after[idx,:,:] = (add_data_to_fd_index_before[idx-7,:,:] == True) & \
                            (add_data_to_fd_index_before[idx,:,:] == False) & \
                            (add_data_to_fd_index_before[idx+7,:,:] == True)
                
        # print(np.count_nonzero(add_data_to_fd_index_after))
    
        add_data_to_fd_index_before = add_data_to_fd_index_before + add_data_to_fd_index_after #now add them together to get the final result for FD binary classification
    
        delta_sesr['fd_sesr'][:,:,:] = add_data_to_fd_index_before
        delta_sesr.to_netcdf(save_file)

        del add_data_to_fd_index_before, add_data_to_fd_index_after
        return(delta_sesr)


def create_percentile_SESR(small_obs, large_obs, percentile_fill_file, indices_to_fill, full_fill_file):
    for Y in range(small_obs.shape[1]):
        for X in range(small_obs.shape[2]):
            # break
            #Make sure there are no np.nan
            if np.all(np.isnan(large_obs[:,Y,X])):
                continue
            elif np.any(np.isnan(large_obs[:,Y,X])):
                #get no nan values
                no_nan_vals_obs_full = large_obs[:,Y,X][~np.isnan(large_obs[:,Y,X])]
                no_nan_indices_obs_full = np.where(~np.isnan(large_obs[:,Y,X]))[0]

                no_nan_vals_obs_small = small_obs[:,Y,X][~np.isnan(small_obs[:,Y,X])]
                no_nan_indices_obs_small = np.where(~np.isnan(small_obs[:,Y,X]))[0]
                percentile_fill_file[no_nan_indices_obs_small,Y,X] = pos(large_obs[no_nan_indices_obs_full,Y,X],small_obs[no_nan_indices_obs_small,Y,X])
            
            else:
                percentile_fill_file[:,Y,X] = pos(large_obs[:,Y,X],small_obs[:,Y,X])

    full_fill_file[indices_to_fill,:,:] = percentile_fill_file
    return(full_fill_file)


# Define a numba-compatible percentile of score function
@njit(parallel=True)
def pos_numba(data, value):
    if len(data) == 0:
        return np.nan
    else:
        return np.sum(data < value) / len(data) * 100

@njit(parallel=True)
def create_percentile_SESR_with_numba(small_obs, large_obs, percentile_fill_file, indices_to_fill, full_fill_file,land_mask):
    for Y in prange(small_obs.shape[1]):
        for X in range(small_obs.shape[2]):
            # Ensure there are no np.nan
            if np.isnan(land_mask[Y,X]):
                pass
            elif np.all(np.isnan(large_obs[:,Y,X])):
                pass
            elif np.any(np.isnan(large_obs[:,Y,X])):
                # Get non-nan values
                valid_large_indices = np.where(~np.isnan(large_obs[:,Y,X]))[0]
                no_nan_vals_obs_full = np.take(large_obs[:,Y,X], valid_large_indices)

                valid_small_indices = np.where(~np.isnan(small_obs[:,Y,X]))[0]
                no_nan_vals_obs_small = np.take(small_obs[:,Y,X], valid_small_indices)
                
                for idx in range(len(valid_small_indices)):
                    percentile_fill_file[valid_small_indices[idx],Y,X] = pos_numba(no_nan_vals_obs_full, no_nan_vals_obs_small[idx])
            
            else:
                for idx in range(small_obs.shape[0]):
                    percentile_fill_file[idx,Y,X] = pos_numba(large_obs[:,Y,X], small_obs[idx,Y,X])

    for idx in range(len(indices_to_fill)):
        full_fill_file[indices_to_fill[idx],:,:] = percentile_fill_file[indices_to_fill[idx],:,:]
        
    return full_fill_file

def SESR_delta_step_2_create_percentile_distribution_from_FD_days(fd_index_step_1, window, year_ranges_tuple, minimum_length_in_weeks):
    save_dir = 'Data/SESR_FD_step_2'
    os.makedirs(save_dir, exist_ok=True)

    save_sesr_percentile = f'{save_dir}/SESR_fd_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    if os.path.exists(save_sesr_percentile):
        return(xr.open_dataset(save_sesr_percentile).load())
    else:
        
        #First get the mask of the FD positive days
        fd_mask = fd_index_step_1['fd_sesr']
    
        sesr_vals = fd_index_step_1['SESR']
        delta_sesr_vals = fd_index_step_1['SESR_delta']
    
        #Retrieve only the values during the FD
        sesr_mask = xr.where(fd_mask ==1,sesr_vals,np.nan)
        delta_sesr_mask = xr.where(fd_mask ==1,delta_sesr_vals,np.nan)
    
        '''add_masked_values_to_dataset'''
        fd_index_step_1['SESR_fd_mask'] = sesr_mask
        fd_index_step_1['SESR_delta_fd_mask'] = delta_sesr_mask
    
        #Now make the percentile of score
        print(f'Working on function SESR_delta_step_2_create_percentile_distribution_from_FD_days for_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.\nAnd finding the percentile of score for each grid cell based on window size {window}.')
        print(f'')
    
        fd_index_step_1['fd_SESR_percentile'] = fd_index_step_1['SESR_percentile'].copy()
        fd_index_step_1['fd_SESR_percentile'][:,:,:] = np.nan
        
        fd_index_step_1['fd_SESR_delta_percentile'] = fd_index_step_1['SESR_percentile'].copy()
    
        fill_sesr = fd_index_step_1['fd_SESR_percentile'].values
        fill_delta_sesr = fd_index_step_1['fd_SESR_delta_percentile'].values

        time_index = pd.Index(fd_index_step_1['time'].values)
        # Convert the time coordinate to a pandas DatetimeIndex
        time_index1 = pd.DatetimeIndex(fd_index_step_1['time'].values)
        
        # Extract the DOY from the time coordinate
        doy1 = time_index1.dayofyear
        
        for idx, doy in enumerate(range(1, 367)):
            # break
            # print(f'Working on doy {doy}')
            subset = mutils.get_data_within_doy(fd_index_step_1, doy, window) #Get all values within the specified window from the doy
    
            # Get the index time labels where the DOY is equal to the desired value
            matching_times = time_index1[doy1 == doy]
            indices_to_fill = [time_index.get_loc(date) for date in matching_times]
            indices_to_fill = np.array(indices_to_fill)
            len(indices_to_fill)
            
            # Get the indices of the selected dates
            indices = [time_index.get_loc(date) for date in subset.time.values]
            len(indices)
        
            all_vals_SESR = subset['SESR'][:,:,:].values # ~(window_distribution, 112, 233) (time, lat, lon)
            all_vals_SESR_delta = subset['SESR_delta'][:,:,:].values # ~(window_distribution, 112, 233) (time, lat, lon)
            
            # Y,X=40,30
            percentile_sesr = np.empty(shape = (len(indices_to_fill),subset.lat.shape[0],subset.lon.shape[0]))
            percentile_sesr[:,:,:] = np.nan
            percentile_sesr_delta = percentile_sesr.copy()
        
            sesr_obs_sub = fd_index_step_1['SESR_fd_mask'][indices_to_fill,:,:].values
            sesr_delta_obs_sub = fd_index_step_1['SESR_fd_mask'][indices_to_fill,:,:].values

            #Create percentiles for SESR when there is FD present in step 1
            fill_sesr = create_percentile_SESR(small_obs=sesr_obs_sub, large_obs=all_vals_SESR,
                                                            percentile_fill_file = percentile_sesr,indices_to_fill = indices_to_fill, 
                                                            full_fill_file = fill_sesr)
            
            #Create percentiles for SESR_delta when there is FD present in step 1
            fill_delta_sesr = create_percentile_SESR(small_obs=sesr_delta_obs_sub, large_obs=all_vals_SESR_delta,
                                                            percentile_fill_file = percentile_sesr_delta,indices_to_fill = indices_to_fill, 
                                                            full_fill_file = fill_delta_sesr)

            # '''With numba'''
            # #Create percentiles for SESR when there is FD present in step 1
            # fill_sesr = create_percentile_SESR_with_numba(small_obs=sesr_obs_sub, large_obs=all_vals_SESR,
            #                                                 percentile_fill_file = percentile_sesr,indices_to_fill = indices_to_fill, 
            #                                                 full_fill_file = fill_sesr, land_mask = land_mask)
            
            # #Create percentiles for SESR_delta when there is FD present in step 1
            # fill_delta_sesr = create_percentile_SESR_with_numba(small_obs=sesr_delta_obs_sub, large_obs=all_vals_SESR_delta,
            #                                                 percentile_fill_file = percentile_sesr_delta,indices_to_fill = indices_to_fill, 
            #                                                 full_fill_file = fill_delta_sesr, land_mask = land_mask)
            

        #Now add back to the file  
        fd_index_step_1['fd_SESR_percentile'][:,:,:] = fill_sesr
        fd_index_step_1['fd_SESR_delta_percentile'][:,:,:] = fill_delta_sesr
        
        fd_index_step_1.to_netcdf(save_sesr_percentile)

        del sesr_mask, delta_sesr_mask
        return(fd_index_step_1)



def SESR_delta_step_3_find_if_mean_SESR_and_delta_SESR_is_under_25th_percentile(fd_index_step_2, window, year_ranges_tuple, minimum_length_in_weeks):
    '''check where fd_sesr variable has a value of 1 and where fd_SESR_percentile or 
    fd_SESR_delta_percentile mean change is below the 25th percentile'''
    
    '''To simplify this, we can actually just do a 35 day running mean (no centering).
    The reason behind this is because currently there are only percentile values for SESR and delta_SESR
    that were computed based on FD occurrence being classified by steps 1-2 previously.'''

    save_dir = 'Data/SESR_FD_step_3'
    os.makedirs(save_dir, exist_ok=True)

    save_file = f'{save_dir}/SESR_fd_step_3_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'

    minimum_length_in_days = (minimum_length_in_weeks*7)+7
    if os.path.exists(save_file):
        return xr.open_dataset(save_file).load()
    else:
        print(f'Working on function SESR_delta_step_3_find_if_mean_SESR_and_delta_SESR_is_under_25th_percentile for years {year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')
        
        running_mean_SESR_percentile=fd_index_step_2['fd_SESR_percentile'].rolling(time=minimum_length_in_days, center=False,).mean(skipna=True)

        fd_index_step_2['fd_SESR_percentile_runmean'] = running_mean_SESR_percentile
        running_mean_delta_SESR_percentile=fd_index_step_2['fd_SESR_delta_percentile'].rolling(time=minimum_length_in_days, center=False,).mean(skipna=True)

        fd_index_step_2['fd_SESR_delta_percentile_runmean'] = running_mean_delta_SESR_percentile

        fd_index_step_2['fd_final_with_SESR'] = xr.where(running_mean_SESR_percentile <= 25,fd_index_step_2['fd_sesr'],0)
        fd_index_step_2['fd_final_with_delta_SESR'] = xr.where(running_mean_delta_SESR_percentile <= 25,fd_index_step_2['fd_sesr'],0)
        
        mask = dutils.load_CONUS_mask()['CONUS_mask'].values[0,:,:]
        
        fd_index_step_3 = xr.where(~np.isnan(mask),fd_index_step_2,np.nan) #mask for the United States
        fd_index_step_3.to_netcdf(save_file)
        
        return(fd_index_step_3, window, year_ranges_tuple, minimum_length_in_weeks)

def calculate_FD_index_with_SESR_no_return_values(window):

    '''Requires
    window = (number of days in which the window distribution around the day of year is created)
    year_ranges_tuple = (tuple of year ranges for calculation of the distribution and other statistics)
    minimum_length_in_weeks = (minimum number of weeks in which FD can occur)

    Outputs:
    The final SESR FD index binary classification as well as the percentile values of the delta_SESR
    under the criteria given by Christian et al. (2023).
    '''
    
    out_dictionary = {}

    year_ranges_tuple_1=(1981,2020)
    year_ranges_tuple_2=(2000,2019)
    minimum_length_in_weeks=4

    
    for year_ranges_tuple in [year_ranges_tuple_1,year_ranges_tuple_2]:
        '''Checking if length >= "40th percentile", SESR less than 20th percentile, and delta_SESR less than 40th percentile.'''
        fd_index_step_1 = SESR_delta_step_1_flash_drought_classification(window,year_ranges_tuple, minimum_length_in_weeks)
        
        '''Next we need to create a new distribution based on the SESR and delta_SESR values when there is a FD classification
        Step 2 find the percentile of score of the SESR and delta_SESR values which lie under the FD conditions from step 1'''
        
        fd_index_step_2 = SESR_delta_step_2_create_percentile_distribution_from_FD_days(fd_index_step_1, window, year_ranges_tuple, minimum_length_in_weeks)
        
        '''Step 3 will then find the values where the mean change in SESR during the entire length of the FD must be less than the 25th percentile'''
        fd_index_step_3 = SESR_delta_step_3_find_if_mean_SESR_and_delta_SESR_is_under_25th_percentile(fd_index_step_2, window, year_ranges_tuple, minimum_length_in_weeks)

    return(0)

def calculate_FD_index_with_SESR_yes_return_values(window):

    '''Requires
    window = (number of days in which the window distribution around the day of year is created)
    year_ranges_tuple = (tuple of year ranges for calculation of the distribution and other statistics)
    minimum_length_in_weeks = (minimum number of weeks in which FD can occur)

    Outputs:
    The final SESR FD index binary classification as well as the percentile values of the delta_SESR
    under the criteria given by Christian et al. (2023).
    '''
    
    '''Checking if length >= "40th percentile", SESR less than 20th percentile, and delta_SESR less than 40th percentile.'''

    out_dictionary = {}

    year_ranges_tuple_1=(1981,2020)
    year_ranges_tuple_2=(2000,2019)
    minimum_length_in_weeks=4
    
    for year_ranges_tuple in [year_ranges_tuple_1,year_ranges_tuple_2]:
        fd_index_step_1 = SESR_delta_step_1_flash_drought_classification(window,year_ranges_tuple, minimum_length_in_weeks)
        
        '''Next we need to create a new distribution based on the SESR and delta_SESR values when there is a FD classification
        Step 2 find the percentile of score of the SESR and delta_SESR values which lie under the FD conditions from step 1'''
        
        fd_index_step_2 = SESR_delta_step_2_create_percentile_distribution_from_FD_days(fd_index_step_1, window, year_ranges_tuple, minimum_length_in_weeks)
        
        '''Step 3 will then find the values where the mean change in SESR during the entire length of the FD must be less than the 25th percentile'''
        fd_index_step_3 = SESR_delta_step_3_find_if_mean_SESR_and_delta_SESR_is_under_25th_percentile(fd_index_step_2, window, year_ranges_tuple, minimum_length_in_weeks)

        out_dictionary[f'{year_ranges_tuple[0]}-{year_ranges_tuple[1]}'] = fd_index_step_3

    out_dictionary['minimum_length_in_weeks'] = minimum_length_in_weeks
    return(out_dictionary)
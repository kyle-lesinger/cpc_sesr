#!/usr/bin/env python3

'''Functions for scripts for Flash drought classification.s'''

import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.stats import percentileofscore as pos
from scipy import stats
from scipy.signal import detrend
import bottleneck as bn
from numba import njit,prange
from datetime import datetime
import config.dataUtils as dutils
import config.metricUtils as mutils
import config.detrendUtils as trendUtils
import config.STATIC as call

def load_data_before_FD_classification_REALTIME(climatology, window, all_dates_or_doy, recompute_ESR_trend_slope,recompute_dzSESR_percentiles,recompute_RZSM_percentiles):
    #First we need to load the full set of data
    esr, rzsm, clim_esr, clim_rzsm, trend_esr, trend_rzsm, clim_sesr = load_full_data_for_realtime_analysis(climatology, window, all_dates_or_doy)
                
    #Then we need to standardize based on climatology_ESR_mean_std
    ESR_detrend_climatology = save_esr_trend_slope_climatology_for_detrending_REALTIME(esr, trend_esr, clim_esr,climatology,recompute_ESR_trend_slope)
    ESR_detrend  = ESR_detrended(esr,ESR_detrend_climatology,climatology)
    '''Next,  remove the trend x - (slope * t + intercept)
    where x is the day of year value, 
    slope is already saved by day of year
    t is the index of the time for each day of year (e.g., 1980-01-01 is t=0 and 1983-01-01 is t =3,
    and intercept is already saved by day of year'''
    
    SESR = standardize_ESR_realtime(ESR_detrend, clim_esr, climatology)
    dzSESR = realtime_dzSESR_change_between_weeks_and_STANDARDIZE(SESR,clim_sesr,climatology)
    dzSESR_percentile = REALTIME_percentile_from_SESR_and_dzSESR_by_day_of_year(window,climatology,dzSESR,recompute_dzSESR_percentiles)
    #Make percentiles from data for SESR and dzSESR by day of year only
    
    #and for RZSM
    RZSM_detrend_climatology = save_RZSM_trend_slope_climatology_for_detrending_REALTIME(rzsm, climatology)
    RZSM_detrend  = RZSM_detrended(rzsm,RZSM_detrend_climatology,climatology)
    RZSM_percentile =  REALTIME_percentile_RZSM_by_day_of_year(window,climatology,RZSM_detrend, rzsm,recompute_RZSM_percentiles)
    '''Now all values are between 0 and 1 (where they should be). This makes it a little more interpretable'''
    
    return esr, rzsm, clim_esr, clim_rzsm, trend_esr, trend_rzsm, clim_sesr, ESR_detrend, dzSESR, dzSESR_percentile, ESR_detrend_climatology,RZSM_detrend_climatology,  RZSM_detrend, RZSM_percentile, climatology, window,  all_dates_or_doy
    


def load_full_data_for_realtime_analysis(climatology, window, all_dates_or_doy):

    esr=xr.open_dataset(f'{call.noah_dir}/esr_rolling_mean_0.50_degrees.nc')
    rzsm=xr.open_dataset(f'{call.noah_dir}/rzsm_0.50_degrees.nc')
    #Then we need to divide by 1000 kg/m2 to get data into m3/m3
    rzsm = rzsm/ 1000
    #Then we need to apply the rolling mean
    rzsm = rzsm.rolling(time=call.mean_rolling_length, center=True).mean()
    #Also be sure to only select the depth_2 (specific to NLDAS RZSM file)
    rzsm = rzsm.isel(depth_2=0)
        
    clim_esr = xr.open_dataset(f'{call.noah_dir}//climatology_ESR_detrend_mean_std/ESR_mean_std_window_size_{window}_years_{climatology[0]}-{climatology[1]}.nc')
    clim_rzsm = xr.open_dataset(f'{call.noah_dir}/climatology_RZSM_percentile/RZSM_percentile_{all_dates_or_doy}_window_size_{window}_years_{climatology[0]}-{climatology[1]}.nc')
    #Make sure we de-trend it
    trend_esr = xr.open_dataset(f'{call.noah_dir}/doy_trend/EVP_pet_refet_ESR_trend_change_per_year_{climatology[0]}-{climatology[1]}.nc')
    trend_rzsm = xr.open_dataset(f'{call.noah_dir}/doy_trend/RZSM_trend_change_per_year_{climatology[0]}-{climatology[1]}.nc')

    clim_sesr = xr.open_dataset(f'{call.noah_dir}/climatology_SESR_delta_mean_std/dzSESR_mean_std_{window}_years_{climatology[0]}-{climatology[1]}.nc')
    return esr, rzsm, clim_esr, clim_rzsm, trend_esr, trend_rzsm, clim_sesr


def detrend_esr_data(esr, trend_esr, clim_esr,climatology):
    out_ESR = trendUtils.make_empty_file_full(esr)
    
    for idx, varname in enumerate(esr.keys()):
        # break
        print(f'Removing the trend from {climatology} for variable {varname} range {esr.time.values[0]} - {esr.time.values[-1]}')
        for idx_for_mean_std,doy in enumerate(range(1,367)):
            # break
            md = pd.to_datetime(esr.isel(time=idx_for_mean_std).time.values)
            month_day = f'2000-{md.month:02}-{md.day:02}'
        
            trnd_val = trend_esr[varname].sel(time=month_day).values
            
            # Create a mask for the target month and day. This selects only the same doy from all the other years
            mask = (esr['time'].dt.month == md.month) & (esr['time'].dt.day == md.day)
            
            # Get the indices and dates
            indices = np.where(mask)[0]

            dates = esr['time'].values[mask] #actual doy values of the days of the year

            detrended = esr[varname][indices,:,:].values - trnd_val
            out_ESR[varname][indices,:,:] = detrended
    return out_ESR

def convert_to_min_max(file):
    # Assuming 'ds' is your xarray dataset with 6 variables
    ds_standardized = xr.Dataset()
    
    for var in file.data_vars:
        # Calculate the minimum and maximum for each grid cell
        min_val = file[var].min(dim='time',skipna=True)
        max_val = file[var].max(dim='time',skipna=True)
        
        # Apply the min-max scaling
        standardized = (file[var] - min_val) / (max_val - min_val)
        
        # Store the standardized variable in the new dataset
        ds_standardized[var] = standardized

    return ds_standardized

def detrend_rzsm_data(rzsm, trend_rzsm, climatology):
    out_ESR = trendUtils.make_empty_file_full(rzsm)
    
    for idx, varname in enumerate(rzsm.keys()):
        # break
        print(f'Removing the trend from {climatology} for variable {varname} range {rzsm.time.values[0]} - {rzsm.time.values[-1]}')
        for idx_for_mean_std,doy in enumerate(range(1,367)):
            # break
            md = pd.to_datetime(rzsm.isel(time=idx_for_mean_std).time.values)
            month_day = f'2000-{md.month:02}-{md.day:02}'
        
            trnd_val = trend_rzsm[varname].sel(time=month_day).values
            
            # Create a mask for the target month and day. This selects only the same doy from all the other years
            mask = (rzsm['time'].dt.month == md.month) & (rzsm['time'].dt.day == md.day)
            
            # Get the indices and dates
            indices = np.where(mask)[0]

            dates = rzsm['time'].values[mask] #actual doy values of the days of the year

            detrended = rzsm[varname][indices,:,:].values - trnd_val
            out_ESR[varname][indices,:,:] = detrended
    return out_ESR

def standardize_ESR_realtime(final_ESR_stand, clim_esr, climatology):

    sesr_esr = final_ESR_stand.copy(deep=True)
    
    for iVar, var in enumerate(['ESR_pet_dtrnd','ESR_refet_dtrnd']):

        sesr_esr[f'S{var}'] = final_ESR_stand[var].copy(deep=True)
        sesr_esr[f'S{var}'][:,:,:] = np.nan
        
        if var == 'ESR_pet':
            mean_ = 'mean_pet'
            std_ = 'std_pet'
        else:
            mean_ = 'mean_refet'
            std_ = 'std_refet'
            
        print(f'Standardizing from {climatology} for variable {var} range {final_ESR_stand.time.values[0]} - {final_ESR_stand.time.values[-1]}')
        
        for idx_for_mean_std,doy in enumerate(range(1,367)):
            # break
            md = pd.to_datetime(final_ESR_stand.isel(time=idx_for_mean_std).time.values)
            month_day = f'2000-{md.month:02}-{md.day:02}'
        
            clim_mean = clim_esr[mean_].sel(time=month_day).values
            clim_std = clim_esr[std_].sel(time=month_day).values
            
            # Create a mask for the target month and day. This selects only the same doy from all the other years
            mask = (final_ESR_stand['time'].dt.month == md.month) & (final_ESR_stand['time'].dt.day == md.day)
            
            # Get the indices and dates
            indices = np.where(mask)[0]

            dates = final_ESR_stand['time'].values[mask] #actual doy values of the days of the year

            standardize = (final_ESR_stand[var][indices,:,:].values - clim_mean)/ clim_std
            sesr_esr[f'S{var}'][indices,:,:] = standardize
    return sesr_esr

def realtime_dzSESR_change_between_weeks_and_STANDARDIZE(SESR,clim_sesr,climatology):

    number_of_weeks = call.num_weeks_difference_SESR  # this is for calculating the difference between periods. We are using a difference of 1 week as the standard
    #We keep this as a constant and not an argument to help with the multi-processing function.
    total_days = number_of_weeks * 7
    '''Here we only need to find the difference between weeks, then standardize with mean and std 
    from the climatology'''
    time_index_short = pd.to_datetime(clim_sesr.sel(time=slice(call.doy_start,call.doy_end)).time.values)
    time_index_full = pd.to_datetime(SESR.time.values)

    final_out = SESR.copy(deep=True)
    
    
    for iVar, var in enumerate(['SESR_pet_dtrnd','SESR_refet_dtrnd']):
        print(f'Computing the weekly difference for var {var} across all years and subtracting the already computed climatology for {climatology}.')
        # iVar, var = 0, 'SESR_pet'
        final_out[f'dz{var}'] = final_out[var].copy(deep=True)
        final_out[f'dz{var}'][:,:,:] = np.nan
        
        # iVar, var = 0, 'ESR_pet'
        fill_array = final_out[var].values
        fill_array[:,:,:] = np.nan
    
        for idx,date in enumerate(time_index_short):
            mean_ = clim_sesr[f'{var.split("_dtrnd")[0]}_mean'].sel(time=date).values
            std_ = clim_sesr[f'{var.split("_dtrnd")[0]}_std'].sel(time=date).values
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
            selected_data = SESR[var].isel(time=true_indices_current_week)
        
            previous_week_indices = np.array([i - total_days for i in true_indices_current_week if i-total_days >=0]) #Make no negative index values
            selected_data_previous = SESR[var].isel(time=previous_week_indices)
        
            # #Sometimes we have a mis-match between years (specifically the number of data points, they must be equal!), so this fixes it
            if len(selected_data_previous.time.values) > len(selected_data.time.values):
                selected_data_previous = selected_data_previous.isel(time = slice(0,len(selected_data.time.values)))
            elif len(selected_data_previous.time.values) < len(selected_data.time.values):
                selected_data = selected_data.isel(time = slice(1,len(selected_data.time.values)))

            #Now find the difference across all years and average
            weekly_difference = selected_data.values - selected_data_previous.values

            #The final correct indices to replace
            new_dates = selected_data.time.values
            
            # Find the time indices in the xarray dataset that match the dates
            time_indices = final_out.time.sel(time=new_dates).time
            
            # If you need the actual indices (integer positions)
            integer_indices = final_out.time.get_index('time').get_indexer(time_indices)
            
            #Now standardize
            fill_array[integer_indices,:,:] =  (weekly_difference - mean_) / std_

        final_out[f'dz{var}'][:,:,:] = fill_array


    return final_out



def return_slope_from_climatology(x):
    """Remove the linear trend from data and return detrended values as a DataArray."""
    t = np.arange(len(x))
    mask = np.isfinite(x)
    
    if np.sum(mask) > 1:  # Ensure there are enough points to fit a line
        slope, intercept, r_value, p_value, std_err = stats.linregress(t[mask], x[mask])
        return xr.DataArray(slope, dims=x.dims, coords=x.coords)
    else:
        # Not enough points to detrend, return original data
        return xr.DataArray(x, dims=x.dims, coords=x.coords)

def return_intercept_from_climatology(x):
    """Remove the linear trend from data and return detrended values as a DataArray."""
    t = np.arange(len(x))
    mask = np.isfinite(x)
    
    if np.sum(mask) > 1:  # Ensure there are enough points to fit a line
        slope, intercept, r_value, p_value, std_err = stats.linregress(t[mask], x[mask])
        return xr.DataArray(intercept, dims=x.dims, coords=x.coords)
    else:
        # Not enough points to detrend, return original data
        return xr.DataArray(x, dims=x.dims, coords=x.coords)


def save_esr_trend_slope_climatology_for_detrending_REALTIME(esr, trend_esr, clim_esr,climatology,recompute_ESR_trend_slope):

    save_dir = f'{call.noah_dir}/climatology_trends'
    os.makedirs(save_dir, exist_ok=True)
    save_file = f'{save_dir}/ESR_climatology_slope_intercept_{climatology[0]}-{climatology[-1]}.nc'
    
    if recompute_ESR_trend_slope==False:
        return xr.open_dataset(save_file)
    else:
        out_ESR = trendUtils.make_empty_file_full(esr.sel(time=slice(call.doy_start,call.doy_end)))
    
        land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
        
        # Loop through each latitude and longitude point
        for iVar, pet_or_refet in enumerate(['ESR_pet','ESR_refet']):
            # break
            out_ESR[f'slope_{pet_or_refet}'] = out_ESR[pet_or_refet].copy(deep=True)
            out_ESR[f'intercept_{pet_or_refet}'] = out_ESR[pet_or_refet].copy(deep=True)
            
            print(f'Working on saving the slope and intercept for {pet_or_refet}_years_{climatology}')
            for Y in range(esr.lat.shape[0]):  # Latitude
                for X in range(esr.lon.shape[0]):  # Longitude
                    if np.isnan(land_mask[Y,X]):
                        pass
                    else:
                        # print(f'Working on Y {Y} and X {X}')
                        ts = esr[pet_or_refet].sel(time=slice(str(climatology[0]),str(climatology[-1])))[:, Y, X]  # Extract time series as 1-d array
                        ts_arr = ts.values
    
                        #get the index values of non np.nan values
                        index_vals = [idx for idx,i in enumerate(ts_arr) if ~np.isnan(i)]
                        
                        no_nan_values = ts[index_vals] #subset only non np.nan values
    
                        if len(no_nan_values) !=0:
                            '''Manually check the trend of a single instance'''
                            # b4 = no_nan_values[no_nan_values['time.dayofyear'] == 1]
                            # plt.plot(np.arange(len(b4.values)),b4.values)
    
                            '''We only need to save the slope for each day of year since they are the same'''
                            slope = no_nan_values.groupby('time.dayofyear').apply(return_slope_from_climatology).sel(time=slice(call.doy_start,call.doy_end))
                            intercept = no_nan_values.groupby('time.dayofyear').apply(return_intercept_from_climatology).sel(time=slice(call.doy_start,call.doy_end))
    
                            out_ESR[f'slope_{pet_or_refet}'][:,Y,X] = slope
                            out_ESR[f'intercept_{pet_or_refet}'][:,Y,X] = intercept
        out_ESR.to_netcdf(save_file)
        return out_ESR


def save_RZSM_trend_slope_climatology_for_detrending_REALTIME(rzsm,climatology):

    save_dir = f'{call.noah_dir}/climatology_trends'
    os.makedirs(save_dir, exist_ok=True)
    save_file = f'{save_dir}/RZSM_climatology_slope_intercept_{climatology[0]}-{climatology[-1]}.nc'
    
    if os.path.exists(save_file):
        return xr.open_dataset(save_file)
    else:
        out_ESR = trendUtils.make_empty_file_full(rzsm.sel(time=slice(call.doy_start,call.doy_end)))
    
        land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
        
        # Loop through each latitude and longitude point
        for iVar, pet_or_refet in enumerate(['RZSM']):
            # break
            out_ESR[f'slope_{pet_or_refet}'] = out_ESR[pet_or_refet].copy(deep=True)
            out_ESR[f'intercept_{pet_or_refet}'] = out_ESR[pet_or_refet].copy(deep=True)
            
            print(f'Working on saving the slope and intercept for {pet_or_refet}_years_{climatology}')
            for Y in range(rzsm.lat.shape[0]):  # Latitude
                for X in range(rzsm.lon.shape[0]):  # Longitude
                    if np.isnan(land_mask[Y,X]):
                        pass
                    else:
                        # print(f'Working on Y {Y} and X {X}')
                        ts = rzsm[pet_or_refet].sel(time=slice(str(climatology[0]),str(climatology[-1])))[:, Y, X]  # Extract time series as 1-d array
                        ts_arr = ts.values
    
                        #get the index values of non np.nan values
                        index_vals = [idx for idx,i in enumerate(ts_arr) if ~np.isnan(i)]
                        
                        no_nan_values = ts[index_vals] #subset only non np.nan values
    
                        if len(no_nan_values) !=0:
                            '''Manually check the trend of a single instance'''
                            # b4 = no_nan_values[no_nan_values['time.dayofyear'] == 1]
                            # plt.plot(np.arange(len(b4.values)),b4.values)
    
                            '''We only need to save the slope for each day of year since they are the same'''
                            slope = no_nan_values.groupby('time.dayofyear').apply(return_slope).sel(time=slice(call.doy_start,call.doy_end))
                            intercept = no_nan_values.groupby('time.dayofyear').apply(return_intercept).sel(time=slice(call.doy_start,call.doy_end))
    
                            out_ESR[f'slope_{pet_or_refet}'][:,Y,X] = slope
                            out_ESR[f'intercept_{pet_or_refet}'][:,Y,X] = intercept
        out_ESR.to_netcdf(save_file)
        return out_ESR


def ESR_detrended(esr,ESR_detrend_climatology,climatology):
    '''Next,  remove the trend x - (slope * t + intercept)
    where x is the day of year value, 
    slope is already saved by day of year
    t is the index of the time for each day of year (e.g., 1980-01-01 is t=0 and 1983-01-01 is t =3,
    and intercept is already saved by day of year
    
    This is taken from scipy.stats but allowed to incorporate np.nan values'''

    
    esr_detrend = esr.copy(deep=True)
    
    for iVar, var in enumerate(['ESR_pet','ESR_refet']):

        esr_detrend[f'{var}_dtrnd'] = esr[var].copy(deep=True)
        esr_detrend[f'{var}_dtrnd'][:,:,:] = np.nan

        fill_array = np.empty_like(esr_detrend[f'{var}_dtrnd'].values)
            
        print(f'Standardizing from climatology {climatology} for variable {var} over range {esr.time.values[0]} - {esr.time.values[-1]}')
        
        for idx_for_mean_std,doy in enumerate(range(1,367)):
            # break
            md = pd.to_datetime(esr.isel(time=idx_for_mean_std).time.values)
            month_day = f'2000-{md.month:02}-{md.day:02}'
        
            clim_slope = ESR_detrend_climatology[f'slope_{var}'].sel(time=month_day).values
            clim_intercept = ESR_detrend_climatology[f'intercept_{var}'].sel(time=month_day).values
            
            # Create a mask for the target month and day. This selects only the same doy from all the other years
            mask = (esr['time'].dt.month == md.month) & (esr['time'].dt.day == md.day)
            
            # Get the indices and dates
            indices = np.where(mask)[0]

            dates = esr['time'].values[mask] #actual doy values of the days of the year
            vals = esr[var][indices,:,:].values

            for t in range(vals.shape[0]):
                vals[t] = vals[t] - (clim_slope * t + clim_slope)

            min_max_standardize = (vals- np.nanmin(vals,axis=0))/(np.nanmax(vals,axis=0)-np.nanmin(vals,axis=0))
            fill_array[indices,:,:] = min_max_standardize

        esr_detrend[f'{var}_dtrnd'][:,:,:] = fill_array
    
    return esr_detrend


def standardize_ESR_realtime_from_climatology(final_ESR_stand, clim_esr, climatology):

    sesr_esr = final_ESR_stand.copy(deep=True)
    
    for iVar, var in enumerate(['ESR_pet','ESR_refet']):

        sesr_esr[f'S{var}'] = final_ESR_stand[var].copy(deep=True)
        sesr_esr[f'S{var}'][:,:,:] = np.nan

        fill_array = np.empty_like(sesr_esr[f'S{var}'].values)
        
        if var == 'ESR_pet':
            mean_ = 'mean_pet'
            std_ = 'std_pet'
        else:
            mean_ = 'mean_refet'
            std_ = 'std_refet'
            
        print(f'Standardizing from {climatology} for variable {var} range {final_ESR_stand.time.values[0]} - {final_ESR_stand.time.values[-1]}')
        
        for idx_for_mean_std,doy in enumerate(range(1,367)):
            # break
            md = pd.to_datetime(final_ESR_stand.isel(time=idx_for_mean_std).time.values)
            month_day = f'2000-{md.month:02}-{md.day:02}'
        
            clim_mean = clim_esr[mean_].sel(time=month_day).values
            clim_std = clim_esr[std_].sel(time=month_day).values
            
            # Create a mask for the target month and day. This selects only the same doy from all the other years
            mask = (final_ESR_stand['time'].dt.month == md.month) & (final_ESR_stand['time'].dt.day == md.day)
            
            # Get the indices and dates
            indices = np.where(mask)[0]

            dates = final_ESR_stand['time'].values[mask] #actual doy values of the days of the year

            standardize = (final_ESR_stand[var][indices,:,:].values - clim_mean)/ clim_std
            np.nanmax(standardize)
            
            fill_array[indices,:,:] = standardize
            
        sesr_esr[f'S{var}'][:,:,:] = fill_array
    return sesr_esr


def RZSM_detrended(rzsm,RZSM_detrend_climatology,climatology):
    '''Next,  remove the trend x - (slope * t + intercept)
    where x is the day of year value, 
    slope is already saved by day of year
    t is the index of the time for each day of year (e.g., 1980-01-01 is t=0 and 1983-01-01 is t =3,
    and intercept is already saved by day of year
    
    This is taken from scipy.stats but allowed to incorporate np.nan values'''

    
    rzsm_detrend = rzsm.copy(deep=True)
    
    for iVar, var in enumerate(['RZSM']):

        rzsm_detrend[f'{var}_dtrnd'] = rzsm[var].copy(deep=True)
        rzsm_detrend[f'{var}_dtrnd'][:,:,:] = np.nan

        fill_array = np.empty_like(rzsm_detrend[f'{var}_dtrnd'].values)
            
        print(f'Standardizing from climatology {climatology} for variable {var} over range {rzsm.time.values[0]} - {rzsm.time.values[-1]}')
        
        for idx_for_mean_std,doy in enumerate(range(1,367)):
            # break
            md = pd.to_datetime(rzsm.isel(time=idx_for_mean_std).time.values)
            month_day = f'2000-{md.month:02}-{md.day:02}'
        
            clim_slope = RZSM_detrend_climatology[f'slope_{var}'].sel(time=month_day).values
            clim_intercept = RZSM_detrend_climatology[f'intercept_{var}'].sel(time=month_day).values
            
            # Create a mask for the target month and day. This selects only the same doy from all the other years
            mask = (rzsm['time'].dt.month == md.month) & (rzsm['time'].dt.day == md.day)
            
            # Get the indices and dates
            indices = np.where(mask)[0]

            dates = rzsm['time'].values[mask] #actual doy values of the days of the year
            vals = rzsm[var][indices,:,:].values

            for t in range(vals.shape[0]):
                vals[t] = vals[t] - (clim_slope * t + clim_slope)

            min_max_standardize = (vals- np.nanmin(vals,axis=0))/(np.nanmax(vals,axis=0)-np.nanmin(vals,axis=0))
            fill_array[indices,:,:] = min_max_standardize

        rzsm_detrend[f'{var}_dtrnd'][:,:,:] = fill_array
    
    return rzsm_detrend


def get_clim_near_window(obs_full, doy, window,climatology):
    # Function to get data within ±n days for a specific day of the year (DOY)
    
    start_doy = (doy - window) % 366
    end_doy = (doy + window) % 366

    # Extract the day of year from the time dimension
    time_doy = obs_full['time'].dt.dayofyear

    if (window==0) and (doy==366):
        mask = (time_doy >= 365) #just include this so that we can have values 
    elif start_doy > end_doy:
        mask = (time_doy >= start_doy) | (time_doy <= end_doy)
    elif doy ==1:
        mask = (time_doy >= 366-window) | (time_doy <= end_doy)
    else:
        mask = (time_doy >= start_doy) & (time_doy <= end_doy)
    
    return obs_full.sel(time=mask).sel(time=slice(str(climatology[0]),str(climatology[1])))



def make_empty_variables_REALTIME(obs_full):
    '''Make empty variables to fill'''
    obs_full['dzSESR_pct_refet_dtrnd'] = obs_full['SESR_pet_dtrnd'].copy(deep=True)
    obs_full['dzSESR_pct_refet_dtrnd'][:,:,:] = np.nan

    obs_full['dzSESR_pct_pet_dtrnd'] = obs_full['dzSESR_pct_refet_dtrnd'].copy(deep=True)
    obs_full['SESR_pct_refet_dtrnd'] = obs_full['dzSESR_pct_refet_dtrnd'].copy(deep=True)
    obs_full['SESR_pct_pet_dtrnd'] = obs_full['dzSESR_pct_refet_dtrnd'].copy(deep=True)

    '''Just do this to ensure that none of the data is actually just a mirror to another object'''
    fill_sesr_pet = np.empty(obs_full['SESR_pct_pet_dtrnd'].shape)
    fill_sesr_pet[:,:,:] = np.nan
    fill_sesr_refet = np.empty(obs_full['SESR_pct_pet_dtrnd'].shape)
    fill_sesr_pet[:,:,:] = np.nan
    fill_dzsesr_pet = np.empty(obs_full['SESR_pct_pet_dtrnd'].shape)
    fill_sesr_pet[:,:,:] = np.nan
    fill_dzsesr_refet = np.empty(obs_full['SESR_pct_pet_dtrnd'].shape)
    fill_sesr_pet[:,:,:] = np.nan
    
    return obs_full, fill_sesr_pet, fill_sesr_refet, fill_dzsesr_pet, fill_dzsesr_refet


def return_indices_to_grab_and_fill_data(subset_clim, subset_after_clim, doy, time_index_full,climatology):
    '''Get indices for data that is the climatology subset (separated from post-climatology data)'''
    time_index_clim = pd.Index(subset_clim['time'].values)
    # Convert the time coordinate to a pandas DatetimeIndex
    time_index1_clim = pd.DatetimeIndex(subset_clim['time'].values)
    doy1_clim = time_index1_clim.dayofyear
    
    clim_index_subset = time_index1_clim[doy1_clim == doy]
    
    '''Get indices for data that is after climatology subset'''
    time_index_af_clim = pd.Index(subset_after_clim['time'].values)
    # Convert the time coordinate to a pandas DatetimeIndex
    time_index1_af_clim = pd.DatetimeIndex(subset_after_clim['time'].values)                    
    doy1_af_clim = time_index1_af_clim.dayofyear


    # Get the index time labels where the DOY is equal to the desired value
    match_clim, match_af_clim = time_index1_clim[doy1_clim == doy], time_index1_af_clim[doy1_af_clim == doy]

    '''We still want to keep the full dataset, so these indices are the locations of the data to be filled after
    the climatology period'''
    indices_to_fill_af_clim = [time_index_full.get_loc(date) for date in match_af_clim if date.year > climatology[-1]]

    '''Since the post-climatology is already split into a different object, these are the correct indices which are 
    going to be ranked with the percentile function'''                       
    indices_to_grab_af_clim = [time_index_af_clim.get_loc(date) for date in match_af_clim if date.year > climatology[-1]]
    
    return indices_to_fill_af_clim, indices_to_grab_af_clim, match_clim, match_af_clim

def REALTIME_percentile_from_SESR_and_dzSESR_by_day_of_year(window,climatology,dzSESR, recompute_dzSESR_percentiles):
    '''This means that for each grid cell, each day of the year has a value ranked at 100% somewhere in the timeseries.
    
    The function make_percentile_from_SESR_and_dzSESR_from_all_dates() is the opposite. There is only 1 value per grid cell in the time series which has a ranking of 100%.'''
    
    save_dir = call.realtime_NOAH_dir
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    # Get today's date
    today_date = datetime.today()
    formatted_date = today_date.strftime("%Y-%m-%d")
    
    save_dzsesr_percentile = f'{save_dir}/dzSESR_pct_by_doy_window_size{window}_clim{climatology[0]}-{climatology[1]}.nc'
    
    if recompute_dzSESR_percentiles == False:
        
        return(xr.open_dataset(save_dzsesr_percentile))
    else:
        # Check if the file exists
        if os.path.exists(save_dzsesr_percentile):
            # Remove the file
            os.remove(save_dzsesr_percentile)
            print(f"File '{save_dzsesr_percentile}' has been removed.")

        obs_full, fill_sesr_pet, fill_sesr_refet, fill_dzsesr_pet, fill_dzsesr_refet = make_empty_variables_REALTIME(dzSESR)
        
        time_index_full = pd.Index(obs_full['time'].values)
        # Convert the time coordinate to a pandas DatetimeIndex
        time_index1_full = pd.DatetimeIndex(obs_full['time'].values)
        
        # Extract the DOY from the time coordinate
        doy1 = time_index1_full.dayofyear

        vars_to_fill = ['dzSESR_pct_refet_dtrnd','dzSESR_pct_pet_dtrnd','SESR_pct_refet_dtrnd','SESR_pct_pet_dtrnd']
        vars_needed = ['dzSESR_refet_dtrnd','dzSESR_pet_dtrnd','SESR_refet_dtrnd','SESR_pet_dtrnd']

        for var_idx, var_fill in enumerate(vars_to_fill):
            # var_idx, var_fill = 0,'dzSESR_pct_refet_dtrnd'
            print(f'Finding {var_fill} percentile of score by day of year for each grid cell window_size_{window}_climatology_{climatology}.')

            # break
            for idx, doy in enumerate(range(1, 367)):

                # idx,doy=0,1
                '''get the climatology only to rank percentiles'''
                subset_clim = get_clim_near_window(obs_full[vars_needed[var_idx]], doy, window,climatology) #Get all values within the specified window from the doy
                subset_after_clim = obs_full[vars_needed[var_idx]].sel(time=slice(str(climatology[1]+1), str(pd.to_datetime(obs_full.time.values[-1]).year)))

                indices_to_fill_af_clim, indices_to_grab_af_clim, match_clim, match_af_clim = return_indices_to_grab_and_fill_data(subset_clim, subset_after_clim, doy,time_index_full,climatology)

                
                if len(indices_to_grab_af_clim) >0:
                    '''VERY IMPORTANT, This ensures that if there are no doy 366 in the after climatology distribution then it won't break the script.'''
                    # clim_indices_just_in_case = [time_index.get_loc(date) for date in match_af_clim if date.year > climatology[-1]]
                    indices_to_fill = np.array(indices_to_fill_af_clim)
    
                    # Y,X=40,30
                    percentile_out = np.empty(shape = (len(indices_to_fill_af_clim),obs_full.lat.shape[0],obs_full.lon.shape[0]))
                    percentile_out[:,:,:] = np.nan
                    
                    '''Percentile of score - without numba'''
                    if var_fill == 'dzSESR_pct_refet_dtrnd':
                        fill_dzsesr_refet = percentile_of_score_without_numba_REALTIME(land_mask, subset_clim, subset_after_clim, percentile_out, fill_dzsesr_refet, indices_to_fill, indices_to_grab_af_clim)
                        obs_full[var_fill][:,:,:] = fill_dzsesr_refet
                    elif var_fill == 'dzSESR_pct_pet_dtrnd':
                        fill_dzsesr_pet = percentile_of_score_without_numba_REALTIME(land_mask, subset_clim, subset_after_clim, percentile_out, fill_dzsesr_pet, indices_to_fill, indices_to_grab_af_clim)
                        obs_full[var_fill][:,:,:] = fill_dzsesr_pet
                    elif var_fill == 'SESR_pct_refet_dtrnd':
                        fill_sesr_refet = percentile_of_score_without_numba_REALTIME(land_mask, subset_clim, subset_after_clim, percentile_out, fill_sesr_refet, indices_to_fill, indices_to_grab_af_clim)
                        obs_full[var_fill][:,:,:] = fill_sesr_refet
                    elif var_fill == 'SESR_pct_pet_dtrnd':
                        fill_sesr_pet = percentile_of_score_without_numba_REALTIME(land_mask, subset_clim, subset_after_clim, percentile_out, fill_sesr_pet, indices_to_fill, indices_to_grab_af_clim)
                        obs_full[var_fill][:,:,:] = fill_sesr_pet


        obs_full.to_netcdf(save_dzsesr_percentile)
                    
    return(xr.open_dataset(save_dzsesr_percentile))

def percentile_of_score_without_numba_REALTIME(land_mask, subset_clim, subset_after_clim, percentile_out_arr, fill_values, indices_to_fill, indices_to_grab_af_clim):
    for Y,_ in enumerate(range(land_mask.shape[0])):
        for X,_ in enumerate(range(land_mask.shape[1])):
            # break
            #Make sure there are no np.nan
            if ~np.isnan(land_mask[Y,X]):
                if np.all(np.isnan(subset_clim[:,Y,X])):
                    pass
                elif np.any(np.isnan(subset_clim[:,Y,X])):
                    #get no nan values
                    no_nan_vals_obs_full = subset_clim[:,Y,X][~np.isnan(subset_clim[:,Y,X])]
                    no_nan_indices_obs_full = np.where(~np.isnan(subset_clim[:,Y,X]))[0]

                    no_nan_vals_obs_small = subset_after_clim[indices_to_grab_af_clim,Y,X][~np.isnan(subset_after_clim[indices_to_grab_af_clim,Y,X])]
                    no_nan_indices_obs_small = np.where(~np.isnan(subset_after_clim[indices_to_grab_af_clim,Y,X]))[0]
                    percentile_out_arr[no_nan_indices_obs_small,Y,X] = pos(subset_clim[no_nan_indices_obs_full,Y,X],subset_after_clim[no_nan_indices_obs_small,Y,X])
                
                else:
                    percentile_out_arr[:,Y,X] = pos(subset_clim[:,Y,X],subset_after_clim[indices_to_grab_af_clim,Y,X])
                    
    fill_values[indices_to_fill,:,:] = percentile_out_arr
    
    return fill_values



def REALTIME_percentile_RZSM_by_day_of_year(window,climatology,RZSM_detrend, rzsm, recompute_RZSM_percentiles):
    '''This means that for each grid cell, each day of the year has a value ranked at 100% somewhere in the timeseries.
    
    The function make_percentile_from_SESR_and_dzSESR_from_all_dates() is the opposite. There is only 1 value per grid cell in the time series which has a ranking of 100%.'''
    
    save_dir = call.realtime_NOAH_dir
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    # Get today's date
    today_date = datetime.today()
    formatted_date = today_date.strftime("%Y-%m-%d")

    save_dzsesr_percentile = f'{save_dir}/RZSM_pct_by_doy_window_size{window}_clim{climatology[0]}-{climatology[1]}.nc'
    
    if recompute_RZSM_percentiles == False:
        return(xr.open_dataset(save_dzsesr_percentile)) 
    else:
        # Check if the file exists
        if os.path.exists(save_dzsesr_percentile):
            # Remove the file
            os.remove(save_dzsesr_percentile)
            print(f"File '{save_dzsesr_percentile}' has been removed.")
            
        RZSM_detrend['RZSM_pct_dtrnd'] = RZSM_detrend['RZSM_dtrnd'].copy(deep=True)

        fill_ = np.empty(RZSM_detrend['RZSM_pct_dtrnd'].shape)
        fill_[:,:,:] = np.nan
        
        time_index_full = pd.Index(RZSM_detrend['time'].values)
        # Convert the time coordinate to a pandas DatetimeIndex
        time_index1_full = pd.DatetimeIndex(RZSM_detrend['time'].values)
        
        # Extract the DOY from the time coordinate
        doy1 = time_index1_full.dayofyear

        vars_to_fill = ['RZSM_pct_dtrnd']
        vars_needed = ['RZSM']

        for var_idx, var_fill in enumerate(vars_to_fill):
            print(f'Finding {var_fill} percentile of score by day of year for each grid cell window_size_{window}_climatology_{climatology}.')

            # break
            for idx, doy in enumerate(range(1, 367)):
                # idx,doy=0,1
                '''get the climatology only to rank percentiles'''
                subset_clim = get_clim_near_window(RZSM_detrend[vars_needed[var_idx]], doy, window,climatology) #Get all values within the specified window from the doy
                subset_after_clim = RZSM_detrend[vars_needed[var_idx]].sel(time=slice(str(climatology[1]+1), str(pd.to_datetime(RZSM_detrend.time.values[-1]).year)))
                
                indices_to_fill_af_clim, indices_to_grab_af_clim, match_clim, match_af_clim = return_indices_to_grab_and_fill_data(subset_clim, subset_after_clim, doy, time_index_full,climatology)

                
                if len(indices_to_grab_af_clim) >0:
                    '''VERY IMPORTANT, This ensures that if there are no doy 366 in the after climatology distribution then it won't break the script.'''
                    # clim_indices_just_in_case = [time_index.get_loc(date) for date in match_af_clim if date.year > climatology[-1]]
                    indices_to_fill = np.array(indices_to_fill_af_clim)
    
                    # Y,X=40,30
                    percentile_out = np.empty(shape = (len(indices_to_fill_af_clim),RZSM_detrend.lat.shape[0],RZSM_detrend.lon.shape[0]))
                    percentile_out[:,:,:] = np.nan
                    percentile_out.shape
                    
                    fill_ = percentile_of_score_without_numba_REALTIME(land_mask, subset_clim, subset_after_clim, percentile_out, fill_, indices_to_fill_af_clim, indices_to_grab_af_clim)
                
                RZSM_detrend[var_fill][:,:,:] = fill_


        RZSM_detrend.to_netcdf(save_dzsesr_percentile)
                    
    return(RZSM_detrend)




def remove_FD_if_longer_than_n_weeks_s4(fd_s2,land_mask, pet_or_refet, num_weeks_FD_to_longterm_drought):
                
    fd_s2[f'fd_4_{pet_or_refet}'] = fd_s2[f'fd_3_{pet_or_refet}'].copy(deep=True)

    '''Check if within the new growing season'''
    for Y,_ in enumerate(range(fd_s2.lat.shape[0])):
        for X,_ in enumerate(range(fd_s2.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                # Y,X=40,30
                '''First get the consecutive ones indicating FD'''
                ones = fdutils_2.consecutive_ones(fd_s2[f'fd_3_{pet_or_refet}'][:,Y,X].values)
                '''Next loop through and find if they are within the growing season'''

                for potential_fd in ones:
                    length_fd = potential_fd[-1] - potential_fd[0]
                    if length_fd >= num_weeks_FD_to_longterm_drought:
                        # print('yes')
                        # break
                        fd_s2[f'fd_4_{pet_or_refet}'][potential_fd[0]:potential_fd[-1],Y,X] = 0
                        
    return fd_s2


def FD_step_3_REALTIME(fd_s2, noah_dir, climatology, window, all_dates_or_doy):

    '''This will actualy create the variable FD_s4 all_dates_or_doy is to remove FD events which are actually part of a long-term drought event'''
    add_text = f'from_{all_dates_or_doy}_percentile'

    num_weeks_FD_to_longterm_drought = call.num_weeks_FD_to_longterm_drought
    save_dir = f'{call.noah_dir}/dzSESR_FD_step_3_REALTIME_{add_text}'
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    date_start, date_end = fdplot.return_date_as_text(fd_s2.time.values[0]), fdplot.return_date_as_text(fd_s2.time.values[-1])
    save_sesr_percentile = f'{save_dir}/dzSESR_fd_step_3_{add_text}_window{window}_clim{climatology[0]}-{climatology[1]}_|_{date_start}_thru_{date_end}.nc'
    
    if os.path.exists(save_sesr_percentile):
        return (xr.open_dataset(save_sesr_percentile))
    else:

        for pet_or_refet in ['pet','refet']:
            print(f'Working on {pet_or_refet} and finding if {all_dates_or_doy} FD transitioned into longterm drought, If longer than {num_weeks_FD_to_longterm_drought} weeks, then we remove the FD event.\nWindow size {window}.\nclimatology_{climatology[0]}-{climatology[1]}.')
            fd_s2 = remove_FD_if_longer_than_n_weeks_s4(fd_s2,land_mask, pet_or_refet, num_weeks_FD_to_longterm_drought)
            # print(np.count_nonzero(add_data_to_fd_index_before))

        fd_s2.to_netcdf(save_sesr_percentile)
    return(fd_s2)

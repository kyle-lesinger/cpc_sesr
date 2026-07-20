#!/usr/bin/env python3

'''Functions for scripts'''

import xarray as xr
import pandas as pd
import numpy as np
import os
import bottleneck as bn
import config.dataUtils as dutils
from scipy.signal import detrend
from scipy import stats
from scipy.stats import linregress
import config.STATIC as call


def detrend_esr_func_by_doy(obs, land_mask, save_sesr):
    # Create an array to hold the detrended data
    obs['ESR_detrend'] = obs['ESR'].copy(deep=True)
    obs['ESR_detrend'][:,:,:] = np.nan

    esr_fill = np.empty_like(obs['ESR_detrend'])
    esr_fill[:,:,:] = np.nan
    
    #For SESR ONLY
    # Loop through each latitude and longitude point
    for Y in range(obs.lat.shape[0]):  # Latitude
        for X in range(obs.lon.shape[0]):  # Longitude
            if np.isnan(land_mask[Y,X]):
                pass
            else:
                ts = obs['ESR'][:, Y, X]  # Extract time series
                ts_arr = ts.values

                index_vals = [idx for idx,i in enumerate(ts_arr) if ~np.isnan(i)]
                
                no_nan_values = ts_arr[index_vals]
                no_nan_values = obs['ESR'][index_vals, Y, X]
                if len(no_nan_values) !=0:
                    # dtrnd = detrend_doy(no_nan_values)
                    #shift back negative values to positive with min/max normalization
                    
                    final_out = no_nan_values.groupby('time.week').apply(detrend_doy)
                    final_dtrnd = (final_out- final_out.min())/(final_out.max()-final_out.min())
                    
                    # detrended_ts = detrend(no_nan_values, axis=0)  # Detrend the time series
                    esr_fill[index_vals, Y, X] = final_dtrnd 


    obs['ESR_detrend'][:, :,:] = esr_fill  # Store the detrended time series

    obs.to_netcdf(save_sesr)
    return(0)


###### For detrending ESR_pet and ESR_refet######
# Define a detrending function
def detrend_doy(x):
    """Remove the linear trend from data."""
    t = np.arange(len(x))
    mask = np.isfinite(x)
    if np.sum(mask) > 1:  # Ensure there are enough points to fit a line
        slope, intercept, r_value, p_value, std_err = stats.linregress(t[mask], x[mask])
        out_val = x - (slope * t + intercept)
        return out_val
    else:
        return x  # Not enough points to detrend, return original



def detrend_ESR_by_doy(recompute=False):

    save_dir = call.noah_dir
    
    year_ranges_tuple_1=call.long_clim
    year_ranges_tuple_2=call.short_clim

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    if recompute==True:
        for year_ranges_tuple in [year_ranges_tuple_1,year_ranges_tuple_2]:
            # year_ranges_tuple = year_ranges_tuple_1
            
            save_esr = f'{save_dir}/ESR_de-trend_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
    
            print('Loading observations.')
            obs_full = xr.open_dataset(f'{call.noah_dir}/esr_rolling_mean_0.50_degrees.nc').load()
            obs_full.close()
            obs = obs_full.sel(time=slice(str(year_ranges_tuple[0]),str(year_ranges_tuple[1]))) #ensures none of the missing values from the previous year
            
            detrend_esr_func(obs, land_mask,save_esr,year_ranges_tuple)
    return(f'Completed {save_esr}.')


# Function to apply min/max standardization
def min_max_standardize(data):
    min_val = np.nanmin(data.values)
    max_val = np.nanmax(data.values)
    standardized_data = (data - min_val) / (max_val - min_val)
    return standardized_data





#Grouping by day of the year and appling min/max standardization
def standardize_by_week_of_year(ds, pet_or_refet):
    ds1 = ds.copy(deep=True)  # Ensure we don't modify the original dataset
    
    # Convert time coordinate to week of the year
    ds1['dayofyear'] = ds1['time.dayofyear']
    
    # Apply min/max standardization grouped by week of the year
    standardized_ds = ds[pet_or_refet].groupby('time.dayofyear').apply(min_max_standardize)
    
    return standardized_ds


def detrend_esr_func(obs, land_mask, save_esr,year_ranges_tuple,min_max_by_week=False):
    # Create an array to hold the detrended data
    obs['ESR_pet_detrend'] = obs['ESR_pet'].copy(deep=True)
    obs['ESR_pet_detrend'][:,:,:] = np.nan
    obs['ESR_refet_detrend'] = obs['ESR_pet_detrend'].copy(deep=True)

    pet_fill = np.empty_like(obs['ESR_refet_detrend'])
    refet_fill = np.empty_like(obs['ESR_refet_detrend'])

    trend_refet_save = obs['ESR_pet_detrend'].sel(time=call.doy_start).to_dataset().rename({'ESR_pet_detrend':'refet_esr_trend'}).copy(deep=True)
    trend_pet_save = obs['ESR_pet_detrend'].sel(time=call.doy_start).to_dataset().rename({'ESR_pet_detrend':'pet_esr_trend'}).copy(deep=True)

    trend_refet_save['refet_esr_trend'][:,:] = 0
    trend_pet_save['pet_esr_trend'][:,:] = 0

    trend_refet_save['min'] = trend_refet_save['refet_esr_trend'].copy(deep=True)
    trend_refet_save['max'] = trend_refet_save['refet_esr_trend'].copy(deep=True)

    trend_pet_save['min'] = trend_refet_save['refet_esr_trend'].copy(deep=True)
    trend_pet_save['max'] = trend_refet_save['refet_esr_trend'].copy(deep=True)
    
    # Loop through each latitude and longitude point
    for pet_or_refet in ['ESR_pet','ESR_refet']:
        print(f'Working on detrending {pet_or_refet}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
        for Y in range(obs.lat.shape[0]):  # Latitude
            for X in range(obs.lon.shape[0]):  # Longitude
                if np.isnan(land_mask[Y,X]):
                    pass
                else:
                    # print(f'Working on Y {Y} and X {X}')
                    ts = obs[pet_or_refet][:, Y, X]  # Extract time series as 1-d array
                    ts_arr = ts.values

                    #get the index values of non np.nan values
                    index_vals = [idx for idx,i in enumerate(ts_arr) if ~np.isnan(i)]
                    
                    no_nan_values = ts[index_vals] #subset only non np.nan values

                    if len(no_nan_values) !=0:
                        '''Manually check the trend of a single instance'''
                        # b4 = no_nan_values[no_nan_values['time.dayofyear'] == 1]
                        # plt.plot(np.arange(len(b4.values)),b4.values)

                        if min_max_by_week == False:
                            ''' min-max normalization applied to every day of the year,not by theday of year'''
                            final_out = no_nan_values.groupby('time.dayofyear').apply(detrend_doy).values
                            final_dtrnd = (final_out- np.nanmin(final_out))/(np.nanmax(final_out)-np.nanmin(final_out))
                            if pet_or_refet == 'ESR_pet':
                                pet_fill[index_vals, Y, X] = final_dtrnd
                            else:
                                refet_fill[index_vals, Y, X] = final_dtrnd
                        else:
                            ''' min-max normalization applied to every day of year only. Consistent with our other analysis.'''
                            final_out = no_nan_values.groupby('time.dayofyear').apply(detrend_doy)
                            final_dtrnd = standardize_by_week_of_year(final_out.to_dataset(), pet_or_refet)
                            
                            if pet_or_refet == 'ESR_pet':
                                pet_fill[index_vals, Y, X] = final_dtrnd.values
                            else:
                                refet_fill[index_vals, Y, X] = final_dtrnd.values
  

                        '''JUst manually add these for plots'''
                        # t = np.arange(len(b4.values))
                        # mask = np.isfinite(b4.values)
    
                        # slope, intercept, r_value, p_value, std_err = stats.linregress(t[mask], tt[mask])
                        # # Calculate the trend line
                        # trend_line = slope * t + intercept
    
                        # # Plot the original data and the trend line
                        # plt.figure(figsize=(10, 6))
                        # plt.plot(t, b4.values, 'o-', label='Original Data')
                        # plt.plot(t, final_out_no_normalization.values, 'o-', label='After detrend')
                        # plt.plot(t, final_dtrnd.values, 'o-', label='After detrend min-max')
                        # plt.plot(t, trend_line, 'r--', label='Trend Line')
                        # plt.xlabel('Time Index')
                        # plt.ylabel('Values')
                        # plt.legend()
                        # plt.savefig('Figures/single_grid_cell_check_detrend.png')
    
                                    
                        # plt.plot(np.arange(len(vals_plt)),vals_plt)
                        
                        
                        # detrended_ts = detrend(no_nan_values, axis=0)  # Detrend the time series

     
    obs['ESR_pet_detrend'][:,:,:] = pet_fill
    obs['ESR_refet_detrend'][:,:,:] = refet_fill

    obs.to_netcdf(save_esr)
    return(0)


def find_trend_by_doy_change_per_year_only(obs,new_obs, land_mask,varname):
    # Create an array to hold the detrended data

    slopes  = np.empty_like(new_obs[list(new_obs.keys())[0]])
    slopes[:,:,:]=np.nan

    print(f'Working on finding the trend change/year for {varname}.')
    for Y in range(obs.lat.shape[0]):  # Latitude
        for X in range(obs.lon.shape[0]):  # Longitude
            if np.isnan(land_mask[Y,X]):
                pass
            else:
                # print(f'Working on Y {Y} and X {X}')
                ts = obs[varname][:, Y, X]  # Extract time series as 1-d array
                ts_arr = ts.values

                #get the index values of non np.nan values
                index_vals = [idx for idx,i in enumerate(ts_arr) if ~np.isnan(i)]
                
                no_nan_values = ts[index_vals] #subset only non np.nan values

                num_years= len(np.unique([i.year for i in pd.to_datetime(no_nan_values.time.values)]))
                if len(no_nan_values) !=0:
                    '''Manually check the trend of a single instance'''
                    # b4 = no_nan_values[no_nan_values['time.dayofyear'] == 1]
                    # plt.plot(np.arange(len(b4.values)),b4.values)

                    '''Before min-max normalization'''
                    group_slopes_yearly_value = no_nan_values.groupby('time.dayofyear').apply(lambda group: xr.DataArray(get_trend_slope(group.values)))

                    slopes[:,Y, X] = group_slopes_yearly_value

    new_obs[varname][:,:,:] = slopes

    return(new_obs)

def find_trend_by_doy_change_per_year_only_return_min_max(obs,new_obs, land_mask,varname):
    # Create an array to hold the detrended data

    slopes  = np.empty_like(new_obs['EVP'])
    slopes[:,:,:]=np.nan

    print(f'Working on finding the trend change/year for {varname}.')
    for Y in range(obs.lat.shape[0]):  # Latitude
        for X in range(obs.lon.shape[0]):  # Longitude
            if np.isnan(land_mask[Y,X]):
                pass
            else:
                # print(f'Working on Y {Y} and X {X}')
                ts = obs[varname][:, Y, X]  # Extract time series as 1-d array
                ts_arr = ts.values

                #get the index values of non np.nan values
                index_vals = [idx for idx,i in enumerate(ts_arr) if ~np.isnan(i)]
                
                no_nan_values = ts[index_vals] #subset only non np.nan values

                num_years= len(np.unique([i.year for i in pd.to_datetime(no_nan_values.time.values)]))
                if len(no_nan_values) !=0:
                    '''Manually check the trend of a single instance'''
                    # b4 = no_nan_values[no_nan_values['time.dayofyear'] == 1]
                    # plt.plot(np.arange(len(b4.values)),b4.values)

                    '''Before min-max normalization'''
                    group_slopes_yearly_value = no_nan_values.groupby('time.dayofyear').apply(lambda group: xr.DataArray(get_trend_slope(group.values)))

                    slopes[:,Y, X] = group_slopes_yearly_value

    new_obs[varname][:,:,:] = slopes

    return(new_obs)

def find_trend_by_doy_change_per_two_decades_only(obs,new_obs_cp, land_mask,varname):
    # Create an array to hold the detrended data

    slopes  = np.empty_like(new_obs_cp['EVP'])
    slopes[:,:,:,:]=np.nan

    decades = [(1981,2000),(1991,2010),(2001,2020),]
    
    
    for iDecade, decade in enumerate(decades):
        print(f'Working on finding the trend change/two decades for period {decade} and variable {varname}.')
        obs_subset= obs[varname].sel(time=slice(str(decade[0]),str(decade[1]))).to_dataset()
        # break
        for Y in range(obs.lat.shape[0]):  # Latitude
            for X in range(obs.lon.shape[0]):  # Longitude
                if np.isnan(land_mask[Y,X]):
                    pass
                else:
                    # print(f'Working on Y {Y} and X {X}')
                    ts = obs_subset[varname][:, Y, X]  # Extract time series as 1-d array
                    ts_arr = ts.values
    
                    #get the index values of non np.nan values
                    index_vals = [idx for idx,i in enumerate(ts_arr) if ~np.isnan(i)]
                    
                    no_nan_values = ts[index_vals] #subset only non np.nan values
    
                    num_years= len(np.unique([i.year for i in pd.to_datetime(no_nan_values.time.values)]))
                    if len(no_nan_values) !=0:
                        '''Manually check the trend of a single instance'''
                        # b4 = no_nan_values[no_nan_values['time.dayofyear'] == 1]
                        # plt.plot(np.arange(len(b4.values)),b4.values)
    
                        '''Before min-max normalization'''
                        group_slopes_yearly_value = no_nan_values.groupby('time.dayofyear').apply(lambda group: xr.DataArray(get_trend_slope(group.values)))
    
                        slopes[iDecade,:,Y, X] = group_slopes_yearly_value

    new_obs_cp[varname][:,:,:,:] = slopes

    return(new_obs_cp)

def find_trend_by_doy_change_per_decade_only(obs,new_obs, land_mask,varname):
    # Create an array to hold the detrended data

    slopes  = np.empty_like(new_obs['EVP'])
    slopes[:,:,:,:]=np.nan

    decades = [(1981,1990),(1991,2000),(2001,2010),(2011,2020)]
    
    
    for iDecade, decade in enumerate(decades):
        print(f'Working on finding the trend change/decade for decade {decade} and variable {varname}.')
        obs_subset= obs[varname].sel(time=slice(str(decade[0]),str(decade[1]))).to_dataset()
        # break
        for Y in range(obs.lat.shape[0]):  # Latitude
            for X in range(obs.lon.shape[0]):  # Longitude
                if np.isnan(land_mask[Y,X]):
                    pass
                else:
                    # print(f'Working on Y {Y} and X {X}')
                    ts = obs_subset[varname][:, Y, X]  # Extract time series as 1-d array
                    ts_arr = ts.values
    
                    #get the index values of non np.nan values
                    index_vals = [idx for idx,i in enumerate(ts_arr) if ~np.isnan(i)]
                    
                    no_nan_values = ts[index_vals] #subset only non np.nan values
    
                    num_years= len(np.unique([i.year for i in pd.to_datetime(no_nan_values.time.values)]))
                    if len(no_nan_values) !=0:
                        '''Manually check the trend of a single instance'''
                        # b4 = no_nan_values[no_nan_values['time.dayofyear'] == 1]
                        # plt.plot(np.arange(len(b4.values)),b4.values)
    
                        '''Before min-max normalization'''
                        group_slopes_yearly_value = no_nan_values.groupby('time.dayofyear').apply(lambda group: xr.DataArray(get_trend_slope(group.values)))
    
                        slopes[iDecade,:,Y, X] = group_slopes_yearly_value

    new_obs[varname][:,:,:,:] = slopes

    return(new_obs)



# Define a function to perform linear regression and return the slope
def get_trend_slope(time_series):
    time_index = np.arange(len(time_series))
    slope, intercept, r_value, p_value, std_err = linregress(time_index, time_series)
    return slope


def make_empty_file(obs):
    new_obs = obs.sel(time=slice('2000-01-01','2000-12-31')).copy(deep=True)
    new_obs = new_obs.apply(lambda x: xr.full_like(x, np.nan)) #convert all variable values to np.nan
    return new_obs

def make_empty_file_full(obs):
    new_obs = obs.copy(deep=True)
    new_obs = new_obs.apply(lambda x: xr.full_like(x, np.nan)) #convert all variable values to np.nan
    return new_obs



def save_netcdf_trends_by_year_and_decade_and_20_years(year_ranges_tuple):

    save_dir = call.doy_trend_dir
    os.makedirs(save_dir, exist_ok=True)
    
    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    print('Loading observations.')
    obs_full = xr.open_dataset(f'{call.noah_dir}/esr_rolling_mean_0.50_degrees.nc').load()
    obs_full.close()
    obs = obs_full.sel(time=slice(str(year_ranges_tuple[0]),str(year_ranges_tuple[1]))) #ensures none of the missing values from the previous year

    '''Plot the change/year'''
    save_esr_per_year = f'{save_dir}/EVP_pet_refet_ESR_trend_change_per_year_{year_ranges_tuple[0]}-{year_ranges_tuple[-1]}.nc'
    if os.path.exists(save_esr_per_year):
        pass
    else:
        new_obs = make_empty_file(obs)

        for varname in list(obs.keys()):
            # break
            new_obs = find_trend_by_doy_change_per_year_only(obs, new_obs, land_mask,varname)
        new_obs.to_netcdf(save_esr_per_year)

    '''Plot the change/decade'''
    save_esr_per_decade = f'{save_dir}/EVP_pet_refet_ESR_trend_change_per_decade_{year_ranges_tuple[0]}-{year_ranges_tuple[-1]}.nc'
    if os.path.exists(save_esr_per_decade):
        pass
    else:
        new_obs = make_empty_file(obs)

        # Add a new dimension 'new_dim' with 4 indexes
        new_dim_size = 4
        new_obs_cp = new_obs.expand_dims(decade=np.arange(new_dim_size)).copy(deep=True)
        
        for varname in list(obs.keys()):
            # break
            new_obs_cp = find_trend_by_doy_change_per_decade_only(obs, new_obs_cp, land_mask,varname)
        new_obs_cp.to_netcdf(save_esr_per_decade)

    '''Plot the change/two decades'''
    save_esr_per_2decade = f'{save_dir}/EVP_pet_refet_ESR_trend_change_per_two_decades_{year_ranges_tuple[0]}-{year_ranges_tuple[-1]}.nc'
    if os.path.exists(save_esr_per_2decade):
        pass
    else:
        new_obs = make_empty_file(obs)

        # Add a new dimension 'new_dim' with 4 indexes
        new_dim_size = 3
        new_obs_cp = new_obs.expand_dims(decade=np.arange(new_dim_size)).copy(deep=True)
        for varname in list(obs.keys()):
            # break
            new_obs_cp = find_trend_by_doy_change_per_two_decades_only(obs, new_obs_cp, land_mask,varname)
        new_obs_cp.to_netcdf(save_esr_per_2decade)
    
    return(f'Completed trends stuff.')



def detrend_RZSM_by_doy():

    save_dir = call.noah_dir
    
    year_ranges_tuple_1=call.long_clim
    year_ranges_tuple_2=call.short_clim

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    obs_full = xr.open_dataset(f'{save_dir}/rzsm_0.50_degrees.nc')['RZSM'].isel(depth_2=0).to_dataset().load()
    obs_full.close()
    
    for year_ranges_tuple in [year_ranges_tuple_1,year_ranges_tuple_2]:
        # year_ranges_tuple = year_ranges_tuple_1
        
        save_ = f'{save_dir}/RZSM_de-trend_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
    
        if os.path.exists(save_):
            print(f'Completed {save_}.')
            pass
        else:
            print('Loading observations.')

            obs = obs_full.sel(time=slice(str(year_ranges_tuple[0]),str(year_ranges_tuple[1]))) #ensures none of the missing values from the previous year
            
            detrend_rzsm_func(obs, land_mask,save_,year_ranges_tuple)
            return(f'Completed {save_}.')


def detrend_rzsm_func(obs, land_mask, save_,year_ranges_tuple,min_max_by_week=False):
    # Create an array to hold the detrended data
    obs['rzsm_detrend'] = obs['RZSM'].copy(deep=True)
    obs['rzsm_detrend'][:,:,:] = np.nan

    fill = np.empty_like(obs['rzsm_detrend'].copy(deep=True))
    
    print(f'Working on detrending RZSM_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
    
    for Y in range(obs.lat.shape[0]):  # Latitude
        for X in range(obs.lon.shape[0]):  # Longitude
            if np.isnan(land_mask[Y,X]):
                pass
            else:
                # print(f'Working on Y {Y} and X {X}')
                ts = obs['RZSM'][:, Y, X]  # Extract time series as 1-d array
                ts_arr = ts.values

                #get the index values of non np.nan values
                index_vals = [idx for idx,i in enumerate(ts_arr) if ~np.isnan(i)]
                
                no_nan_values = ts[index_vals] #subset only non np.nan values

                if len(no_nan_values) !=0:
                    '''Manually check the trend of a single instance'''
                    # b4 = no_nan_values[no_nan_values['time.dayofyear'] == 1]
                    # plt.plot(np.arange(len(b4.values)),b4.values)

                    if min_max_by_week == False:
                        ''' min-max normalization applied to every day of the year,not by theday of year'''
                        final_out = no_nan_values.groupby('time.dayofyear').apply(detrend_doy).values
                        final_dtrnd = (final_out- np.nanmin(final_out))/(np.nanmax(final_out)-np.nanmin(final_out))

                        fill[index_vals, Y, X] = final_dtrnd

                    else:
                        ''' min-max normalization applied to every day of year only. Consistent with our other analysis.'''
                        final_out = no_nan_values.groupby('time.dayofyear').apply(detrend_doy)
                        final_dtrnd = standardize_by_week_of_year(final_out.to_dataset(), pet_or_refet)

                        fill[index_vals, Y, X] = final_dtrnd.values

 
    obs['rzsm_detrend'][:,:,:] = fill


    obs.to_netcdf(save_)
    return(0)

def make_empty_file(obs):
    new_obs = obs.sel(time=slice('2000-01-01','2000-12-31')).copy(deep=True)
    new_obs = new_obs.apply(lambda x: xr.full_like(x, np.nan)) #convert all variable values to np.nan
    return new_obs

def compute_trends_by_year_and_decade_and_20_years(year_ranges_tuple):

    save_dir = call.doy_trend_dir
    os.makedirs(save_dir, exist_ok=True)
        
    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    print('Loading observations.')
    obs_full = xr.open_dataset(f'{call.noah_dir}/esr_rolling_mean_0.50_degrees.nc').load()
    obs_full.close()
    obs = obs_full.sel(time=slice(str(year_ranges_tuple[0]),str(year_ranges_tuple[1]))) #ensures none of the missing values from the previous year

    '''Plot the change/year'''
    save_esr_per_year = f'{save_dir}/EVP_pet_refet_ESR_trend_change_per_year_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
    if os.path.exists(save_esr_per_year):
        pass
    else:
        new_obs = make_empty_file(obs)

        '''Add some data for the min and max to re-scale'''
        for varname in list(obs.keys()):
            # break
            new_obs = find_trend_by_doy_change_per_year_only(obs, new_obs, land_mask,varname)
        new_obs.to_netcdf(save_esr_per_year)

    '''Plot the change/decade'''
    save_esr_per_decade = f'{save_dir}/EVP_pet_refet_ESR_trend_change_per_decade_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
    if os.path.exists(save_esr_per_decade):
        pass
    else:
        new_obs = make_empty_file(obs)

        # Add a new dimension 'new_dim' with 4 indexes
        new_dim_size = 4
        new_obs_cp = new_obs.expand_dims(decade=np.arange(new_dim_size)).copy(deep=True)
        
        for varname in list(obs.keys()):
            # break
            new_obs_cp = find_trend_by_doy_change_per_decade_only(obs, new_obs_cp, land_mask,varname)
        new_obs_cp.to_netcdf(save_esr_per_decade)

    '''Plot the change/two decades'''
    save_esr_per_2decade = f'{save_dir}/EVP_pet_refet_ESR_trend_change_per_two_decades_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
    if os.path.exists(save_esr_per_2decade):
        pass
    else:
        new_obs = make_empty_file(obs)

        # Add a new dimension 'new_dim' with 4 indexes
        new_dim_size = 3
        new_obs_cp = new_obs.expand_dims(decade=np.arange(new_dim_size)).copy(deep=True)
        for varname in list(obs.keys()):
            # break
            new_obs_cp = find_trend_by_doy_change_per_two_decades_only(obs, new_obs_cp, land_mask,varname)
        new_obs_cp.to_netcdf(save_esr_per_2decade)
    
    return(f'Completed trends for EVP/PET/refet/ESR.')


def RZSM_trends_by_year_and_decade_and_20_years(year_ranges_tuple):

    save_dir = call.trend_dir
    os.makedirs(save_dir, exist_ok=True)
        
    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    print('Loading observations.')
    obs_full = xr.open_dataset(f'{call.noah_dir}/RZSM_rolling_mean_0.50_degrees_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').load()
    obs_full.close()
    obs = obs_full.sel(time=slice(str(year_ranges_tuple[0]),str(year_ranges_tuple[1]))) #ensures none of the missing values from the previous year

    '''Plot the change/year'''
    save_esr_per_year = f'{save_dir}/RZSM_trend_change_per_year_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
    if os.path.exists(save_esr_per_year):
        pass
    else:
        new_obs = make_empty_file(obs)

        '''Add some data for the min and max to re-scale'''
        for varname in list(obs.keys()):
            # break
            new_obs = find_trend_by_doy_change_per_year_only(obs, new_obs, land_mask,varname)
            
        new_obs.to_netcdf(save_esr_per_year)

    # '''Plot the change/decade'''
    # save_esr_per_decade = f'{save_dir}/RZSM_trend_change_per_decade_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
    # if os.path.exists(save_esr_per_decade):
    #     pass
    # else:
    #     new_obs = make_empty_file(obs)

    #     # Add a new dimension 'new_dim' with 4 indexes
    #     new_dim_size = 4
    #     new_obs_cp = new_obs.expand_dims(decade=np.arange(new_dim_size)).copy(deep=True)
        
    #     for varname in list(obs.keys()):
    #         # break
    #         new_obs_cp = find_trend_by_doy_change_per_decade_only(obs, new_obs_cp, land_mask,varname)
    #     new_obs_cp.to_netcdf(save_esr_per_decade)

    # '''Plot the change/two decades'''
    # save_esr_per_2decade = f'{save_dir}/RZSM_trend_change_per_two_decades_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
    # if os.path.exists(save_esr_per_2decade):
    #     pass
    # else:
    #     new_obs = make_empty_file(obs)

    #     # Add a new dimension 'new_dim' with 4 indexes
    #     new_dim_size = 3
    #     new_obs_cp = new_obs.expand_dims(decade=np.arange(new_dim_size)).copy(deep=True)
    #     for varname in list(obs.keys()):
    #         # break
    #         new_obs_cp = find_trend_by_doy_change_per_two_decades_only(obs, new_obs_cp, land_mask,varname)
    #     new_obs_cp.to_netcdf(save_esr_per_2decade)
    
    return(f'Completed trends for RZSM.')

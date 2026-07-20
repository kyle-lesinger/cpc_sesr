#!/usr/bin/env python3

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import fastkde
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator
import matplotlib.colors as mcolors
import config.dataUtils as dutils
import calendar
import config.fdUtils_2 as fdutils_2
from geopy.geocoders import Nominatim
from opencage.geocoder import OpenCageGeocode
import config.FDtimeSeriesPlot as fdplot
import config.STATIC as call


'''Use commong functions for all FD classifications'''



def run_FD_time_series_plots_REALTIME(all_dates_or_doy, pet_or_refet, year_ranges_tuple, window):
    fd = xr.open_dataset(f'{call.noah_dir}/dzSESR_FD_step_4_REALTIME_from_by_doy_percentile/dzSESR_fd_step_4_from_by_doy_percentile_window0_clim1981-2020_|_2021-01-06_thru_2024-07-24.nc')
    rzsm = xr.open_dataset(f'{call.noah_dir}/dzSESR_percentile_REALTIME/RZSM_pct_by_doy_window_size0_clim1981-2020.nc')

    rzsm = rzsm.sel(time=fd.time.values)
    usdm = xr.open_dataset(f'{call.USDM_dir}/fd_start_USDM_classification_0.50_degrees.nc')
    # for k,v in enumerate(list(fd.keys())):
    #     print(v)
    select_lat_lon = [(28,46), (11,80), (28,98), (23,66)]
    
    # select_lat_lon = [(31,76)]

    for idPlot,lat_lon in enumerate(select_lat_lon):
        print(f'Plotting grid cell number {idPlot} out of {len(select_lat_lon)}.')
        # lat_lon=select_lat_lon[0]
        Y=lat_lon[0]
        X=lat_lon[1]

        
        Y_degree = fd.lat.values[Y]
        X_degree = fd.lon.values[X]
    
        state_name = fdplot.return_state_name(Y_degree,X_degree)
        print(f'Working on index lat Y {Y} and X {X} for state/region {state_name}.')
        varname=f'fd_5_{pet_or_refet}' #set as the last step of FD that has been run currently.
        num_dates_b4_and_after=4 #Add +/- 3 for the plot specifically so we can see conditions before and after
        ones = fdutils_2.consecutive_ones(fd[varname][:,Y,X].values)
        
    
        REALTIME_all_FD_events_not_in_winter(fd,pet_or_refet,Y,X,ones,state_name,year_ranges_tuple,all_dates_or_doy,rzsm,usdm,num_dates_b4_and_after)
    return 0

def REALTIME_all_FD_events_not_in_winter(fd,pet_or_refet,Y,X,ones,state_name,year_ranges_tuple,all_dates_or_doy,rzsm,usdm,num_dates_b4_and_after):

    for idx,vals in enumerate(ones):
        # idx,vals = 0,ones[0]
        # Y,X=40,30
        vals = (vals[0],vals[-1]-1) #We need to simply subtract 1 to ensure the index is correct
        print(f'Working on index time values {vals}.')
        # if pd.to_datetime(fd.time.values[vals[0]]).year ==2012:
        #     break
        if (pd.to_datetime(fd.time.values[vals[0]]).month in [12,1,2]) or (pd.to_datetime(fd.time.values[vals[-1]]).month in [12,1,2]):
            pass
        else:
            save_with_RZSM(pet_or_refet, all_dates_or_doy, fd, num_dates_b4_and_after, idx, vals,state_name, Y, X, year_ranges_tuple,rzsm)
            save_without_RZSM(pet_or_refet, all_dates_or_doy, fd, num_dates_b4_and_after,idx, vals,state_name, Y, X, year_ranges_tuple)
            save_with_RZSM_and_USDM(pet_or_refet, all_dates_or_doy, fd, num_dates_b4_and_after,idx, vals,state_name, Y, X, year_ranges_tuple,rzsm,usdm)
    return (print('Completed all of the FD events for a particular grid cell'))



def save_with_RZSM(pet_or_refet, all_dates_or_doy, fd, num_dates_b4_and_after,idx, vals,state_name, Y, X, year_ranges_tuple,rzsm):

    save_dir = f'{call.fig_dir}/REALTIME_individual_grid_cell_FD_events_with_RZSM/{state_name}_Y_{Y}_X_{X}'
    os.makedirs(save_dir, exist_ok=True)
    
    save_name= f'{save_dir}/fd_with_RZSM_{pet_or_refet}_{all_dates_or_doy}_{fdplot.return_date_as_text(fd.time.values[vals[0]])}_{fdplot.return_date_as_text(fd.time.values[vals[-1]])}_for_climatology_{year_ranges_tuple[0]}-{year_ranges_tuple[-1]}.png'
    
    dates_before= vals[0]-num_dates_b4_and_after
    dates_after= vals[-1]+num_dates_b4_and_after
    
    time_points = fd.time.values[dates_before:dates_after] 
    delta_sesr_percentile = fd[f'dzSESR_pct_{pet_or_refet}_dtrnd'][dates_before:dates_after,Y,X].values
    sesr_values = fd[f'SESR_{pet_or_refet}_dtrnd'][dates_before:dates_after,Y,X].values
    sesr_percentile = fd[f'SESR_pct_{pet_or_refet}_dtrnd'][dates_before:dates_after,Y,X].values
    rzsm_percentile_values = rzsm['RZSM_pct_dtrnd'][dates_before:dates_after,Y,X].values
    
    rapid_intensification_start_value = [i for i in sesr_percentile if i < 20][0]
    rapid_intensification_start_date = fd.time.values[vals[0]]
    rapid_intensification_idx = [idx for idx,i in enumerate(sesr_percentile) if i < 20][0]
    
    
    rapid_intensification_end_value = fd[f'SESR_pct_{pet_or_refet}_dtrnd'][vals[1],Y,X].values
    rapid_intensification_end_date = fd.time.values[vals[1]]
    rapid_intensification_end_idx = vals[1]
    markers = [vals[0], vals[1]]  # Example markers for P4, P6, etc.
    
    mean_delta_sesr = fd[f'dzSESR_pct_{pet_or_refet}_dtrnd'][vals[0]:vals[-1],Y,X].values
    mean_ = round(np.nanmean(mean_delta_sesr),1)
    
    
    start_index = np.where(time_points == rapid_intensification_start_date)[0][0]
    end_index = np.where(time_points == rapid_intensification_end_date)[0][0]
    
    dt_txt = [pd.to_datetime(fdplot.return_date_as_text(i)) for i in time_points]
    
    plt.figure(figsize=(10, 6))
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(2, 1, figsize=(13, 10))
    
    # Plot SESR values
    ax[0].plot(time_points, sesr_values, 'o-', color='red', label='SESR')
    # ax[0].plot(time_points, rzsm_percentile_values, 'o-', color='blue', label='RZSM')
    
    ax[0].plot([rapid_intensification_start_date, rapid_intensification_end_date], 
             [sesr_values[np.where(time_points == rapid_intensification_start_date)[0][0]], 
              sesr_values[np.where(time_points == rapid_intensification_end_date)[0][0]]], 
             'r--', lw=1)
    
    
    ax[0].axvline(x=rapid_intensification_start_date, color='black', linestyle='--')
    ax[0].axvline(x=rapid_intensification_end_date, color='black', linestyle='--')
    
    # Add text below the vertical lines
    ax[0].text(rapid_intensification_start_date, np.min(sesr_values) + 0.17, 'Start FD', ha='center', va='center_baseline', fontsize=15, rotation=0, color='red')
    ax[0].text(rapid_intensification_end_date, np.min(sesr_values) + 0.17, 'End FD', ha='center', va='top', fontsize=15, rotation=0, color='red')
    
    # Annotations for dzSESR percentile for each week
    for i, txt in enumerate(delta_sesr_percentile):
        # val = f'{round(delta_sesr_percentile[i], 1)}%' #This makes the numbers have a lot of decimals
        val_dzsesr = round(delta_sesr_percentile[i], 1)
        val_rzsm = round(rzsm_percentile_values[i], 1)
    
        if (sesr_values[i] < sesr_values[i-1]) and (i !=0):
            change = -6
        elif i == 0:
            change = i
        else:
            change = 7
    
        if i == 0:
            ax[0].annotate(f'{val_dzsesr}% dzSESR/\n{val_rzsm}% RZSM', (time_points[i], sesr_values[i]),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(3,5+change), fontsize=8)
        else:
            ax[0].annotate(f'{val_dzsesr}% dzSESR', (time_points[i], sesr_values[i]),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(3,5+change), fontsize=8)
    
    ax2 = ax[0].twinx()
    ax2.plot(time_points, rzsm_percentile_values, 'o-', color='Blue', label='RZSM (m3/m3)')
    
    # Annotations for dzSESR percentile for each week
    for i, txt in enumerate(rzsm_percentile_values):
        # val = f'{round(delta_sesr_percentile[i], 1)}%' #This makes the numbers have a lot of decimals
        val_rzsm = round(rzsm_percentile_values[i], 1)
    
        if (val_rzsm < rzsm_percentile_values[i-1]) and (i !=0):
            change = -1
        elif i == 0:
            change = i
        else:
            change = 2
        if i == 0:
            pass
        else:
            ax2.annotate(f'{val_rzsm}% RZSM', (time_points[i], rzsm_percentile_values[i]),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(3,5+change), fontsize=8)
    
    #Annotations for the mean dzSESR percentile change
    middle_date = rapid_intensification_start_date + (rapid_intensification_end_date - rapid_intensification_start_date) / 2
    location_ = (start_index + ((end_index - start_index)//2))+1
    avg_sesr_vals = (sesr_values[start_index] + sesr_values[end_index])/2
    ax[0].annotate(mean_, (time_points[location_], avg_sesr_vals),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(0,3))
    ax[0].annotate('dzSESR mean %', (time_points[location_], avg_sesr_vals),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(17.5,11), fontweight='bold')
    
    # # Labels and title
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('SESR')
    ax[0].set_title(f'Time series schematic illustrating the  criteria used in the flash drought identification methodology.\nTime Range({fdplot.return_date_as_text(fd.time.values[vals[0]])} thru {fdplot.return_date_as_text(fd.time.values[vals[-1]])})')
    # Rotate and change the size of xtick labels
    formatted_dates = [date.strftime('%Y-%m-%d') for date in dt_txt]
    ax[0].set_xticks(formatted_dates)
    ax[0].set_xticklabels(formatted_dates, rotation=45, fontsize=10)
    
    '''Only for the plot of the US to show where the location is'''
    plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed for more or less spacing
    
    # Plotting the map in the second subplot
    ax_map = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax_map.set_extent([-128, -60, 25, 50], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND)
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')
    ax_map.add_feature(cfeature.STATES, linestyle=':')
    
    lon_converted = fd.lon.values[X]-360
    lat_converted = int(float(f'{fd.lat.values[Y]}'))
    # Mark the specific location
    ax_map.plot(lon_converted, lat_converted, 'ro', markersize=10, transform=ccrs.PlateCarree())
    ax_map.text(lon_converted, lat_converted, f'  ({fd.lon.values[X]}E, {fd.lat.values[Y]}N)', horizontalalignment='left', transform=ccrs.PlateCarree())
    
    
    gl = ax_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    plt.savefig(save_name)
    plt.close()
    return('Done')


def save_without_RZSM(pet_or_refet, all_dates_or_doy, fd, num_dates_b4_and_after,idx, vals,state_name, Y, X, year_ranges_tuple):

    save_dir = f'{call.fig_dir}/REALTIME_individual_grid_cell_FD_events_no_RZSM/{state_name}_Y_{Y}_X_{X}'
    os.makedirs(save_dir, exist_ok=True)
    
    save_name= f'{save_dir}/fd_{pet_or_refet}_{all_dates_or_doy}_{fdplot.return_date_as_text(fd.time.values[vals[0]])}_{fdplot.return_date_as_text(fd.time.values[vals[-1]])}_for_climatology_{year_ranges_tuple[0]}-{year_ranges_tuple[-1]}.png'
    
    dates_before= vals[0]-num_dates_b4_and_after
    dates_after= vals[-1]+num_dates_b4_and_after
    
    time_points = fd.time.values[dates_before:dates_after] 
    delta_sesr_percentile = fd[f'dzSESR_pct_{pet_or_refet}_dtrnd'][dates_before:dates_after,Y,X].values
    sesr_values = fd[f'SESR_{pet_or_refet}_dtrnd'][dates_before:dates_after,Y,X].values
    sesr_percentile = fd[f'SESR_pct_{pet_or_refet}_dtrnd'][dates_before:dates_after,Y,X].values
    
    rapid_intensification_start_value = [i for i in sesr_percentile if i < 20][0]
    rapid_intensification_start_date = fd.time.values[vals[0]]
    rapid_intensification_idx = [idx for idx,i in enumerate(sesr_percentile) if i < 20][0]
    
    
    rapid_intensification_end_value = fd[f'SESR_pct_{pet_or_refet}_dtrnd'][vals[1],Y,X].values
    rapid_intensification_end_date = fd.time.values[vals[1]]
    rapid_intensification_end_idx = vals[1]
    markers = [vals[0], vals[1]]  # Example markers for P4, P6, etc.
    
    mean_delta_sesr = fd[f'dzSESR_pct_{pet_or_refet}_dtrnd'][vals[0]:vals[1],Y,X].values
    mean_ = round(np.nanmean(mean_delta_sesr),1)
    
    
    start_index = np.where(time_points == rapid_intensification_start_date)[0][0]
    end_index = np.where(time_points == rapid_intensification_end_date)[0][0]
    
    dt_txt = [pd.to_datetime(fdplot.return_date_as_text(i)) for i in time_points]
    
    plt.figure(figsize=(10, 6))
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(2, 1, figsize=(13, 10))
    
    # Plot SESR values
    ax[0].plot(time_points, sesr_values, 'o-', color='red', label='SESR')
    # ax[0].plot(time_points, rzsm_percentile_values, 'o-', color='blue', label='RZSM')
    
    ax[0].plot([rapid_intensification_start_date, rapid_intensification_end_date], 
             [sesr_values[np.where(time_points == rapid_intensification_start_date)[0][0]], 
              sesr_values[np.where(time_points == rapid_intensification_end_date)[0][0]]], 
             'r--', lw=1)
    
    
    ax[0].axvline(x=rapid_intensification_start_date, color='black', linestyle='--')
    ax[0].axvline(x=rapid_intensification_end_date, color='black', linestyle='--')
    
    # Add text below the vertical lines
    ax[0].text(rapid_intensification_start_date, np.min(sesr_values) + 0.17, 'Start FD', ha='center', va='center_baseline', fontsize=15, rotation=0, color='red')
    ax[0].text(rapid_intensification_end_date, np.min(sesr_values) + 0.17, 'End FD', ha='center', va='top', fontsize=15, rotation=0, color='red')
    
    # Annotations for dzSESR percentile for each week
    for i, txt in enumerate(delta_sesr_percentile):
        # val = f'{round(delta_sesr_percentile[i], 1)}%' #This makes the numbers have a lot of decimals
        val_dzsesr = round(delta_sesr_percentile[i], 1)
        if (sesr_values[i] < sesr_values[i-1]) and (i !=0):
            change = -9
        elif i == 0:
            change = i
        else:
            change = 14
        ax[0].annotate(f'{val_dzsesr}% - dzSESR', (time_points[i], sesr_values[i]),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(3,5))
    
    #Annotations for the mean dzSESR percentile change
    middle_date = rapid_intensification_start_date + (rapid_intensification_end_date - rapid_intensification_start_date) / 2
    location_ = (start_index + ((end_index - start_index)//2))+1
    avg_sesr_vals = (sesr_values[start_index] + sesr_values[end_index])/2
    ax[0].annotate(mean_, (time_points[location_], avg_sesr_vals),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(0,3))
    ax[0].annotate('dzSESR mean %', (time_points[location_], avg_sesr_vals),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(17.5,11), fontweight='bold')
    
    # # Labels and title
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('SESR')
    ax[0].set_title(f'Time series schematic illustrating the  criteria used in the flash drought identification methodology.\nTime Range({fdplot.return_date_as_text(fd.time.values[vals[0]])} thru {fdplot.return_date_as_text(fd.time.values[vals[-1]])})')
    # Rotate and change the size of xtick labels
    formatted_dates = [date.strftime('%Y-%m-%d') for date in dt_txt]
    ax[0].set_xticks(formatted_dates)
    ax[0].set_xticklabels(formatted_dates, rotation=45, fontsize=10)
    
    '''Only for the plot of the US to show where the location is'''
    plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed for more or less spacing
    
    # Plotting the map in the second subplot
    ax_map = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax_map.set_extent([-128, -60, 25, 50], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND)
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')
    ax_map.add_feature(cfeature.STATES, linestyle=':')
    
    lon_converted = fd.lon.values[X]-360
    lat_converted = int(float(f'{fd.lat.values[Y]}'))
    # Mark the specific location
    ax_map.plot(lon_converted, lat_converted, 'ro', markersize=10, transform=ccrs.PlateCarree())
    ax_map.text(lon_converted, lat_converted, f'  ({fd.lon.values[X]}E, {fd.lat.values[Y]}N)', horizontalalignment='left', transform=ccrs.PlateCarree())
    
    
    gl = ax_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    plt.savefig(save_name)
    plt.close()
    return('Done')

def save_with_RZSM_and_USDM(pet_or_refet, all_dates_or_doy, fd, num_dates_b4_and_after,idx, vals,state_name, Y, X, year_ranges_tuple,rzsm,usdm):


    save_dir = f'{call.fig_dir}/REALTIME_individual_grid_cell_FD_events_RZSM_USDM/{state_name}_Y_{Y}_X_{X}'
    os.makedirs(save_dir, exist_ok=True)
    
    save_name= f'{save_dir}/fd_USDM_RZSM_{pet_or_refet}_{all_dates_or_doy}_{fdplot.return_date_as_text(fd.time.values[vals[0]])}_{fdplot.return_date_as_text(fd.time.values[vals[-1]])}_for_climatology_{year_ranges_tuple[0]}-{year_ranges_tuple[-1]}.png'

    dates_before= vals[0]-num_dates_b4_and_after
    dates_after= vals[-1]+num_dates_b4_and_after

    if pd.to_datetime(fd.time.values[dates_before]).year < 2000:
        print('Cannot make this plot for USDM because it is before the year 2000')
        pass
    else:
        '''Get the correct dates since USDM has a different scale'''
        
        
        time_points = fd.time.values[dates_before:dates_after] 

        usdm_subset = usdm['fd_start'].sel(time=slice(time_points[0]-np.timedelta64('7','D'),time_points[-1]))
        
        delta_sesr_percentile = fd[f'dzSESR_pct_{pet_or_refet}_dtrnd'][dates_before:dates_after,Y,X].values
        sesr_values = fd[f'SESR_{pet_or_refet}_dtrnd'][dates_before:dates_after,Y,X].values
        sesr_percentile = fd[f'SESR_pct_{pet_or_refet}_dtrnd'][dates_before:dates_after,Y,X].values
        rzsm_percentile_values = rzsm['RZSM_pct_dtrnd'][dates_before:dates_after,Y,X].values
        usdm_classification = usdm_subset[:,Y,X].values

        start_end_usdm = {idx: i for idx, i in enumerate(usdm_classification) if i ==1}
        indi = list(start_end_usdm.keys())
        
        if len(indi) == 0:
            start_USDM = '2000-01-01'
            USDM_has_NO_VALUES(indi, start_end_usdm, sesr_percentile, vals, usdm_subset, fd, pet_or_refet,Y,X,time_points,sesr_values,delta_sesr_percentile,rzsm_percentile_values,save_name,start_USDM)
        else:
            start_USDM = fdplot.return_date_as_text(usdm_subset.time.values[indi[0]])
            end_USDM = fdplot.return_date_as_text(usdm_subset.time.values[indi[-1]])
            USDM_has_values(indi,start_end_usdm,sesr_percentile,vals,usdm_subset,fd, pet_or_refet,Y,X,time_points,sesr_values,delta_sesr_percentile,rzsm_percentile_values,save_name,usdm_classification,start_USDM)
           

    return('Done')



def USDM_has_NO_VALUES(indi,start_end_usdm,sesr_percentile,vals,usdm_subset,fd, pet_or_refet,Y,X,time_points,sesr_values,delta_sesr_percentile,rzsm_percentile_values,save_name,start_USDM):

    rapid_intensification_start_value = [i for i in sesr_percentile if i < 20][0]
    rapid_intensification_start_date = fd.time.values[vals[0]]
    rapid_intensification_idx = [idx for idx,i in enumerate(sesr_percentile) if i < 20][0]

    rapid_intensification_end_value = fd[f'SESR_pct_{pet_or_refet}_dtrnd'][vals[1],Y,X].values
    rapid_intensification_end_date = fd.time.values[vals[1]]
    rapid_intensification_end_idx = vals[1]
    markers = [vals[0], vals[1]]  # Example markers for P4, P6, etc.
    
    mean_delta_sesr = fd[f'dzSESR_pct_{pet_or_refet}_dtrnd'][vals[0]:vals[1],Y,X].values
    mean_ = round(np.nanmean(mean_delta_sesr),1)
    
    
    start_index = np.where(time_points == rapid_intensification_start_date)[0][0]
    end_index = np.where(time_points == rapid_intensification_end_date)[0][0]
    
    dt_txt = [pd.to_datetime(fdplot.return_date_as_text(i)) for i in time_points]
    
    plt.figure(figsize=(10, 6))
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(2, 1, figsize=(13, 10))
    
    # Plot SESR values
    ax[0].plot(time_points, sesr_values, 'o-', color='red', label='SESR')
    # ax[0].plot(time_points, rzsm_percentile_values, 'o-', color='blue', label='RZSM')
    
    ax[0].plot([rapid_intensification_start_date, rapid_intensification_end_date], 
             [sesr_values[np.where(time_points == rapid_intensification_start_date)[0][0]], 
              sesr_values[np.where(time_points == rapid_intensification_end_date)[0][0]]], 
             'r--', lw=1)
    
    
    ax[0].axvline(x=rapid_intensification_start_date, color='black', linestyle='--')
    ax[0].axvline(x=rapid_intensification_end_date, color='black', linestyle='--')
    
    # Add text below the vertical lines
    ax[0].text(rapid_intensification_start_date, np.min(sesr_values) + 0.17, 'Start FD', ha='center', va='center_baseline', fontsize=15, rotation=0, color='red')
    ax[0].text(rapid_intensification_end_date, np.min(sesr_values) + 0.17, 'End FD', ha='center', va='top', fontsize=15, rotation=0, color='red')
    
    # Annotations for dzSESR percentile for each week
    for i, txt in enumerate(delta_sesr_percentile):
        # val = f'{round(delta_sesr_percentile[i], 1)}%' #This makes the numbers have a lot of decimals
        val_dzsesr = round(delta_sesr_percentile[i], 1)
        val_rzsm = round(rzsm_percentile_values[i], 1)
    
        if (sesr_values[i] < sesr_values[i-1]) and (i !=0):
            change = -6
        elif i == 0:
            change = i
        else:
            change = 7
    
        if i == 0:
            ax[0].annotate(f'{val_dzsesr}% dzSESR/\n{val_rzsm}% RZSM', (time_points[i], sesr_values[i]),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(3,5+change), fontsize=8)
        else:
            ax[0].annotate(f'{val_dzsesr}% dzSESR', (time_points[i], sesr_values[i]),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(3,5+change), fontsize=8)
    
    ax2 = ax[0].twinx()
    ax2.plot(time_points, rzsm_percentile_values, 'o-', color='Blue', label='RZSM (m3/m3)')
    
    # Annotations for dzSESR percentile for each week
    for i, txt in enumerate(rzsm_percentile_values):
        # val = f'{round(delta_sesr_percentile[i], 1)}%' #This makes the numbers have a lot of decimals
        val_rzsm = round(rzsm_percentile_values[i], 1)
    
        if (val_rzsm < rzsm_percentile_values[i-1]) and (i !=0):
            change = -1
        elif i == 0:
            change = i
        else:
            change = 2
        if i == 0:
            pass
        else:
            ax2.annotate(f'{val_rzsm}% RZSM', (time_points[i], rzsm_percentile_values[i]),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(3,5+change), fontsize=8)
                
    #Annotations for the mean dzSESR percentile change
    middle_date = rapid_intensification_start_date + (rapid_intensification_end_date - rapid_intensification_start_date) / 2
    location_ = (start_index + ((end_index - start_index)//2))+1
    avg_sesr_vals = (sesr_values[start_index] + sesr_values[end_index])/2
    ax[0].annotate(mean_, (time_points[location_], avg_sesr_vals),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(0,3))
    ax[0].annotate('dzSESR mean %', (time_points[location_], avg_sesr_vals),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(17.5,11), fontweight='bold')

    # Add a single gold star to the legend
    # ax[0].plot([], [], 'y*', markersize=12, label='USDM Start Date')
    
    # Display only the gold star in the legend
    ax[0].legend(loc='center', fontsize=10, bbox_to_anchor=(0.4, 0.9))
    ax2.legend(loc='center', fontsize=10, bbox_to_anchor=(0.6, 0.9))

    # # Labels and title
    ax[0].set_xlabel(f'USDM DID NOT INDICATE A FLASH DROUGHT OCCURRED!!!',color='red')
    ax[0].set_ylabel('SESR')
    ax[0].set_title(f'Time series schematic illustrating the  criteria used in the flash drought identification methodology.\nTime Range({fdplot.return_date_as_text(fd.time.values[vals[0]])} thru {fdplot.return_date_as_text(fd.time.values[vals[-1]])})')
    # Rotate and change the size of xtick labels
    formatted_dates = [date.strftime('%Y-%m-%d') for date in dt_txt]
    ax[0].set_xticks(formatted_dates)
    ax[0].set_xticklabels(formatted_dates, rotation=45, fontsize=10)
    
    '''Only for the plot of the US to show where the location is'''
    plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed for more or less spacing
    
    # Plotting the map in the second subplot
    ax_map = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax_map.set_extent([-128, -60, 25, 50], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND)
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')
    ax_map.add_feature(cfeature.STATES, linestyle=':')
    
    lon_converted = fd.lon.values[X]-360
    lat_converted = int(float(f'{fd.lat.values[Y]}'))
    # Mark the specific location
    ax_map.plot(lon_converted, lat_converted, 'ro', markersize=10, transform=ccrs.PlateCarree())
    ax_map.text(lon_converted, lat_converted, f'  ({fd.lon.values[X]}E, {fd.lat.values[Y]}N)', horizontalalignment='left', transform=ccrs.PlateCarree())
    
    
    gl = ax_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    plt.savefig(save_name)
    plt.close()

def USDM_has_values(indi,start_end_usdm,sesr_percentile,vals,usdm_subset,fd, pet_or_refet,Y,X,time_points,sesr_values,delta_sesr_percentile,rzsm_percentile_values,save_name,usdm_classification,start_USDM):

    rapid_intensification_start_value = [i for i in sesr_percentile if i < 20][0]
    rapid_intensification_start_date = fd.time.values[vals[0]]
    rapid_intensification_idx = [idx for idx,i in enumerate(sesr_percentile) if i < 20][0]

    ''''difference between the USDM and the dual SESR and RZSM index'''
    days_diff = (rapid_intensification_start_date - usdm_subset.time.values[indi[0]]) / np.timedelta64(1, 'D')
    
    rapid_intensification_end_value = fd[f'SESR_pct_{pet_or_refet}_dtrnd'][vals[1],Y,X].values
    rapid_intensification_end_date = fd.time.values[vals[1]]
    rapid_intensification_end_idx = vals[1]
    markers = [vals[0], vals[1]]  # Example markers for P4, P6, etc.
    
    mean_delta_sesr = fd[f'dzSESR_pct_{pet_or_refet}_dtrnd'][vals[0]:vals[1],Y,X].values
    mean_ = round(np.nanmean(mean_delta_sesr),1)
    
    
    start_index = np.where(time_points == rapid_intensification_start_date)[0][0]
    end_index = np.where(time_points == rapid_intensification_end_date)[0][0]
    
    dt_txt = [pd.to_datetime(fdplot.return_date_as_text(i)) for i in time_points]
    
    plt.figure(figsize=(10, 6))
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(2, 1, figsize=(13, 10))
    
    # Plot SESR values
    ax[0].plot(time_points, sesr_values, 'o-', color='red', label='SESR')
    # ax[0].plot(time_points, rzsm_percentile_values, 'o-', color='blue', label='RZSM')
    
    ax[0].plot([rapid_intensification_start_date, rapid_intensification_end_date], 
             [sesr_values[np.where(time_points == rapid_intensification_start_date)[0][0]], 
              sesr_values[np.where(time_points == rapid_intensification_end_date)[0][0]]], 
             'r--', lw=1)
    
    
    ax[0].axvline(x=rapid_intensification_start_date, color='black', linestyle='--')
    ax[0].axvline(x=rapid_intensification_end_date, color='black', linestyle='--')
    
    # Add text below the vertical lines
    ax[0].text(rapid_intensification_start_date, np.min(sesr_values) + 0.17, 'Start FD', ha='center', va='center_baseline', fontsize=15, rotation=0, color='red')
    ax[0].text(rapid_intensification_end_date, np.min(sesr_values) + 0.17, 'End FD', ha='center', va='top', fontsize=15, rotation=0, color='red')
    
    # Annotations for dzSESR percentile for each week
    for i, txt in enumerate(delta_sesr_percentile):
        # val = f'{round(delta_sesr_percentile[i], 1)}%' #This makes the numbers have a lot of decimals
        val_dzsesr = round(delta_sesr_percentile[i], 1)
        val_rzsm = round(rzsm_percentile_values[i], 1)
    
        if (sesr_values[i] < sesr_values[i-1]) and (i !=0):
            change = -6
        elif i == 0:
            change = i
        else:
            change = 7
    
        if i == 0:
            ax[0].annotate(f'{val_dzsesr}% dzSESR/\n{val_rzsm}% RZSM', (time_points[i], sesr_values[i]),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(3,5+change), fontsize=8)
        else:
            ax[0].annotate(f'{val_dzsesr}% dzSESR', (time_points[i], sesr_values[i]),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(3,5+change), fontsize=8)
    
    ax2 = ax[0].twinx()
    ax2.plot(time_points, rzsm_percentile_values, 'o-', color='Blue', label='RZSM (m3/m3)')
    
    # Annotations for dzSESR percentile for each week
    for i, txt in enumerate(rzsm_percentile_values):
        # val = f'{round(delta_sesr_percentile[i], 1)}%' #This makes the numbers have a lot of decimals
        val_rzsm = round(rzsm_percentile_values[i], 1)
    
        if (val_rzsm < rzsm_percentile_values[i-1]) and (i !=0):
            change = -1
        elif i == 0:
            change = i
        else:
            change = 2
        if i == 0:
            pass
        else:
            ax2.annotate(f'{val_rzsm}% RZSM', (time_points[i], rzsm_percentile_values[i]),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(3,5+change), fontsize=8)
    # Plot USDM classification stars on the correct dates
    num_stars = 0
    for i, (date, usdm_value) in enumerate(zip(time_points, usdm_classification)):
        # break
        if (usdm_value == 1):
            if num_stars ==0:
                ax[0].plot(date, sesr_values[i], 'y*', markersize=25, label='USDM Classification')
                # ax[0].text(date, sesr_values[i] - 0.5, fdplot.return_date_as_text(date), ha='center', va='top', fontsize=8, color='gold')
                num_stars+=1
                
    #Annotations for the mean dzSESR percentile change
    middle_date = rapid_intensification_start_date + (rapid_intensification_end_date - rapid_intensification_start_date) / 2
    location_ = (start_index + ((end_index - start_index)//2))+1
    avg_sesr_vals = (sesr_values[start_index] + sesr_values[end_index])/2
    ax[0].annotate(mean_, (time_points[location_], avg_sesr_vals),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(0,3))
    ax[0].annotate('dzSESR mean %', (time_points[location_], avg_sesr_vals),  ha='center', xycoords = 'data',textcoords='offset points',xytext=(17.5,11), fontweight='bold')

    # Add a single gold star to the legend
    # ax[0].plot([], [], 'y*', markersize=12, label='USDM Start Date')
    
    # Display only the gold star in the legend
    ax[0].legend(loc='center', fontsize=10, bbox_to_anchor=(0.4, 0.9))
    ax2.legend(loc='center', fontsize=10, bbox_to_anchor=(0.6, 0.9))

    # # Labels and title
    ax[0].set_xlabel(f'USDM FD start: {start_USDM} [{days_diff} days difference from SESR & RZSM multi-indicator]')
    ax[0].set_ylabel('SESR')
    ax[0].set_title(f'Time series schematic illustrating the  criteria used in the flash drought identification methodology.\nTime Range({fdplot.return_date_as_text(fd.time.values[vals[0]])} thru {fdplot.return_date_as_text(fd.time.values[vals[-1]])})')
    # Rotate and change the size of xtick labels
    formatted_dates = [date.strftime('%Y-%m-%d') for date in dt_txt]
    ax[0].set_xticks(formatted_dates)
    ax[0].set_xticklabels(formatted_dates, rotation=45, fontsize=10)
    
    '''Only for the plot of the US to show where the location is'''
    plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed for more or less spacing
    
    # Plotting the map in the second subplot
    ax_map = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax_map.set_extent([-128, -60, 25, 50], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND)
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')
    ax_map.add_feature(cfeature.STATES, linestyle=':')
    
    lon_converted = fd.lon.values[X]-360
    lat_converted = int(float(f'{fd.lat.values[Y]}'))
    # Mark the specific location
    ax_map.plot(lon_converted, lat_converted, 'ro', markersize=10, transform=ccrs.PlateCarree())
    ax_map.text(lon_converted, lat_converted, f'  ({fd.lon.values[X]}E, {fd.lat.values[Y]}N)', horizontalalignment='left', transform=ccrs.PlateCarree())
    
    
    gl = ax_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    plt.savefig(save_name)
    plt.close()

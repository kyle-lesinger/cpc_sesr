#!/usr/bin/env python3



import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.stats import percentileofscore as pos
import bottleneck as bn
from numba import njit,prange
import config.dataUtils as dutils
import config.metricUtils as mutils
import config.fdUtils_2 as fdutils_2
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator
import config.STATIC as call

'''Additional steps which may improve the FD identification

1.) If the FD is longer than 
##################################################################################################################################################################
2A) In the criteria, Christian states "The mean change in SESR during the entire lenght of the FD must be less than the 25th percentile... The criterion were taken from the distribution of change in SESR at each grid point for pentads that were encomapassed within the flah dorught event for all years used from the dataset.".

(2A-1) Does this mean we simply take the average of the change in SESR across the span of the drought?
(2A-2) Or does this mean that we take the SESR which was already placed into percentiles, and then we re-create a new distribution looking at only change in SESR during the FD?

Right now, let's just assumed (2A-1) is correct.

############################################################################################################

'''
def remove_winter_events(ds_full,land_mask):
    '''We are removing any event if it started in winter, or it ended in winter. This was a subjective choice'''
    all_time = pd.to_datetime(ds_full.time.values)
    for Y,_ in enumerate(range(ds_full.lat.shape[0])):
        for X,_ in enumerate(range(ds_full.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                fd_events = fdutils_2.consecutive_ones(ds_full[:,Y,X])

                def find_if_start_or_end_are_within_winter(fd_events,all_time,ds_full):
                    for fd_event in fd_events:
                        if (all_time[fd_event[0]].month in [12,1,2]) or (all_time[fd_event[-1]].month in [12,1,2]):
                            # print('Within the winter')
                            ds_full[fd_event[0]:fd_event[-1],Y,X] = 0
                    return ds_full
                ds_full = find_if_start_or_end_are_within_winter(fd_events,all_time,ds_full)
    return ds_full
    
def percent_of_years_with_a_FD(fd_stats,fd_final, data, pet_or_refet, s1_or_s2_or_s3):
    fd_yearly_sum = fd_final.resample(time='YE').sum()
    fd_yearly_sum= xr.where(fd_yearly_sum >=1,1,0)
    vals_tmp = fd_yearly_sum.mean(dim='time').values
    vals_tmp = np.where(vals_tmp==0,np.nan,vals_tmp)
    fd_stats['fd_all_pct_yrs'][:,:] = vals_tmp
    return fd_stats

def mean_median_max_FD_length(fd_stats,fd_final, data, pet_or_refet, s1_or_s2_or_s3,land_mask):
    fd_stats['mean_length'] = fd_stats['fd_all_pct_yrs'].copy(deep=True)
    fd_stats['mean_length'][:,:] = np.nan
    fd_stats['median_length'] = fd_stats['mean_length'].copy(deep=True)
    fd_stats['max_length'] = fd_stats['mean_length'].copy(deep=True)
    
    for Y,_ in enumerate(range(fd_final.lat.shape[0])):
        for X,_ in enumerate(range(fd_final.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                lengths = []
                fd_events = fdutils_2.consecutive_ones(fd_final[:,Y,X])
                if len(fd_events) != 0:
                    for fd_event in fd_events:
                        lengths.append(int(fd_event[-1] - fd_event[0]))
                    fd_stats['mean_length'][Y,X] = round(np.nanmean(lengths),1)
                    fd_stats['median_length'][Y,X] = round(np.nanmedian(lengths),1)
                    fd_stats['max_length'][Y,X] = round(np.nanmax(lengths),1)
    return fd_stats


def percent_of_years_with_a_FD_intensification_cats(fd_stats,tmp):
    fd_yearly_sum = tmp.resample(time='YE').sum()
    fd_yearly_sum= xr.where(fd_yearly_sum >=1,1,0)
    
    for cat in np.arange(1,5):
        vals_tmp = fd_yearly_sum[f'fd{cat}'].mean(dim='time').values
        vals_tmp = np.where(vals_tmp==0,np.nan,vals_tmp)
    
        fd_stats[f'FD{cat}_pct_yrs'][:,:] = vals_tmp
    return fd_stats


def FD_intensification_categories(fd_stats,fd_final, data, pet_or_refet, s1_or_s2_or_s3,land_mask):

    fd_stats['FD1_pct_yrs'] = fd_stats['fd_all_pct_yrs'].copy(deep=True)
    fd_stats['FD1_pct_yrs'][:,:] = 0
    fd_stats['FD2_pct_yrs'] = fd_stats['FD1_pct_yrs'].copy(deep=True)
    fd_stats['FD3_pct_yrs'] = fd_stats['FD1_pct_yrs'].copy(deep=True)
    fd_stats['FD4_pct_yrs'] = fd_stats['FD1_pct_yrs'].copy(deep=True)

    tmp = data.copy(deep=True)
    tmp[f'fd_{pet_or_refet}_{s1_or_s2_or_s3}'][:,:,:] = 0
    tmp['fd1'] = tmp[f'fd_{pet_or_refet}_{s1_or_s2_or_s3}'].copy(deep=True)
    tmp['fd2'] = tmp['fd1'].copy(deep=True)
    tmp['fd3'] = tmp['fd1'].copy(deep=True)
    tmp['fd4'] = tmp['fd1'].copy(deep=True)
    
    for Y,_ in enumerate(range(fd_final.lat.shape[0])):
        for X,_ in enumerate(range(fd_final.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                fd_events = fdutils_2.consecutive_ones(fd_final[:,Y,X])
                if len(fd_events) != 0:
                    for fd_event in fd_events:
                        dzSESR_mean_pct = np.nanmean(data[f'mean_dzsesr_{pet_or_refet}_pct_change'][fd_event[0]: fd_event[1],Y,X].values)
                        if (dzSESR_mean_pct >20) and (dzSESR_mean_pct <=25):
                            tmp['fd1'][fd_event[0]: fd_event[1],Y,X] = 1
                        elif (dzSESR_mean_pct >15) and (dzSESR_mean_pct <=20):
                            tmp['fd2'][fd_event[0]: fd_event[1],Y,X] = 1
                        elif (dzSESR_mean_pct >10) and (dzSESR_mean_pct <=15):
                            tmp['fd3'][fd_event[0]: fd_event[1],Y,X] = 1
                        elif (dzSESR_mean_pct <=10):
                            tmp['fd4'][fd_event[0]: fd_event[1],Y,X] = 1

    fd_stats = percent_of_years_with_a_FD_intensification_cats(fd_stats,tmp, )

    return fd_stats

    
def min_max_percent_of_years(fd_stats):
    pct = [i for i in fd_stats.keys() if 'pct' in i]
    min_, max_ = [],[]
    for i in pct:
        min_.append(np.nanmin(fd_stats[i].values))
        max_.append(np.nanmax(fd_stats[i].values))
        
    return(np.nanmin(min_), np.nanmax(max_)) 

def min_max_for_median_mean_and_max(fd_stats):
    pct = [i for i in fd_stats.keys() if 'pct' not in i]
    pct = [i for i in fd_stats.keys() if 'max' not in i]
    min_, max_ = [],[]
    for i in pct:
        min_.append(np.nanmin(fd_stats[i].values))
        max_.append(np.nanmax(fd_stats[i].values))
        
    return(np.nanmin(min_), np.nanmax(max_)) 

def plot_fd_stats(fd_stats,all_dates_or_only_doy_percentile,window, year_ranges_tuple,save_fig):
    pct_min, pct_max = min_max_percent_of_years(fd_stats)
    v_pct = np.linspace(np.nanmin(pct_min), np.nanmax(pct_max), 6, endpoint=True)
    
    len_min, len_max = min_max_for_median_mean_and_max(fd_stats)
    v_len = np.linspace(np.nanmin(len_min), np.nanmax(len_max), 6, endpoint=True)

    max_min, max_max = np.nanmin(fd_stats['max_length'].values), np.nanmax(fd_stats['max_length'].values)
    v_max = np.linspace(np.nanmin(max_min), np.nanmax(max_max), 6, endpoint=True)    
    
    cmap = plt.get_cmap('YlOrRd')
    fig, axs = plt.subplots(
        nrows = 2, ncols= 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 7),dpi=300)

    lon = fd_stats.lon.values
    lat = fd_stats.lat.values
    axs = axs.flatten()

    axs_start = 0
    for key in list(fd_stats.keys()):
        data = fd_stats[key].values
        map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
          llcrnrlon=-128, urcrnrlon=-60, resolution='l')
        x, y = map(*np.meshgrid(lon, lat))

        if 'pct' in key:
            im = axs[axs_start].contourf(x, y, data, levels=v_pct, extend='both',
                      transform=ccrs.PlateCarree(), cmap=cmap)
            out_leg='% of years w/ FD'
        elif 'length' in key:
            out_leg=f'FD {key} (weeks)'
            if key == 'max_length':
                im = axs[axs_start].contourf(x, y, data, levels=v_max, extend='both',
                     transform=ccrs.PlateCarree(), cmap=cmap)
            else:
                im = axs[axs_start].contourf(x, y, data, levels=v_len, extend='both',
                          transform=ccrs.PlateCarree(), cmap=cmap)    

        gl = axs[axs_start].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                   linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = True
        gl.bottom_labels = True
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        axs[axs_start].coastlines()
        axs[axs_start].set_aspect('equal')  # this makes the plots better
        axs[axs_start].set_title(f'{key}: {year_ranges_tuple[0]}-{year_ranges_tuple[1]}',fontsize=12)
        cbar = fig.colorbar(im, ax = axs[axs_start], orientation='horizontal')
        
        cbar.set_label(out_leg, fontsize=10, labelpad=5)
        axs_start+=1

    out_name_title = f'{all_dates_or_only_doy_percentile}_percentile_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}'
    plt.suptitle(out_name_title,fontsize=15)
    plt.tight_layout()
    plt.savefig(save_fig)


def plot_percent_of_years_with_FD(window,year_ranges_tuple_1,year_ranges_tuple_2,all_dates_or_only_doy_percentile,s1_or_s2_or_s3,pet_or_refet):

    stats_dir = f'{call.noah_dir}/FD_statistics'
    os.makedirs(stats_dir, exist_ok=True)
    # def FD_step_2(window):
    save_dir = f'{call.fig_dir}/FD_percent_of_years_all_FD_steps'
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    
    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
    
        # year_ranges_tuple = year_ranges_tuple_1
        
        save_fig = f'{save_dir}/FD_{s1_or_s2_or_s3}_{pet_or_refet}_percent_of_years_{all_dates_or_only_doy_percentile}_percentile_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.png'
        save_netcdf = f'{stats_dir}/FD_{s1_or_s2_or_s3}_{pet_or_refet}_percent_of_years_{all_dates_or_only_doy_percentile}_percentile_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'

        if os.path.exists(save_netcdf):
            fd_stats = xr.open_dataset(save_netcdf)
            plot_fd_stats(fd_stats,all_dates_or_only_doy_percentile,window, year_ranges_tuple,save_fig) 
        else:
            #Load data
            print(f'Making percent of years with FD plots for window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')
    
            data = xr.open_dataset(f'{call.noah_dir}/dzSESR_FD_step_4_from_{all_dates_or_only_doy_percentile}_percentile/dzSESR_fd_step_4_from_{all_dates_or_only_doy_percentile}_percentile_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
    
            '''Final output will have 8 different statistics. 
            1.) Percent of years with a FD for only events that occurred during the growing season. If they started or ended in winter
            then they are discarded and not counted
            2-4) Mean, Median, and Max length of FD events
            4-8) FD intensification categories FD1-FD4 (see below)
    
            Mean change in SESR flash drought intensity categories
            FD1 Moderate flash drought 20th–25th percentile
            FD2 Severe flash drought 15th–20th percentile
            FD3 Extreme flash drought 10th–15th percentile
            FD4 Exceptional flash drought ,10th percentile
    
            file:///cpc/home/klesinger/Downloads/noaa_22122_DS1.pdf --- Souce for categories
            '''
            ds_full = data[f'fd_{pet_or_refet}_{s1_or_s2_or_s3}'].copy(deep=True)
            fd_final = remove_winter_events(ds_full,land_mask)
    
            #These should be different values below (sanity check)
            # np.count_nonzero(data[f'fd_{pet_or_refet}_{s1_or_s2_or_s3}']==1) 
            # np.count_nonzero(ds_full.values==1)
    
            '''Now calculate the different statistics'''
            fd_stats = ds_full.to_dataset().copy(deep=True).rename({f'fd_{pet_or_refet}_{s1_or_s2_or_s3}':'fd_all_pct_yrs'}).isel(time=0)
            fd_stats = percent_of_years_with_a_FD(fd_stats,fd_final, data, pet_or_refet, s1_or_s2_or_s3)
            fd_stats = mean_median_max_FD_length(fd_stats,fd_final, data, pet_or_refet, s1_or_s2_or_s3,land_mask)
            fd_stats = FD_intensification_categories(fd_stats,fd_final, data, pet_or_refet, s1_or_s2_or_s3,land_mask)
            fd_stats.to_netcdf(save_netcdf)
            plot_fd_stats(fd_stats,all_dates_or_only_doy_percentile,window, year_ranges_tuple,save_fig)
    
    return('Completed percent of years with FD plots')

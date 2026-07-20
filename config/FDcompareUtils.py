#!/usr/bin/env python3

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import fastkde
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator
import matplotlib.colors as mcolors
import config.dataUtils as dutils
import calendar
import pickle
import config.STATIC as call




def compute_FD_similarity_statistics(window, year_ranges_tuple_1, year_ranges_tuple_2,all_dates_or_only_doy_percentile, ):

    add_text = f'from_{all_dates_or_only_doy_percentile}_percentile'
    
    save_dir = f'{call.noah_dir}/FD_statistics_comparison_{add_text}'
    os.makedirs(save_dir, exist_ok=True)
    
    save_figdir = f'{call.fig_dir}/FD_statistics_comparison_{add_text}'
    os.makedirs(save_figdir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
        # year_ranges_tuple=year_ranges_tuple_1
        # window=0

        rzsm = xr.open_dataset(f'{call.noah_dir}/climatology_RZSM_percentile/RZSM_percentile_{all_dates_or_only_doy_percentile}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').load()
            
        save_file = f'{save_dir}/FD_all_dates_comparison_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.pkl'
        save_file_season = f'{save_dir}/FD_seasonal_comparison_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.pkl'
        
        if (os.path.exists(save_file) and os.path.exists(save_file_season)):
            # Open the file in binary read mode and load the object
            with open(save_file, 'rb') as file:
                out_dictionary_all_time = pickle.load(file)
            with open(save_file_season, 'rb') as file:
                out_dictionary_seasons = pickle.load(file)
        else:
            out_dictionary_all_time, out_dictionary_seasons = create_dictionary_of_statistics(save_file,save_file_season, add_text, window,year_ranges_tuple,land_mask, rzsm,)
        
        '''Now plot the final results'''
        plot_fd_comparison_stats_all_time(add_text, window,year_ranges_tuple,land_mask, rzsm,out_dictionary_all_time,save_figdir,all_dates_or_only_doy_percentile)
        plot_fd_comparison_stats_by_season(add_text, window,year_ranges_tuple,land_mask, rzsm,out_dictionary_seasons,save_figdir,all_dates_or_only_doy_percentile)
        
        plot_false_positive_percentage_RZSM_seasons(add_text, window,year_ranges_tuple,land_mask, rzsm,out_dictionary_seasons,save_figdir,all_dates_or_only_doy_percentile)
        plot_false_positive_percentage_RZSM_all_time(add_text, window,year_ranges_tuple,land_mask, rzsm,out_dictionary_all_time,save_figdir,all_dates_or_only_doy_percentile)
    return('Completed FD pickle statistics creation.')


def plot_false_positive_percentage_RZSM_all_time(add_text, window,year_ranges_tuple,land_mask, rzsm,out_dictionary_all_time,save_figdir,all_dates_or_only_doy_percentile):
    
    keys_ = [i for i in out_dictionary_all_time.keys() if 'RZSM_comparison' in i][0]
    keys_dict = {'s5_RZSM_comparison':'Criteria: a.) If step s4 == 1 (FD) and step s5 ==0 (no FD)'}
    out_name_title = f'seasonal_rzsm_comparison_{all_dates_or_only_doy_percentile}_percentile_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.png'

    v_pct = np.linspace(0, 1, 9, endpoint=True)
    
    cmap = plt.get_cmap('YlOrBr')
    fig, axs = plt.subplots(
        nrows = 1, ncols= 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(7, 3),dpi=300)
    
    lon = rzsm.lon.values
    lat = rzsm.lat.values
    axs = axs.flatten()
    
    axs_start = 0

    for iSource,pet_or_refet in enumerate(['pet','refet']):
        # break
        data = 1 - out_dictionary_all_time[keys_][f'pct_{pet_or_refet}_correct'].values
        map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
          llcrnrlon=-128, urcrnrlon=-60, resolution='l')
        x, y = map(*np.meshgrid(lon, lat))
    

        im = axs[axs_start].contourf(x, y, data, levels=v_pct, extend='both',
                  transform=ccrs.PlateCarree(), cmap=cmap)
    
        # gl = axs[axs_start].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
        #                            linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
        # gl.top_labels = False
        # gl.right_labels = False
        # gl.left_labels = True
        # gl.bottom_labels = True
        # gl.xformatter = LongitudeFormatter()
        # gl.yformatter = LatitudeFormatter()
        axs[axs_start].coastlines()
        axs[axs_start].set_aspect('equal')  # this makes the plots better
        axs[axs_start].set_title(f'{pet_or_refet}',fontsize=12)
    
        axs_start+=1
    
    cbar_ax = fig.add_axes([0.10, 0.025, .83, .02]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label(f'Percentage of FD overestimation by SESR conditional on RZSM percentile', labelpad=7)
    plt.suptitle(f'{out_name_title}',fontsize=10)

    out = f'{save_figdir}/{out_name_title}'
    plt.savefig(out)

def plot_false_positive_percentage_RZSM_seasons(add_text, window,year_ranges_tuple,land_mask, rzsm,out_dictionary_seasons,save_figdir,all_dates_or_only_doy_percentile):
    '''We previously found the number of FD events in which both SESR said FD (step s4) and in which RZSM was < 30th percentile (step s5)
    Now we check how many events (proportionally) were considered FD positive in s4 and then were now considered FD negative in s5.
    Since I already created the percentage of correct, we just do 1-pct_correct to get the percent false. This is a very simple statistic.'''
    

    keys_ = [i for i in out_dictionary_seasons.keys() if 'RZSM_comparison' in i][0]
    keys_dict = {'s5_RZSM_comparison':'Criteria: a.) If step s4 == 1 (FD) and step s5 ==0 (no FD)'}
    out_name_title = f'seasonal_rzsm_comparison_{all_dates_or_only_doy_percentile}_percentile_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.png'

    v_pct = np.linspace(0, 1, 9, endpoint=True)
    
    cmap = plt.get_cmap('YlOrBr')
    fig, axs = plt.subplots(
        nrows = 2, ncols= 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(7, 3),dpi=300)
    
    lon = rzsm.lon.values
    lat = rzsm.lat.values
    axs = axs.flatten()
    
    axs_start = 0

    for iSource,pet_or_refet in enumerate(['pet','refet']):
        # break
        for iSeason,season in enumerate(out_dictionary_seasons[keys_][f'pct_{pet_or_refet}_correct'].season.values):
            # break
            data = 1 - out_dictionary_seasons[keys_][f'pct_{pet_or_refet}_correct'].isel(season=iSeason).values
            map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
              llcrnrlon=-128, urcrnrlon=-60, resolution='l')
            x, y = map(*np.meshgrid(lon, lat))
        
    
            im = axs[axs_start].contourf(x, y, data, levels=v_pct, extend='both',
                      transform=ccrs.PlateCarree(), cmap=cmap)
        
            # gl = axs[axs_start].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
            #                            linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
            # gl.top_labels = False
            # gl.right_labels = False
            # gl.left_labels = True
            # gl.bottom_labels = True
            # gl.xformatter = LongitudeFormatter()
            # gl.yformatter = LatitudeFormatter()
            axs[axs_start].coastlines()
            axs[axs_start].set_aspect('equal')  # this makes the plots better
            axs[axs_start].set_title(f'{pet_or_refet} - {season}',fontsize=7)
        
            axs_start+=1

                            # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.10, 0.15, .83, .02]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label(f'Percentage of FD overestimation by SESR conditional on RZSM percentile', labelpad=7)
    plt.suptitle(f'{out_name_title}',fontsize=10)
    # plt.tight_layout()

    out = f'{save_figdir}/{out_name_title}'
    plt.savefig(out)



def min_max_percent_of_years_all_time(keys_,dictionary):

    min_, max_ = [],[]
    for i in keys_:
        # break
        min_.append(np.nanmin(dictionary[i]['pct_correct'].values))
        max_.append(np.nanmax(dictionary[i]['pct_correct'].values))
        
    return(np.nanmin(min_), np.nanmax(max_)) 

def min_max_percent_of_years_by_season(keys_,out_dictionary_seasons):

    min_, max_ = [],[]
    for i in keys_:
        # break
        min_ = out_dictionary_seasons[i]['pct_correct'].min(dim=['lon','lat'])
        max = out_dictionary_seasons[i]['pct_correct'].max(dim=['lon','lat'])
        
    return(np.nanmin(min_), np.nanmax(max_)) 

def plot_fd_comparison_stats_by_season(add_text, window,year_ranges_tuple,land_mask, rzsm,out_dictionary_seasons,save_figdir,all_dates_or_only_doy_percentile):
    keys_ = [i for i in out_dictionary_seasons.keys() if 'pet_refet_FD_comparison' in i]
    keys_dict = {'s1_pet_refet_FD_comparison':'Criteria: a.)length >=4 weeks,b.)SESR <=20th percentile,c.)dzSESR <= 40th percentile,\nd.) Can be 1 week of deviation of dzSESR',
                's2_pet_refet_FD_comparison':'Criteria: a.) find if flash drought is between planting andharvest for individual grid cell\n(based on 11 different crops)',
                's3_pet_refet_FD_comparison':'Criteria: a.) mean dzSESR percentile <=25th percentile',
                's4_pet_refet_FD_comparison':'Criteria: a.) if the flash drought does not transition into a longterm drought (<=6 months))',
                's5_pet_refet_FD_comparison':'Criteria: a.) mean RZSM percentile <=30th percentile'}
    out_name_title = f'seasonal_et_refet_{all_dates_or_only_doy_percentile}_percentile_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.png'

    v_pct = np.linspace(0, 1, 9, endpoint=True)
    
    cmap = plt.get_cmap('YlOrBr')
    fig, axs = plt.subplots(
        nrows = 5, ncols= 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8),dpi=300)
    
    lon = rzsm.lon.values
    lat = rzsm.lat.values
    axs = axs.flatten()
    
    axs_start = 0

    for iKey,key in enumerate(keys_):
        for iSeason,season in enumerate(out_dictionary_seasons[key]['pct_correct'].season.values):
            
            data = out_dictionary_seasons[key]['pct_correct'].isel(season=iSeason).values
            map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
              llcrnrlon=-128, urcrnrlon=-60, resolution='l')
            x, y = map(*np.meshgrid(lon, lat))
        
    
            im = axs[axs_start].contourf(x, y, data, levels=v_pct, extend='both',
                      transform=ccrs.PlateCarree(), cmap=cmap)
        
            # gl = axs[axs_start].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
            #                            linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
            # gl.top_labels = False
            # gl.right_labels = False
            # gl.left_labels = True
            # gl.bottom_labels = True
            # gl.xformatter = LongitudeFormatter()
            # gl.yformatter = LatitudeFormatter()
            axs[axs_start].coastlines()
            axs[axs_start].set_aspect('equal')  # this makes the plots better
            axs[axs_start].set_title(f'{season} - {key}',fontsize=7)
        
            axs_start+=1
        
    cbar_ax = fig.add_axes([0.10, 0.08, .83, .02]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label(f'Percent Agreement', labelpad=7)
    plt.suptitle(f'{out_name_title}',fontsize=10)

    out = f'{save_figdir}/{out_name_title}'
    plt.savefig(out)

def plot_fd_comparison_stats_all_time(add_text, window,year_ranges_tuple,land_mask, rzsm,out_dictionary_all_time,save_figdir,all_dates_or_only_doy_percentile):
    keys_ = [i for i in out_dictionary_all_time.keys() if 'pet_refet_FD_comparison' in i]
    keys_dict = {'s1_pet_refet_FD_comparison':'Criteria: a.)length >=4 weeks,b.)SESR <=20th percentile,c.)dzSESR <= 40th percentile,\nd.) Can be 1 week of deviation of dzSESR',
                's2_pet_refet_FD_comparison':'Criteria: a.) find if flash drought is between planting andharvest for individual grid cell\n(based on 11 different crops)',
                's3_pet_refet_FD_comparison':'Criteria: a.) mean dzSESR percentile <=25th percentile',
                's4_pet_refet_FD_comparison':'Criteria: a.) if the flash drought does not transition into a longterm drought (<=6 months))',
                's5_pet_refet_FD_comparison':'Criteria: a.) mean RZSM percentile <=30th percentile'}

    out_name_title = f'et_refet_{all_dates_or_only_doy_percentile}_percentile_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.png'

    pct_min, pct_max = min_max_percent_of_years_all_time(keys_,out_dictionary_all_time)
    v_pct = np.linspace(np.nanmin(pct_min), np.nanmax(pct_max), 9, endpoint=True)
    
    cmap = plt.get_cmap('YlOrBr')
    fig, axs = plt.subplots(
        nrows = 5, ncols= 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(7, 10),dpi=300)
    
    lon = rzsm.lon.values
    lat = rzsm.lat.values
    axs = axs.flatten()
    
    axs_start = 0
    for iKey,key in enumerate(keys_):
        data = out_dictionary_all_time[key]['pct_correct'].values
        map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
          llcrnrlon=-128, urcrnrlon=-60, resolution='l')
        x, y = map(*np.meshgrid(lon, lat))
    
    
        im = axs[axs_start].contourf(x, y, data, levels=v_pct, extend='both',
                  transform=ccrs.PlateCarree(), cmap=cmap)
    
        # gl = axs[axs_start].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
        #                            linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
        # gl.top_labels = False
        # gl.right_labels = False
        # gl.left_labels = True
        # gl.bottom_labels = True
        # gl.xformatter = LongitudeFormatter()
        # gl.yformatter = LatitudeFormatter()
        axs[axs_start].coastlines()
        axs[axs_start].set_aspect('equal')  # this makes the plots better
        axs[axs_start].set_title(f'{keys_dict.get(key)}',fontsize=7)
    
        axs_start+=1
        
    cbar_ax = fig.add_axes([0.10, 0.08, .83, .02]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label(f'Percent Agreement', labelpad=7)
    plt.suptitle(f'{out_name_title}',fontsize=15)

    out = f'{save_figdir}/{out_name_title}'
    plt.savefig(out)

def create_dictionary_of_statistics(save_file,save_file_season, add_text, window,year_ranges_tuple,land_mask, rzsm,):
    #Load data
    print(f'Making comparison for {save_file}')
    open_dzSESR_percentile = f'{call.noah_dir}/dzSESR_FD_step_4_{add_text}/dzSESR_fd_step_4_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
    dsesr = xr.open_dataset(open_dzSESR_percentile).load()

    # list(dsesr.keys())
    out_dictionary_all_time = {}
    out_dictionary_seasons = {}
    #Compare pet and refet for each FD step to see if they have similar weeks under FD
    for iStep, step in enumerate(['s1','s2','s3','s4','s5']):
        # iStep, step = 0,'s1'
        out_dictionary_all_time = out_dictionary_all_time = find_number_of_weeks_in_FD_the_same(dsesr,land_mask, rzsm, out_dictionary_all_time, step)
        out_dictionary_seasons= find_number_of_weeks_in_FD_the_same_by_season(dsesr,land_mask, rzsm, out_dictionary_seasons, step)
        if step=='s5':
            out_dictionary_all_time = RZSM_comparison(dsesr,land_mask, rzsm, out_dictionary_all_time, step)
            out_dictionary_seasons = RZSM_comparison_by_season(dsesr,land_mask, rzsm, out_dictionary_seasons, step)

    with open(save_file, 'wb') as file:
        pickle.dump(out_dictionary_all_time, file)
    with open(save_file_season, 'wb') as file:
        pickle.dump(out_dictionary_seasons, file)
        
    return out_dictionary_all_time, out_dictionary_seasons


def RZSM_comparison_by_season(dsesr,land_mask, rzsm, out_dictionary, step):
    '''Need to only compare steps s4 and s5'''
    cp = dsesr.copy(deep=True)
    '''Returns a 1 for fd_pet_s5 if both fd_pet_s5==1 and fd_refet_s5==1'''
    s5_pet_true = cp[f'fd_pet_s5'].where((cp[f'fd_pet_s5'] == 1) & (cp[f'fd_pet_s4']==1)).to_dataset(name='pet_correct')
    s5_pet_true = s5_pet_true.groupby('time.season').sum()
    '''Returns a 1 for fd_pet_s5 if both fd_pet_s5==1 and fd_refet_s5==1'''
    s5_pet_false = cp[f'fd_pet_s4'].where((cp[f'fd_pet_s4'] == 1) & (cp[f'fd_pet_s5']==0))
    s5_pet_false = s5_pet_false.groupby('time.season').sum()
    
    s5_refet_true = cp[f'fd_refet_s5'].where((cp[f'fd_refet_s5'] == 1) & (cp[f'fd_refet_s4']==1)).to_dataset(name='refet_correct')
    s5_refet_true = s5_refet_true.groupby('time.season').sum()
    s5_refet_false = cp[f'fd_refet_s4'].where((cp[f'fd_refet_s4'] == 1) & (cp[f'fd_refet_s5']==0))
    s5_refet_false = s5_refet_false.groupby('time.season').sum()
    
    pct_correct_pet = s5_pet_true.pet_correct.values/(s5_pet_true.pet_correct.values+s5_pet_false.values)
    pct_correct_refet = s5_refet_true.refet_correct.values/(s5_refet_true.refet_correct.values+s5_refet_false.values)
    
    s5_pet_true['pet_incorrect'] = s5_pet_true['pet_correct'].copy(deep=True)
    s5_pet_true['pet_incorrect'][:,:,:] = s5_pet_false.values

    s5_pet_true['pct_pet_correct'] = s5_refet_true['refet_correct'].copy(deep=True)
    s5_pet_true['pct_pet_correct'][:,:,:] = pct_correct_pet  
    
    s5_pet_true['refet_correct'] = s5_refet_true['refet_correct'].copy(deep=True)
    s5_pet_true['refet_correct'][:,:,:] = s5_refet_false.values

    s5_pet_true['refet_incorrect'] = s5_pet_true['pet_correct'].copy(deep=True)
    s5_pet_true['refet_incorrect'][:,:,:] = s5_refet_false.values

    s5_pet_true['pct_refet_correct'] = s5_refet_true['refet_correct'].copy(deep=True)
    s5_pet_true['pct_refet_correct'][:,:,:] = pct_correct_refet  
    
    out_dictionary[f'{step}_RZSM_comparison'] = s5_pet_true
    del cp
                        
    return out_dictionary


def find_number_of_weeks_in_FD_the_same_by_season(dsesr,land_mask, rzsm, out_dictionary, step):

    cp = dsesr.copy(deep=True)
    
    true = cp[f'fd_refet_{step}'].where((dsesr[f'fd_refet_{step}'] == 1) & (dsesr[f'fd_pet_{step}']==1)).to_dataset(name='correct')
    true = true.groupby('time.season').sum()
    
    false = cp[f'fd_refet_{step}'].where((dsesr[f'fd_refet_{step}'] == 1) & (dsesr[f'fd_pet_{step}']==0))
    false = false.groupby('time.season').sum()
    
    pct_correct = true.correct.values/(true.correct.values+false.values)
    true['incorrect'] = true['correct'].copy(deep=True)
    true['incorrect'][:,:,:] = false.values

    true['pct_correct'] = true['correct'].copy(deep=True)
    true['pct_correct'][:,:,:] = pct_correct
    
    out_dictionary[f'{step}_pet_refet_FD_comparison'] = true
    del cp
                        
    return out_dictionary


def find_number_of_weeks_in_FD_the_same(dsesr,land_mask, rzsm, out_dictionary, step):

    cp = dsesr.copy(deep=True)
    
    true = cp[f'fd_refet_{step}'].where((dsesr[f'fd_refet_{step}'] == 1) & (dsesr[f'fd_pet_{step}']==1)).to_dataset(name='correct').sum(dim='time')
    false = cp[f'fd_refet_{step}'].where((dsesr[f'fd_refet_{step}'] == 1) & (dsesr[f'fd_pet_{step}']==0)).sum(dim='time')
    pct_correct = true.correct.values/(true.correct.values+false.values)
    true['incorrect'] = true['correct'].copy(deep=True)
    true['incorrect'][:,:] = false.values


    
    true['pct_correct'] = true['correct'].copy(deep=True)
    true['pct_correct'][:,:] = pct_correct
    
    out_dictionary[f'{step}_pet_refet_FD_comparison'] = true
    del cp
                        
    return out_dictionary

def RZSM_comparison(dsesr,land_mask, rzsm, out_dictionary, step):
    '''Need to only compare steps s4 and s5'''
    cp = dsesr.copy(deep=True)
    
    s5_pet_true = cp[f'fd_pet_s5'].where((cp[f'fd_pet_s5'] == 1) & (cp[f'fd_pet_s4']==1)).to_dataset(name='pet_correct').sum(dim='time')
    s5_pet_false = cp[f'fd_pet_s4'].where((cp[f'fd_pet_s4'] == 1) & (cp[f'fd_pet_s5']==0)).sum(dim='time')
    
    s5_refet_true = cp[f'fd_refet_s5'].where((cp[f'fd_refet_s5'] == 1) & (cp[f'fd_refet_s4']==1)).to_dataset(name='refet_correct').sum(dim='time')
    s5_refet_false = cp[f'fd_refet_s4'].where((cp[f'fd_refet_s4'] == 1) & (cp[f'fd_refet_s5']==0)).sum(dim='time')
    
    pct_correct_pet = s5_pet_true.pet_correct.values/(s5_pet_true.pet_correct.values+s5_pet_false.values)

    pct_correct_refet = s5_refet_true.refet_correct.values/(s5_refet_true.refet_correct.values+s5_refet_false.values)
    
    s5_pet_true['pet_incorrect'] = s5_pet_true['pet_correct'].copy(deep=True)
    s5_pet_true['pet_incorrect'][:,:] = s5_pet_false.values

    s5_pet_true['pct_pet_correct'] = s5_refet_true['refet_correct'].copy(deep=True)
    s5_pet_true['pct_pet_correct'][:,:] = pct_correct_pet  
    
    s5_pet_true['refet_correct'] = s5_refet_true['refet_correct'].copy(deep=True)
    s5_pet_true['refet_correct'][:,:] = s5_refet_false.values

    s5_pet_true['refet_incorrect'] = s5_pet_true['pet_correct'].copy(deep=True)
    s5_pet_true['refet_incorrect'][:,:] = s5_refet_false.values

    s5_pet_true['pct_refet_correct'] = s5_refet_true['refet_correct'].copy(deep=True)
    s5_pet_true['pct_refet_correct'][:,:] = pct_correct_refet  
    
    out_dictionary[f'{step}_RZSM_comparison'] = s5_pet_true
    del cp
                        
    return out_dictionary



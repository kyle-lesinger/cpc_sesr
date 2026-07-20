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
import config.STATIC as call

def get_min_max(c_year_mean):
    min_ = c_year_mean.min()
    max_ = c_year_mean.max()

    final_min = 100
    final_max = -100
    
    for i in list(c_year_mean.keys()):
        if min_[i].values < final_min:
            final_min = min_[i].values
        if min_[i].values > final_max:
            final_max = max_[i].values     

    v = np.linspace(final_min, final_max, 100, endpoint=True)
    v = [i for i in v if i < 0] + [0] + [i for i in v if i > 0]
    v = np.unique(np.array([round(i,3) for i in v]))
    v = np.linspace(v[0], v[-1], 10, endpoint=True)
    return final_min, final_max, v


def get_min_max_trial2(c_year_mean):
    def return_spacing(min_, max_, var):
        v = np.linspace(min_[var].values, max_[var].values, 7, endpoint=True)
        v = [i for i in v if i < 0] + [0] + [i for i in v if i > 0]
        v = np.unique(np.array([round(i,3) for i in v]))
        v = [i if i != -0 else 0 for i in v]
        return(v)
    
    min_ = c_year_mean.min()
    max_ = c_year_mean.max()

    all_min_max = {}
    
    for var in list(min_.keys()):
        # break
        all_min_max[var] = return_spacing(min_, max_, var)

    return all_min_max

def plot_change_per_year_by_season():
    
    #First we need to load each file
    dir_ = f'{call.doy_trend_dir}'
    
    save_fig = f'{call.fig_dir}/Trends/trend_change_per_1_year_all_seasons.png'
    
    '''change/year, plot by season, total of 20 plots'''
    c_year = xr.open_dataset(f'{dir_}/EVP_pet_refet_ESR_trend_change_per_year_1981-2020.nc')
    c_year
    
    #For each variable, groupby the season and take the average and plot
    c_year_mean = c_year.groupby("time.season").mean(dim="time")
    c_year_mean
    
    cmap = plt.get_cmap('RdBu')
    # bad_color = cmap.get_bad()
    # print(bad_color)
    
    save_dir = f'{call.fig_dir}/Trends'
    os.system(f'mkdir -p {save_dir}')
    
    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    lon = c_year_mean.lon.values
    lat = c_year_mean.lat.values
    
    fig, axs = plt.subplots(
    nrows = 5, ncols= 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 12),dpi=300)
    
    axs = axs.flatten()
    
    final_min, final_max, v = get_min_max(c_year_mean)
    all_min_max = get_min_max_trial2(c_year_mean)
    
    axs_start = 0
    for idx, varname in enumerate(c_year_mean.keys()):
        # idx, varname = 0, 'EVP'
        v=all_min_max[varname]
        norm = mcolors.TwoSlopeNorm(vmin=v[0] , vcenter=0, vmax=v[-1] )
        for idSeason, season in enumerate(['DJF','MAM','JJA','SON']):
            print(f'Plotting trend for {varname} for season {season}.')
            # break
            # print(land_mask)
            data = c_year_mean[varname].sel(season=season).values
            data = np.where(np.isnan(land_mask),np.nan,data)
            # data = np.ma.masked_invalid(data)
            for Y in range(data.shape[0]):
                for X in range(data.shape[1]):
                    if np.isnan(land_mask[Y,X]) or (data[Y,X] == 0):
                        data[Y,X]=np.nan
    
            map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
                  llcrnrlon=-128, urcrnrlon=-60, resolution='l')
            x, y = map(*np.meshgrid(lon, lat))
        
            im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                          transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
            cbar = fig.colorbar(im, ax = axs[axs_start], orientation='horizontal')
            if varname in ['EVP','refET','PEVPR']:
                cbar.set_label('change/year (mm)', fontsize=10, labelpad=5)
            else:
                cbar.set_label('change/year (unitless)', fontsize=10, labelpad=5)
            
            cbar.set_ticks([round(t, 3) for t in cbar.get_ticks()])
            cbar.ax.tick_params(labelsize=8)
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
            if varname == 'PEVPR':
                vname = 'pet'
            elif varname == 'refET':
                vname='refet'
            else:
                vname = varname
            axs[axs_start].set_title(f'{vname} - {season}',fontsize=12,pad=5)
            axs_start+=1
    
    plt.suptitle(f'Trend change/year over climatology\n1981-2020',fontsize=15)
    plt.tight_layout()
    plt.savefig(save_fig)
    plt.close()
    return(f'Completed and saved into {save_fig}')



def get_min_max_decade_returns_by_season(c_year_mean):
    def return_spacing(min_, max_, var,season):
        v = np.linspace(min_[var].sel(season=season).values, max_[var].sel(season=season).values, 7, endpoint=True)
        v = [i for i in v if i < 0] + [0] + [i for i in v if i > 0]
        v = np.unique(np.array([round(i,3) for i in v]))
        v = [i if i != -0 else 0 for i in v]
        return(v)
    
    min_ = c_year_mean.min(dim=['decade','lat','lon'])
    max_ = c_year_mean.max(dim=['decade','lat','lon'])

    all_min_max = {}

    for iSeason,season in enumerate(c_year_mean.season.values):
        for var in list(min_.keys()):
            # break
            all_min_max[f'{var}_{season}'] = return_spacing(min_, max_, var,season)

    return all_min_max


def plot_change_per_two_decades_by_season():
    
    #First we need to load each file
    dir_ = f'{call.doy_trend_dir}'
    

    '''change/year, plot by season, total of 20 plots'''
    c_year = xr.open_dataset(f'{dir_}/EVP_pet_refet_ESR_trend_change_per_two_decades_1981-2020.nc')
    c_year
    
    #For each variable, groupby the season and take the average and plot
    c_year_mean = c_year.groupby("time.season").mean(dim="time")
    c_year_mean
    
    cmap = plt.get_cmap('RdBu')
    # bad_color = cmap.get_bad()
    # print(bad_color)
    
    save_dir = f'{call.fig_dir}/Trends'
    os.system(f'mkdir -p {save_dir}')
    
    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    lon = c_year_mean.lon.values
    lat = c_year_mean.lat.values

    
    final_min, final_max, v = get_min_max(c_year_mean)
    all_min_max = get_min_max_decade_returns_by_season(c_year_mean)
    
    for idSeason, season in enumerate(['DJF','MAM','JJA','SON']):
            
        fig, axs = plt.subplots(
        nrows = len(list(c_year.keys())), ncols= len(c_year.decade.values), subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 12),dpi=300)
        
        axs = axs.flatten()
        # idSeason,season=0,'DJF'
        save_fig = f'{call.fig_dir}/Trends/trend_change_per_2_decades_{season}.png'
    
        axs_start = 0
        for idx, varname in enumerate(c_year_mean.keys()):
            # idx, varname = 0, 'EVP'
            v=all_min_max[f'{varname}_{season}']
            norm = mcolors.TwoSlopeNorm(vmin=v[0] , vcenter=0, vmax=v[-1] )
                
            for iDecade, decade in enumerate(['1981-2000','1991-2010','2001-2020']):

                print(f'Plotting trend for {varname} {decade} for season {season}.')
                # break
                # print(land_mask)
                data = c_year_mean[varname].sel(season=season).isel(decade=iDecade).values
                data = np.where(np.isnan(land_mask),np.nan,data)
                # data = np.ma.masked_invalid(data)
                for Y in range(data.shape[0]):
                    for X in range(data.shape[1]):
                        if np.isnan(land_mask[Y,X]) or (data[Y,X] == 0):
                            data[Y,X]=np.nan
        
                map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
                      llcrnrlon=-128, urcrnrlon=-60, resolution='l')
                x, y = map(*np.meshgrid(lon, lat))
            
                im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                              transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
                cbar = fig.colorbar(im, ax = axs[axs_start], orientation='horizontal')
                if varname in ['EVP','refET','PEVPR']:
                    cbar.set_label('change/year (mm)', fontsize=10, labelpad=5)
                else:
                    cbar.set_label('change/year (unitless)', fontsize=10, labelpad=5)
                
                cbar.set_ticks([round(t, 3) for t in cbar.get_ticks()])
                cbar.ax.tick_params(labelsize=10)
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
                if varname == 'PEVPR':
                    vname = 'pet'
                elif varname == 'refET':
                    vname='refet'
                else:
                    vname = varname
                axs[axs_start].set_title(f'{vname} - ({decade})',fontsize=12,pad=5)
                axs_start+=1
        
        plt.suptitle(f'{season} trend change/20 years over climatology\n1981-2020',fontsize=15)
        plt.tight_layout()
        plt.savefig(save_fig)
        plt.close()
        print(f'Completed and saved into {save_fig}')
    return(print('Completed plotting 20-year trend plots for different ET, PET, refET, ESR_pet, ESR_refet'))

def plot_change_per_1_decade_by_season():
    
    #First we need to load each file
    dir_ =f'{call.doy_trend_dir}'
    

    '''change/year, plot by season, total of 20 plots'''
    c_year = xr.open_dataset(f'{dir_}/EVP_pet_refet_ESR_trend_change_per_decade_1981-2020.nc')
    c_year
    
    #For each variable, groupby the season and take the average and plot
    c_year_mean = c_year.groupby("time.season").mean(dim="time")
    c_year_mean
    
    cmap = plt.get_cmap('RdBu')
    # bad_color = cmap.get_bad()
    # print(bad_color)
    
    save_dir = f'{call.fig_dir}/Trends'
    os.system(f'mkdir -p {save_dir}')
    
    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    lon = c_year_mean.lon.values
    lat = c_year_mean.lat.values

    
    final_min, final_max, v = get_min_max(c_year_mean)
    all_min_max = get_min_max_decade_returns_by_season(c_year_mean)
    
    for idSeason, season in enumerate(['DJF','MAM','JJA','SON']):
            
        fig, axs = plt.subplots(
        nrows = len(list(c_year.keys())), ncols= len(c_year.decade.values), subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 12),dpi=300)
        
        axs = axs.flatten()
        # idSeason,season=0,'DJF'
        save_fig = f'{call.fig_dir}/Trends/trend_change_per_1_decade_{season}.png'
    
        axs_start = 0
        for idx, varname in enumerate(c_year_mean.keys()):
            # idx, varname = 0, 'EVP'
            v=all_min_max[f'{varname}_{season}']
            norm = mcolors.TwoSlopeNorm(vmin=v[0] , vcenter=0, vmax=v[-1] )
                
            for iDecade, decade in enumerate(['1981-1990','1991-2000','2001-2010','2011-2020']):

                print(f'Plotting trend for {varname} {decade} for season {season}.')
                # break
                # print(land_mask)
                data = c_year_mean[varname].sel(season=season).isel(decade=iDecade).values
                data = np.where(np.isnan(land_mask),np.nan,data)
                # data = np.ma.masked_invalid(data)
                for Y in range(data.shape[0]):
                    for X in range(data.shape[1]):
                        if np.isnan(land_mask[Y,X]) or (data[Y,X] == 0):
                            data[Y,X]=np.nan
        
                map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
                      llcrnrlon=-128, urcrnrlon=-60, resolution='l')
                x, y = map(*np.meshgrid(lon, lat))
            
                im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                              transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
                cbar = fig.colorbar(im, ax = axs[axs_start], orientation='horizontal')
                if varname in ['EVP','refET','PEVPR']:
                    cbar.set_label('change/year (mm)', fontsize=10, labelpad=5)
                else:
                    cbar.set_label('change/year (unitless)', fontsize=10, labelpad=5)
                
                cbar.set_ticks([round(t, 3) for t in cbar.get_ticks()])
                cbar.ax.tick_params(labelsize=10)
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
                if varname == 'PEVPR':
                    vname = 'pet'
                elif varname == 'refET':
                    vname='refet'
                else:
                    vname = varname
                axs[axs_start].set_title(f'{vname} - ({decade})',fontsize=12,pad=5)
                axs_start+=1
        
        plt.suptitle(f'{season} trend change/10 years over climatology\n1981-2020',fontsize=15)
        plt.tight_layout()
        plt.savefig(save_fig)
        plt.close()
        print(f'Completed and saved into {save_fig}')
    return(print('Completed plotting 20-year trend plots for different ET, PET, refET, ESR_pet, ESR_refet'))
    
    


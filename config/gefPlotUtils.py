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


def plot_CRPS_pearson_correlation(season_metric,wk_lead_indexes,varname):
    '''Only plotting CRPS, pearson function doesnt work'''
    cmap_crps = plt.get_cmap('YlOrRd')

    save_dir = f'{call.fig_dir}/GEFSv12_hindcast/CRPS_pearson_r'
    os.system(f'mkdir -p {save_dir}')
    save_fig = f'{save_dir}/{varname}_crps_by_lead_2000-2019.png'

    key_ = list(season_metric.keys())
    
    lon = season_metric[key_[0]].lon.values
    lat = season_metric[key_[0]].lat.values

    metric = 'crps'


    dname = list(season_metric[f'{metric}_DJF'].keys())[0]
    
    fig, axs = plt.subplots(
        nrows = 5, ncols= 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(8, 10),dpi=300)
    
    axs,axs_start = axs.flatten(),0
    min_, max_ = get_min_max_by_metric(season_metric,metric,wk_lead_indexes,dname)
    v = np.linspace(min_, max_, 100, endpoint=True)
    
    for iLead, lead in wk_lead_indexes.items():
        for iSeason,season in enumerate(['DJF','MAM','JJA','SON']):
            
            data =  season_metric[f'{metric}_{season}'][dname][lead,:,:].values
        
            map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
                  llcrnrlon=-128, urcrnrlon=-60, resolution='l')
            x, y = map(*np.meshgrid(lon, lat))
    
            im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                          transform=ccrs.PlateCarree(), cmap=cmap_crps)
            gl = axs[axs_start].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                       linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = False
            gl.bottom_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            axs[axs_start].coastlines()
            axs[axs_start].set_aspect('equal')  # this makes the plots better
    
            axs[axs_start].set_title(f'{iLead}-{season}',fontsize=12)    
            axs_start+=1
    #left, bottom, width, height
    cbar_ax = fig.add_axes([0.10, 0.054, .83, .01]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label(f'{metric}', labelpad=5)
    plt.suptitle(f'{metric} over weekly leads compared with\nNLDAS Noah (no detrending performed on either dataset).',fontsize=13)
    plt.tight_layout()
    plt.savefig(save_fig)
    plt.close()            
    return print(f'Completed making CRPS image and saved into {call.fig_dir}/GEFSv12_hindcast/CRPS_pearson_r')

def get_min_max_by_metric(season_metric,metric,wk_lead_indexes,dname):
    min_ = []
    max_ = []
    
    for k,lead_index in wk_lead_indexes.items(): 
        for iSeason,season in enumerate(['DJF','MAM','JJA','SON']):
            vals = season_metric[f'{metric}_{season}'][dname][lead_index,:,:].values
            min_.append(np.nanmin(vals))
            max_.append(np.nanmax(vals))
    return min(min_), max(max_)



# Make plots 
def GEF_yearly_average_statistics_of_accumulated_value(recompute):
    cmap = plt.get_cmap('YlOrRd')
    # bad_color = cmap.get_bad()
    # print(bad_color)
    
    save_dir = f'{call.fig_dir}/et_pet_refet_monthly_and_yearly_sum_statistics'
    os.system(f'mkdir -p {save_dir}')

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    #Open and take yearly average
    refet = xr.open_dataset(f'{call.gefs_dir}/GEFSv12_hindcast_lagged_averaged_ensemble/refet_lagged_avg_ensemble.nc').sel(time=slice(call.hind_start, call.hind_end)).resample(time="YE").sum()
    et = xr.open_dataset(f'{call.gefs_dir}/GEFSv12_hindcast_lagged_averaged_ensemble/et_lagged_avg_ensemble.nc').sel(time=slice(call.hind_start, call.hind_end)).resample(time="YE").sum()
    cpc_refet = xr.open_dataset(f'{call.gefs_dir}/GEFSv12_hindcast_lagged_averaged_ensemble/cpc_refet_lagged_avg_ensemble.nc').sel(time=slice(call.hind_start, call.hind_end)).resample(time="YE").sum()

    refet = xr.where(~np.isnan(land_mask),refet,np.nan).transpose('time','lat','lon') #mask ocean and other water bodies
    et = xr.where(~np.isnan(land_mask),et,np.nan).transpose('time','lat','lon') #mask ocean and other water bodies
    cpc_refet = xr.where(~np.isnan(land_mask),cpc_refet,np.nan).transpose('time','lat','lon') #mask ocean and other water bodies
    
    data_arrays = [et['EVP'],refet['refet'],cpc_refet['refet']]
   
    lon = refet.lon.values
    lat = refet.lat.values

    
    fig, axs = plt.subplots(
    nrows = 3, ncols= 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 9),dpi=300)
    
    axs = axs.flatten()

    axs_start = 0
    for idx, (arr, var_name) in enumerate(zip(data_arrays,['EVP','refet','cpc_refet'])):
        # break
        # print(land_mask)
        if var_name == 'EVP':
            '''trying to remove the ocean values but I cannot make it work'''
            data_ = arr.mean(dim='time')
            lout = land_mask.copy()
            
            for Y,yy in enumerate(data_.lat.values):
                for X,xx in enumerate(data_.lon.values):
                    if np.isnan(land_mask[Y,X]):
                        lout[Y,X]=np.nan
                    else:
                        lout[Y,X] = data_[Y,X].values
            data = np.where(np.isnan(land_mask),np.nan,lout)
        else:
            data = arr.mean(dim='time').values
            data = np.where(np.isnan(land_mask),np.nan,data)
            # data = np.ma.masked_invalid(data)
            for Y in range(data.shape[0]):
                for X in range(data.shape[1]):
                    if np.isnan(land_mask[Y,X]) or (data[Y,X] == 0):
                        data[Y,X]=np.nan

            data = np.where(np.isnan(land_mask),np.nan,data)

        if var_name == 'EVP':
            v = np.linspace(np.nanmin(data), 1500, 9, endpoint=True)
        else:
            v = np.linspace(np.nanmin(data), np.nanmax(data), 9, endpoint=True)
        
        map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
              llcrnrlon=-128, urcrnrlon=-60, resolution='l')
        
        x, y = map(*np.meshgrid(lon, lat))

        im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
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
        axs[axs_start].set_title(f'{var_name}: 2000-2019',fontsize=12)

        cbar = fig.colorbar(im, ax = axs[axs_start], orientation='horizontal')
        cbar.set_label('mm/year', fontsize=10, labelpad=5)
        axs_start+=1

    plt.suptitle(f'Accumulated yearly average of variable.\nThen averaged over the period. ',fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/GEFSv12_yearly_EVP_PET_refET_sum.png')

    return('Completed yearly sum plots')



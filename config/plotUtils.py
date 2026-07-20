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



def make_ESR_plots(year_ranges_tuple_1,year_ranges_tuple_2):
    '''For some reason, this function begins to slowly eat up memory. Not sure why, but it 
    is written to not re-run after it has been completed once'''
    
    obs = xr.open_dataset(f'{call.noah_dir}/esr_rolling_mean_0.50_degrees.nc').load()

    # For each doy in julian day 90-336, look at yearly statistics and plot in Figures
    # 1.) ESR for each doy 1981-2020
    # 2.) Difference between each year for ESR 1981-2020 (plotting 1982-2020)
    # 3.) 10 year running mean for ESR 1981-2020 (plotting 1991-2020)
    
    for year_ranges_tuple in [year_ranges_tuple_1]:
        make_plot_for_ESR_by_year_by_doy(obs,year_ranges_tuple)
    
    for year_ranges_tuple in [year_ranges_tuple_1]:
        make_ESR_anomaly_by_year_by_doy(obs,year_ranges_tuple)
        
    # You can change the year ranges to be different years. Due to difficulties in getting the KDE plots
    # to generate quickly, the fastkde function was determined to be the best. But I could not figure out a 
    # way to add it to the subplots function so I just made each plot individually. 
    # Saved into Figures/Distribution_and_yearly_sum
    
    plot_kde(obs,year_ranges_tuple_1,year_ranges_tuple_2)

    
    # Plot the yearly summation of EVP, PET, refet to verify that these are good to go for further analysis. 
    # Saved into Figures/monthly_and_yearly_sum_statistics
    
    plot_yearly_average_statistics_of_accumulated_value(year_ranges_tuple_1, year_ranges_tuple_2)

    # # Plot the monthly summation of EVP, PET, refet to verify that these are good to go for further analysis. 
    # # Saved into Figures/monthly_and_yearly_sum_statistics
    
    plot_monthly_average_statistics_of_accumulated_value(obs, year_ranges_tuple_1, year_ranges_tuple_2)
    return (0)



def plot_kde(obs: xr.DataArray, year_ranges_tuple_1, year_ranges_tuple_2):
    '''Make the KDE plots for EVP, PEVPR, and ESR
    Save into Figures directory for each variable and time segment that is selected'''
    save_dir = f'{call.fig_dir}/Distribution_kde_plots'
    os.system(f'mkdir -p {save_dir}')

    obs_plot1 = obs.sel(time=slice(str(year_ranges_tuple_1[0]),str(year_ranges_tuple_1[1]))) 
    obs_plot2 = obs.sel(time=slice(str(year_ranges_tuple_2[0]),str(year_ranges_tuple_2[1]))) 
                        
    def remove_np_nan(file):
        return(file[~np.isnan(file)])

    #Now just manually plot the variables
    et_1 = remove_np_nan(obs_plot1['EVP'].values)
    et_2 = remove_np_nan(obs_plot2['EVP'].values)
    
    pet_1 = remove_np_nan(obs_plot1['PEVPR'].values)
    pet_2 = remove_np_nan(obs_plot2['PEVPR'].values)

    refet_1 = remove_np_nan(obs_plot1['refET'].values)
    refet_2 = remove_np_nan(obs_plot2['refET'].values)

    esr_pet_1 = remove_np_nan(obs_plot1['ESR_pet'].values)
    esr_pet_2 = remove_np_nan(obs_plot2['ESR_pet'].values)

    esr_refet_1 = remove_np_nan(obs_plot1['ESR_refet'].values)
    esr_refet_2 = remove_np_nan(obs_plot2['ESR_refet'].values)

    data_arrays = [et_1,pet_1,refet_1 ,esr_pet_1,esr_refet_1, et_2,pet_2,refet_2 ,esr_pet_2,esr_refet_2]

    for idx, (arr, var_name) in enumerate(zip(data_arrays,['EVP','PET', 'refET','ESR_pet', 'ESR_refet','EVP','PET', 'refET','ESR_pet', 'ESR_refet',])):
        # break
        # #Test 1
        # sns.kdeplot(arr, ax=axs[idx_start])

        # #Test 2
        # # Fit the KDE model
        # kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
        # kde.fit(arr[:, None])

        # #Create x_values based on data
        # x_vals = np.linspace(0, np.max(arr), 1000)[:, None]

        # # Evaluate the KDE
        # log_density = kde.score_samples(x_vals)
        # density = np.exp(log_density)

        # axs[idx_start].plot(x_vals[:, 0], density, label='KDE')

        #Test 3
        plt.figure(figsize=(3,3),dpi=300)
        PDF = fastkde.pdf(arr, var_names = [var_name])
        PDF.plot()
        
        if idx >=3:
            #Add the years to the plot ( if even then its the 2nd tuple)
            plt.title(f'KDE 7day_mean distribution\n{var_name} {year_ranges_tuple_2[0]}-{year_ranges_tuple_2[1]}', fontsize=10)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{var_name}_kde_{year_ranges_tuple_2[0]}-{year_ranges_tuple_2[1]}.png',dpi=300)
            print(f'Completed {var_name} in range {year_ranges_tuple_2[0]}-{year_ranges_tuple_2[1]}')
        else:
            plt.title(f'KDE 7day_mean distribution\n{var_name} {year_ranges_tuple_1[0]}-{year_ranges_tuple_1[1]}', fontsize=10)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{var_name}_kde_{year_ranges_tuple_1[0]}-{year_ranges_tuple_1[1]}.png',dpi=300)
            print(f'Completed {var_name} in range {year_ranges_tuple_1[0]}-{year_ranges_tuple_1[1]}')

    return('Completed KDE plots')


   
def plot_yearly_average_statistics_of_accumulated_value(year_ranges_tuple_1, year_ranges_tuple_2):
    cmap = plt.get_cmap('YlOrRd')
    cmap_diff = plt.get_cmap('RdBu_r')
    # bad_color = cmap.get_bad()
    # print(bad_color)
    
    save_dir = f'{call.fig_dir}/et_pet_refet_monthly_and_yearly_sum_statistics'
    os.system(f'mkdir -p {save_dir}')

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    
    yr_plot1 = xr.open_dataset(f'{call.noah_dir}/esr_annual_sum_average_0.50_degrees_{year_ranges_tuple_1[0]}-{year_ranges_tuple_1[1]}.nc')
    yr_plot2 = xr.open_dataset(f'{call.noah_dir}/esr_annual_sum_average_0.50_degrees_{year_ranges_tuple_2[0]}-{year_ranges_tuple_2[1]}.nc')

    data_arrays = [yr_plot1['EVP'],yr_plot1['PEVPR'],yr_plot1['refET'],yr_plot2['EVP'],yr_plot2['PEVPR'],yr_plot2['refET'],
                  yr_plot2['EVP'].mean(dim='time') - yr_plot1['EVP'].mean(dim='time') , yr_plot2['PEVPR'].mean(dim='time') - yr_plot1['PEVPR'].mean(dim='time'), yr_plot2['refET'].mean(dim='time') - yr_plot1['refET'].mean(dim='time')]

    
    lon = yr_plot1.lon.values
    lat = yr_plot1.lat.values

    fig, axs = plt.subplots(
    nrows = 3, ncols= 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 9),dpi=300)
    
    axs = axs.flatten()

    axs_start = 0
    for idx, (arr, var_name) in enumerate(zip(data_arrays,['EVP','PET','refET','EVP','PET','refET','diff(EVP)','diff(PET)','diff(refET)'])):
        # break
        # print(land_mask)
        if 'diff' in var_name:
            data = arr.values
        else:
            data = arr.mean(dim='time').values
            data = np.where(np.isnan(land_mask),np.nan,data)
            # data = np.ma.masked_invalid(data)
            for Y in range(data.shape[0]):
                for X in range(data.shape[1]):
                    if np.isnan(land_mask[Y,X]) or (data[Y,X] == 0):
                        data[Y,X]=np.nan

        # print(data)
        if 'diff' in var_name:
            v = np.linspace(np.nanmin(data), np.nanmax(data), 9, endpoint=True)
            v = [i for i in v if i < 0] + [0] + [i for i in v if i > 0]
            norm = mcolors.TwoSlopeNorm(vmin=v[0], vcenter=0, vmax=v[-1])
        else:
            v = np.linspace(np.nanmin(data), np.nanmax(data), 8, endpoint=True)

            
        map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
              llcrnrlon=-128, urcrnrlon=-60, resolution='l')
        x, y = map(*np.meshgrid(lon, lat))

        if 'diff' in var_name:
            im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                          transform=ccrs.PlateCarree(), cmap=cmap_diff,norm=norm)
        else:
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
        if axs_start <=2:
            axs[axs_start].set_title(f'{var_name}: {year_ranges_tuple_1[0]}-{year_ranges_tuple_1[1]}',fontsize=12)
        elif axs_start in [3,4,5]:
            axs[axs_start].set_title(f'{var_name}: {year_ranges_tuple_2[0]}-{year_ranges_tuple_2[1]}',fontsize=12)
        else:
            axs[axs_start].set_title(f'{var_name}\n[bottom - top]',fontsize=12)
        cbar = fig.colorbar(im, ax = axs[axs_start], orientation='horizontal')
        cbar.set_label('mm/year', fontsize=10, labelpad=5)
        axs_start+=1

    plt.suptitle(f'Accumulated yearly average of variable.\nThen averaged over the period. ',fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/yearly_EVP_PET_refET_sum.png')

    return('Completed yearly sum plots')





def make_plot_for_ESR_by_year_by_doy(obs,year_ranges_tuple):
    save_esr_dir = f'{call.fig_dir}/ESR_doy_vals_by_year_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}'
    os.makedirs(save_esr_dir, exist_ok=True)

    save_diff_esr_dir = f'{call.fig_dir}/diff_ESR_doy_vals_by_year_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}'
    os.makedirs(save_diff_esr_dir, exist_ok=True)
    
    save_10_yr_mean_esr_dir = f'{call.fig_dir}/ESR_10_year_mean_doy_vals_by_year_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}'
    os.makedirs(save_10_yr_mean_esr_dir, exist_ok=True)
    
    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    obs1 = obs.sel(time=slice(str(year_ranges_tuple[0]),str(year_ranges_tuple[1]))) #ensures none of the missing values from the previous year

    for idx,doy in enumerate(range(1,366)):
        for refet_or_PET in ['ESR_refet','ESR_pet']:
        # idx,doy=0,1
            obs_subset = obs1.sel(time=obs1['time.dayofyear']==doy)
    
            save_esr = f'{save_esr_dir}/{refet_or_PET}_doy{doy}.png'
            if os.path.exists(save_esr):
                pass
            else:
                print(f'Working on plotting ESR for doy {doy}.')
                plot_ESR_by_year(obs_subset,year_ranges_tuple, save_esr, land_mask, doy, refet_or_PET)

            '''Difference in ESR by 1 year'''
            save_diff_esr = f'{save_diff_esr_dir}/diff_{refet_or_PET}_doy{doy}.png'
            obs_diff = obs_subset.diff(dim='time', n=1)
            if os.path.exists(save_diff_esr):
                pass
            else:
                print(f'Working on plotting diff(ESR) for doy {doy}.')
                plot_ESR_diff_by_year(obs_diff, year_ranges_tuple, save_diff_esr, land_mask, doy, refet_or_PET)
            
            '''10-year running mean'''
            save_mean_esr = f'{save_10_yr_mean_esr_dir}/mean_{refet_or_PET}_doy{doy}.png'
            num_years_mean = 10
            
            running_mean = obs_subset.rolling(time=num_years_mean, center=False).mean()
            start_good_year = pd.to_datetime(running_mean.time.values[num_years_mean-1]).year
            
            years = [i.year for i in pd.to_datetime(running_mean.time.values)]
    
            if os.path.exists(save_mean_esr):
                pass
            else:
                print(f'Working on plotting mean(ESR) for doy {doy}.')
                plot_ESR_10_year_mean_by_doy(running_mean, year_ranges_tuple, save_mean_esr, land_mask, doy, start_good_year, num_years_mean, refet_or_PET)    
                
            # '''10-year running mean but a 5 year difference'''
            # save_mean_esr = f'{save_10_yr_mean_esr_dir}/mean_{refet_or_PET}_doy{doy}.png'
            # num_years_mean = 10
            
            # running_mean = obs_subset.rolling(time=num_years_mean, center=False).mean()
            # start_good_year = pd.to_datetime(running_mean.time.values[num_years_mean-1]).year
            
            # years = [i.year for i in pd.to_datetime(running_mean.time.values)]
    
            # if os.path.exists(save_mean_esr):
            #     pass
            # else:
            #     print(f'Working on plotting mean(ESR) for doy {doy}.')
            #     plot_ESR_10_year_mean_by_doy(running_mean, year_ranges_tuple, save_mean_esr, land_mask, doy, start_good_year, num_years_mean, refet_or_PET)      
    
        

'''add this to the plotUtils.py script'''
def plot_ESR_by_year(obs_subset: xr.DataArray, year_ranges_tuple, save_esr, land_mask, doy, refet_or_PET):
    cmap = plt.get_cmap('YlOrBr')    
    # Define a TwoSlopeNorm that divides the data at 0.5
    # norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    v = np.linspace(0, 1, 100, endpoint=True)
    
    lon = obs_subset.lon.values
    lat = obs_subset.lat.values

    fig, axs = plt.subplots(
    nrows = len(obs_subset.time.values)//5, ncols= 5, 
        subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 20), dpi=150)
    
    axs = axs.flatten()

    axs_start = 0
    for idx, year in enumerate(range(year_ranges_tuple[0],year_ranges_tuple[1])):
        # break
        try:
            data = obs_subset[refet_or_PET].sel(time=str(year)).values[0,:,:]
            data = np.where(np.isnan(land_mask),np.nan,data)
            data = np.ma.masked_invalid(data) #This actually masks the values
            
            map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
                  llcrnrlon=-128, urcrnrlon=-60, resolution='l')
            x, y = map(*np.meshgrid(lon, lat))
    
            im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                          transform=ccrs.PlateCarree(), cmap=cmap)
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
    
            axs[axs_start].set_title(f'{year_ranges_tuple[0]+idx}',fontsize=12)
    
            axs_start+=1
        except KeyError:
            pass

                            #left, bottom, width, height
    cbar_ax = fig.add_axes([0.10, 0.025, .83, .01]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label(f'ESR', labelpad=5)
    plt.suptitle(f'{refet_or_PET} - DOY {doy}.',fontsize=30)
    plt.tight_layout()
    plt.savefig(save_esr)
    plt.close()

    return(f'Completed DOY {doy}.')

def plot_ESR_diff_by_year(obs_diff: xr.DataArray, year_ranges_tuple, save_file, land_mask, doy, refet_or_PET):

    # cmap = plt.get_cmap('YlOrBr')  
    cmap = plt.get_cmap('RdBu_r')  
    lon = obs_diff.lon.values
    lat = obs_diff.lat.values

    if year_ranges_tuple[0] <1999:
        siz1,siz2=15,20
    else:
        siz1,siz2=15,12
        
    fig, axs = plt.subplots(
    nrows = (len(obs_diff.time.values)+1)//5, ncols= 5, 
        subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(siz1,siz2))
    
    axs = axs.flatten()

    axs_start = 0
    for idx, year in enumerate(range(pd.to_datetime(obs_diff.time.values[0]).year,pd.to_datetime(obs_diff.time.values[-1]).year+1)):
        # break
        data = obs_diff[refet_or_PET].sel(time=str(year)).values[0,:,:]
        data = np.where(np.isnan(land_mask),np.nan,data)

        min_, max_ = np.nanmin(data), np.nanmax(data)
        # Define a TwoSlopeNorm that divides the data for plotting
        if min_ < 0:
            norm = mcolors.TwoSlopeNorm(vmin=min_, vcenter=0, vmax=max_)
        else:
            norm = mcolors.TwoSlopeNorm(vmin=min_, vcenter=0.5, vmax=max_)
        
        v = np.linspace(min_, max_, 99, endpoint=True)
        v = [i for i in v if i < 0] + [0] + [i for i in v if i > 0]
        
        data = np.ma.masked_invalid(data) #This actually masks the values
        
        map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
              llcrnrlon=-128, urcrnrlon=-60, resolution='l')
        x, y = map(*np.meshgrid(lon, lat))

        try:
            im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                          transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
            axs[axs_start].coastlines()
            axs[axs_start].set_aspect('equal')  # this makes the plots better
    
            axs[axs_start].set_title(f'diff({year}-{year-1})',fontsize=12)

            axs_start+=1
        except ValueError:
            pass
            # gl = axs[axs_start].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
            #                            linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
            # gl.top_labels = False
            # gl.right_labels = False
            # gl.left_labels = False
            # gl.bottom_labels = False
            # gl.xformatter = LongitudeFormatter()
            # gl.yformatter = LatitudeFormatter()


                            #left, bottom, width, height
    cbar_ax = fig.add_axes([0.10, 0.023, .83, .01]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label('Left year had lower ESR than right year <<<------------  diff(ESR)  ------------>>> Left year had higher ESR than right year', labelpad=5)
    plt.suptitle(f'diff({refet_or_PET}) - DOY {doy})',fontsize=30)
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()

    return(f'Completed DOY {doy}.')

def plot_ESR_10_year_mean_by_doy(running_mean: xr.DataArray, year_ranges_tuple, save_file, land_mask, doy, start_good_year, num_years_mean, refet_or_PET):
    cmap = plt.get_cmap('YlOrBr')    
    # Define a TwoSlopeNorm that divides the data at 0.5
    # norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

    min_, max_ = np.nanmin(running_mean[refet_or_PET].values), np.nanmax(running_mean[refet_or_PET].values)
    v = np.linspace(min_, max_, 100, endpoint=True)
    
    lon = running_mean.lon.values
    lat = running_mean.lat.values

    fig, axs = plt.subplots(
    nrows =7, ncols= 5, 
        subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 20))
    
    axs = axs.flatten()

    axs_start = 0
    for idx, year in enumerate(range(start_good_year,pd.to_datetime(running_mean.time.values[-1]).year+1)):
        # break
        data = running_mean[refet_or_PET].sel(time=str(year)).values[0,:,:]
        data = np.where(np.isnan(land_mask),np.nan,data)
        data = np.ma.masked_invalid(data) #This actually masks the values
        
        map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
              llcrnrlon=-128, urcrnrlon=-60, resolution='l')
        x, y = map(*np.meshgrid(lon, lat))

        im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                      transform=ccrs.PlateCarree(), cmap=cmap)

        # Add black contour lines in increments of 0.10
        contour_levels = np.arange(min(0.0,min_), min(max_,0.9) + 0.2, 0.2)
        cs = axs[axs_start].contour(x, y, data, levels=contour_levels, colors='black',
                                    linewidths=0.5, linestyles='solid', transform=ccrs.PlateCarree())
        # plt.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
        
        # gl = axs[axs_start].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
        #                            linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
        # gl.top_labels = False
        # gl.right_labels = False
        # gl.left_labels = False
        # gl.bottom_labels = False
        # gl.xformatter = LongitudeFormatter()
        # gl.yformatter = LatitudeFormatter()
        axs[axs_start].coastlines()
        axs[axs_start].set_aspect('equal')  # this makes the plots better

        axs[axs_start].set_title(f'{start_good_year+idx}',fontsize=12)

        axs_start+=1

                            #left, bottom, width, height
    cbar_ax = fig.add_axes([0.10, 0.023, .83, .01]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label('mean(ESR)', labelpad=5)
    plt.suptitle(f'10 year running mean({refet_or_PET}) - DOY {doy}.',fontsize=30)
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()

    return(f'Completed DOY {doy}.')




def detrend_esr_func_by_doy_EXTRA_CODE_TO_MAKE_PLOTS(obs, land_mask, save_sesr):
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

                    '''Manually check the trend of a single instance'''
                    b4 = no_nan_values[no_nan_values['time.dayofyear'] == 1]
                    plt.plot(np.arange(len(b4.values)),b4.values)

                    '''Before min-max normalization'''
                    final_out = no_nan_values.groupby('time.dayofyear').apply(detrend_doy)
                    final_out_no_normalization = final_out[final_out['time.dayofyear'] == 1]
                    plt.plot(np.arange(len(final_out.values)),final_out.values)

                    '''After min-max normalization'''
                    final_dtrnd = (final_out_no_normalization- final_out_no_normalization.min())/(final_out_no_normalization.max()-final_out_no_normalization.min())

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
                    esr_fill[index_vals, Y, X] = final_dtrnd 
 

    obs['ESR_detrend'][:, :,:] = esr_fill  # Store the detrended time series

    obs.to_netcdf(save_sesr)



def plot_monthly_average_statistics_of_accumulated_value(obs: xr.DataArray, year_ranges_tuple_1, year_ranges_tuple_2):
    cmap = plt.get_cmap('YlOrBr')    
    
    save_dir = f'{call.fig_dir}/et_pet_refet_monthly_and_yearly_sum_statistics'
    os.system(f'mkdir -p {save_dir}')

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    monthly_sum_per_year = obs.groupby('time.year').apply(lambda x: x.groupby('time.month').sum(dim='time'))
    # Define custom month order starting from November (month 11)
    # custom_month_order = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # Reindex the monthly sums to match the custom month order
    # monthly_sum_per_year = monthly_sum_per_year.reindex(month=custom_month_order)
    
    yr_plot1 = monthly_sum_per_year.sel(year=slice(str(year_ranges_tuple_1[0]),str(year_ranges_tuple_1[1]))).mean(dim='year')
    yr_plot2 = monthly_sum_per_year.sel(year=slice(str(year_ranges_tuple_2[0]),str(year_ranges_tuple_2[1]))).mean(dim='year')

    data_arrays1 = [yr_plot1['EVP'],yr_plot1['PEVPR'],yr_plot1['refET']]
    data_arrays2 = [yr_plot2['EVP'],yr_plot2['PEVPR'],yr_plot2['refET']]
   
    lon = yr_plot1.lon.values
    lat = yr_plot1.lat.values

    for idx1,data_arrays in enumerate([data_arrays1,data_arrays2]):

        for idx2, (arr, var_name) in enumerate(zip(data_arrays,['EVP','PET','refET',])):
            fig, axs = plt.subplots(
                nrows = 4, ncols= 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 15),dpi=300)
                
            axs = axs.flatten()
            
            axs_start = 0
            for idx3,month in enumerate(arr.month.values):
                month_name = calendar.month_name[month]
                # break
                data = arr.sel(month=month).values
                data = np.where(np.isnan(land_mask),np.nan,data)
                # data = np.ma.masked_invalid(data)
                for Y in range(data.shape[0]):
                    for X in range(data.shape[1]):
                        if np.isnan(land_mask[Y,X]) or (data[Y,X] == 0):
                            data[Y,X]=np.nan
                            
                v = np.linspace(np.nanmin(data), np.nanmax(data), 8, endpoint=True)
                v=np.array([round(i) for i in v])
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

                if idx1==0:
                    axs[axs_start].set_title(f'{month_name} (1981-2020)',fontsize=12)
                else:
                    axs[axs_start].set_title(f'{month_name} (2000-2019)',fontsize=12)
                    
                cbar = fig.colorbar(im, ax = axs[axs_start], orientation='horizontal')
                cbar.set_label('mm/year', fontsize=10, labelpad=5)
                axs_start+=1
    
            plt.suptitle(f'Accumulated monthly accumulated {var_name}.\nThen averaged over the period. ',fontsize=20)
            if idx1 == 0:
                plt.savefig(f'{save_dir}/monthly_{var_name}_sum_{year_ranges_tuple_1[0]}-{year_ranges_tuple_1[1]}.png')
            else:
                plt.savefig(f'{save_dir}/monthly_{var_name}_sum_{year_ranges_tuple_2[0]}-{year_ranges_tuple_2[1]}.png')

    return('Completed monthly sum plots')
    return(0)

def plot_dzSESR_values_for_case_studies(window, year_ranges_tuple, start_date, num_weeks, pet_or_refet):
    # window=0
    # year_ranges_tuple=year_ranges_tuple_1
    # start_date='2012-05-01'
    # num_weeks=12
    # pet_or_refet='pet'
    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    start = pd.to_datetime(start_date)
    time = pd.date_range(start, periods=num_weeks, freq='W')

    print(f'Plotting the dzSESR values for the weekly time increments {time}.')
    
    data = xr.open_dataset(f'{call.noah_dir}/dzSESR_standardized/dzSESR_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').sel(time=time)
    
    cmap = plt.get_cmap('RdBu')
    
    save_dir = f'{call.fig_dir}/dzSESR_case_studies'
    os.system(f'mkdir -p {save_dir}')

    lon = data.lon.values
    lat = data.lat.values

    fig, axs = plt.subplots(
    nrows = len(time)//3, ncols= len(time)//4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 9),dpi=300)
    
    axs = axs.flatten()

    axs_start = 0

    # min_, max_ = np.nanmin(data[f'dzSESR_{pet_or_refet}'].values), np.nanmax(data[f'dzSESR_{pet_or_refet}'].values)

    '''Mask the ocean values to make sure the colorbar is good'''

    all_values = data[f'dzSESR_{pet_or_refet}'].values
    all_values = np.where(np.isnan(land_mask),np.nan,all_values)
    min_, max_ = np.nanmin(all_values), np.nanmax(all_values)
    v = np.linspace(min_, max_, 100, endpoint=True)
    v = [i for i in v if i < 0] + [0] + [i for i in v if i > 0]
    norm = mcolors.TwoSlopeNorm(vmin=v[0], vcenter=0, vmax=v[-1])

    for idx, date in enumerate(data.time.values):
        # break
        dat = data[f'dzSESR_{pet_or_refet}'][idx,:,:].values

        dat = np.where(np.isnan(land_mask),np.nan,dat)
        # data = np.ma.masked_invalid(data)
        for Y in range(dat.shape[0]):
            for X in range(dat.shape[1]):
                if np.isnan(land_mask[Y,X]) or (dat[Y,X] == 0):
                    dat[Y,X]=np.nan
        

        map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
              llcrnrlon=-128, urcrnrlon=-60, resolution='l')
        x, y = map(*np.meshgrid(lon, lat))

        im = axs[axs_start].contourf(x, y, dat, levels=v, extend='both',
                      transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
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

        axs[axs_start].set_title(f'dzSESR_{pet_or_refet}:  {pd.to_datetime(date)}',fontsize=12)
        axs_start+=1
                           #[left, bottom, width, height]
    cbar_ax = fig.add_axes([0.08, 0.06, .85, .02]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label(f'dzSESR_{pet_or_refet}', fontsize=12, labelpad=5)
    
    plt.suptitle(f'dzSESR_{pet_or_refet} with window size of {window}. ',fontsize=20)
    plt.savefig(f'{save_dir}/dzSESR_{pet_or_refet}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.png')

    return(f'Completed dzSESR_{pet_or_refet}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}')


def find_closest_date(diff_dates):
    for idx,i in enumerate(diff_dates):
        if i.days == 0:
            final = i
            fin_idx = idx
        elif i.days == 1:
            final = i
            fin_idx = idx
        elif i.days == 2:
            final = i
            fin_idx = idx
        elif i.days == 3:
            final = i
            fin_idx = idx
        elif i.days == 4:
            final = i
            fin_idx = idx
        elif i.days == 5:
            final = i
            fin_idx = idx
        elif i.days == 6:
            final = i
            fin_idx = idx
        elif i.days == 7:
            final = i
            fin_idx = idx
        elif i.days == 8:
            final = i
            fin_idx = idx
        elif i.days == 9:
            final = i
            fin_idx = idx
        elif i.days == 10:
            final = i
            fin_idx = idx
    return final, fin_idx

def plot_binary_FD_values_for_case_studies(window, year_ranges_tuple, start_date, num_weeks, pet_or_refet,all_dates_or_only_doy_percentile,s1_or_s2_or_s3):

    # window=0
    # year_ranges_tuple=year_ranges_tuple_1
    # start_date='2012-05-01'
    # num_weeks=12
    # pet_or_refet='pet'
    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    data = xr.open_dataset(f'{call.noah_dir}/dzSESR_FD_step_4_from_{all_dates_or_only_doy_percentile}_percentile/dzSESR_fd_step_4_from_{all_dates_or_only_doy_percentile}_percentile_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
    
    initial_date = pd.to_datetime(start_date)
    # Find the closest date
    diff_dates = ([pd.to_datetime(i) - initial_date for i in data.time.values])
    diff_val = [i  for i in data.time.values]
    
    
    final, fin_idx = find_closest_date(diff_dates)
    
    data2 = data.isel(time=slice(fin_idx,fin_idx+num_weeks))
    
    time = pd.date_range(diff_val[fin_idx], periods=num_weeks, freq='W')
    
    print(f'Plotting the FD values for the weekly time increments {time}.')
    
    
    cmap = plt.get_cmap('binary')  
    
    save_dir = f'{call.fig_dir}/FD_case_studies_{initial_date.year}'
    os.system(f'mkdir -p {save_dir}')
    
    lon = data2.lon.values
    lat = data2.lat.values
    
    fig, axs = plt.subplots(
    nrows = len(time)//3, ncols= len(time)//4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 9),dpi=300)
    
    axs = axs.flatten()
    
    axs_start = 0
    
    # min_, max_ = np.nanmin(data[f'dzSESR_{pet_or_refet}'].values), np.nanmax(data[f'dzSESR_{pet_or_refet}'].values)
    
    '''Mask the ocean values to make sure the colorbar is good'''
    
    all_values = data2[f'fd_{pet_or_refet}_{s1_or_s2_or_s3}'].values
    all_values = np.where(np.isnan(land_mask),np.nan,all_values)
    min_, max_ = np.nanmin(all_values), np.nanmax(all_values)
    v = np.linspace(0, 1, 2, endpoint=True)
    
    for idx, date in enumerate(data2.time.values):
        # break
        dat = data[f'fd_{pet_or_refet}_{s1_or_s2_or_s3}'][idx,:,:].values
    
        dat = np.where(np.isnan(land_mask),np.nan,dat)
        # data = np.ma.masked_invalid(data)
        for Y in range(dat.shape[0]):
            for X in range(dat.shape[1]):
                if np.isnan(land_mask[Y,X]) or (dat[Y,X] == 0):
                    dat[Y,X]=np.nan
        
    
        map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
              llcrnrlon=-128, urcrnrlon=-60, resolution='l')
        x, y = map(*np.meshgrid(lon, lat))
    
        im = axs[axs_start].contourf(x, y, dat, levels=v, extend='both',
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
    
        axs[axs_start].set_title(f'fd_{pet_or_refet}_{s1_or_s2_or_s3}:  {pd.to_datetime(date)}',fontsize=12)
        axs_start+=1
                           #[left, bottom, width, height]
    cbar_ax = fig.add_axes([0.08, 0.06, .85, .02]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label(f'fd_{pet_or_refet}_{s1_or_s2_or_s3}', fontsize=12, labelpad=5)
    
    plt.suptitle(f'fd_{pet_or_refet} {s1_or_s2_or_s3} with window size of {window}.Years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]} ',fontsize=20)
    plt.savefig(f'{save_dir}/fd_{pet_or_refet}_{s1_or_s2_or_s3}_from_{all_dates_or_only_doy_percentile}_percentile_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.png')

    return(f'Completed fd_{pet_or_refet}_{s1_or_s2_or_s3}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}')



def plot_dzSESR_monthly_average_statistics_of_average_value(window):

    
    obs_long = xr.open_dataset(f'{call.noah_dir}/dzSESR_standardized/dzSESR_window_size_{window}_years_1981-2020.nc')
    obs_short = xr.open_dataset(f'{call.noah_dir}/dzSESR_standardized/dzSESR_window_size_{window}_years_2000-2019.nc')
    
    cmap = plt.get_cmap('RdBu')    
    
    save_dir = f'{call.fig_dir}/dzSESR_monthly_statistics'
    os.system(f'mkdir -p {save_dir}')

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    monthly_long = obs_long.groupby('time.year').apply(lambda x: x.groupby('time.month').mean(dim='time'))
    monthly_short = obs_short.groupby('time.year').apply(lambda x: x.groupby('time.month').mean(dim='time'))
    # Define custom month order starting from November (month 11)
    # custom_month_order = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # Reindex the monthly sums to match the custom month order
    # monthly_long = monthly_long.reindex(month=custom_month_order)
    # monthly_short = monthly_short.reindex(month=custom_month_order)
    
    yr_plot1 = monthly_long.mean(dim='year')
    yr_plot2 = monthly_short.mean(dim='year')

    data_arrays1 = [yr_plot1['dzSESR_pet'],yr_plot1['dzSESR_refet'],yr_plot1['SESR_pet'],yr_plot1['SESR_refet']]
    data_arrays2 = [yr_plot2['dzSESR_pet'],yr_plot2['dzSESR_refet'],yr_plot2['SESR_pet'],yr_plot2['SESR_refet']]
   
    lon = yr_plot1.lon.values
    lat = yr_plot1.lat.values

    for idx1,data_arrays in enumerate([data_arrays1,data_arrays2]):

        for idx2, (arr, var_name) in enumerate(zip(data_arrays,['dzSESR_pet','dzSESR_refet','SESR_pet','SESR_refet',])):
            fig, axs = plt.subplots(
                nrows = 4, ncols= 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(17, 15),dpi=300)
                
            axs = axs.flatten()
            
            axs_start = 0
            for idx3,month in enumerate(arr.month.values):
                month_name = calendar.month_name[month]
                # break
                data = arr.sel(month=month).values
                data = np.where(np.isnan(land_mask),np.nan,data)
                # data = np.ma.masked_invalid(data)
                for Y in range(data.shape[0]):
                    for X in range(data.shape[1]):
                        if np.isnan(land_mask[Y,X]) or (data[Y,X] == 0):
                            data[Y,X]=np.nan

                #We want the same number of values on each side of the legend
                v = np.linspace(np.nanmin(data), np.nanmax(data), 6, endpoint=True)
                
                v = [i for i in v if i < 0] + [0] + [i for i in v if i > 0]
                v = np.linspace(v[0], v[0], 6, endpoint=True)
                v = [i for i in v if i < 0] + [0] + [i for i in v if i > 0]
                norm = mcolors.TwoSlopeNorm(vmin=v[0], vcenter=0, vmax=v[-1])
                map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
                      llcrnrlon=-128, urcrnrlon=-60, resolution='l')
                x, y = map(*np.meshgrid(lon, lat))
        
                im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                              transform=ccrs.PlateCarree(), cmap=cmap, norm = norm)
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

                if idx1==0:
                    axs[axs_start].set_title(f'{month_name} (1981-2020)',fontsize=12)
                else:
                    axs[axs_start].set_title(f'{month_name} (2000-2019)',fontsize=12)
                cbar = fig.colorbar(im, ax = axs[axs_start], orientation='horizontal')
                cbar.set_label('dzSESR', fontsize=10, labelpad=5)
                cbar.set_ticks([round(t, 3) for t in cbar.get_ticks()])
                axs_start+=1
    
            plt.suptitle(f'Average monthly {var_name}.\nThen averaged over the period. ',fontsize=20)
            if idx1 == 0:
                plt.savefig(f'{save_dir}/monthly_{var_name}_average_1981-2020.png')
            else:
                plt.savefig(f'{save_dir}/monthly_{var_name}_average_2000-2019.png')

    return('Completed monthly sum plots')

def make_ESR_anomaly_by_year_by_doy(obs,year_ranges_tuple):
    save_esr_dir = f'{call.fig_dir}/ESR_doy_anomaly_by_year_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}'
    os.makedirs(save_esr_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values
    
    obs1 = obs.sel(time=slice(str(year_ranges_tuple[0]),str(year_ranges_tuple[1]))) #ensures none of the missing values from the previous year

    # Group by day of year and calculate the mean
    climatology = obs1.groupby('time.dayofyear').mean('time')
    
    # Subtract the climatology to get the anomaly
    obs1 = obs1.groupby('time.dayofyear') - climatology
    
    # `anomaly` now contains the anomalies by subtracting the mean for each day of the year

    
    for idx,doy in enumerate(range(1,366)):
        for refet_or_PET in ['ESR_refet','ESR_pet']:
        # idx,doy=0,1
            obs_subset = obs1.sel(time=obs1['time.dayofyear']==doy)
    
            save_esr = f'{save_esr_dir}/{refet_or_PET}_doy{doy}.png'
            if os.path.exists(save_esr):
                pass
            else:
                print(f'Working on plotting ESR anomaly for doy {doy}.')
                plot_ESR_anomaly_by_year(obs_subset,year_ranges_tuple, save_esr, land_mask, doy, refet_or_PET)

    return 0

'''add this to the plotUtils.py script'''
def plot_ESR_anomaly_by_year(obs_subset: xr.DataArray, year_ranges_tuple, save_esr, land_mask, doy, refet_or_PET):
    cmap = plt.get_cmap('RdBu')    
    # Define a TwoSlopeNorm that divides the data at 0.5
    # norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    v = np.linspace(np.nanmin(obs_subset[refet_or_PET].values), np.nanmax(obs_subset[refet_or_PET].values), 99, endpoint=True)
    v = [i for i in v if i < 0] + [0] + [i for i in v if i > 0]
    norm = mcolors.TwoSlopeNorm(vmin=v[0] , vcenter=0, vmax=v[-1])
    
    lon = obs_subset.lon.values
    lat = obs_subset.lat.values

    fig, axs = plt.subplots(
    nrows = len(obs_subset.time.values)//5, ncols= 5, 
        subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 20), dpi=150)
    
    axs = axs.flatten()

    axs_start = 0
    for idx, year in enumerate(range(year_ranges_tuple[0],year_ranges_tuple[1])):
        # break
        try:
            data = obs_subset[refet_or_PET].sel(time=str(year)).values[0,:,:]
            data = np.where(np.isnan(land_mask),np.nan,data)
            data = np.ma.masked_invalid(data) #This actually masks the values
            
            map = Basemap(projection='cyl', llcrnrlat=25, urcrnrlat=50,
                  llcrnrlon=-128, urcrnrlon=-60, resolution='l')
            x, y = map(*np.meshgrid(lon, lat))
    
            im = axs[axs_start].contourf(x, y, data, levels=v, extend='both',
                          transform=ccrs.PlateCarree(), cmap=cmap, norm = norm)
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
    
            axs[axs_start].set_title(f'{year_ranges_tuple[0]+idx}',fontsize=12)
    
            axs_start+=1
        except KeyError:
            pass

                            #left, bottom, width, height
    cbar_ax = fig.add_axes([0.10, 0.025, .83, .01]) 
    cbar = fig.colorbar(im, cax = cbar_ax, orientation='horizontal')
    cbar.set_label(f'(Red = Evaporation low and {refet_or_PET.split("ESR_")[-1]} high relative to mean) <<< ------ ESR anomaly ------ >>> (Blue = Evaporation high and {refet_or_PET.split("ESR_")[-1]} high relative to mean)', labelpad=5)
    plt.suptitle(f'{refet_or_PET} - DOY {doy}.',fontsize=30)
    plt.tight_layout()
    plt.savefig(save_esr)
    plt.close()

    return(f'Completed DOY {doy}.')
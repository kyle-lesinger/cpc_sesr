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
import config.STATIC as call

'''Additional steps which may improve the FD identification

1.) If the average RZSM percentile during the FD event is less than the 30th percentile, then the FD continues. Else we remove the event as not being in flash drought
##################################################################################################################################################################

'''

def remove_FD_if_greater_than_selected_RZSM_percentile(dsesr,land_mask, pet_or_refet, min_RZSM_percentile, rzsm):
                
    dsesr[f'fd_{pet_or_refet}_s5'] = dsesr[f'fd_{pet_or_refet}_s4'].copy(deep=True)

    '''Check if within the new growing season'''
    for Y,_ in enumerate(range(dsesr.lat.shape[0])):
        for X,_ in enumerate(range(dsesr.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                # Y,X=40,30
                '''First get the consecutive ones indicating FD'''
                ones = fdutils_2.consecutive_ones(dsesr[f'fd_{pet_or_refet}_s4'][:,Y,X].values)
                '''Next loop through and find if they are within the growing season'''

                for potential_fd in ones:
                    rzsm_mean = np.nanmean(rzsm['rzsm_dtrnd_pct'][potential_fd[0]:potential_fd[-1],Y,X].values)
                    if rzsm_mean >  min_RZSM_percentile:
                        # print('yes')
                        # break
                        dsesr[f'fd_{pet_or_refet}_s5'][potential_fd[0]:potential_fd[-1],Y,X] = 0
                        
    return dsesr


def FD_step_4(window, year_ranges_tuple_1, year_ranges_tuple_2,all_dates_or_only_doy_percentile, min_RZSM_percentile):

    add_text = f'from_{all_dates_or_only_doy_percentile}_percentile'
    
    save_dir = f'{call.noah_dir}/dzSESR_FD_step_4_{add_text}'
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
        # year_ranges_tuple=year_ranges_tuple_1
        # window=0

        rzsm = xr.open_dataset(f'{call.rzsm_clim_perc}/RZSM_percentile_{all_dates_or_only_doy_percentile}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc').load()
            
        save_sesr_percentile = f'{save_dir}/dzSESR_fd_step_4_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        
        if os.path.exists(save_sesr_percentile):
            print(f'Completed dzSESR_fd_step_4_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
            pass
        else:
            #Load data
            open_dzSESR_percentile = f'{call.noah_dir}/dzSESR_FD_step_3_{add_text}/dzSESR_fd_step_3_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
            dsesr = xr.open_dataset(open_dzSESR_percentile).load()

            for pet_or_refet in ['pet','refet']:
                print(f'Working on {pet_or_refet} and finding if {all_dates_or_only_doy_percentile} FD transitioned into longterm drought, Finding if the RZSM percentile is lower than {min_RZSM_percentile} percentile.\nWindow size {window}.\nYears_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')
                dsesr = remove_FD_if_greater_than_selected_RZSM_percentile(dsesr,land_mask, pet_or_refet, min_RZSM_percentile, rzsm)
                # print(np.count_nonzero(add_data_to_fd_index_before))

            dsesr.to_netcdf(save_sesr_percentile)
    return('Completed FD steps 4')

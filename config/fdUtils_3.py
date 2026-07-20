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

1.) If the FD is longer than 
##################################################################################################################################################################
2A) In the criteria, Christian states "The mean change in SESR during the entire lenght of the FD must be less than the 25th percentile... The criterion were taken from the distribution of change in SESR at each grid point for pentads that were encomapassed within the flah dorught event for all years used from the dataset.".

(2A-1) Does this mean we simply take the average of the change in SESR across the span of the drought?
(2A-2) Or does this mean that we take the SESR which was already placed into percentiles, and then we re-create a new distribution looking at only change in SESR during the FD?

Right now, let's just assumed (2A-1) is correct.

############################################################################################################

'''

def remove_FD_if_longer_than_n_weeks_s4(dsesr,land_mask, pet_or_refet, num_weeks_FD_to_longterm_drought):
                
    dsesr[f'fd_{pet_or_refet}_s4'] = dsesr[f'fd_{pet_or_refet}_s3'].copy(deep=True)

    '''Check if within the new growing season'''
    for Y,_ in enumerate(range(dsesr.lat.shape[0])):
        for X,_ in enumerate(range(dsesr.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                # Y,X=40,30
                '''First get the consecutive ones indicating FD'''
                ones = fdutils_2.consecutive_ones(dsesr[f'fd_{pet_or_refet}_s3'][:,Y,X].values)
                '''Next loop through and find if they are within the growing season'''

                for potential_fd in ones:
                    length_fd = potential_fd[-1] - potential_fd[0]
                    if length_fd >= num_weeks_FD_to_longterm_drought:
                        # print('yes')
                        # break
                        dsesr[f'fd_{pet_or_refet}_s4'][potential_fd[0]:potential_fd[-1],Y,X] = 0
                        
    return dsesr


def FD_step_3(window, year_ranges_tuple_1, year_ranges_tuple_2,all_dates_or_only_doy_percentile, num_weeks_FD_to_longterm_drought):

    '''This will actualy create the variable FD_s4 which is to remove FD events which are actually part of a long-term drought event'''
    add_text = f'from_{all_dates_or_only_doy_percentile}_percentile'
    
    save_dir = f'{call.noah_dir}/dzSESR_FD_step_3_{add_text}'
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
        # year_ranges_tuple=year_ranges_tuple_1
        # window=0
        
        save_sesr_percentile = f'{save_dir}/dzSESR_fd_step_3_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        
        if os.path.exists(save_sesr_percentile):
            print(f'Completed dzSESR_fd_step_3_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
            pass
        else:
            #Load data
            open_dzSESR_percentile = f'{call.noah_dir}/dzSESR_FD_step_2_{add_text}/dzSESR_fd_step_2_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
            dsesr = xr.open_dataset(open_dzSESR_percentile).load()

            for pet_or_refet in ['pet','refet']:
                print(f'Working on {pet_or_refet} and finding if {all_dates_or_only_doy_percentile} FD transitioned into longterm drought, If longer than {num_weeks_FD_to_longterm_drought} weeks, then we remove the FD event.\nWindow size {window}.\nYears_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')
                dsesr = remove_FD_if_longer_than_n_weeks_s4(dsesr,land_mask, pet_or_refet, num_weeks_FD_to_longterm_drought)
                # print(np.count_nonzero(add_data_to_fd_index_before))

            dsesr.to_netcdf(save_sesr_percentile)
    return('Completed FD steps 2 and 3')

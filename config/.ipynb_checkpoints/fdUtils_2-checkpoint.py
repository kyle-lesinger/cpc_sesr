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
import config.STATIC as call

'''Step 2 (and 3) of the FD function has already idenfied the following:

if FD is within the Growing season for a particular set of crops AND if the mean change in SESR (dzSESR percentile) over the FD is less than the 25th percentile.

Here are things to do in Step 2 which are not clear in the code given by Christian nor explicity in the 2 differetn papers

1A) Growing season start - much ambiguity, so I downloaded the crop calendar from https://sage.nelson.wisc.edu/data-and-models/datasets/crop-calendar-dataset/netcdf-0-5-degree/

With this crop calendar, if the grid cell has a crop which falls within the planting.start or harvest.end and is under flash drought from step 1, then keep
as flash drought. Else delete as flash drought

##################################################################################################################################################################
2A) In the criteria, Christian states "The mean change in SESR during the entire lenght of the FD must be less than the 25th percentile... The criterion were taken from the distribution of change in SESR at each grid point for pentads that were encomapassed within the flah dorught event for all years used from the dataset.".

(2A-1) Does this mean we simply take the average of the change in SESR across the span of the drought?
(2A-2) Or does this mean that we take the SESR which was already placed into percentiles, and then we re-create a new distribution looking at only change in SESR during the FD?

Right now, let's just assumed (2A-1) is correct.

############################################################################################################

'''

def consecutive_ones(array):
    # Find the indices where the value changes
    changes = np.diff(array, prepend=array[0])
    
    # Identify the start and end of sequences of 1s
    start_indices = np.where(changes == 1)[0]
    end_indices = np.where(changes == -1)[0]
    
    # If the array ends with a sequence of 1s, append the last index
    if array[-1] == 1:
        end_indices = np.append(end_indices, len(array))
    
    # Combine start and end indices
    consecutive_ones = list(zip(start_indices, end_indices))
    return(consecutive_ones)

def return_growing_season():
    '''The current data is formatted so that each day of the year has value for the planting.start and harvest.end.

    We want to find if the grid cell has a flash drought that is currently classified as (1) within either
    fd_pet or fd_refet, and see if it is also within the growing season'''
    
        
def return_unique_names(growing_season):
    names = []
    for var in list(growing_season.keys()):
        names.append(f"{var.split('_')[0]}_{var.split('_')[1]}")
    names = [i for i in names if 'Barley_Winter' not in i] #must remove this because there isn't a planting
    return(np.unique(names))

def return_crops_which_are_in_FD_which_are_in_season(growing_season, Y,X, doys):
    positive_labelled_crops=[]
    
    for idd, crop in enumerate(list(return_unique_names(growing_season))):
        # break
        if len(positive_labelled_crops) == 0:
            # break
            name_to_check_if_complete = crop.split('_')[0]
            if name_to_check_if_complete in positive_labelled_crops:
                break
            else:
                '''Now check if the current FD doy is between plant.start and harvest.end'''
                try:
                    start_grow = int(growing_season[f'{crop}_plant.start'][Y,X].values)
                    end_harvest = int(growing_season[f'{crop}_harvest.end'][Y,X].values)

                    '''Make growing season conditions'''
                    if start_grow < end_harvest:
                        #Something like start plant is 85 and harvest end is 267
                        if ((start_grow < doys[0]) and (doys[0] < end_harvest)) or ((start_grow < doys[-1]) and (doys[-1] < end_harvest)):
                            positive_labelled_crops.append(crop)
                    elif start_grow > end_harvest:
                        #Something like Oats_Winter start plant is 258 and harvest end is 244
                        if ((doys[0] > start_grow) or (doys[0] < end_harvest)) or ((doys[-1] > start_grow) or (doys[0] < end_harvest)):
                            positive_labelled_crops.append(crop)
                    #Don't do Winter crops just yet
                   
                    if (len(positive_labelled_crops) == 1) and ('Winter' in positive_labelled_crops[0]):
                        positive_labelled_crops = []
                except ValueError:
                    #Sometimes there are np.nan dates so this ignores them
                    pass
    return positive_labelled_crops

def find_if_within_the_growing_season_s2(dsesr,land_mask, pet_or_refet, growing_season):
                
    dsesr[f'fd_{pet_or_refet}_s2'] = dsesr[f'fd_{pet_or_refet}_s1'].copy(deep=True)

    '''Check if within the new growing season'''
    for Y,_ in enumerate(range(dsesr.lat.shape[0])):
        for X,_ in enumerate(range(dsesr.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                # Y,X=40,30
                '''First get the consecutive ones indicating FD'''
                ones = consecutive_ones(dsesr[f'fd_{pet_or_refet}_s1'][:,Y,X].values)
                '''Next loop through and find if they are within the growing season'''

                for potential_fd in ones:
                    arr = dsesr[f'fd_{pet_or_refet}_s1'][potential_fd[0]:potential_fd[-1],Y,X]
                    doys=pd.to_datetime(arr.time.values).dayofyear 

                    positive_labelled_crops = return_crops_which_are_in_FD_which_are_in_season(growing_season, Y,X, doys)

                    if len(positive_labelled_crops) >=1:
                        pass
                    else:
                        dsesr[f'fd_{pet_or_refet}_s2'][potential_fd[0]:potential_fd[-1],Y,X] = 0
    return dsesr


def find_if_mean_dzSESR_is_less_than_25_percentile_s3(dsesr,land_mask, pet_or_refet):

    dsesr[f'fd_{pet_or_refet}_s3'] = dsesr[f'fd_{pet_or_refet}_s2'].copy(deep=True)
    for Y,_ in enumerate(range(dsesr.lat.shape[0])):
        for X,_ in enumerate(range(dsesr.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                # Y,X=40,30
                '''First get the consecutive ones indicating FD'''
                ones = consecutive_ones(dsesr[f'fd_{pet_or_refet}_s2'][:,Y,X].values)
                '''Next loop through and find if they are within the growing season'''
                
                for potential_fd in ones:
                    #This was technically already averaged in a previous function, so the values are identical
                    mean_dzSESR_change = dsesr[f'mean_dzsesr_{pet_or_refet}_pct_change'][potential_fd[0]:potential_fd[-1],Y,X].values
                    final_mean = np.nanmean(mean_dzSESR_change) 

                    if final_mean <= 25:
                        pass
                    else:
                        dsesr[f'fd_{pet_or_refet}_s3'][potential_fd[0]:potential_fd[-1],Y,X] = 0

    return dsesr


def re_check_FD_length_after_s3(dsesr,land_mask, pet_or_refet):

    '''Check if within the new growing season'''
    for Y,_ in enumerate(range(dsesr.lat.shape[0])):
        for X,_ in enumerate(range(dsesr.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                ones = consecutive_ones(dsesr[f'fd_{pet_or_refet}_s3'][:,Y,X].values)
    
                for i in ones:
                    if len(np.arange(i[0],i[-1])) < 4:
                        dsesr[f'fd_{pet_or_refet}_s3'][i[0]:i[-1],Y,X] = 0
    return dsesr


def FD_step_2(window, year_ranges_tuple_1, year_ranges_tuple_2,all_dates_or_only_doy_percentile):

    growing_season = xr.open_dataset(f'{call.mask_dir}/crop_calendar/combine_crops_by_harvest_plant_only/all_good_crops_plant_start_and_harvest_end.nc')
    
    min_change_SESR_avg_percentile = 25

    add_text = f'from_{all_dates_or_only_doy_percentile}_percentile'
    
    save_dir = f'{call.noah_dir}/dzSESR_FD_step_2_{add_text}'
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
        # year_ranges_tuple=year_ranges_tuple_1
        # window=0
        
        save_sesr_percentile = f'{save_dir}/dzSESR_fd_step_2_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        
        if os.path.exists(save_sesr_percentile):
            print(f'Completed dzSESR_fd_step_2_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
            pass
        else:
            #Load data
            open_dzSESR_percentile = f'{call.noah_dir}/dzSESR_FD_step_1_{add_text}/dzSESR_fd_step_1_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
            dsesr = xr.open_dataset(open_dzSESR_percentile).load()
            dsesr = dsesr.rename({'fd_pet':'fd_pet_s1'})
            dsesr = dsesr.rename({'fd_refet':'fd_refet_s1'})

            for pet_or_refet in ['pet','refet']:
                print(f'Working on {pet_or_refet} and finding if {all_dates_or_only_doy_percentile} FD step 1 is within the growing season for different crops.\nWindow size {window}.\nYears_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')
                dsesr = find_if_within_the_growing_season_s2(dsesr,land_mask, pet_or_refet, growing_season)
                # print(np.count_nonzero(add_data_to_fd_index_before))
                print(f'Checking if the mean delta sesr change is below the 25th percentile during the FD.')
                dsesr = find_if_mean_dzSESR_is_less_than_25_percentile_s3(dsesr,land_mask, pet_or_refet,)

                #Re-verify that the FD is at least 4 weeks in length
                dsesr = re_check_FD_length_after_s3(dsesr,land_mask, pet_or_refet,)
            dsesr.to_netcdf(save_sesr_percentile)
    return('Completed FD steps 2 and 3')




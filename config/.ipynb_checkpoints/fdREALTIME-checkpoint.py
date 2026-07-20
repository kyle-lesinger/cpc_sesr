#!/usr/bin/env python3

'''Functions for scripts for Flash drought classification.s'''

import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.stats import percentileofscore as pos
import bottleneck as bn
from numba import njit,prange
import config.fdUtils_1 as fdutils_1
import config.fdUtils_2 as fdutils_2
import config.fdUtils_3 as fdutils_3
import config.fdUtils_4 as fdutils_4
import config.dataUtils as dutils
import config.metricUtils as mutils
import config.FDtimeSeriesPlot as fdplot
import config.STATIC as call


def restrict_to_post_climatology(dzSESR_percentile, RZSM_percentile, climatology):
    dzSESR_percentile_af_clim = dzSESR_percentile.sel(time=slice(str(climatology[1]+1),None,None))
    RZSM_percentile_af_clim = RZSM_percentile.sel(time=slice(str(climatology[1]+1),None,None))
    return dzSESR_percentile_af_clim, RZSM_percentile_af_clim
    
def FD_step_1_REALTIME(dzSESR_percentile, climatology, window, all_dates_or_doy, RZSM_percentile, day_of_week_to_analyze_FD, recompute):

    add_text = f'from_{call.all_dates_or_doy}_percentile'

    if day_of_week_to_analyze_FD=='Wednesday':
        num = 2
    elif day_of_week_to_analyze_FD=='Thursday':
        num = 3
    elif day_of_week_to_analyze_FD=='Friday':
        num = 4
    elif day_of_week_to_analyze_FD=='Saturday':
        num = 5
    elif day_of_week_to_analyze_FD=='Sunday':
        num = 6
    elif day_of_week_to_analyze_FD=='Monday':
        num = 0
    elif day_of_week_to_analyze_FD=='Tuesday':
        num = 1
    
    minimum_length_in_weeks = call.minimum_length_in_weeks #Event must last at least 4 weeks in length to be considered a FD
    save_dir = f'{call.noah_dir}/dzSESR_FD_step_1_REALTIME_{add_text}'
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    '''Grab only data after climatology'''
    dzSESR_percentile_af_clim, RZSM_percentile_af_clim = restrict_to_post_climatology(dzSESR_percentile, RZSM_percentile, climatology)
    dzSESR_percentile_af_clim


    date_start, date_end = fdplot.return_date_as_text(RZSM_percentile_af_clim.time.values[0]), fdplot.return_date_as_text(RZSM_percentile_af_clim.time.values[-1])
    save_sesr_percentile = f'{save_dir}/dzSESR_fd_step_1_{add_text}_window{window}_clim{climatology[0]}-{climatology[1]}_|_{date_start}_thru_{date_end}.nc'
    
    if recompute==False:
        return xr.open_dataset(save_sesr_percentile)
    else:
        remove_old_file(save_sesr_percentile)
            
        call.start_REALTIME ### include this to begin FD analysis (nothing before these dates -- we have already completed this analysis)

        dzSESR_percentile_af_clim #should contain all the data
        dzSESR_percentile_af_clim['fd_1_pet'] = dzSESR_percentile_af_clim['EVP'].copy(deep=True)
        dzSESR_percentile_af_clim['fd_1_pet'][:,:,:] = 0

        dzSESR_percentile_af_clim['fd_1_refet'] = dzSESR_percentile_af_clim['fd_1_pet'].copy(deep=True)
        
        day_of_week = dzSESR_percentile_af_clim.sel(time=dzSESR_percentile_af_clim.time.dt.dayofweek == num)
        day_of_week['mean_dzsesr_pet_pct_change'] = day_of_week['fd_1_pet'].copy(deep=True)
        day_of_week['mean_dzsesr_refet_pct_change'] = day_of_week['fd_1_pet'].copy(deep=True)
        
        #because it must be at least 28 days, we can start from the first 28 days
        minimum_length = (call.minimum_length_in_weeks*7) + 7 #must include the plus 7 days because we can account for a single week of above the 40th percentile
        
        add_data_to_fd_index_before = np.array(day_of_week.fd_1_refet.values)
        add_data_to_fd_index_after = add_data_to_fd_index_before.copy()
        
        # print(np.count_nonzero(add_data_to_fd_index_before))
        
        '''Check if SESR is less than 20th and delta_SESR is less than 40th'''
        
        dzsesr_pet_percentile_vals = day_of_week['dzSESR_pct_pet_dtrnd'].values
        dzsesr_refet_percentile_vals = day_of_week['dzSESR_pct_refet_dtrnd'].values

        sesr_pet_percentile_vals = day_of_week['SESR_pct_pet_dtrnd'].values
        sesr_refet_percentile_vals = day_of_week['SESR_pct_refet_dtrnd'].values
        
        # Initialize variables
        min_delta_SESR = call.min_delta_SESR
        min_SESR = call.min_SESR
        
        growing_start = call.growing_start #start month March for growing season FD
        growing_end = call.growing_end #end month November for growing season FD

        vars_to_fill = ['fd_1_pet','fd_1_refet']
        pet_refet_name = ['pet','refet']
        dz_percentile_needed = [dzsesr_pet_percentile_vals,dzsesr_refet_percentile_vals,]
        sesr_percentile_needed = [sesr_pet_percentile_vals,sesr_refet_percentile_vals,]
        

        # Loop through each spatial point
        for idx_num, pet_or_refet in enumerate(vars_to_fill):
            # break
            print(f'\nWorking on {pet_or_refet}.\nUsing percentiles from {all_dates_or_doy}.\nWindow size {window}.\nDates_{date_start}_thru_{date_end}.')
            print(f'\nChecking if length >= {minimum_length_in_weeks} weeks,\nSESR less than {min_SESR}th percentile,\ndzSESR less than {min_delta_SESR}th percentile from climatology_{climatology[0]}-{climatology[1]}.\n#############################################################\n')
            # break
            for Y,_ in enumerate(range(day_of_week.lat.shape[0])):
                for X,_ in enumerate(range(day_of_week.lon.shape[0])):
                    if ~np.isnan(land_mask[Y,X]):
                        #This is the first pass at getting all the data into binary format
                        day_of_week = classify_FD_step_1_REALTIME(Y,X,day_of_week,
                                                                  delta_sesr_percentile_vals=dz_percentile_needed[idx_num],
                                                                  sesr_percentile_vals = sesr_percentile_needed[idx_num],
                                                                  min_delta_SESR=min_delta_SESR,
                                                                  min_SESR=min_SESR,
                                                                  minimum_length_in_weeks=minimum_length_in_weeks,
                                                                  growing_start=growing_start,growing_end=growing_end,
                                                                  vars_to_fill=vars_to_fill,
                                                                  pet_refet_name=pet_refet_name,
                                                                  dz_percentile_needed=dz_percentile_needed,
                                                                  sesr_percentile_needed=sesr_percentile_needed,
                                                                  idx_num=idx_num)
       
        day_of_week.to_netcdf(save_sesr_percentile)

        return(day_of_week)        

def classify_FD_step_1_REALTIME(Y,X,day_of_week,delta_sesr_percentile_vals,sesr_percentile_vals,min_delta_SESR,min_SESR,minimum_length_in_weeks,growing_start,growing_end,vars_to_fill, pet_refet_name, 
                                                dz_percentile_needed,sesr_percentile_needed,idx_num):
    new_flash = 0
    continuous_flash = 0
    was_flash = 0
    was_interrupted = 0
    flash_interruption = 0
    length_flash_drought = 0
    length_flash_drought_new = 0

    # Loop through all weeks of the dataset
    for idx,week_val in enumerate(day_of_week.time.values):
        
        if (idx <=minimum_length_in_weeks) or (idx == len(day_of_week.time.values)-1):
            #We must have a minimum number of observations
            pass
        else:
            # break
                
            if new_flash != 1 and continuous_flash != 1:
                if delta_sesr_percentile_vals[idx,Y,X] <= min_delta_SESR:
                    #Less than the 40th percentile for delta_SESR
                    new_flash = 1
                    length_flash_drought += 1
                    # break
            elif new_flash == 1 and continuous_flash != 1:
                # break
                if delta_sesr_percentile_vals[idx,Y,X] <= min_delta_SESR:
                    continuous_flash = 1
                    new_flash = 0
                    flash_interruption = 0
                    if length_flash_drought == 0:
                        length_flash_drought = length_flash_drought_new + 1
                    else:
                        length_flash_drought += 1
                elif delta_sesr_percentile_vals[idx,Y,X] > min_delta_SESR:
                    if flash_interruption == 0:
                        continuous_flash = 1
                        new_flash = 0
                        flash_interruption = 1
                    elif flash_interruption == 1:
                        new_flash = 0
                        was_flash = 1
                        flash_interruption = 0
                else:
                    new_flash = 0
                    continuous_flash = 1
            elif new_flash != 1 and continuous_flash == 1:
                # break
                if delta_sesr_percentile_vals[idx,Y,X] <= min_delta_SESR:
                    if flash_interruption == 1:
                        if day_of_week[f'SESR_{pet_refet_name[idx_num]}_dtrnd'][idx + 1,Y,X].values < day_of_week[f'SESR_{pet_refet_name[idx_num]}_dtrnd'][idx - 1,Y,X].values:
                            if length_flash_drought == 0:
                                length_flash_drought = length_flash_drought_new + 2
                            else:
                                length_flash_drought += 2
                            flash_interruption = 0
                            was_interrupted = 1
                        else:
                            continuous_flash = 0
                            was_flash = 1
                            flash_interruption = 0
                            new_flash = 1
                            length_flash_drought_new = 1
                    elif flash_interruption == 0:
                        length_flash_drought += 1
                elif delta_sesr_percentile_vals[idx,Y,X] > min_delta_SESR:
                    if flash_interruption == 0:
                        flash_interruption = 1
                    elif flash_interruption == 1:
                        continuous_flash = 0
                        was_flash = 1
                        flash_interruption = 0
                else:
                    if np.isnan(sesr_percentile_vals[idx, Y,X]):
                        continuous_flash = 0
                        was_flash = 1
                        flash_interruption = 0
                    else:
                        continuous_flash = 1
                        flash_interruption = 1
            
            if was_flash == 1:
                # break
                # Check if flash drought is in growing season
                length_flash_drought_adj = length_flash_drought + 1
                fd_n_begin = idx - length_flash_drought_adj
                fd_n_end = idx - 1


                '''For clarity (and because I cannot directly communicate with Jordan Christian about his code, 
                I am making 2 tests. But also just breaking them down into a step by step procedure to produce
                as many possible variations to hopefully attain the methodlogy that he conducted

                So just start with making sure that the length is at least n weeks long and 
                
                current_month = pd.to_datetime(week_val).month
                start_month = pd.to_datetime(day_of_week.time.values[fd_n_begin]).month
                end_month = pd.to_datetime(day_of_week.time.values[fd_n_end]).month

                if (current_month >= growing_start and current_month <= growing_end) or (start_month == 2 and end_month <= 12) :
                    in_grow_seas = 1
                else:
                    in_grow_seas = 0
               
                Check if flash drought ended in drought and is minimum length
                Write sequentially to keep everything in order'''
                if ((length_flash_drought >= minimum_length_in_weeks) and (np.any((sesr_percentile_vals[fd_n_end-1,Y,X]<=min_SESR, sesr_percentile_vals[fd_n_end-2,Y,X] <=min_SESR)))):
                    day_of_week[f'{vars_to_fill[idx_num]}'][fd_n_begin:fd_n_end,Y,X] = 1
                    day_of_week[f'mean_dzsesr_{pet_refet_name[idx_num]}_pct_change'][fd_n_begin:fd_n_end,Y,X] = np.nanmean(day_of_week[f'dzSESR_pct_{pet_refet_name[idx_num]}_dtrnd'][fd_n_begin:fd_n_end,Y,X].values)

                was_flash = 0
                was_interrupted = 0
                length_flash_drought = 0
                length_flash_drought_new = 0
    return(day_of_week)

def remove_old_file(save_sesr_percentile):
    if os.path.exists(save_sesr_percentile):
        os.remove(save_sesr_percentile)
        print(f'Removed {save_sesr_percentile}.')

def FD_step_2_REALTIME(fd_s1,climatology, window, all_dates_or_doy,recompute):

    print('Loading combined growing season crop file.')
    growing_season = xr.open_dataset(f'{call.mask_dir}/crop_calendar/combine_crops_by_harvest_plant_only/all_good_crops_plant_start_and_harvest_end.nc')
    
    min_change_SESR_avg_percentile = call.min_change_SESR_avg_percentile

    add_text = f'from_{all_dates_or_doy}_percentile'
    
    save_dir = f'{call.noah_dir}/dzSESR_FD_step_2_REALTIME_{add_text}'
    os.makedirs(save_dir, exist_ok=True)

    save_dir_s1 = f'{call.noah_dir}/dzSESR_FD_step_1_REALTIME_{add_text}'

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    date_start, date_end = fdplot.return_date_as_text(fd_s1.time.values[0]), fdplot.return_date_as_text(fd_s1.time.values[-1])
    save_sesr_percentile = f'{save_dir}/dzSESR_fd_step_2_{add_text}_window{window}_clim{climatology[0]}-{climatology[1]}_|_{date_start}_thru_{date_end}.nc'
    

    if recompute == False:
        return xr.open_dataset(save_sesr_percentile)
    else:
        remove_old_file(save_sesr_percentile)

        for pet_or_refet in ['pet','refet']:
            # break
            print(f'Working on {pet_or_refet} and finding if {all_dates_or_doy} FD step 1 is within the growing season for different crops.\nWindow size {window}.\nclimatology_{climatology[0]}-{climatology[1]}.')
            fd_s1 = find_if_within_the_growing_season_s2(fd_s1,land_mask, pet_or_refet, growing_season)
            # print(np.count_nonzero(add_data_to_fd_index_before))
            print(f'Checking if the mean delta sesr change is below the {call.min_change_SESR_avg_percentile}th percentile during the FD.')
            fd_s1 = find_if_mean_dzSESR_is_less_than_25_percentile_s3(fd_s1,land_mask, pet_or_refet,)
    
            #Re-verify that the FD is at least 4 weeks in length
            fd_s1 = re_check_FD_length_after_s3(fd_s1,land_mask, pet_or_refet,)
            
        fd_s1.to_netcdf(save_sesr_percentile)
        return(fd_s1)

        
    
    



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

def find_if_within_the_growing_season_s2(fd_s1,land_mask, pet_or_refet, growing_season):
                
    fd_s1[f'fd_2_{pet_or_refet}'] = fd_s1[f'fd_1_{pet_or_refet}'].copy(deep=True)

    '''Check if within the new growing season'''
    for Y,_ in enumerate(range(fd_s1.lat.shape[0])):
        for X,_ in enumerate(range(fd_s1.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                # Y,X=40,30
                '''First get the consecutive ones indicating FD'''
                ones = consecutive_ones(fd_s1[f'fd_1_{pet_or_refet}'][:,Y,X].values)
                '''Next loop through and find if they are within the growing season'''

                for potential_fd in ones:
                    arr = fd_s1[f'fd_1_{pet_or_refet}'][potential_fd[0]:potential_fd[-1],Y,X]
                    doys=pd.to_datetime(arr.time.values).dayofyear 

                    positive_labelled_crops = return_crops_which_are_in_FD_which_are_in_season(growing_season, Y,X, doys)

                    if len(positive_labelled_crops) >=1:
                        pass
                    else:
                        fd_s1[f'fd_2_{pet_or_refet}'][potential_fd[0]:potential_fd[-1],Y,X] = 0
    return fd_s1


def find_if_mean_dzSESR_is_less_than_25_percentile_s3(fd_s1a,land_mask, pet_or_refet):

    fd_s1a[f'fd_3_{pet_or_refet}'] = fd_s1a[f'fd_2_{pet_or_refet}'].copy(deep=True)
    for Y,_ in enumerate(range(fd_s1a.lat.shape[0])):
        for X,_ in enumerate(range(fd_s1a.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                # Y,X=40,30
                '''First get the consecutive ones indicating FD'''
                ones = consecutive_ones(fd_s1a[f'fd_2_{pet_or_refet}'][:,Y,X].values)
                '''Next loop through and find if they are within the growing season'''
                
                for potential_fd in ones:
                    #This was technically already averaged in a previous function, so the values are identical
                    mean_dzSESR_change = fd_s1a[f'mean_dzsesr_{pet_or_refet}_pct_change'][potential_fd[0]:potential_fd[-1],Y,X].values
                    final_mean = np.nanmean(mean_dzSESR_change) 

                    if final_mean <= call.min_change_SESR_avg_percentile:
                        pass
                    else:
                        fd_s1a[f'fd_3_{pet_or_refet}'][potential_fd[0]:potential_fd[-1],Y,X] = 0

    return fd_s1a


def re_check_FD_length_after_s3(fd_s1a,land_mask, pet_or_refet):

    '''Check if within the new growing season'''
    for Y,_ in enumerate(range(fd_s1a.lat.shape[0])):
        for X,_ in enumerate(range(fd_s1a.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                ones = consecutive_ones(fd_s1a[f'fd_3_{pet_or_refet}'][:,Y,X].values)
    
                for i in ones:
                    if len(np.arange(i[0],i[-1])) < 4:
                        fd_s1a[f'fd_3_{pet_or_refet}'][i[0]:i[-1],Y,X] = 0
    return fd_s1a



def remove_FD_if_greater_than_selected_RZSM_percentile(fd_s3,land_mask, pet_or_refet, min_RZSM_percentile, RZSM_percentile):
                
    fd_s3[f'fd_5_{pet_or_refet}'] = fd_s3[f'fd_4_{pet_or_refet}'].copy(deep=True)

    '''Check if within the new growing season'''
    for Y,_ in enumerate(range(fd_s3.lat.shape[0])):
        for X,_ in enumerate(range(fd_s3.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                # Y,X=40,30
                '''First get the consecutive ones indicating FD'''
                ones = fdutils_2.consecutive_ones(fd_s3[f'fd_4_{pet_or_refet}'][:,Y,X].values)
                '''Next loop through and find if they are within the growing season'''

                for potential_fd in ones:
                    rzsm_mean = np.nanmean(RZSM_percentile['RZSM_pct_dtrnd'][potential_fd[0]:potential_fd[-1],Y,X].values)
                    if rzsm_mean >  min_RZSM_percentile:
                        # print('yes')
                        # break
                        fd_s3[f'fd_5_{pet_or_refet}'][potential_fd[0]:potential_fd[-1],Y,X] = 0
                        
    return fd_s3


def FD_step_4_REALTIME(fd_s3, RZSM_percentile, climatology, window, all_dates_or_doy,recompute):

    min_RZSM_percentile = call.min_RZSM_percentile
    add_text = f'from_{all_dates_or_doy}_percentile'
    
    save_dir = f'{call.noah_dir}/dzSESR_FD_step_4_REALTIME_{add_text}'
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    date_start, date_end = fdplot.return_date_as_text(fd_s3.time.values[0]), fdplot.return_date_as_text(fd_s3.time.values[-1])
    save_sesr_percentile = f'{save_dir}/dzSESR_fd_step_4_{add_text}_window{window}_clim{climatology[0]}-{climatology[1]}_|_{date_start}_thru_{date_end}.nc'

    if recompute == False:
        return xr.open_dataset(save_sesr_percentile)
    else:
        remove_old_file(save_sesr_percentile)
        for pet_or_refet in ['pet','refet']:
            print(f'Working on {pet_or_refet} and finding if the RZSM percentile is lower than {min_RZSM_percentile} percentile.\nWindow size {window}.\nclimatology_{climatology[0]}-{climatology[1]}.')
            fd_s3 = remove_FD_if_greater_than_selected_RZSM_percentile(fd_s3,land_mask, pet_or_refet, min_RZSM_percentile, RZSM_percentile)
            # print(np.count_nonzero(add_data_to_fd_index_before))
    
        fd_s3.to_netcdf(save_sesr_percentile)
        return(fd_s3)

def remove_FD_if_longer_than_n_weeks_s4(fd_s2,land_mask, pet_or_refet, num_weeks_FD_to_longterm_drought):
                
    fd_s2[f'fd_4_{pet_or_refet}'] = fd_s2[f'fd_3_{pet_or_refet}'].copy(deep=True)

    '''Check if within the new growing season'''
    for Y,_ in enumerate(range(fd_s2.lat.shape[0])):
        for X,_ in enumerate(range(fd_s2.lon.shape[0])):
            if ~np.isnan(land_mask[Y,X]):
                # Y,X=40,30
                '''First get the consecutive ones indicating FD'''
                ones = fdutils_2.consecutive_ones(fd_s2[f'fd_3_{pet_or_refet}'][:,Y,X].values)
                '''Next loop through and find if they are within the growing season'''

                for potential_fd in ones:
                    length_fd = potential_fd[-1] - potential_fd[0]
                    if length_fd >= num_weeks_FD_to_longterm_drought:
                        # print('yes')
                        # break
                        fd_s2[f'fd_4_{pet_or_refet}'][potential_fd[0]:potential_fd[-1],Y,X] = 0
                        
    return fd_s2


def FD_step_3_REALTIME(fd_s1, climatology, window, all_dates_or_doy, recompute):

    '''This will actualy create the variable FD_s4 all_dates_or_doy is to remove FD events which are actually part of a long-term drought event'''
    add_text = f'from_{all_dates_or_doy}_percentile'

    num_weeks_FD_to_longterm_drought = call.num_weeks_FD_to_longterm_drought
    save_dir = f'{call.noah_dir}/dzSESR_FD_step_3_REALTIME_{add_text}'
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    date_start, date_end = fdplot.return_date_as_text(fd_s1.time.values[0]), fdplot.return_date_as_text(fd_s1.time.values[-1])
    save_sesr_percentile = f'{save_dir}/dzSESR_fd_step_3_{add_text}_window{window}_clim{climatology[0]}-{climatology[1]}_|_{date_start}_thru_{date_end}.nc'
    
    if recompute==False:
        return (xr.open_dataset(save_sesr_percentile))
    else:
        remove_old_file(save_sesr_percentile)
        for pet_or_refet in ['pet','refet']:
            print(f'Working on {pet_or_refet} and finding if {all_dates_or_doy} FD transitioned into longterm drought, If longer than {num_weeks_FD_to_longterm_drought} weeks, then we remove the FD event.\nWindow size {window}.\nclimatology_{climatology[0]}-{climatology[1]}.')
            fd_s1 = remove_FD_if_longer_than_n_weeks_s4(fd_s1,land_mask, pet_or_refet, num_weeks_FD_to_longterm_drought)
            # print(np.count_nonzero(add_data_to_fd_index_before))

        fd_s1.to_netcdf(save_sesr_percentile)
    return(fd_s1)


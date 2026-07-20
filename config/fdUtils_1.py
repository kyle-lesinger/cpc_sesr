#!/usr/bin/env python3

'''Functions for scripts for Flash drought classification.s'''

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

    


def FD_step_1(window, year_ranges_tuple_1, year_ranges_tuple_2,all_days_or_only_doy_percentile, day_of_week_to_analyze_FD = 'Wednesday', ):

    add_text = f'from_{all_days_or_only_doy_percentile}_percentile'

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
    
    minimum_length_in_weeks = 4 #Event must last at least 4 weeks in length to be considered a FD
    save_dir = f'{call.noah_dir}/dzSESR_FD_step_1_{add_text}'
    os.makedirs(save_dir, exist_ok=True)

    land_mask = dutils.load_CONUS_mask()['CONUS_mask'][0,:,:].values

    for year_ranges_tuple in [year_ranges_tuple_1, year_ranges_tuple_2]:
        # year_ranges_tuple=year_ranges_tuple_1
        #window=0
        
        save_sesr_percentile = f'{save_dir}/dzSESR_fd_step_1_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
        
        if os.path.exists(save_sesr_percentile):
            print(f'Completed dzSESR_fd_step_1_{add_text}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc')
            pass
        else:
            #Load data
            open_dzSESR_percentile = f'{call.dzSESR_perc_dir}/dzSESR_percentile_{all_days_or_only_doy_percentile}_window_size_{window}_years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.nc'
            dsesr = xr.open_dataset(open_dzSESR_percentile).load()
            dsesr['fd_pet'] = dsesr['EVP'].copy(deep=True)
            dsesr['fd_pet'][:,:,:] = 0

            dsesr['fd_refet'] = dsesr['fd_pet'].copy(deep=True)
            
            day_of_week = dsesr.sel(time=dsesr.time.dt.dayofweek == num)
            day_of_week['mean_dzsesr_pet_pct_change'] = day_of_week['fd_pet'].copy(deep=True)
            day_of_week['mean_dzsesr_refet_pct_change'] = day_of_week['fd_pet'].copy(deep=True)
            
            #because it must be at least 28 days, we can start from the first 28 days
            minimum_length = (minimum_length_in_weeks*7) + 7 #must include the plus 7 days because we can account for a single week of above the 40th percentile
            
            add_data_to_fd_index_before = np.array(day_of_week.fd_refet.values)
            add_data_to_fd_index_after = add_data_to_fd_index_before.copy()
            
            # print(np.count_nonzero(add_data_to_fd_index_before))
            
            '''Check if SESR is less than 20th and delta_SESR is less than 40th'''
            
            dzsesr_pet_percentile_vals = day_of_week['dzSESR_pct_pet'].values
            dzsesr_refet_percentile_vals = day_of_week['dzSESR_pct_refet'].values

            sesr_pet_percentile_vals = day_of_week['SESR_pct_pet'].values
            sesr_refet_percentile_vals = day_of_week['SESR_pct_refet'].values
            
            # Initialize variables
            min_delta_SESR = 40
            min_SESR = 20
            
            growing_start = 3 #start month March for growing season FD
            growing_end = 11 #end month November for growing season FD

            vars_to_fill = ['fd_pet','fd_refet']
            pet_refet_name = ['pet','refet']
            dz_percentile_needed = [dzsesr_pet_percentile_vals,dzsesr_refet_percentile_vals,]
            sesr_percentile_needed = [sesr_pet_percentile_vals,sesr_refet_percentile_vals,]
            
            
            # Loop through each spatial point
            for idx_num, pet_or_refet in enumerate(vars_to_fill):
                # break
                print(f'Working on {pet_or_refet} using percentiles from {all_days_or_only_doy_percentile}.\nWindow size {window}.\nYears_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.')
                print(f'\nChecking if length >= {minimum_length_in_weeks} weeks,\nSESR less than 20th percentile,\ndzSESR less than 40th percentile for years_{year_ranges_tuple[0]}-{year_ranges_tuple[1]}.\nChecking if drought is within the growing season\nChecking if the mean dzSESR percentile during the FD is less than the 25th percentile.\nChecking if the final SESR value is below the 20th percentile.')
                # break
                for Y,_ in enumerate(range(day_of_week.lat.shape[0])):
                    for X,_ in enumerate(range(day_of_week.lon.shape[0])):
                        if ~np.isnan(land_mask[Y,X]):
                            #This is the first pass at getting all the data into binary format
                            day_of_week = classify_FD_step_1(Y,X,day_of_week,dz_percentile_needed[idx_num],
                                                             sesr_percentile_needed[idx_num],min_delta_SESR,min_SESR,minimum_length_in_weeks,
                                                             growing_start,growing_end,vars_to_fill, pet_refet_name, 
                                                             dz_percentile_needed,sesr_percentile_needed,idx_num)

                                
            day_of_week.to_netcdf(save_sesr_percentile)
    return(f'Completed making step 1 FD for window size {window} for all conditions.')        

def classify_FD_step_1(Y,X,day_of_week,delta_sesr_percentile_vals,sesr_percentile_vals,min_delta_SESR,min_SESR,minimum_length_in_weeks,growing_start,growing_end,vars_to_fill, pet_refet_name, 
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
                if delta_sesr_percentile_vals[idx,Y,X] < min_delta_SESR:
                    #Less than the 40th percentile for delta_SESR
                    new_flash = 1
                    length_flash_drought += 1
                    # break
            elif new_flash == 1 and continuous_flash != 1:
                # break
                if delta_sesr_percentile_vals[idx,Y,X] < min_delta_SESR:
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
                if delta_sesr_percentile_vals[idx,Y,X] < min_delta_SESR:
                    if flash_interruption == 1:
                        if day_of_week[f'SESR_{pet_refet_name[idx_num]}'][idx + 1,Y,X].values < day_of_week[f'SESR_{pet_refet_name[idx_num]}'][idx - 1,Y,X].values:
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
                    day_of_week[f'mean_dzsesr_{pet_refet_name[idx_num]}_pct_change'][fd_n_begin:fd_n_end,Y,X] = np.nanmean(day_of_week[f'dzSESR_pct_{pet_refet_name[idx_num]}'][fd_n_begin:fd_n_end,Y,X].values)

                was_flash = 0
                was_interrupted = 0
                length_flash_drought = 0
                length_flash_drought_new = 0
    return(day_of_week)



# def classify_FD_step_1_v2(Y,X,day_of_week,delta_sesr_percentile_vals,sesr_percentile_vals,min_delta_SESR,min_SESR,minimum_length_in_weeks,growing_start,growing_end,vars_to_fill, pet_refet_name, 
#                                                 dz_percentile_needed,sesr_percentile_needed,idx_num):
#     new_flash = 0
#     continuous_flash = 0
#     was_flash = 0
#     was_interrupted = 0
#     flash_interruption = 0
#     length_flash_drought = 0
#     length_flash_drought_new = 0

#     # Loop through all weeks of the dataset
#     for idx,week_val in enumerate(day_of_week.time.values):
        
#         if (idx <=minimum_length_in_weeks) or (idx == len(day_of_week.time.values)-1):
#             #We must have a minimum number of observations
#             pass
#         else:
#             # break
#             if continuous_flash != 1
#                if (delta_sesr_percentile_vals[idx,Y,X] < min_delta_SESR)
#                    continuous_flash = 1
#                    length_flash_drought += 1
#             else:
#                 if (delta_sesr_percentile_vals[idx,Y,X] < min_delta_SESR)
#                     continuous_flash = 1;
#                     length_flash_drought += 1
#                 else
#                     continuous_flash = 0
#                     was_flash = 1            

#             if was_flash == 1
#                 #Check if flash drought is in growing season
#                 length_flash_drought_adj = length_flash_drought + 1
#                 fd_n_begin = n - length_flash_drought
#                 fd_n_end = n
#                 in_grow_seas = 0
                
#                 # Check if flash drought ended in drought and is minimum length
#                 if (length_flash_drought >= minimum_length_in_weeks) and (np.any(sesr_percentile_vals[fd_n_end-1,Y,X].values <=min_sesr_percentile, sesr_percentile_vals[fd_n_end-2,Y,X].values <=min_sesr_percentile):
#                     day_of_week[f'{vars_to_fill[idx_num]}'][fd_n_begin:fd_n_end,Y,X] = 1
#                     if mean_sesr_change >= min_mean_sesr_change(1)
                    

#                     n_y_begin = mod(fd_n_begin,73);
#                     if n_y_begin == 0; n_y_begin = 73; end
#                     n_y_end = mod(fd_n_end,73);
#                     if n_y_end == 0; n_y_end = 73; end
#                     if n_y_end == 1; n_y_end = 74; end
#                     if n_y_end - n_y_begin > 0
#                         sesr_change_distribution =...
#                             standardized_sesr_change_3d(i,n_y_begin:(n_y_end - 1),:);
#                     else
#                         sesr_change_distribution =...
#                             standardized_sesr_change_3d(i,n_y_begin:73,:);
#                         sesr_change_distribution =...
#                             [sesr_change_distribution,...
#                             standardized_sesr_change_3d(i,1:(n_y_end - 1),:)];
#                     end
#                     min_mean_sesr_change =...
#                         prctile(sesr_change_distribution(:),[25,15,7.5,2.5]); % [25,20,15,10] [30,25,20,15] [25,15,7.5,2.5]
# %                     clear sesr_change_distribution
                    
#                     % Store missing criteria data
#                     if mean_sesr_change >= min_mean_sesr_change(1)
#                     flash_drought_directory{i}.not_fast_enough(y_idx) =...
#                         flash_drought_directory{i}.not_fast_enough(y_idx) +...
#                         1;
#                     end
                    

            
#             if new_flash != 1 and continuous_flash != 1:
#                 if delta_sesr_percentile_vals[idx,Y,X] < min_delta_SESR:
#                     #Less than the 40th percentile for delta_SESR
#                     new_flash = 1
#                     length_flash_drought += 1
#                     # break
#             elif new_flash == 1 and continuous_flash != 1:
#                 # break
#                 if delta_sesr_percentile_vals[idx,Y,X] < min_delta_SESR:
#                     continuous_flash = 1
#                     new_flash = 0
#                     flash_interruption = 0
#                     if length_flash_drought == 0:
#                         length_flash_drought = length_flash_drought_new + 1
#                     else:
#                         length_flash_drought += 1
#                 elif delta_sesr_percentile_vals[idx,Y,X] > min_delta_SESR:
#                     if flash_interruption == 0:
#                         continuous_flash = 1
#                         new_flash = 0
#                         flash_interruption = 1
#                     elif flash_interruption == 1:
#                         new_flash = 0
#                         was_flash = 1
#                         flash_interruption = 0
#                 else:
#                     new_flash = 0
#                     continuous_flash = 1
#             elif new_flash != 1 and continuous_flash == 1:
#                 # break
#                 if delta_sesr_percentile_vals[idx,Y,X] < min_delta_SESR:
#                     if flash_interruption == 1:
#                         if day_of_week[f'SESR_{pet_refet_name[idx_num]}'][idx + 1,Y,X].values < day_of_week[f'SESR_{pet_refet_name[idx_num]}'][idx - 1,Y,X].values:
#                             if length_flash_drought == 0:
#                                 length_flash_drought = length_flash_drought_new + 2
#                             else:
#                                 length_flash_drought += 2
#                             flash_interruption = 0
#                             was_interrupted = 1
#                         else:
#                             continuous_flash = 0
#                             was_flash = 1
#                             flash_interruption = 0
#                             new_flash = 1
#                             length_flash_drought_new = 1
#                     elif flash_interruption == 0:
#                         length_flash_drought += 1
#                 elif delta_sesr_percentile_vals[idx,Y,X] > min_delta_SESR:
#                     if flash_interruption == 0:
#                         flash_interruption = 1
#                     elif flash_interruption == 1:
#                         continuous_flash = 0
#                         was_flash = 1
#                         flash_interruption = 0
#                 else:
#                     if np.isnan(sesr_percentile_vals[idx, Y,X]):
#                         continuous_flash = 0
#                         was_flash = 1
#                         flash_interruption = 0
#                     else:
#                         continuous_flash = 1
#                         flash_interruption = 1
            
#             if was_flash == 1:
#                 # break
#                 # Check if flash drought is in growing season
#                 length_flash_drought_adj = length_flash_drought + 1
#                 fd_n_begin = idx - length_flash_drought_adj
#                 fd_n_end = idx - 1

#                 current_month = pd.to_datetime(week_val).month
#                 start_month = pd.to_datetime(day_of_week.time.values[fd_n_begin]).month
#                 end_month = pd.to_datetime(day_of_week.time.values[fd_n_end]).month

#                 if (current_month >= growing_start and current_month <= growing_end) or (start_month == 2 and end_month <= 12) :
#                     in_grow_seas = 1
#                 else:
#                     in_grow_seas = 0
               
#                 # Check if flash drought ended in drought and is minimum length
#                 '''Write sequentially to keep everything in order'''
#                 if (length_flash_drought >= minimum_length_in_weeks):
#                     if np.any(sesr_percentile_vals[fd_n_end-1,Y,X].values <=20, sesr_percentile_vals[fd_n_end-2,Y,X].values <=20:
#                         day_of_week[f'{vars_to_fill[idx_num]}'][fd_n_begin:fd_n_end,Y,X] = 1
#                         day_of_week[f'mean_dzsesr_{pet_refet_name[idx_num]}_pct_change'][fd_n_begin:fd_n_end,Y,X] = np.nanmean(day_of_week[f'dzSESR_pct_{pet_refet_name[idx_num]}'][fd_n_begin:fd_n_end,Y,X].values)
#                 was_flash = 0
#                 was_interrupted = 0
#                 length_flash_drought = 0
#                 length_flash_drought_new = 0
#     return(day_of_week)


# '''Use these for Step 2

# if (np.nanmean(day_of_week[f'dzSESR_pct_{pet_refet_name[idx_num]}'][fd_n_begin:fd_n_end,Y,X].values) <= 25):
#     if in_grow_seas:
# '''



    
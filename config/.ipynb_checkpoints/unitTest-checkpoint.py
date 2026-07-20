#!/usr/bin/env python3

'''Functions for scripts'''

import xarray as xr
import pandas as pd
import numpy as np
import os
import datetime as dt
from glob import glob
import pickle
import config.STATIC as call

def find_bad_merged_files_and_save_pickle(var):

    '''Returns the bad data saved as a pickle file (also as an object)'''
    def name(file):
        return list(file.keys())[0]
        
    print(f'Checking {var} to find which files have np.nan values across different models and leads')
    file_path = f'{call.gefs_dir}/GEFSv12_merged/bad_files_{var}.pkl'
    os.chdir(f'{call.gefs_dir}/GEFSv12_merged/{var}')
    f_list = sorted(glob('*.nc'))
    print(f'File list is {f_list}')
    bad_data = {}
    
    for f in f_list:
        a=xr.open_dataset(f).load()
        a.close()
        max_grid_cells = (a.X.shape[0] * a.Y.shape[0])
        bad_data[f] = []
        
        for iMod,model in enumerate(a.M.values):
            for iLead,lead in enumerate(a.L.values):
                if np.count_nonzero(np.isnan(a.data[0,iMod,iLead,:,:].values)) >= max_grid_cells:
                    bad_data[f].append(f'M{iMod}_L{iLead}')
        del a
    
                    
    # Save the object to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(bad_data, file)
    

    return bad_data


def find_bad_grib_to_netcdf_files_and_delete(var):

    '''Returns the bad data saved as a pickle file (also as an object)'''
    def name(file):
        return list(file.keys())[0]
        
    print(f'Checking {var} to find which files have np.nan values across different models and leads')
    file_path = f'{call.gefs_dir}/GEFSv12_merged/bad_grib_conversion_files_{var}.pkl'
    os.chdir(f'{call.gefs_dir}/GEFSv12_raw/{var}/regrid')
    f_list = sorted(glob('*.nc'))
    # print(f'File list is {f_list}')
    bad_data = {}
    
    for f in f_list:
        a=xr.open_dataset(f).load()
        a.close()
        max_grid_cells = (a.lon.shape[0] * a.lat.shape[0])
        bad_data[f] = []

        bad_steps=0
        for iStep,step in enumerate(a.step.values):
            while bad_steps < 3:
                if np.count_nonzero(np.isnan(a[name(a)][iStep,:,:].values)) >= max_grid_cells:
                    bad_steps+=1
        if bad_steps <=3:
            continue
        else:
            os.remove(f)
            print(f'Deleted {f}')
    return 0


# def load_bad_pickle_files_and_delete_vals(var):
#     # var='dswrf_sfc'
#     # Load the object from the pickle file
#     bad_data = find_bad_merged_files_and_save_pickle(var)
#     if len(bad_data) == 0:
#         print(f'There is no bad data for variable {var}.')
#     else:
#         for init,missing_mods_and_leads in bad_data.items():
#             fname = f'/glade/work/klesinger/sesr/Data/GEFSv12/GEFSv12_merged/{var}/{init}'
#             # break
#             missing_control = [True if 'M0' in item else False for item in missing_mods_and_leads]

#             if ((len(missing_mods_and_leads) > 30) or (True in missing_control)):
#                 os.remove(fname)
#             else:
#                 '''replace the values from the control'''
#                 op = xr.open_dataset(fname)
#                 op.close()
#                 for kk,vv in bad_files[k].items():
#                     # break
#                     #Now replace the data
#                     bad_mod = int(vv.split('_')[0].split('M')[-1])
#                     bad_lead = int(vv.split('_')[1].split('L')[-1])
#                     '''replace with control'''
#                     op[name(op)][0,bad_mod, bad_lead, :,:] = op[name(op)][0,0, bad_lead, :,:].values
#                 os.system(f'rm {fname}')
#                 op.to_netcdf(fname)
#     return 0


def delete_bad_files_check_2(vars_to_process):
    '''Deletes files from a different script which created a dual variable list (which was incorrect)'''
    for var in vars_to_process:
        os.chdir(f'{call.gefs_dir}/GEFSv12_raw/{var}/regrid')
        f_list = sorted(glob('*.nc'))
        for f in f_list:
            a=xr.open_dataset(f)
            a.close()
            ke = list(a.keys())
            for k in ke:
                if ('avg6h' in k ) or ('avg3h' in k):
                    try:
                        os.remove(f)
                        print(f'Removed {f}.')
                    except FileNotFoundError:
                        continue

def delete_bad_files_check_3(vars_to_process):
    '''Deletes files from a different script which created a dual variable list (which was incorrect)'''
    for var in vars_to_process:
        os.chdir(f'{call.gefs_dir}/GEFSv12_raw/{var}/regrid')
        f_list = sorted(glob('*.nc'))
        for f in f_list:
            a=xr.open_dataset(f)
            a.close()
            try:
                a.forecast_time0
                os.remove(f)
                print(f'Removed {f}.')
            except AttributeError:
                ke = list(a.keys())
                for k in ke:
                    if ('avg6h' in k ) or ('avg3h' in k):
                        try:
                            os.remove(f)
                            print(f'Removed {f}.')
                        except FileNotFoundError:
                            continue
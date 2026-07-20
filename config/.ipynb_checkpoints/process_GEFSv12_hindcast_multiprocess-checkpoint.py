#!/usr/bin/env python3

import os
import datetime as dt
import numpy as np
import xarray as xr
from glob import glob
from multiprocessing import Pool
import sys
import STATIC as call

print(f'directory to save is {call.large_mem_dir}')
#init 
# var = 'dswrf_sfc'
global num_processes, var, raw_dir, save_dir, mask
num_processes = sys.argv[1]
var=sys.argv[3]
raw_dir = f'{call.large_mem_dir}/GEFSv12_raw/{var}'
save_dir = f'{call.large_mem_dir}/GEFSv12_raw/{var}/regrid'
mask =  f'{call.mask_dir}/nldas_0.50.grd'
os.makedirs(save_dir, exist_ok=True)
os.chdir(f'{raw_dir}')
file_list = sorted(glob('*.grib2'), reverse=sys.argv[2].lower() == 'true')


def GEFSv12_grib_to_netcdf(file):
    outName=f"{file.split('grib2')[0]}nc"
    # break
    if os.path.exists(f'{save_dir}/{outName}'):
        pass
    else:
        print(f'Converting {file} to netcdf.')
        #Because there are different files, we need to use the proper opening format for each $
        '''Either cf or pf, depending on the file name'''

        if file.split('_')[-1].split('.')[0] == 'c00':
            try:
                grib_o = xr.open_dataset(file,filter_by_keys={'dataType': 'cf'}) #This works
                grib_o.to_netcdf(outName)
                os.system(f'cdo -remapbil,{mask} {outName} {save_dir}/{outName}')
                os.remove(outName)
                
            except EOFError:
                pass #error caused by realization having no data
            except ValueError:
                pass #error caused by realization having no data
        else:
            try:
                grib_o = xr.open_dataset(file,filter_by_keys={'dataType': 'pf'}) #This works$
                grib_o.to_netcdf(outName)
                os.system(f'cdo -remapbil,{mask} {outName} {save_dir}/{outName}')
                os.remove(outName)
            except EOFError:
                pass
            except ValueError:
                pass #error caused by realization having no data
    return 0



'''Data was previously download from AWS bucket using a different HPC system'''
vars_to_process=call.noah_vars




if __name__ == '__main__':
    p = Pool(int(num_processes))
    p.map(GEFSv12_grib_to_netcdf,file_list)

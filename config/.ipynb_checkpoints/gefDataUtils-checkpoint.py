#!/usr/bin/env python3

'''Functions for scripts'''

import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.signal import detrend
from sklearn.neighbors import KernelDensity
import config.detrendUtils as trendUtils
import datetime as dt
from glob import glob
import pickle
import config.climpredUtils as clim
import config.FDtimeSeriesPlot as fdplot
import config.STATIC as call

def name(file):
    '''Returns the very first variable in an xarray dataset'''
    return(list(file.keys())[0])

def weekly_lead_indexes_required_because_of_mean_being_centered(mean_rolling_length=call.mean_rolling_length):
    if mean_rolling_length == 7:
        return {'Wk1':3,'Wk2':10,'Wk3':17,'Wk4':24,'Wk5':31}
    else:
        assert num_days_in_rolling_mean == 7, 'We have not added any additional code for any rolling mean other than 7.'

def get_init_date_list(path):
    file_list = sorted(glob(path))
    date_list = [i.split('.')[0].split('_')[-1] for i in file_list]
    return date_list

def load_elevation_file():
    '''NLDAS elevation file, source: https://ldas.gsfc.nasa.gov/nldas/elevation'''
    elevationAll=xr.open_dataset(f'{call.mask_dir}/NLDAS_elevation_0.50.nc4')
    elevationMean=elevationAll['NLDAS_elev']
    elevationMean = elevationMean.rename({'lon':'X'})
    elevationMean = elevationMean.rename({'lat':'Y'})
    return elevationMean

#In case there are errors in preprocessing (making duplicate files)
def drop_duplicates_along_all_dims(obj, keep="first"):
    '''This is primarily used to fix any duplicate dates that may be present'''
    all_dims = obj.dims
    indexes = {dim: ~obj.get_index(dim).duplicated(keep=keep) for dim in all_dims}
    return obj.isel(indexes)

def merge_CPC_reference_ET_files():

    print('Merging reference ET files directly from the CPC (we will compare later).')

    eto_dir = f'{call.gefs_dir}/ETo_hindcast_cpc_source/raw_data'
    all_files = glob(f'{eto_dir}/ETrs*')
    template_GEFSv12 = xr.open_dataset(sorted(glob(f'{call.gefs_dir}/ETo_hindcast/*julian*'))[0])
    template_GEFSv12.close()
    
    save_dir = f'{call.gefs_dir}/ETo_hindcast_cpc_source/merged_inits'
    os.makedirs(save_dir, exist_ok=True)
    
    for year in range(2000,2020):
        y=year
        for month in range(1,13):
            m = f'{month:02}'
            for day in range(1,32):
                d = f'{day:02}'
                
                save_file = f'{save_dir}/refet_julian_{y}-{m}-{d}.nc'
                day_subset = [i for i in all_files if f'{y}{m}{d}' in i]
                if len(day_subset) > 0:
                    if os.path.exists(save_file):
                        pass
                    else:
                        fill_template = template_GEFSv12.copy(deep=True)
                        fill_template.ETo_Penman[:,:,:,:,:] = np.nan
                        tempDay = xr.open_dataset(day_subset[0])
                        
                        fill_template['S'] = np.atleast_1d(fdplot.return_date_as_text(pd.to_datetime(tempDay.time.values[0])))
                        '''Now we have all the files, so merge them and save
                        Reference ET is already in mm/day'''
        
                        for iFile, file in enumerate(day_subset):
                            realization = int(file.split('/ETrs')[-1].split('.nc')[0][-2:])-1 #must subtract 1 due to python indexing
                            fill_template['ETo_Penman'][:,realization,:,:,:] = xr.open_dataset(day_subset[0]).refet.values
                            
                        fill_template.to_netcdf(save_file)
                        


def return_date_info(_date,var):
    out_date_create = pd.to_datetime(_date)
    out_date = f'{out_date_create.year}-{out_date_create.month:02}-{out_date_create.day:02}'
    final_out_name = f'{var}_{out_date}.nc'
    return out_date_create, out_date, final_out_name
    
def return_files_by_lead_d10_or_d35(lead_splices,var,_date, template_GEFS_initial):
    '''Used for returning the single GEFSv12 hindcast files to assist with merging'''
    template_GEFS_initial[:,:,:,:,:] = np.nan
    
    all_files_d10 = sorted(glob(f'{lead_splices[0]}_{var}_{_date.year}{_date.month:02}{_date.day:02}*.nc'))
    all_files_d35 = sorted(glob(f'{lead_splices[1]}_{var}_{_date.year}{_date.month:02}{_date.day:02}*.nc'))
    return template_GEFS_initial,all_files_d10, all_files_d35

def check_file_character_length(all_files_d10,all_files_d35):
    '''Finds any anomalies and removes them. Specifically if there is a duplicate and it adds a (2) to the file, this will remove that duplicate'''
    #some files have doubles (rsync error or from HPC when converting)
    file_len = [len(i) for i in all_files_d10]
    mode = max(set(file_len), key=file_len.count)
    #Replace files
    all_files_d10 = [i for i in all_files_d10 if len(i) == mode]
    all_files_d35 = [i for i in all_files_d35 if len(i) == mode]
    return all_files_d10,all_files_d35

def remove_step_in_name_if_present(open_d10):
    try:
        var_name = [i for i in list(open_d10.keys()) if 'step' not in i][0]
    except IndexError:
        pass
    return var_name
    

def add_data_to_empty_array(template_GEFS_initial,open_d10,open_d35,ensemble_number,var_name):
    #First get the dates of the files
    '''Take average of first 7 timesteps if d10 file. I have verified this is correct when looking at HPC'''
    start_ = 0
    steps = {}
    for i in range(35):
        if i ==0:
            steps[str(i)] = open_d10[name(open_d10)][start_:start_+7,:,:].mean(dim=['step']).values
            start_+=7 #needed to begin the next index to keep up with proper dates
        elif (i!=0) and (i<10):
            # step
            steps[str(i)] = open_d10[name(open_d10)][start_:start_+8,:,:].mean(dim=['step']).values #eight total possible values until last time step
            start_+= 8 #Need to add one because we don't want to re-index the same day
        elif i == 10:
            try:
                #Need to take from the first file (time 00:00:00), and combine with d35 files
                s1 = (open_d10[name(open_d10)][-1,:,:] + open_d35[name(open_d10)][0,:,:] + \
                    open_d35[name(open_d35)][1,:,:] + open_d35[name(open_d35)][2,:,:]) /4
                steps[str(i)] = s1
                start_ = 3 #start count over, 4th file is the new date in d35 files
            except IndexError:
                pass
                #Some ensembles have broken members
        elif (i>10) and (i <=34):
            # break
            steps[str(i)] = open_d35[name(open_d35)][start_:start_+4,:,:].mean(dim=['step']).values
            start_+=4

    # print(steps.keys())
    #Add to file
    for step,lead_day in enumerate(steps.keys()):
        # print(step)
        # print(lead_day)
        template_GEFS_initial[:,int(ensemble_number),int(step),:,:] = steps[str(lead_day)]

    return template_GEFS_initial[:,:,:,:,:]



def c_lat_lon_when_opening(file):
    file['lat'] =np.array([25. , 25.5, 26. , 26.5, 27. , 27.5, 28. , 28.5, 29. , 29.5, 30. ,
       30.5, 31. , 31.5, 32. , 32.5, 33. , 33.5, 34. , 34.5, 35. , 35.5,
       36. , 36.5, 37. , 37.5, 38. , 38.5, 39. , 39.5, 40. , 40.5, 41. ,
       41.5, 42. , 42.5, 43. , 43.5, 44. , 44.5, 45. , 45.5, 46. , 46.5,
       47. , 47.5, 48. , 48.5, 49. , 49.5, 50. , 50.5, 51. , 51.5, 52. ,
       52.5, 53. ])

    file['lon'] = np.array([235. , 235.5, 236. , 236.5, 237. , 237.5, 238. , 238.5, 239. ,
       239.5, 240. , 240.5, 241. , 241.5, 242. , 242.5, 243. , 243.5,
       244. , 244.5, 245. , 245.5, 246. , 246.5, 247. , 247.5, 248. ,
       248.5, 249. , 249.5, 250. , 250.5, 251. , 251.5, 252. , 252.5,
       253. , 253.5, 254. , 254.5, 255. , 255.5, 256. , 256.5, 257. ,
       257.5, 258. , 258.5, 259. , 259.5, 260. , 260.5, 261. , 261.5,
       262. , 262.5, 263. , 263.5, 264. , 264.5, 265. , 265.5, 266. ,
       266.5, 267. , 267.5, 268. , 268.5, 269. , 269.5, 270. , 270.5,
       271. , 271.5, 272. , 272.5, 273. , 273.5, 274. , 274.5, 275. ,
       275.5, 276. , 276.5, 277. , 277.5, 278. , 278.5, 279. , 279.5,
       280. , 280.5, 281. , 281.5, 282. , 282.5, 283. , 283.5, 284. ,
       284.5, 285. , 285.5, 286. , 286.5, 287. , 287.5, 288. , 288.5,
       289. , 289.5, 290. , 290.5, 291. , 291.5, 292. , 292.5, 293. ])
    return(file)

def take_mean_if_all_realizations_are_present(all_files_d10, all_files_d35,template_GEFS_initial,all_possible_ensemble_members,var):
    #If all possible files are there, then this is the easy code processing to add data to single file
    soil_layer_depth=3
    wind_height_value=10
    count=0
    for ensemble_number,files in enumerate(zip(all_files_d10,all_files_d35)):
        # break
        # ensemble_number, files = 0, ('d10_tmax_2m_2000010500_c00.nc', 'd35_tmax_2m_2000010500_c00.nc')
        if var != 'soilw_bgrnd' and var != 'hgt_pres' and var != 'ugrd_hgt' and var != 'vgrd_hgt':
            open_d10, open_d35 = xr.open_dataset(files[0]), xr.open_dataset(files[1])
            # var_name = remove_step_in_name_if_present(open_d10)
            var_name = name(open_d10)
    
        elif var == 'hgt_pres':
            #only grab 500mb height
            open_d10 = xr.open_dataset(files[0]).sel(isobaricInhPa=height_value)
            open_d35 = xr.open_dataset(files[1]).sel(isobaricInhPa=height_value)
            # var_name = remove_step_in_name_if_present(open_d10)
            var_name = name(open_d10)
            
        elif var in ['ugrd_hgt','vgrd_hgt']: 
            #only grab 500mb height

            open_d10 = xr.open_dataset(files[0])
            open_d35 = xr.open_dataset(files[1])
            var_name = name(open_d10)
            # var_name = remove_step_in_name_if_present(open_d10)
    
        elif var == 'soilw_bgrnd':
            #Take the sum of the columns
            open_d10= xr.open_dataset(files[0])
            open_d35 = xr.open_dataset(files[1])
            var_name = name(open_d10)
            # var_name = remove_step_in_name_if_present(open_d10)
            #TODO: Take the summation of the first 3 soil layers (0-100cm)
            open_d10 = open_d10[f'{var_name}'][:,0:soil_layer_depth,:,:].sum(dim=['depthBelowLandLayer']).to_dataset()
            open_d35 = open_d35[f'{var_name}'][:,0:soil_layer_depth,:,:].sum(dim=['depthBelowLandLayer']).to_dataset()
            
        # print(var_name)
        open_d10, open_d35 = c_lat_lon_when_opening(open_d10), c_lat_lon_when_opening(open_d35)
        template_GEFS_initial = add_data_to_empty_array(template_GEFS_initial,open_d10,open_d35,ensemble_number,var_name)
        count+=1
    return template_GEFS_initial[:,:,:,:,:], open_d10

def take_mean_if_missing_realizations_by_duplicating_previous_step(all_files_d10, all_files_d35,template_GEFS_initial,all_possible_ensemble_members,var):
    '''Some files just won't download because they are corrupted from source (< 5 across all variables)'''
    soil_layer_depth=3
    wind_height_value=10
    #Some ensembles are missing, split to get the name of ensemble members
    avail_ensemble_members_d10 = [i.split('_')[-1].split('.')[0] for i in all_files_d10]
    avail_ensemble_members_d35 = [i.split('_')[-1].split('.')[0] for i in all_files_d35]
    #if missing only the exact same data
    if len(list(set(avail_ensemble_members_d10).difference(avail_ensemble_members_d35))) == 0:
        #Find a way to append the missing ensemble files with np.nan
        for idx,ensemble in enumerate(all_possible_ensemble_members):
            if ensemble not in avail_ensemble_members_d10:
                pass
            else:
                if var in ['soilw_bgrnd','vgrd_hgt','ugrd_hgt']:
                    outMean='depthBelowLandLayer'
                else:
                    outMean='step'
                idx_num = avail_ensemble_members_d10.index(ensemble)
                
                if var == 'hgt_pres':
                    open_d10=xr.open_dataset(all_files_d10[idx_num]).sel(isobaricInhPa=height_value)
                    open_d35=xr.open_dataset(all_files_d35[idx_num]).sel(isobaricInhPa=height_value)
                elif var == 'soilw_bgrnd':
                    open_d10 = open_d10[f'{var_name}'][:,0:soil_layer_depth,:,:].sum(dim=[outMean]).to_dataset()
                    open_d35 = open_d35[f'{var_name}'][:,0:soil_layer_depth,:,:].sum(dim=[outMean]).to_dataset()
                elif var in ['ugrd_hgt','vgrd_hgt']: 
                    #only grab 500mb height
                    open_d10 = xr.open_dataset(files[0])
                    open_d35 = xr.open_dataset(files[1])
                    var_name = name(open_d10)
                else:
                    open_d10=xr.open_dataset(all_files_d10[idx_num])
                    open_d35=xr.open_dataset(all_files_d35[idx_num]) 
                
                var_name = name(open_d10)

                open_d10, open_d35 = c_lat_lon_when_opening(open_d10), c_lat_lon_when_opening(open_d35)
                template_GEFS_initial = add_data_to_empty_array(template_GEFS_initial,open_d10,open_d35,idx,var_name)

    #if missing different ensemble members -- MISSING different members
    elif len(list(set(avail_ensemble_members_d10).difference(avail_ensemble_members_d35)))>=1:
         #Because there are missing ensemble members that are supposed to be aligned,
         #we must delete those ensemble members
          
         missing_members = list(set(avail_ensemble_members_d10).difference(avail_ensemble_members_d35))
         
         #remove missing members
         out_10 = [i for i in avail_ensemble_members_d10 if i not in missing_members]
         out_35 = [i for i in avail_ensemble_members_d35 if i not in missing_members]
        
         #replace empty file with the control file. Deep learning doesn't like np.nan values
         #there are so few of these missing files that it should be fine
         for i in all_files_d10:
             #for each set of files
             for m in all_possible_ensemble_members:
                 if f'd10_{i[4:24]}_{m}.nc' in all_files_d10:
                     pass
                 #for each set of members
                 #we need to see if the file exists, if not create a blank one
                 else:
                     temp_10=xr.open_dataset(all_files_d10[0]) #make a temporary file as the blank file
                     # temp_10[name(temp_10)][:,:,:] = np.nan
                     temp_10.to_netcdf(f'd10_{i[4:24]}_{m}.nc')
                 
         #replace empty file with the control file. Deep learning doesn't like np.nan values
         #there are so few of these missing files that it should be fine
         for i in all_files_d35:
             # print(i)
             #for each set of files
             for m in all_possible_ensemble_members:
                 # print(m)
                 if f'd35_{i[4:24]}_{m}.nc' in all_files_d35:
                     pass
                 #for each set of members
                 #we need to see if the file exists, if not create a blank one
                 else:
                     temp_10=xr.open_dataset(all_files_d35[0]) #make a temporary file as the blank file
                     # temp_10[name(temp_10)][:,:,:] = np.nan
                     temp_10.to_netcdf(f'd35_{i[4:24]}_{m}.nc')
         #'''Take average of first 7 timesteps if d10 file. I have verified
         #this is correct when looking at HPC'''
         
         for idx,ensemble in enumerate(all_possible_ensemble_members):
             if ensemble not in avail_ensemble_members_d10:
                 pass
             else:
                 idx_num = avail_ensemble_members_d10.index(ensemble)
                 open_d10=xr.open_dataset(all_files_d10[idx_num])
                 open_d35=xr.open_dataset(all_files_d35[idx_num])
                 var_name = [i for i in list(open_d10.keys()) if 'step' not in i][0]
                 
                 open_d10, open_d35 = c_lat_lon_when_opening(open_d10), c_lat_lon_when_opening(open_d35)
                 template_GEFS_initial = add_data_to_empty_array(template_GEFS_initial,open_d10,open_d35,ensemble_number,var_name)       

    return template_GEFS_initial[:,:,:,:,:], open_d10


def julian_date(_date,template_GEFS_initial):
    #Return julian date for anomaly calculation
    a_date_in= template_GEFS_initial.shape[2]
    #get the start date
    a_start_date = pd.to_datetime(_date) 

    a_date_out=[]
    for a_i in range(a_date_in):
        a_date_out.append((a_start_date + np.timedelta64(a_i,'D')).timetuple().tm_yday)

    return(a_date_out)


def julian_date_HINDCAST(_date,num_leads):
    #Return julian date for anomaly calculation
    a_start_date = pd.to_datetime(_date) 

    a_date_out=[]
    for a_i in range(num_leads):
        a_date_out.append((a_start_date + np.timedelta64(a_i,'D')).timetuple().tm_yday)

    return(a_date_out)

def save_file(_date, template_GEFS_initial, var,  save_dir, final_out_name, open_d10):
    #Add julian date
    # julian_list = julian_date(_date,template_GEFS_initial)
    #Instead of replacing the below lines, lets just make it a 35 day lead

    
    GEFS_out = return_netcdf_file(var, open_d10, julian_list, template_GEFS_initial, _date)

def merge_ensemble_members(var):   
    import datetime as dt
    print(f'Working on variable {var} to merge ensemble members.')
    source_dir = f'{call.gefs_dir}/GEFSv12_raw/{var}/regrid'
    os.chdir(source_dir)
    save_dir = f'{call.gefs_dir}/GEFSv12_merged/{var}'
    os.makedirs(save_dir, exist_ok=True)
    
    height_value=250 #choose 500 or 250, must manually select it here only for geopotential height
    soil_layer_depth=3 #0-100cm. Can do 0-2m if number =4

    #GEFS long-term (multi-ensemble) forecasts are only initialized on Wednesdays
    start_date, end_date = dt.date(2000, 1, 1), dt.date(2019, 12, 31)
    dates = [start_date + dt.timedelta(days=d) for d in range(0, end_date.toordinal() - start_date.toordinal() + 1)]
    #from date time, Wednesday is a 2. (Monday is a 0) https://docs.python.org/3/library/datetime.html#datetime.datetime.weekday
    dates = [i for i in dates if i.weekday() ==2]

    '''Manually insert the shape of the file here, we could have it load a previous file but its fine for right now'''
    template_GEFS_initial, lead_splices, all_possible_ensemble_members = np.empty(shape=(1,11,35,57,117)), ['d10','d35'], ['c00', 'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10']

    for _date in dates:
        print(_date)
        # _date=dates[0]
        out_date_create, out_date, final_out_name = return_date_info(_date,var)

        if os.path.exists(f"{save_dir}/{final_out_name}"):
            print(f'Already completed {var} for date {out_date}.')
        else:
            '''Now combine the files, because there are some missing ensemble members (not sure why)
            we need to account for files with 1.) ALL members, 2.) some members.
            It gets more complicated when the missing ensemble members are different between each
            d10 (first 10 days) and d35 (last 25 days), but I have figured it out'''

            # lead_day=0 #keeps up with which index is correct in template_GEFS_initial (resets with each date)

            template_GEFS_initial, all_files_d10, all_files_d35 = return_files_by_lead_d10_or_d35(lead_splices,var,_date, template_GEFS_initial)
            all_files_d10,all_files_d35 = check_file_character_length(all_files_d10,all_files_d35)
            
            #TODO:If all realizations are present
            if (len(all_files_d10) == 11) and (len(all_files_d35) == 11):
                # print('here')
                template_GEFS_initial, open_d10 = take_mean_if_all_realizations_are_present(all_files_d10, all_files_d35,template_GEFS_initial,all_possible_ensemble_members,var)
                # save_file(_date, template_GEFS_initial, var, save_dir, final_out_name,open_d10)
            #If all ensembles are missing, do nothing
            elif (len(all_files_d10) == 0 )and (len(all_files_d35) == 0):
                print(f'{_date} has no values for {var}.')
                pass
            #If there are a differnet number of ensembles between leads
            elif (len(all_files_d10) != 11) or (len(all_files_d35) != 11):
                template_GEFS_initial,open_d10 = take_mean_if_missing_realizations_by_duplicating_previous_step(all_files_d10, all_files_d35,template_GEFS_initial,all_possible_ensemble_members,var)
                # save_file(_date, template_GEFS_initial, var, save_dir, final_out_name,open_d10)

            int_index_list=np.arange(35)
            julian_list = julian_date(_date,template_GEFS_initial)

            # assert ~np.isnan(np.nanmax(template_GEFS_initial)), f'The maximum value for var {var} {_date} is np.nan, this should not occur'
            
            GEFS_out = xr.Dataset(
                data_vars = dict(
                    data = (['S','M','L','Y','X'], template_GEFS_initial[:,:,:,:,:]),
                ),
                coords = dict(
                  
                    X = open_d10.lon.values,
                    Y = open_d10.lat.values,
                    L = julian_list,
                    M = range(template_GEFS_initial.shape[1]),
                    S = np.atleast_1d(pd.to_datetime(_date)),
                ),
                attrs = dict(
                    Description = 'Daily average already computed. All ensembles and Ls in one file')
            )

            GEFS_out.to_netcdf(path = f"{save_dir}/{final_out_name}")
            GEFS_out.close()



def load_REFET_and_hindcast():
    # Include analysis before detrending
    
    obs = xr.open_dataset(f'{call.noah_dir}/RefET_daily_halfdeg.19800101-20240729.nc')
    obs
    
    eto_list = sorted(glob(f'{call.gefs_dir}/ETo_hindcast/refet*.nc'))
    int_list = [i for i in eto_list if 'julian' not in i] 
    julian_list = [i for i in eto_list if 'julian' in i] 
    
    # Open each file separately and store in a list
    datasets_int = [xr.open_dataset(f) for f in int_list]
    print('Loading GEFSv12 hindcast reference ETo Penman Monteith using the integers as leads.')
    gef = xr.concat(datasets_int, dim='S')
    #Cannot chunk without an error .chunk({'S': 1, 'L': 35}) --- doesn't work
    gef = clim.rename_subx_for_climpred(gef)

    return obs,gef


def load_cpc_REFET_and_hindcast():
    # Include analysis before detrending
    
    obs = xr.open_dataset(f'{call.noah_dir}/RefET_daily_halfdeg.19800101-20240729.nc')
    obs
    
    eto_list = sorted(glob(f'{call.gefs_dir}/ETo_hindcast_cpc_source/merged_inits/refet*.nc'))

    julian_list = [i for i in eto_list if 'julian' in i] 
    
    # Open each file separately and store in a list
    datasets_int = [xr.open_dataset(f) for f in julian_list]
    datasets_int = convert_julian_to_integer(datasets_int)
    
    print('Loading GEFSv12 hindcast reference ETo Penman Monteith directly from NOAA CPC calculation using the integers as leads.')
    gef = xr.concat(datasets_int, dim='S')
    #Cannot chunk without an error .chunk({'S': 1, 'L': 35}) --- doesn't work
    gef = clim.rename_subx_for_climpred(gef)

    return obs,gef

def convert_julian_to_integer(datasets_int):
    for idx, i in enumerate(datasets_int):
        # print(f'File is {i}.')
        convert_to_int = i
        convert_to_int['L'] = np.arange(len(convert_to_int.L.values))
        datasets_int[idx] = convert_to_int
    return datasets_int

def load_EVP_and_hindcast():
    # Include analysis before detrending
    obs = xr.open_dataset(f'{call.noah_dir}/et_0.50_degrees.nc')

    et_list = sorted(glob(f'{call.gefs_dir}/GEFSv12_merged/lhtfl_sfc/*.nc'))

    # Open each file separately and store in a list (convert from W/m2 to mm/day)
    print('Loading GEFSv12 hindcast reference EVP using the integers as leads.')
    
    datasets_int = [xr.open_dataset(f) for f in et_list]

    '''Fix the integer leads because they are formulated as julian values'''
    
    for ds in datasets_int:
        ds.close()

    print('Converting julian days to integers for evaporation')
    datasets_int = convert_julian_to_integer(datasets_int)

    gef = xr.concat(datasets_int, dim='S')

    gef = (gef * 0.0864)/2.45

    ''' Source for converting between W/m2 and mm/day for (gef * 0.0864)/2.45
    https://ldas.gsfc.nasa.gov/faq/ldas#:~:text=Tags:-,LDAS,kg%201%20m%20day%20day
    '''
        
    #Cannot chunk without an error .chunk({'S': 1, 'L': 35}) --- doesn't work
    gef = clim.rename_subx_for_climpred(gef)
    
    del datasets_int

    return obs,gef



def load_ESR_and_hindcast(cpc_source):
    # Include analysis before detrending
    obs = xr.open_dataset(f'{call.noah_dir}/esr_rolling_mean_0.50_degrees.nc')
    obs

    if cpc_source == True:
        esr_list = sorted(glob(f'{call.gefs_dir}/ESR_hindcast_cpc_source/*.nc'))
    else:
        esr_list = sorted(glob(f'{call.gefs_dir}/ESR_hindcast/*.nc'))

    ''' Source for converting between W/m2 and mm/day
    https://ldas.gsfc.nasa.gov/faq/ldas#:~:text=Tags:-,LDAS,kg%201%20m%20day%20day
    '''
        
    # Open each file separately and store in a list
    datasets_int = [xr.open_dataset(f) for f in esr_list]
    if cpc_source:
        use = 'CPC'
    else:
        use = 'my own REFET calculation'
    print(f'Loading GEFSv12 hindcast ESR from {use} and using the integers as leads.')
    gef = xr.concat(datasets_int, dim='S')
    #Cannot chunk without an error .chunk({'S': 1, 'L': 35}) --- doesn't work
    gef = clim.rename_subx_for_climpred(gef)

    return obs,gef
    

def change_latent_heat_flux_julian_files_to_integer_files():
    '''This is required because it eats up a lot of memory creating julian listed files'''
    
    os.chdir(f'{call.gefs_dir}/GEFSv12_merged/lhtfl_sfc')
    flist=sorted(glob('*.nc'))
    for f in flist:
        op=xr.open_dataset(f)
        op.close()
        op['L'] = np.arange(35)
        int_out = f'{f[0:10]}int_{f[10:]}'
        op.to_netcdf(int_out)
        os.system(f'rm {f}')
        os.system(f'mv {int_out} {f}')
    return ('Completed changing the latent heat flux files from julian to integer for processing')





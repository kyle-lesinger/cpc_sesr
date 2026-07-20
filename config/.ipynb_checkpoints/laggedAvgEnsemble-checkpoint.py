#!/usr/bin/env python3



import xarray as xr
import pandas as pd
import numpy as np
import os
from climpred import metrics as cmet
from climpred.options import OPTIONS
from climpred import HindcastEnsemble
import climpred
import config.gefDataUtils as gData
import config.dataUtils as dutils
import config.STATIC as call


def lagged_function(fill_array,gef_mean,gef_arr,num_values_TEST,open_f_lead_0_ordinal, beginning_start_date_ordinal):
    
    for iLead,lead in enumerate(gef_mean.lead.values):
        lead_final_ordinal = open_f_lead_0_ordinal + lead
        # date_count = 0 #For understanding the starting date of the file
        fill_array[(lead_final_ordinal - beginning_start_date_ordinal),:,:] += gef_arr[lead, :,:]

        values_present = np.where(gef_arr[lead, :,:] != 0.0,1,0)
        num_values_TEST[(lead_final_ordinal - beginning_start_date_ordinal),:,:] += values_present
        #Checks the number of values actually inserted because not all days will have the same number of subX forecasts (num_values_TEST)
        #so we can take the average
            
    return (fill_array, num_values_TEST)

def lagged_average_ensemble_mean(variable,recompute):
    '''This function will create a dataset that looks the same as NLDAS Noah with a 3-d time,lat,lon time series.
    I have first done a ensemble mean, then we piece each day into a file. We keep a seperate file to know how many
    different realizations occurred. So we will divide the by the number of realizations'''

    assert variable in ['et','refet','cpc_refet'], 'We have only set this code up for et, refet, and cpc_refet for GEFSv12 hindcast'
    
    def name(xr_object):
        return(list(xr_object.keys())[0])

    save_dir = f'{call.gefs_dir}/GEFSv12_hindcast_lagged_averaged_ensemble'
    os.makedirs(save_dir, exist_ok=True)

    save_file = f'{save_dir}/{variable}_lagged_avg_ensemble.nc'

    if recompute:
        print('Recomputing lagged average ensemble')
        if variable == 'et':
            obs, gef = gData.load_EVP_and_hindcast()
        elif variable == 'refet':
            obs, gef = gData.load_REFET_and_hindcast()
        elif variable == 'cpc_refet':
            obs, gef = gData.load_cpc_REFET_and_hindcast()
        
        dates = gef.init.values

        gef = gef.where(gef >= 0, 0)
            
        ending_date = np.datetime64(dates[-1]) + np.timedelta64(gef.lead.values[-1],'D')
        obs = obs.sel(time=slice(dates[0],ending_date)) #manually entered this information for this project
    
        #Get date information from the beginning of the files
        try:
            beginning_start_date_ordinal = pd.to_datetime(dates[0], format= '%Y/%m/%d').toordinal()
        except ValueError:
            beginning_start_date_ordinal = pd.to_datetime(dates[0], format= '%Y-%m-%d').toordinal()
        beginning_start_date = pd.to_datetime(dates[0])
        beginining_index = list(dates).index(dates[0])
    
        fill_array = np.zeros_like(obs[name(obs)].values) #array to fill with data
        num_values_TEST = np.zeros_like(obs[name(obs)].values) #array to fill with integers to determine how 
    
        '''Just take the mean because it will be easier to work with'''
        gef_mean = gef.mean(dim='member')
        gef_mean = gef_mean.fillna(0)
        
        for iDate,date in enumerate(dates):
            # iDate,date = 0,np.datetime64('2000-01-05T00:00:00.000000000')
            if ~isinstance(date, np.datetime64):
                if isinstance(date, str):
                    date=np.datetime64(pd.to_datetime(date))
            try:
                open_f_ordinal = pd.to_datetime(date, format= '%Y/%m/%d').toordinal()
                #Get date info about opened, intialized file
                open_f_lead_0_date = str(pd.to_datetime(gef.init.values[iDate], format= '%Y/%m/%d'))[0:10] #Character string of lead 0 day (day of initialization)
    
            except ValueError:
                open_f_ordinal = pd.to_datetime(date, format= '%Y-%m-%d').toordinal()
                #Get date info about opened, intialized file
                open_f_lead_0_date = str(pd.to_datetime(gef.init.values[iDate], format= '%Y-%m-%d'))[0:10] #Character string of lead 0 day (day of initialization)
                    

            open_f_lead_0_ordinal = pd.to_datetime(open_f_lead_0_date).toordinal()
    
            #Convert to numpy array
            gef_arr = gef_mean[name(gef_mean)][iDate,:,:,:].values
            # print(f'GEFSv12 array mean shape is {gef_arr.shape}')
            fill_array, num_values_TEST = lagged_function(fill_array,gef_mean,gef_arr,num_values_TEST,open_f_lead_0_ordinal, beginning_start_date_ordinal)
    
        
        final_out = fill_array/num_values_TEST #This gives the correct number of values
        final_out[np.isinf(final_out)] = np.nan
        np.nanmax(final_out)
        
        obs[name(obs)][:,:,:] = final_out
        obs.to_netcdf(save_file)

    return f'Completed lagged average ensemble for variable {variable}.'



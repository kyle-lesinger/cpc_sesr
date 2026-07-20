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
import config.gefDataUtils as gData
import config.dataUtils as dUtils
import pyet
import math
import config.conversionUtils as cUtils
import config.STATIC as call

def create_reference_surface_pressure_in_millibars():
    elevation = gData.load_elevation_file()
    atm_pressure_kPA = (101.3 * ((293 - 0.0065 * elevation)/293)**5.26) # kPa --- LOOKS FINE AFTER INSPECTION
    atm_pressure_mb = np.multiply(atm_pressure_kPA,10)
    return elevation,atm_pressure_kPA,atm_pressure_mb

def load_conus_mask_and_rename_lat_lon():
    conus_mask = dUtils.load_CONUS_mask()
    conus_mask = conus_mask['NLDAS_mask']
    conus_mask = conus_mask.rename({'lon':'X'})
    conus_mask = conus_mask.rename({'lat':'Y'})
    return conus_mask

def convert_latitude_to_radians(elevation):
    #Must save latitude in radians for PET calculation
    latitude_radians = xr.zeros_like(elevation)
    for Y in range(latitude_radians.Y.shape[0]):
        for X in range(latitude_radians.X.shape[0]):
            latitude_radians.data[0,Y,X]=pyet.utils.deg_to_rad(latitude_radians.Y[Y].values)
        # print(pyet.deg_to_rad(latitude.latitude[Y].values))
    return latitude_radians

def compute_windSpeed(windU, windV) -> float:
    ''' Returns average daily wind at 10-m height m/s'''
    #Square u and v and take the square root to get windspeed
    windSpeed = np.sqrt((np.square(windU.data) + np.square(windV.data)))
    return (windSpeed)

def wind_speed_2m(ws, z):
    """
    Convert wind speed measured at different heights above the soil
    surface to wind speed at 2 m above the surface, assuming a short grass
    surface.
    Based on FAO equation 47 in Allen et al (1998).
    :param ws: Measured wind speed [m s-1]
    :param z: Height of wind measurement above ground surface [m]
    :return: Wind speed at 2 m above the surface [m s-1]
    :rtype: float
    """
    windspeed = xr.zeros_like(ws)
    windspeed= ws * (4.87 / math.log((67.8 * z) - 5.42))
    #check if it calculated it correctly
    # windspeed.windspeed[0,0,0,10,10].values
    # ws.windspeed[0,0,0,10,10].values
    return(windspeed)


     
def qair2rh (qair, temp, press):
    ##' Convert specific humidity to relative humidity NCEP surface flux data does not have RH
    ##' from Bolton 1980 The computation of Equivalent Potential Temperature 
    ##' \url{http://www.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html}
    ##' @title qair2rh
    ##' @param qair specific humidity, dimensionless (e.g. kg/kg) ratio of water mass / total air mass
    ##' @param temp degrees C
    ##' @param press pressure in mb
    ##' @return rh relative humidity, ratio of actual water mixing ratio to saturation mixing ratio
    ##' @export
    ##' @author David LeBauer   
    # press = np.multiply(press,10) #convert from kPa to millibars
    es =  6.112 * np.exp((17.67 * temp)/(temp + 243.5))
    e = (qair *press) / (0.378 * qair + 0.622)
    rh = e / es
    
    #can have more than 1 for humidity; or less than 0
    rh[rh>1]=1
    rh[rh<0]=0

    return(rh)


def convert_units_for_vars(allVars, atm_pressure_mb):
    
    for iVar,v in enumerate(allVars.keys()):
        # break
        if isinstance(allVars[v], xr.Dataset):
            varValues = allVars[v]
        elif isinstance(allVars[v], xr.DataArray):
            varValues = allVars[v].to_dataset()

        if v in ['tmp_2m', 'tmax_2m', 'tmin_2m']:
            KelvintoCelsius=273.15
            
            temp_kelvin = varValues.data.values
            temp_celsius = temp_kelvin - 273.15
            cUtils.check_temperature_units(temp_kelvin, temp_celsius, v)
            allVars[v] = varValues - KelvintoCelsius #Change to Celsius
            
        elif v in ['dswrf_sfc',]:
            #Since we need to compute a reference ETo, multiply by (1-0.23): 0.23 is the reference albedo
            convert_W_to_MJ = 0.0864
            albedo=0.23
            allVars[v] = (varValues * convert_W_to_MJ)*(1-albedo) #Creates net_shortwave_radiation
        elif v in ['spfh_2m']:

            rel_humidity_from_huss = xr.zeros_like(allVars[v])

            for model_n in range(allVars[v].M.shape[0]):
                for i_lead in range(allVars[v].L.shape[0]):

                    rel_humidity_from_huss[(gData.name(allVars[v]))][0,model_n,i_lead,:,:] = qair2rh(qair=allVars[v][(gData.name(allVars[v]))][0,model_n,i_lead,:,:].values, \
                                 press=atm_pressure_mb[0,:, :].values, \
                                     temp = (allVars['tmp_2m'][(gData.name(allVars['tmp_2m']))][0,model_n, i_lead, :, :].values) - 273.15)
            allVars[v] = rel_humidity_from_huss
        '''Now convert windspeed'''
        
    '''convert_10m_wind_to_2m_windspeed'''
    windSpeed_10m = compute_windSpeed(windU = allVars['ugrd_hgt'], windV = allVars['vgrd_hgt']).to_dataset(name='data')
    # cUtils.check_wind_speed_units(allVars['ugrd_hgt'].data, allVars['vgrd_hgt'].data, windSpeed_10m.data)
    allVars['windspeed'] = wind_speed_2m(ws=windSpeed_10m,z=10)
        
    return allVars








def open_all_files_by_date(_date, fileDir, vars):
    allVars = {}
    for var in vars:
        allVars[var] = xr.open_dataset(f'{fileDir}/{var}/{var}_{_date}.nc')
    return allVars

def inverse_relative_distance_Earth_Sun_dr_var(allVars):
    dr = allVars['dswrf_sfc'].copy(deep=True)
    year_ = pd.to_datetime(allVars['dswrf_sfc'].S.values)[0].year
    
    if year_%4==0:
        num_days=366
    else:
        num_days=365

    for idx,i in enumerate(allVars['dswrf_sfc'].L.values):
        # print(idx,i)
        dr_value = (0.033*np.cos((2*math.pi/num_days)*i)) + 1
        dr.data[0,:,idx,:,:] = dr_value
    return dr, num_days

def solar_declination(allVars,num_days):
    solar_dec= allVars['dswrf_sfc'].copy(deep=True)
    for idx,i in enumerate(solar_dec.L.values):
        # print(idx,i)
        gamma = 2 * np.pi / num_days * (i - 1)

        # Calculate the solar declination
        declination = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) -
                       0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma) -
                       0.002697 * np.cos(3 * gamma) + 0.001480 * np.sin(3 * gamma))

        solar_dec.data[0,:,idx,:,:] = declination
    return solar_dec

def sunset_hour_angle(allVars,latitude_radians,solar_dec):
    sunset = allVars['dswrf_sfc'].copy(deep=True)
    for model_n in range(sunset.M.shape[0]):
        for i_lead in range(sunset.L.shape[0]):
            #count the number of np.nan values
            sunset.data[0,model_n,i_lead,:,:] = np.arccos(np.multiply(-np.tan(latitude_radians[0,:,:].values),np.tan(solar_dec.data[0,model_n,i_lead,:,:])))
    return sunset

def lat_radians_full(allVars,sunset_hr_angle,latitude_radians):
    latitude_radians_full = allVars['dswrf_sfc'].copy(deep=True)
    for model_n in range(sunset_hr_angle.M.shape[0]):
        for i_lead in range(sunset_hr_angle.L.shape[0]):
            latitude_radians_full.data[0,model_n,i_lead,:,:] = latitude_radians[0,:,:]
    return latitude_radians_full

def compute_extraterrestrial_radiation(allVars, solar_constant, dr, sunset_hr_angle,latitude_radians_full,solar_dec):
    radiation_extraterrestrial = allVars['dswrf_sfc'].copy(deep=True)

    part_a = np.multiply((24*60/math.pi) * solar_constant,dr)
    part_b = np.multiply(np.multiply(sunset_hr_angle.data,np.sin(latitude_radians_full.data)),np.sin(solar_dec.data))
    part_c = np.multiply(np.multiply(np.cos(latitude_radians_full.data),np.cos(solar_dec.data)),np.sin(sunset_hr_angle.data))
    
    radiation_extraterrestrial = np.multiply(np.add(part_b,part_c),part_a)
    return radiation_extraterrestrial

def compute_net_longwave_radiation(allVars,avp,elevation,radiation_extraterrestrial):
    #Part A
    tmax_4 = np.power(allVars['tmax_2m']+273.15,4) #Convert back to kelvin
    tmin_4 = np.power(allVars['tmin_2m']+273.15,4)
    boltzman_ = (((tmax_4+tmin_4)/2) * (4.903*10**-9))
    
    #Part B
    e_regularized = np.multiply(np.sqrt(avp),-0.14)
    e_regularized = np.add(0.34,e_regularized)
    
    #Part C
    # Ratio between Rs and Rso
    Rs = allVars['dswrf_sfc']
    
    Rso = Rs.copy(deep=True)
    for model_n in range(Rso.M.shape[0]):
        for i_lead in range(Rso.L.shape[0]):
            #count the number of np.nan values
            Rso.data[0,model_n,i_lead,:,:] = np.multiply(np.add(np.multiply(2e-5,elevation.data[0,:,:]),0.75),radiation_extraterrestrial.data[0,model_n,i_lead,:,:])
            

    rs_rso = (((Rs/Rso)*1.35)-0.35)
    
    net_longwave_radiation = np.multiply(np.multiply(boltzman_.data,e_regularized.data),rs_rso.data).to_dataset()
    return net_longwave_radiation, Rso

def compute_longwave_radiation_and_other_radiation(allVars,latitude_radians,elevation,good_template):
    '''Calculating radiation from reference data https://www.fao.org/3/x0490e/x0490e06.htm'''
    
    dr, num_days = inverse_relative_distance_Earth_Sun_dr_var(allVars)
    solar_dec = solar_declination(allVars,num_days)
    sunset_hr_angle = sunset_hour_angle(allVars,latitude_radians,solar_dec)
                
    solar_constant = 0.0820 #MJ m-2 min-1
    
    latitude_radians_full = lat_radians_full(allVars,sunset_hr_angle,latitude_radians)

    radiation_extraterrestrial = compute_extraterrestrial_radiation(allVars, solar_constant, dr, sunset_hr_angle,latitude_radians_full,solar_dec)

    # pyet.meteo_utils.calc_ea(tmean=None, tmax=None, tmin=None, rhmax=None, rhmin=None, rh=None, ea=None), all in Celsius (temp), rh in (%)
    avp = pyet.meteo_utils.calc_ea(tmean=allVars['tmp_2m'].data,tmax=allVars['tmax_2m'].data,
                                   tmin=allVars['tmin_2m'].data,
                                   rh=np.multiply(allVars['spfh_2m'].data,100)).to_dataset()

    avp['Y'] = good_template.Y.values
    avp['X'] = good_template.X.values
    
    #Now calculate the longwave radiation (net)
    # np.power(2,4)

    net_longwave_radiation, Rso = compute_net_longwave_radiation(allVars,avp,elevation,radiation_extraterrestrial)
    # np.add(net_shortwave_radiation.dswrf,net_shortwave_radiation.dswrf)
    '''We have to subtract longwave in this case since the total energy is positive and is leaving the system'''
    total_radiation = (allVars['dswrf_sfc'].data - net_longwave_radiation.data).to_dataset()
    return allVars, total_radiation, avp, Rso


def compute_reference_ET_FAO56_and_save_no_julian(allVars, total_radiation,atm_pressure_kPA,elevation, latitude_radians, avp, Rso,fileOUT_name):

    output_ETo = xr.zeros_like(allVars['dswrf_sfc']).to_dataset().rename(data='ETo_Penman')

    # mask_radiation = xr.where(~np.isnan(total_radiation.data),total_radiation.data,0) #must mask or else the function will die from ocean values
    # mask_radiation = xr.where(mask_radiation < 100,mask_radiation,0) #must mask or else the function will die from ocean values
    # mask_rel_humidity= xr.where(~np.isnan(allVars['spfh_2m'].data),allVars['spfh_2m'].data,1) #must mask or else the function will die from ocean values

    #Calculate ETo for each model and lead time
    for model_n in range(allVars['dswrf_sfc'].M.shape[0]):
        for i_lead,val in enumerate(allVars['dswrf_sfc'].L.values):
            output_ETo.ETo_Penman[0,model_n,i_lead,:,:]=pyet.combination.pm_fao56(tmean=allVars['tmp_2m'].data[0,model_n,i_lead,:,:],
                                      wind=allVars['windspeed'].data[0,model_n,i_lead,:,:].values,
                                      rn=mask_radiation.data[0,model_n,i_lead,:,:],
                                      tmax=allVars['tmax_2m'].data[0,model_n,i_lead,:,:],
                                      tmin=allVars['tmin_2m'].data[0,model_n,i_lead,:,:],
                                      rh=np.multiply(mask_rel_humidity[0,model_n,i_lead,:,:],100),
                                      pressure=atm_pressure_kPA.data[0,:,:],
                                      elevation=elevation.data[0,:,:],
                                      lat=latitude_radians.data[0,:,:],
                                      ea = avp.data[0,model_n,i_lead,:,:].values,
                                      rso = Rso.data[0,model_n,i_lead,:,:],clip_zero=False)
        
    #Save as a netcdf for later processing
    output_ETo.to_netcdf(path = fileOUT_name, mode ='w')
    return 0

def compute_reference_ET_FAO56_and_save_julian_days(allVars, total_radiation,atm_pressure_kPA,elevation, latitude_radians, avp, Rso,julian_list,fileOUT_name):

    output_ETo = xr.zeros_like(allVars['dswrf_sfc']).rename(data='ETo_Penman')
    output_ETo['L'] = julian_list
        
    mask_radiation = xr.where(~np.isnan(total_radiation.data),total_radiation.data,0) #must mask or else the function will die from ocean values
    mask_radiation = xr.where(mask_radiation < 100,mask_radiation,0) #must mask or else the function will die from ocean values
    mask_rel_humidity= xr.where(~np.isnan(allVars['spfh_2m'].data),allVars['spfh_2m'].data,1) #must mask or else the function will die from ocean values

    #Calculate ETo for each model and lead time
    for model_n in range(allVars['dswrf_sfc'].M.shape[0]):
        for i_lead in range(allVars['dswrf_sfc'].L.shape[0]):
            output_ETo.ETo_Penman[0,model_n,i_lead,:,:]=pyet.combination.pm_fao56(tmean=allVars['tmp_2m'].data[0,model_n,i_lead,:,:],
                                      wind=allVars['windspeed'].data[0,model_n,i_lead,:,:].values,
                                      rn=mask_radiation.data[0,model_n,i_lead,:,:],
                                      tmax=allVars['tmax_2m'].data[0,model_n,i_lead,:,:],
                                      tmin=allVars['tmin_2m'].data[0,model_n,i_lead,:,:],
                                      rh=np.multiply(mask_rel_humidity[0,model_n,i_lead,:,:],100),
                                      pressure=atm_pressure_kPA.data[0,:,:],
                                      elevation=elevation.data[0,:,:],
                                      lat=latitude_radians.data[0,:,:],
                                      ea = avp.data[0,model_n,i_lead,:,:].values,
                                      rso = Rso.data[0,model_n,i_lead,:,:],clip_zero=False)
        
    #Save as a netcdf for later processing
    output_ETo.to_netcdf(path = fileOUT_name, mode ='w')
    return output_ETo

def compute_reference_ET_FAO56_and_save_no_julian(allVars, total_radiation,atm_pressure_kPA,elevation, latitude_radians, avp, Rso,fileOUT_name):

    output_ETo = xr.zeros_like(allVars['dswrf_sfc']).rename(data='ETo_Penman')
    output_ETo['L'] = np.arange(len(output_ETo.L.values))
    mask_radiation = xr.where(~np.isnan(total_radiation.data),total_radiation.data,0) #must mask or else the function will die from ocean values
    mask_radiation = xr.where(mask_radiation < 100,mask_radiation,0) #must mask or else the function will die from ocean values
    mask_rel_humidity= xr.where(~np.isnan(allVars['spfh_2m'].data),allVars['spfh_2m'].data,1) #must mask or else the function will die from ocean values

    #Calculate ETo for each model and lead time
    for model_n in range(allVars['dswrf_sfc'].M.shape[0]):
        for i_lead,val in enumerate(allVars['dswrf_sfc'].L.values):
            output_ETo.ETo_Penman[0,model_n,i_lead,:,:]=pyet.combination.pm_fao56(tmean=allVars['tmp_2m'].data[0,model_n,i_lead,:,:],
                                      wind=allVars['windspeed'].data[0,model_n,i_lead,:,:].values,
                                      rn=mask_radiation.data[0,model_n,i_lead,:,:],
                                      tmax=allVars['tmax_2m'].data[0,model_n,i_lead,:,:],
                                      tmin=allVars['tmin_2m'].data[0,model_n,i_lead,:,:],
                                      rh=np.multiply(mask_rel_humidity[0,model_n,i_lead,:,:],100),
                                      pressure=atm_pressure_kPA.data[0,:,:],
                                      elevation=elevation.data[0,:,:],
                                      lat=latitude_radians.data[0,:,:],
                                      ea = avp.data[0,model_n,i_lead,:,:].values,
                                      rso = Rso.data[0,model_n,i_lead,:,:],clip_zero=False)
        
    #Save as a netcdf for later processing
    output_ETo.to_netcdf(path = fileOUT_name, mode ='w')
    return 0
    
def create_FAO56_refET_GEFSv12(_date):
    fileDir=f'{call.gefs_dir}/GEFSv12_merged'
    vars=call.gefs_vars

    saveDir=f'{call.gefs_dir}/ETo_hindcast'
    os.makedirs(saveDir, exist_ok=True)
    fileOUT_name_julian=f'{saveDir}/refet_julian_{_date}.nc'
    fileOUT_name=f'{saveDir}/refet_{_date}.nc'
    
    tempName=vars[0]

    mod, var = 'EMC', 'refet'

    if (os.path.exists(fileOUT_name) and os.path.exists(fileOUT_name_julian)):
        pass
    else:
        good_template = xr.open_dataset(f'{call.gefs_dir}/GEFSv12_merged/soilw_bgrnd/soilw_bgrnd_2000-01-05.nc')
        print(f'Working on date {_date} to calculate Penman-Monteith ETo for GEFSv12.')
        
        elevation,atm_pressure_kPA,atm_pressure_mb = create_reference_surface_pressure_in_millibars()
        atm_pressure_mb.values
    
        conus_mask = load_conus_mask_and_rename_lat_lon()
        latitude_radians = convert_latitude_to_radians(elevation)
        allVars = open_all_files_by_date(_date, fileDir, vars)
        allVars =  convert_units_for_vars(allVars, atm_pressure_mb)
        allVars, total_radiation, avp, Rso = compute_longwave_radiation_and_other_radiation(allVars,latitude_radians,elevation,good_template)
        julian_list = gData.julian_date(_date,allVars['spfh_2m'].data.values)
        '''Save the file with julian dates already saved'''
        output_ETo = compute_reference_ET_FAO56_and_save_julian_days(allVars, total_radiation,atm_pressure_kPA,elevation, latitude_radians, avp, Rso, julian_list,fileOUT_name_julian)
        
        lead_list = np.arange(len(allVars['dswrf_sfc'].L.values))
        change = allVars['dswrf_sfc']
        change['L'] =lead_list
        allVars['dswrf_sfc'] = change
        compute_reference_ET_FAO56_and_save_no_julian(allVars, total_radiation,atm_pressure_kPA,elevation, latitude_radians, avp, Rso, fileOUT_name)
        print(f'Completed ETo_Penman for date {_date}')


        """
        Notes:
            
        Penman monteith formula for ETo. 
        Parameters
    
        YES - tmean (pandas.Series/xarray.DataArray) – average day temperature [°C]
    
        YES - wind (float/pandas.Series/xarray.DataArray) – mean day wind speed [m/s]
    
        NO - rs (float/pandas.Series/xarray.DataArray, optional) – incoming solar radiation [MJ m-2 d-1]
    
        YES - rn (float/pandas.Series/xarray.DataArray, optional) – net radiation [MJ m-2 d-1]
    
        NO - g (float/pandas.Series/xarray.DataArray, optional) – soil heat flux [MJ m-2 d-1]
    
        YES - tmax (float/pandas.Series/xarray.DataArray, optional) – maximum day temperature [°C]
    
        YES - tmin (float/pandas.Series/xarray.DataArray, optional) – minimum day temperature [°C]
    
        rhmax (float/pandas.Series/xarray.DataArray, optional) – maximum daily relative humidity [%]
    
        rhmin (float/pandas.Series/xarray.DataArray, optional) – mainimum daily relative humidity [%]
    
        rh (float/pandas.Series/xarray.DataArray optional) – mean daily relative humidity [%]
    
        pressure (float/pandas.Series/xarray.DataArray, optional) – atmospheric pressure [kPa]
    
        elevation (float/xarray.DataArray, optional) – the site elevation [m]
    
        lat (float/xarray.DataArray, optional) – the site latitude [rad]
    
        n (float/pandas.Series/xarray.DataArray, optional) – actual duration of sunshine [hour]
    
        nn (float/pandas.Series/xarray.DataArray, optional) – maximum possible duration of sunshine or daylight hours [hour]
    
        rso (float/pandas.Series/xarray.DataArray, optional) – clear-sky solar radiation [MJ m-2 day-1]
    
        a (float, optional) – empirical coefficient for Net Long-Wave radiation [-]
    
        b (float, optional) – empirical coefficient for Net Long-Wave radiation [-]
    
        ea (float/pandas.Series/xarray.DataArray, optional) – actual vapor pressure [kPa]
    
        albedo (float, optional) – surface albedo [-]
    
        kab (float, optional) – coefficient derived from as1, bs1 for estimating clear-sky radiation [degrees].
    
        as1 (float, optional) – regression constant, expressing the fraction of extraterrestrial reaching the earth on overcast days (n = 0) [-]
    
        bs1 (float, optional) – empirical coefficient for extraterrestrial radiation [-]
    
        clip_zero (bool, optional) – if True, replace all negative values with 0.
    
        Returns
    
        Potential evapotranspiration [mm d-1].
    
        #Calculate potential evapotranspiration: 
        #source: https://pypi.org/project/pyet/
        #additional api info: https://pyet.readthedocs.io/en/latest/api/generated/generated/pyet.combination.pm_fao56.html
    
    
    """



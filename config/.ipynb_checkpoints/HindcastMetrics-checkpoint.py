#!/usr/bin/env python3

'''Functions for scripts'''

import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy.stats import percentileofscore as pos
import bottleneck as bn
from numba import njit,prange
from climpred import metrics as cmet
from climpred.options import OPTIONS
from climpred import HindcastEnsemble
import climpred
import config.dataUtils as dutils
import config.gefDataUtils as gData
import config.climpredUtils as clim
import config.STATIC as call
import config.gefPlotUtils as gPlot


def compute_CRPS_on_hindcast_data_reference_ET_only(num_days_in_rolling_mean):
    '''To save on memory, we will do ET separately'''
    print('Calculating REFET - CRPS and pearson correlation over all grid cells. Takes about 80GB of memory')
    obs, gef = gData.load_REFET_and_hindcast()
    obs = obs.rename({'refet':'ETo_Penman'})
    #Perform a rolling mean to match observations
    gef = gef.rolling(init=num_days_in_rolling_mean, center=True).mean()
    wk_lead_indexes = gData.weekly_lead_indexes_required_because_of_mean_being_centered(num_days_in_rolling_mean)
    season_metric = clim.compute_CRPS_and_pearson_correlation(obs,gef)
    '''Now plot CRPS and pearson correlation'''
    gPlot.plot_CRPS_pearson_correlation(season_metric,wk_lead_indexes,varname='refet')
    return 0



def compute_CRPS_on_hindcast_data_ET_only(num_days_in_rolling_mean):

    print('Calculating EVP - CRPS and pearson correlation over all grid cells. Takes about 80GB of memory')
    obs, gef = gData.load_EVP_and_hindcast()
    obs = obs.rename({'EVP':'data'})
    #Perform a rolling mean to match observations
    gef = gef.rolling(init=num_days_in_rolling_mean, center=True).mean()
    wk_lead_indexes = gData.weekly_lead_indexes_required_because_of_mean_being_centered(num_days_in_rolling_mean)
    season_metric = clim.compute_CRPS_and_pearson_correlation(obs,gef)
    '''Now plot CRPS and pearson correlation'''
    gPlot.plot_CRPS_pearson_correlation(season_metric,wk_lead_indexes,varname='et')
    return 0

def compute_CRPS_on_hindcast_data(num_days_in_rolling_mean, varname,cpc_source=True):

    if varname == 'et':
        obs, gef = gData.load_EVP_and_hindcast()
        obs = obs.rename({'EVP':'data'})
    elif varname == 'refet':
        obs, gef = gData.load_REFET_and_hindcast()
        obs = obs.rename({'refet':'data'})
    elif varname == 'ESR_refet':
        obs, gef = gData.load_ESR_and_hindcast()
        obs = obs.rename({varname:'data'})
    elif varname == 'cpc_refet':
        obs, gef = gData.load_cpc_REFET_and_hindcast()
        obs = obs.rename({'refet':'data'})
        gef = gef.rename({'ETo_Penman':'data'})
        # gef['init'] = [pd.to_datetime(i) for i in gef.init.values]
    elif varname == 'cpc_ESR_refet':
        obs, gef = gData.load_ESR_and_hindcast(cpc_source)
        obs = obs.rename({'ESR_refet':'data'})
        
    gef['init'] = [pd.to_datetime(i) for i in gef.init.values]

    obs = obs['data'].to_dataset()
    #Perform a rolling mean to match observations
    gef = gef.rolling(init=num_days_in_rolling_mean, center=True).mean()
    wk_lead_indexes = gData.weekly_lead_indexes_required_because_of_mean_being_centered(num_days_in_rolling_mean)
    season_metric = clim.compute_CRPS_and_pearson_correlation(obs,gef)
    '''Now plot CRPS and pearson correlation'''
    gPlot.plot_CRPS_pearson_correlation(season_metric,wk_lead_indexes,varname)
    return 0
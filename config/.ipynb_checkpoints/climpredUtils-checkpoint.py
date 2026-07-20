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

def rename_subx_for_climpred(file):
    #https://climpred.readthedocs.io/en/stable/examples/subseasonal/daily-subx-example.html
    file = file.rename(S='init')
    file = file.rename(L='lead')
    file["lead"].attrs = {"units": "days"}
    file = file.rename(M='member')
    file = file.rename(X='lon')
    file = file.rename(Y='lat')
    file = file.assign_attrs(lead='days')
    return(file)

def rename_obs_for_climpred(file):
    file = file.rename(X='lon')
    file = file.rename(Y='lat')
    return(file)

def compute_CRPS_and_pearson_correlation(obs,gef):
    #Next we need to compute the pearson correlation with climpred and the CRPS
    #Loop through each season
    season_metric = {}
    
    for iSeason,season in enumerate(['DJF','MAM','JJA','SON']):
        print(f'Computing CRPS and pearson correlation for season {season}')
        # iSeason, season = 0, 'DJF'
        hind = HindcastEnsemble(gef[gData.name(gef)].sel(init=(gef['init.season']==f'{season}'))).add_observations(obs[gData.name(obs)])
        crps = hind.verify(metric='crps', comparison="m2o", dim=["init","member"], alignment="maximize")
        # pearson = hind.verify(metric='pearson_r', comparison="e2o", dim=["init"], alignment="maximize")
        # pearson_pValue = hind.verify(metric='pearson_r_p_value', comparison="e2o", dim=["init"], alignment="maximize")
        season_metric[f'crps_{season}'] = crps
        # season_metric[f'pearson_{season}'] = pearson
        # season_metric[f'pearson_pValue_{season}'] = pearson_pValue
        
    return season_metric
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


def check_temperature_units(temp_kelvin, temp_celsius, v):
    # Convert Kelvin to Celsius
    expected_celsius = temp_kelvin - 273.15
    mask = ~np.isnan(temp_celsius) & ~np.isnan(expected_celsius)
    # Check if the conversion is correct
    assert np.allclose(temp_celsius[mask], expected_celsius[mask]), f"{v} variable kelvin/celsius check failed."
    print('Temperature units check passed.')

def check_wind_speed_units(windU, windV, wind_speed):
    # Calculate expected wind speed
    expected_wind_speed = np.sqrt(np.square(windU) + np.square(windV))
    # Check if the wind speed calculation is correct
    assert np.allclose(wind_speed, expected_wind_speed), "Wind speed calculation error!"
    print("Wind speed units check passed.")

def check_radiation_units(dswrf_sfc, net_shortwave_radiation):
    # Convert W/m^2 to MJ/m^2/day
    expected_net_shortwave_radiation = dswrf_sfc * 0.0864
    # Check if the radiation calculation is correct
    assert np.allclose(net_shortwave_radiation, expected_net_shortwave_radiation), "Radiation units check failed."


#!/bin/bash

module load cdo
module load nco

'''Must do this for NLDAS files prior to additional cdo processing'''
#After we have donwloaded the data and performed the daymean we need to continue processing

#TODO: First we need to regrid it, but the current format is not good for cdo operators so lets fix it
ncap2 -O -s 'lon@units="degrees_east";lon@standard_name="longitude";lat@units="degrees_north";lat@standard_name="latitude"' et_pet_noah_daymean_0.25_degrees.nc et_pet_noah_daymean_0.25_degrees_correct_attributes.nc

# Perform the remapping
cdo remapbil,nldas_0.50.grd et_pet_noah_daymean_0.25_degrees_correct_attributes.nc et_pet_noah_daymean_0.50_degrees.nc

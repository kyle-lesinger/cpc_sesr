#!/bin/bash

vars=("spfh_2m" "tmax_2m" "tmin_2m" "soilw_bgrnd" "dswrf_sfc" "ugrd_hgt" "vgrd_hgt" "tmp_2m")

#cd /glade/derecho/scratch/klesinger/sesr_data/GEFSv12_raw

#for var in "${vars[@]}"; do
#    echo $var number of RAW files:
#    ls $var/ | wc -l;
#done

#echo ########################################################

#for var in "${vars[@]}"; do
#    cd /glade/derecho/scratch/klesinger/sesr_data/GEFSv12_raw/$var/regrid
#    echo "$var number of NETCDF CONUS files (total should be 22946):"
#    ls *.nc | wc -l;
#done

#echo ########################################################
#Check completely processed files

for var in "${vars[@]}"; do
    cd /glade/work/klesinger/sesr/Data/GEFSv12/GEFSv12_merged/$var
    echo "$var number of completed files (total should be 1043):"
    ls *.nc | wc -l;
done

#!/bin/bash

module load conda
conda init
conda activate cfgrib
module load cdo

vars=("lhtfl_sfc" "spfh_2m" "tmax_2m" "tmin_2m" "soilw_bgrnd" "dswrf_sfc" "ugrd_hgt" "vgrd_hgt" "tmp_2m")
vars=("tmax_2m" "tmin_2m" "soilw_bgrnd" "ugrd_hgt" "vgrd_hgt" "tmp_2m"  "dswrf_sfc" "lhtfl_sfc" "spfh_2m" )

for var in "${vars[@]}"; do
    #arg 1 Number of parallel processes; arg 2 if we reverse it or not when processing; arg 3 is variable
    python3 config/process_GEFSv12_hindcast_multiprocess.py "5" "False" $var
done


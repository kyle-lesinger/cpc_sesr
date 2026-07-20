#!/bin/bash

#required modules
module load cdo
module load ncl

#First make the netcdf files and regrid
vars=("spfh_2m" "tmax_2m" "tmin_2m" "soilw_bgrnd" "dswrf_sfc" "ugrd_hgt" "vgrd_hgt" "tmp_2m")
#bash download_data_scripts/download_data.sh
# number = number of multi-processes
python create_GEFSv12_create_wget.py 40

mask=/glade/work/klesinger/sesr/Data/masks/nldas_0.50.grd

year=$1

process_var() {
    var=$1
    yearfunc=$2
    dir=/glade/work/klesinger/sesr/Data/GEFSv12/GEFSv12_raw/$var
    mkdir $dir
    cd $dir || { echo "Directory GEFSv12/$var not found"; return; }
    mkdir regrid
    for file in "*${yearfunc}*.grib2"; do
        nc_file="${file%.grib2}.nc"
        if [[ -f regrid/$nc_file ]]; then
            echo "File $nc_file already exists, skipping."
            rm $nc_file #Just add this because we are saving on space and ensuring no memory leaks
        else
            ncl_convert2nc $file
            cdo remapbil,$mask $nc_file regrid/$nc_file
            rm $nc_file
        fi
    done
}

process_var_reverse() {
    var=$1
    dir=/glade/work/klesinger/sesr/Data/GEFSv12/GEFSv12_raw/$var
    mkdir $dir
    cd $dir || { echo "Directory GEFSv12/$var not found"; return; }
    mkdir regrid

    #Reverse the array
    files=(*grib2)
    reversed_files=()
    length=${#files[@]}

    for (( i==$length-1; i>=0; i-- )); do
        reversed_files+=("${files[$i]}")
    done
    
    for file in "${reversed_files[@]}"; do
        nc_file="${file%.grib2}.nc"
        if [[ -f regrid/$nc_file ]]; then
            echo "File $nc_file already exists, skipping."
        else
            ncl_convert2nc $file
            cdo remapbil,$mask $nc_file regrid/$nc_file
            rm $nc_file
        fi
    done
}

process_var_multi_reverse() {
    var=$1
    dir=/glade/work/klesinger/sesr/Data/GEFSv12/GEFSv12_raw/$var
    mkdir $dir
    cd $dir || { echo "Directory GEFSv12/$var not found"; return; }
    mkdir regrid
    
    #Reverse the array
    files=(*grib2)
    reversed_files=()
    length=${#files[@]}

    for (( i==$length-1; i>=0; i-- )); do
        reversed_files+=("${files[$i]}")
    done
    
    process_batch() {
    for file in "${reversed_files[@]}"; do
        nc_file="${file%.grib2}.nc"
        if [[ -f regrid/$nc_file ]]; then
            echo "File $nc_file already exists, skipping."
            #rm $nc_file
        else
            ncl_convert2nc $file
            cdo remapbil,$mask $nc_file regrid/$nc_file
            rm $nc_file
        fi
    done
    }
    
    # Process files in batches of 10
    batch_size=10
    for (( i=0; i<$length; i+=batch_size )); do
        batch=("${reversed_files[@]:i:batch_size}")
        process_batch "${batch[@]}" &
        ((active_batches++))

        #Wait for background processes to finish
        if (( active_batches >= 1 )); then
            wait
            active_batches=0
        fi
    
    done

#Just wait for any remaining background processes to die.
wait

}

for var in "${vars[@]}"; do
    process_var $var $year &

done
#wait


# for var in "${vars[@]}"; do
#    bash "wget_GEFSv12_${var}.sh" &
#    bash "wget_GEFSv12_${var}_reverse.sh" 
#    wait

    #process_var $var &
#done
#wait

# process_var $var &




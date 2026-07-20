#!/usr/bin/env python3

'''Set paths and constants to run throughout scripts.'''

base_dir = '/glade/work/klesinger/sesr'
USDM_dir = f'{base_dir}/Data/USDM'
noah_dir = f'{base_dir}/Data/Noah'
SESR_clim_dir = f'{noah_dir}/climatology_SESR_delta_mean_std'
SESR_dir = f'{noah_dir}/SESR'
rzsm_clim_perc = f'{noah_dir}/climatology_RZSM_percentile'
dzSESR_dir = f'{noah_dir}/dzSESR_standardized'
dzSESR_perc_dir = f'{noah_dir}/dzSESR_percentile'
doy_trend_dir = f'{noah_dir}/doy_trend'
fig_dir = f'{base_dir}/Figures'
mask_dir = f'{base_dir}/Data/masks'
gefs_dir = f'{base_dir}/Data/GEFSv12'
realtime_NOAH_dir = f'{noah_dir}/dzSESR_percentile_REALTIME'
large_mem_dir = '/glade/derecho/scratch/klesinger/sesr_data'

# Climatologies
long_clim=(1981,2020)
short_clim=(2000,2019)

#Noah variables
noah_vars = ['lhtfl_sfc','soilw_bgrnd','dswrf_sfc','spfh_2m','tmp_2m','ugrd_hgt','vgrd_hgt','tmin_2m','tmax_2m', ]

#GEFSv12 hindcast variables
gefs_vars = ['tmp_2m','lhtfl_sfc','soilw_bgrnd','dswrf_sfc','spfh_2m','ugrd_hgt','vgrd_hgt','tmin_2m','tmax_2m', ]

#GEFSv12 hindcast dates
hind_start = '2000-01-01'
hind_end = '2019-12-31'

# Length of rolling mean to be applied
mean_rolling_length = 7

# Decision of choosing percentiles based on the day of year or by all dates (slightly different results are produced)
# It was decided that choosing "by_doy" is the only appropriate choice. So each day of the year will have its own
# distribution. Meaning that doy 1 (Jan. 1) will have a 0% and 100% value. And Jan. 2 will also have a 0% and 100%, etc. 
all_dates_or_doy = 'by_doy'

# Final date of files available for NLDAS NOAH (must be both RZSM and EVP, PEVPR, and REFET)
noah_start = '1980-01-01'
noah_end = '2024-07-29'

# day of year climatology base (useful for just keeping tracking of day of year things since the year
# 2000 has 366 days
doy_start = '2000-01-01'
doy_end = '2000-12-31'

# Dates for when we consider "REALTIME" to start (outside of the historical distribution)
start_REALTIME = '2021-01-01'

# Selection of how many weeks we want to use between differencing for SESR when computing dzSESR
num_weeks_difference_SESR = 1

# Hindcast frequency (GEFSv12 specific)
init_day_of_week = 'Wednesday'

#################### Window to select values from distribution to create percentile values #################### 
window = 0 #selects ~40 values per grid cell over a 40 year distribution for each day of year

# window = 1 
#****  When window = 1; selects ~120 values per grid cell over a 40 year distribution. But instead of the day
#Before and day after, it is 4 days before and after (because the rolling mean was already applied)
#This method reduces redundancies in data which aren't meaningful

# window = 2
#****  When window = 2; selects ~200 values per grid cell over a 40 year distribution.it is 4 days and 8 days
#before and after (because the rolling mean was already applied)

# window = 3
#****  When window = 3; selects ~280 values per grid cell over a 40 year distribution.it is 4 days and 8 days
# and 12 days before and after (because the rolling mean was already applied)



# Flash drought metrics
#For step 1
minimum_length_in_weeks = 4

#minimum change in dzSESR (change in SESR percentiles)
min_delta_SESR = 40
min_SESR = 20
min_change_SESR_avg_percentile = 25

#Long-term drought transition length (weeks)
num_weeks_FD_to_longterm_drought = 48

#Growing season months
growing_start = 3 #start month March for growing season FD
growing_end = 11 #end month November for growing season FD

#Minimum average RZSM percentile
min_RZSM_percentile = 30
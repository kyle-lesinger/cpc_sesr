#!/usr/bin/env python3

'''Create wget script for GEFSv12 model to setup download of files in parallel.
Source:
https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf

'''
import os
import datetime as dt
import numpy as np
import sys

num_processes = int(sys.argv[1])


#for Easley cluster
home_dir = '/glade/work/klesinger/sesr/Data'
model = 'GEFSv12'
save_wget_scripts_dir = f'{home_dir}/{model}'
save_data_dir = f'{save_wget_scripts_dir}/{model}_raw'
os.makedirs(save_data_dir, exist_ok=True)

#https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf
vars_to_download = ['dswrf_sfc','tmp_2m', 'soilw_bgrnd','tmax_2m','tmin_2m','spfh_2m', 'ugrd_hgt','vgrd_hgt', 'lhtfl_sfc']

#GEFS long-term (multi-ensemble) forecasts are only initialized on Wednesdays
start_date, end_date = dt.date(2000, 1, 1), dt.date(2020, 1, 5)

#dates
dates = [start_date + dt.timedelta(days=d) for d in range(0, end_date.toordinal() - start_date.toordinal() + 1)]

#from date time, Wednesday is a 2. (Monday is a 0) https://docs.python.org/3/library/datetime.html#datetime.datetime.weekday
dates = [i for i in dates if i.weekday() ==2]

#name of ensemble models
ensem_names = ['c00','p01','p02','p03','p04','p05','p06','p07','p08','p09','p10']

#Need specific days for the output leads (either 1.) days 1-10   or 2.) days 10-35
lead_splices = ['Days:1-10','Days:10-35']

url0 = 'https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast'

'''Because the lead splices will have the same name in GEFS when downloading, make sure
to change the name of the save file'''


#TODO: Change to only look at each variable first. Then save the output as a seratre text file

for v_i, var in enumerate(vars_to_download):
    count=0    
    output=[]
    
    for year in range(2000,2020):
        yearly_dates = [i for i in dates if i.year==year]
        daily_months = []
        for idx,i in enumerate(yearly_dates):
            daily_months.append(f'{yearly_dates[idx].month:02}')
        daily_days = []
        for idx,i in enumerate(yearly_dates):
            daily_days.append(f'{yearly_dates[idx].day:02}')
        
        for idx,dates_ in enumerate(yearly_dates):
            ymd = f'{year}{daily_months[idx]}{daily_days[idx]}00'
            url1= f'{url0}/{year}/{ymd}'
            
            for ensemble in ensem_names:
                for splices in lead_splices:
                    if splices == 'Days:1-10':
                        name_out = 'd10'
                    else:
                        name_out = 'd35'

                    url2= f'{url1}/{ensemble}/{splices}'
                    if var == 'hgt_pres' and splices=='Days:1-10':
                        url3= f'{url2}/{var}_abv700mb_{ymd}_{ensemble}.grib2'
                        n1 = url3.split('/')[-1]
                        n2=f'{n1[:8]}_{n1[-20:]}'
                        out_name_new = f'{name_out}_{n2}'
                    else:
                        url3= f'{url2}/{var}_{ymd}_{ensemble}.grib2'
   

                        #split url3 to get the correct name
                        n1 = url3.split('/')[-1]
                        out_name_new = f'{name_out}_{n1}'
                    #GEFSv12
                    command = f"wget -nc --no-proxy -O {save_data_dir}/{var}/{out_name_new} {url3} &"
                    command_idx = f"wget -nc --no-proxy -O {save_data_dir}/{var}/{out_name_new}.idx {url3}.idx &"
                    if count % num_processes == 0:
                        output.append('wait')
                    
                    count+=1
                    output.append(command)
                    output.append(command_idx)

    def save_commands(reverse,output, model, save_wget_scripts_dir, var):
        name = f'wget_{model}'
        if reverse == False:
            np.savetxt(f'{save_wget_scripts_dir}/{name}_{var}.txt',output, fmt="%s")
            os.system(f'mv {save_wget_scripts_dir}/{name}_{var}.txt {save_wget_scripts_dir}/{name}_{var}.sh')
        else:
            output.reverse()
            np.savetxt(f'{save_wget_scripts_dir}/{name}_{var}_reverse.txt',output, fmt="%s")
            os.system(f'mv {save_wget_scripts_dir}/{name}_{var}_reverse.txt {save_wget_scripts_dir}/{name}_{var}_reverse.sh')

        return 'Done'

    save_commands(False,output, model, save_wget_scripts_dir, var)
    save_commands(True,output, model, save_wget_scripts_dir, var)
    

    
                            



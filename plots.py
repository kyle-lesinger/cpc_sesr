#!/usr/bin/env python3

year_ranges_tuple_1=(1981,2020)
year_ranges_tuple_2=(2000,2019)

import config.plotUtils as putils

putils.make_ESR_plots(year_ranges_tuple_1,year_ranges_tuple_2)
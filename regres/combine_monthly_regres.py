import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

var  = ['TREFHT','PRECT']  #   TREFHT, PRECT
exps = ['ghg','aer']               #   ghg, aer, res

#   Load selected data
for exp in exps:
    for v in var:
        mondat = []
        for mon in range(1,13):
            mondattmp = xr.open_dataset('../data/regres/{}_oecd_{}_mon{}_dT_results.nc'.format(
                                        exp,v,str(mon).zfill(2)))
            mondattmp = mondattmp.assign_coords({'mon':mon})
            mondat.append(mondattmp)

        mondat = xr.concat(mondat,'mon')

        mondat.to_netcdf('../data/regres/{}_oecd_{}_mon_dT_results.nc'.format(exp,v))


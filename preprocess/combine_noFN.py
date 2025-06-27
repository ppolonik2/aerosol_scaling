import numpy as np
import xarray as xr
import os
from functools import reduce
import pandas as pd
import subprocess
import random
import datetime

def reorient_netcdf(fp):
    """  
    Function to orient and save netcdf wrt -180,180 longitude (modified from Jacob)
    :param fp: file to be reoriented
    """
    f = xr.open_dataset(fp)
    if np.max(f.coords['lon'] > 180):
        new_lon = [-360.00 + num if num > 180 else num for num in f.coords['lon'].values]
        f = f.assign_coords({'lon':new_lon})
        f.assign_coords(lon=(np.mod(f.lon + 180, 360) - 180))
        f = f.sortby(f.coords['lon'])
        f['lon'].attrs = {'standard_name':'longitude',
                          'long_name':'longitude',
                          'units':'degrees_east',
                          'axis':'X'}
    return f

def dateshift_netCDF(ds):
    """
    Function to shift and save netcdf with dates at midpoint of month.
    :param ds: dataarray to be reoriented
    """
    f = ds
    if np.unique(ds.indexes['time'].day)[0]==1 & len(np.unique(ds.indexes['time'].day))==1:
        new_time = ds.indexes['time']-datetime.timedelta(days=16)
        f = f.assign_coords({'time':new_time})
    return f


#   Figure out which ensembles are available for all desired variables
variables = ['TREFHT','PRECT']
monthly_or_annual = 'monthly'
exp = 'ghg' #   ghg, aer, ee, xaer
Nsubset = 5

datdir = '../data/sflens/{}/'.format(exp)
allfiles = [f for f in os.listdir(datdir) if f.endswith('.nc')]

ensavail = {}
for v in variables:
    vfiles = [f for f in allfiles if f.startswith(v)]
    uens   = np.unique([vf.split('.')[1] for vf in vfiles])
    ensavail[v] = uens

#   Find unique ensembles that are available everywhere using stackoverflow magic
#   https://stackoverflow.com/questions/69525490/get-intersection-of-all-values-from-a-dictionary-with-lists
enss = reduce(set.intersection, map(set, ensavail.values()))

#   Loop through and gather full (not meaned) variables
if Nsubset>0:
    enss = random.sample(list(enss),Nsubset)
vdatfull = []
for v in variables:
    ensdat = []
    for ens in enss:
        tmpdat = reorient_netcdf('{}/{}.{}.{}.nc'.format(datdir,v,ens,monthly_or_annual))[v]
        tmpdat = dateshift_netCDF(tmpdat)
        ensdat.append(tmpdat)

    #   Combine ensembles
    ensdat = xr.concat(ensdat,pd.Index(enss,name='ens'))
    vdatfull.append(ensdat)

#   Do it again for the means
vdatmean = []
for v in variables:
    ensdat = []
    for ens in enss:
        tmpdat = xr.open_dataset('{}/{}.{}.{}.mean.nc'.format(datdir,v,ens,monthly_or_annual))[v]
        tmpdat = dateshift_netCDF(tmpdat)
        tmpdat = tmpdat.squeeze().drop(['lat','lon'])
        tmpdat.name = v+'_mean'
        ensdat.append(tmpdat)

    #   Combine ensembles
    ensdat = xr.concat(ensdat,pd.Index(enss,name='ens'))
    vdatmean.append(ensdat)
   
vdat = xr.merge(vdatfull+vdatmean)
vdat.attrs = {}

fullvar = variables

#   Cut off 2100 because it's only Jan
vdat = vdat.sel(time=vdat.time[:-1].values)

#   Re-order so that time is first to allow CDO commands to work
vdat = vdat.transpose('time','lat','lon','ens').sortby('ens')

#   Also change ens to integer to avoid CDO issues
vdat['ens'] = vdat['ens'].astype(int)

#   Save just the non-mean variables
#   Because for some reason CDO messes up the means
#       So just do the non-means 
#       Then calculate the spatial mean
#       Then combine again
vdat[fullvar].to_netcdf('../data/sflens/'+exp+'_'.join(['']+variables)+'_'+monthly_or_annual+'.nc')

#   Then run CDO to create regridded version
#   However, this unfortunately messes up the 1D variables (without lat/lon) for some reason
#   Run python lines below to fix
subprocess.run(['cdo', '-remapbil,so2_tmp.nc', 
                '../data/sflens/{}_{}_{}.nc'.format(exp,'_'.join(fullvar),monthly_or_annual),
                '../data/sflens/{}_{}_05deg_{}.nc'.format(exp,'_'.join(fullvar),monthly_or_annual)])
subprocess.run(['cdo', 'fldmean', 
                '../data/sflens/{}_{}_05deg_{}.nc'.format(exp,'_'.join(fullvar),monthly_or_annual),
                '../data/sflens/{}_{}_05deg_{}_mean.nc'.format(exp,'_'.join(fullvar),monthly_or_annual)])

vdat05 = xr.open_dataset('../data/sflens/{}_{}_05deg_{}.nc'.format(exp,'_'.join(fullvar),monthly_or_annual))
vdat05_mean = xr.open_dataset('../data/sflens/{}_{}_05deg_{}_mean.nc'.format(exp,'_'.join(fullvar),monthly_or_annual))
vdat05_mean = vdat05_mean.squeeze().drop(['lat','lon'])
for v in fullvar:
    vdat05[v+'_mean'] = vdat05_mean[v]

vdat05.to_netcdf('../data/sflens/{}_{}_05deg_{}_wmean.nc'.format(exp,'_'.join(fullvar),monthly_or_annual))

#   Remove duplicate since now there's a version with and without means
os.remove('../data/sflens/{}_{}_05deg_{}.nc'.format(exp,'_'.join(fullvar),monthly_or_annual))


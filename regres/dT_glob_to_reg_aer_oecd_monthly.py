#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   
#   Take global FaIR runs that use SSP3 emissions
#   Use the ratio of dT_aer_OECD and dT_aer_nonOECD to partition CESM dT
#   Use the two resulting CESM dTs as x variable in pattern scaling
#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import cftime
import subprocess
import statsmodels.formula.api as smf    
import pdb

#EXPs     = ['ghg','aer'] #   ghg or aer
EXPs     = ['aer']   #   ghg or aer
YVAR     =  'PRECT' #   PRECT, TREFHT
logP     =   True     #   use log(PRECT) for YVAR = PRECT; Doesn't work

hist_only = False
out_dir = '../data/regres/'

def integrate_emis(ds):
    #   Created area netcdf using;
    #   cdo gridarea /home/ppolonik/code/RoadsToParis/netcdfs/unmodified/CO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_190001-194912.nc emis_gridarea.nc
    area = xr.open_dataset('../data/geo/emis_gridarea.nc')
    #   kg/m2/s * m2 * s / 1e12  -> GT emissions
    emissum = ds * area['cell_area'] * 3.1536e7 / 1e12
    return emissum

def get_files(path2emis,emisname='CO2'):
    emisfiles = os.listdir(path2emis)
    emisfiles = [ef for ef in emisfiles if ef.startswith(emisname)]
    emisfiles = [ef for ef in emisfiles if 'ssp5' not in ef.lower()]
    emisfiles = [os.path.join(path2emis,cf) for cf in emisfiles]
    return emisfiles

def weighted_temporal_mean(ds, var):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month
    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)
    # Subset our dataset for our variable
    obs = ds[var]
    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")
    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")
    # Return the weighted average
    return obs_sum / ones_out

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
    return f

def dateshift_netCDF(fp):
    """
    Function to shift and save netcdf with dates at midpoint of month.
    :param fp: dataarray to be reoriented
    """
    f = fp
    if np.unique(fp.indexes['time'].day)[0]==1 & len(np.unique(fp.indexes['time'].day))==1:
        new_time = fp.indexes['time']-datetime.timedelta(days=16)
        f = f.assign_coords({'time':new_time})
    return f

def r2_func(true,resid):
    return 1 - resid / np.sum((true-np.mean(true))**2)

#   Create annual SO2 if necessary
#   This is now in annualize.bash
#   if not 'so2_annual.nc' in os.listdir('./SO2dat/'):
#       subprocess.run(['cdo','-yearmean','-cat','./SO2dat/SO2*.nc','./SO2dat/so2_annual.nc'])

#   Read regional sums of SO2 
#   so2orig = xr.open_dataset('./SO2dat/so2_{}_sums.nc'.format(group))

#   Changed SO2 to forcing-weighted emissions
#   aeremisorig = xr.open_dataset('./emisdat/spec_force.nc'.format(group))

#   Open dTs from FaIR SSP3
#   And format to make it easier to interpolate to model output times
fair_dT = xr.open_dataset('../data/fair/globxr_ssp3.nc').mean('config')
fair_dT = fair_dT.rename({'date':'time'}).to_dataframe()
fair_dT = fair_dT[[c for c in fair_dT.columns if c.startswith('dT')]]
fair_dT.index = [cftime.DatetimeNoLeap(d.year,d.month,d.day) for d in pd.to_datetime(fair_dT.index)]
fair_dT.index.name = 'time'
fair_dT = fair_dT.to_xarray()

grps = ['OECD','nonOECD']

formulas = {
    'ghg':'{} ~ dT_ghg'.format(YVAR),
    'aer':'{} ~ {}'.format(YVAR,' + '.join(['dT_aer_'+g for g in grps])),
    }

#   For selected experiments, read data and run regressions
for mon in range(1,13):
    print(mon)
    dss = {}
    for EXP in EXPs:
        path2dat = '../data/sflens/'.format(EXP)
        dat = reorient_netcdf('{}/{}_TREFHT_PRECT_05deg_monthly_wmean.nc'.format(path2dat,EXP))
        if logP:
            dat['PRECT'] = np.log(dat['PRECT'])

        #   Rolling 12 month mean with weights for days of month
        days_in_mon = xr.DataArray(dat['time'].to_index().days_in_month,
                                   coords=[dat['time']],name='days_in_mon')
        rollingwsum   = (dat['TREFHT_mean']*days_in_mon).rolling(time=12,center=True).sum()
        rollingweight = (days_in_mon).rolling(time=12,center=True).sum()
        dat['TREFHT_mean'] = rollingwsum / rollingweight

        dat = dat.isel({'time':slice(6,-5)}) #   Drop the nans
        dat = dat.loc[{'time':dat.time.dt.month==mon}]
    
        if hist_only:
            dat = dat.sel(time=slice('1850-01-01','2014-12-31'))
    
        dat['TREFHT'] = dat['TREFHT'] - 273.15
        dat['TREFHT_mean'] = dat['TREFHT_mean'] - 273.15
        dat = dat.mean('ens')
    
        #   Just annual SO2 data, but now changed to group sums
        #   so2 = xr.open_dataset('./SO2dat/so2_annual.nc').sum('sector')['SO2_em_anthro']
        #   so2 = so2.interp(time=dat.time)
    
        #   Combine so2 with rest of data
        #   With a slightly janky way of merging the times (find closest, overwite)
        #       this is just because the date chosen for the year midpoint isn't necessary identical
        #   aeremis = aeremisorig.interp(time=dat.time)
        if EXP=='aer':
            fair_dT = fair_dT.interp(time=dat.time)
            fair_dT_aersum = fair_dT['dT_aer_OECD'] + fair_dT['dT_aer_nonOECD']
            TREFHT_mean_dT = dat['TREFHT_mean'] - dat['TREFHT_mean'][1]
            fair_dT['dT_aer_OECD']    = TREFHT_mean_dT * fair_dT['dT_aer_OECD']    / fair_dT_aersum
            fair_dT['dT_aer_nonOECD'] = TREFHT_mean_dT * fair_dT['dT_aer_nonOECD'] / fair_dT_aersum
            dat = xr.merge([dat,fair_dT[['dT_aer_OECD','dT_aer_nonOECD']]])
            
        else:
            dat = dat.rename({'TREFHT_mean':'dT_ghg'})
    
        #   Run regressions
        lats = dat.lat
        lons = dat.lon
    
        xvars = [x.strip() for x in formulas[EXP].split('~')[1].split('+')]
        xvarsreg = ['const'] + xvars
    
        coefs = np.zeros((len(lats),len(lons),len(xvars)+1))
        pvals = np.zeros((len(lats),len(lons),len(xvars)+1))
        r2s   = np.zeros((len(lats),len(lons)))
    
        for lati, lat in enumerate(lats):
            print(lat.values)
            for loni, lon in enumerate(lons):
        
                datloc = dat.sel({'lat':lat,'lon':lon})[xvars+[YVAR]].to_dataframe()
                datloc['const'] = 1
    
                #   Regressions with statsmodel
                fit = smf.ols(formulas[EXP],datloc).fit()
                r2 = fit.rsquared
                coefs[lati,loni,:] = fit.params.values
                pvals[lati,loni,:] = fit.pvalues.values
                r2s[lati,loni] = r2
        
        ds = xr.Dataset(
            data_vars = 
                {xvr:         (['lat','lon'],coefs[:,:,i]) for i,xvr in enumerate(xvarsreg)} |
                {xvr+'_pval': (['lat','lon'],pvals[:,:,i]) for i,xvr in enumerate(xvarsreg)} |
                {'r2':        (['lat','lon'],r2s)},
            coords    = {'lat':(['lat'],lats.data),
                         'lon':(['lon'],lons.data)} 
                         )
         
        if hist_only:
            ds.to_netcdf('{}{}_{}_{}_mon{:02}_histonly_dT_results.nc'.format(
                        out_dir,EXP,'oecd',YVAR,mon))
        else:
            ds.to_netcdf('{}{}_{}_{}_mon{:02}_dT_results.nc'.format(out_dir,EXP,'oecd',YVAR,mon))
        dss[EXP] = ds


#   Get anomalies
#ghganom  = ghg.groupby('time.month')  - ghg.groupby('time.month').mean()
#aeranom  = aer.groupby('time.month')  - aer.groupby('time.month').mean()
#lensanom = lens.groupby('time.month') - lens.groupby('time.month').mean()
#resanom  = ghganom+aeranom-lensanom
#
##   Fit and plot
#fit_ghg  = np.polyfit(ghganom.TREFHT_mean, resanom.TREFHT_mean,1)
#fit_lens = np.polyfit(lensanom.TREFHT_mean,resanom.TREFHT_mean,1)
#
#plt.figure(figsize=(10,4));
#plt.subplot(1,2,1)
#plt.scatter(lensanom.TREFHT_mean,resanom.TREFHT_mean,10,c=[(v.year) for v in dat.time.values])
#plt.xlabel('LENS T anom')
#plt.ylabel('Residual T anom')
#plt.plot(lensanom.TREFHT_mean,np.polyval(fit_lens,lensanom.TREFHT_mean),'.',color='orange')
#plt.colorbar()
#plt.subplot(1,2,2)
#plt.scatter(ghganom.TREFHT_mean,resanom.TREFHT_mean,10,c=[(v.year) for v in dat.time.values])
#plt.xlabel('GHG T anom')
#plt.ylabel('Residual T anom')
#plt.plot(ghganom.TREFHT_mean,np.polyval(fit_ghg,ghganom.TREFHT_mean),'.',color='orange')
#plt.colorbar()
#plt.tight_layout()





#   Global dT regression

#   Select a point to plot regression
#   latind, lonind = 125, 234 #   Poor fit in china
#   latind, lonind = 138, 141 #   Pretty good fit in mid Spain
#   
#   Tsel = datan.isel({'lat':latind,'lon':lonind})['TREFHT']
#   
#   plt.figure(figsize=(14,5))
#   plt.subplot(1,2,1)
#   plt.imshow(r2s,origin='lower',vmin=0.80,vmax=1)
#   plt.plot(lonind,latind,'rx')
#   plt.title('R$^2$')
#   plt.colorbar()
#   
#   plt.subplot(1,2,2)
#   plt.plot(datglob,Tsel.values,'.')
#   sorti = np.argsort(datglob)
#   plt.plot(datglob[sorti],np.polyval(coefs[latind,lonind,:],datglob[sorti]),'--k')
#   plt.xlabel('Global mean T')
#   plt.ylabel('Local T')
#   plt.title('lat: {:.2f}, lon: {:.2f}, r$^2$: {:.2f}'.format(Tsel.lat.values,
#                                                              Tsel.lon.values,
#                                                              r2s[latind,lonind]))
#   plt.tight_layout()

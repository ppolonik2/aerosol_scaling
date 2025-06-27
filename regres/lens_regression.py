import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

remove_trend = False

lens  = xr.open_dataset('../data/lens/LENS_TREFHT_PRECT_ensmean_05deg_wmean.nc')
lens  = lens[['TREFHT_mean','TREFHT']]

# FIX: This is annual but I need monthly
lens5 = xr.open_dataset('../data/lens/TREFHT_PRECT_05deg_mean_sub5_wmean.nc')
lens5 = lens5[['TREFHT_mean','TREFHT']]

#   Take first 30 years as baseline
lensbase  = lens['TREFHT_mean'].isel(time=range(12*30)).groupby('time.month').mean()
lensbase5 = lens5['TREFHT_mean'].isel(time=range(12*30)).groupby('time.month').mean()

#   Get difference from that baseline
lensanom  = lens['TREFHT_mean'].groupby('time.month')  - lensbase
lensanom5 = lens5['TREFHT_mean'].groupby('time.month') - lensbase5

#   Take a rolling mean to remove any trends
lensanomflat = lensanom - lensanom.rolling(time=12,center=True).mean()

#   Read pattern scaling
aerscale = xr.open_dataset('../data/regres/aer_oecd_TREFHT_mon_dT_results.nc')
ghgscale = xr.open_dataset('../data/regres/ghg_oecd_TREFHT_mon_dT_results.nc')

# Take global area-weighted mean of pattern scaling coefficients
aerscale = aerscale.rename({'mon':'month'})
ghgscale = ghgscale.rename({'mon':'month'})
weights05 = np.cos(np.deg2rad(aerscale.lat))
weights05.name = "weights"
aerscale_mean = aerscale.weighted(weights05).mean(('lon','lat'))
ghgscale_mean = ghgscale.weighted(weights05).mean(('lon','lat'))

#   Set up regression to predict LENS seasonal cycle change based on pattern scaling seasonality
cycles = pd.DataFrame(
    data={'ghg':ghgscale_mean['dT_ghg'].values,
          'oecd':aerscale_mean['dT_aer_OECD'].values,
          'nonoecd':aerscale_mean['dT_aer_nonOECD'].values}
)
cycles.index = ghgscale_mean.month

formula = 'TREFHT_mean ~ ghg + oecd + nonoecd'

lensanomflatdf = lensanomflat.to_dataframe().dropna()
startyr = lensanomflatdf[lensanomflatdf.month==1].index[0].year
endyr   = lensanomflatdf[lensanomflatdf.month==12].index[-1].year

# Model de-trended LENS data using monthly pattern scaling coefficients
runyrs = range(startyr,endyr+1)
coefs = pd.DataFrame(index=runyrs,columns=list(cycles.columns)+['r2'])
for yr in runyrs:
    Ttmp = lensanomflatdf[lensanomflatdf.index.year==yr]['TREFHT_mean']
    Ttmp.index = Ttmp.index.month
    annual = cycles.join(Ttmp)
    fit = smf.ols(formula=formula,data=annual).fit()
    coefs.loc[yr,'ghg'] = fit.params.loc['ghg'] 
    coefs.loc[yr,'oecd'] = fit.params.loc['oecd'] 
    coefs.loc[yr,'nonoecd'] = fit.params.loc['nonoecd'] 
    coefs.loc[yr,'r2'] = fit.rsquared_adj

#   Use obs (and weight by area in regression)

#   Read in half degree version (regridded with CDO)
ecmwf = xr.open_dataset('../data/ecmwf/ecmwf_t2m_monthly_reanalysis_05deg.nc')

#   Convert dates to actual date format
dates = pd.DatetimeIndex(['{}-{}-{}'.format(d[0:4],d[4:6],d[6:8]) 
            for d in ecmwf.date.values.astype(str)])
ecmwf = ecmwf.assign_coords({'date':dates})
ecmwf_wmean = ecmwf.weighted(weights05).mean(('lon','lat'))
lens_wmean  = lens.weighted(weights05).mean(('lon','lat'))
lens_wmean5 = lens5.weighted(weights05).mean(('lon','lat'))

#   Take rolling mean and use first 10 years to subtract out 1940s seasonal cycle
ecmwfflat = ecmwf - ecmwf.rolling(date=12,center=True).mean()
base_seasonal = ecmwfflat.isel(date=range(10*12)).groupby('date.month').mean()

if remove_trend:
    ecmwf_anom = ecmwfflat.groupby('date.month') - base_seasonal
else:
    mean40s = ecmwf.isel(date=range(10*12)).mean('date')
    ecmwf_anom = ecmwf.groupby('date.month') - base_seasonal - mean40s

ecmwf_anom = ecmwf_anom.assign_coords({'year':ecmwf['date.year'],'month':ecmwf['date.month']})
ecmwf_anom = xr.merge([ecmwf_anom,weights05])
ecmwf_anomdf = ecmwf_anom.to_dataframe().dropna()

#   Do the same for LENS
lense  = lens.sel(time=slice('1940-01-01','2024-08-31'))
lense5 = lens5.sel(time=slice('1940-01-01','2024-08-31'))
lensSflat  = lense  - lense.rolling(time=12,center=True).mean()
lensSflat5 = lense5 - lense5.rolling(time=12,center=True).mean()
lensS_seasonal  = lensSflat.isel(time=range(10*12)).groupby('time.month').mean()
lensS_seasonal5 = lensSflat5.isel(time=range(10*12)).groupby('time.month').mean()

if remove_trend:
    lensS_anom  = lensSflat.groupby('time.month') - lensS_seasonal
    lensS_anom5 = lensSflat5.groupby('time.month') - lensS_seasonal5
else:
    mean40sl  = lense.isel(time=range(10*12)).mean('time')
    mean40sl5 = lense5.isel(time=range(10*12)).mean('time')
    lensS_anom  = lense.groupby('time.month')  - lensS_seasonal - mean40sl
    lensS_anom5 = lense5.groupby('time.month') - lensS_seasonal5 - mean40sl5

lensS_anom = lensS_anom.assign_coords({'year':lens['time.year'],'month':lens['time.month']})
lensS_anom = xr.merge([lensS_anom,weights05])
lensS_anomdf = lensS_anom.to_dataframe().dropna()

lensS_anom5 = lensS_anom5.assign_coords({'year':lens5['time.year'],'month':lens5['time.month']})
lensS_anom5 = xr.merge([lensS_anom5,weights05])
lensS_anomdf5 = lensS_anom5.to_dataframe().dropna()

#   Create spatial cycles
spatial_cycles = xr.merge([ghgscale['dT_ghg'],aerscale['dT_aer_OECD'],aerscale['dT_aer_nonOECD']])
spatial_cycles = spatial_cycles.to_dataframe()
spatial_cycles = spatial_cycles.rename(columns={'dT_ghg':'ghg',
                                                'dT_aer_OECD':'oecd',
                                                'dT_aer_nonOECD':'nonoecd'})

#   Get complete start/end years (rolling mean creates nans)
startyre = ecmwf_anomdf[ecmwf_anomdf.month==1].iloc[[0]].year.iloc[0]
endyre   = ecmwf_anomdf[ecmwf_anomdf.month==12].iloc[[-1]].year.iloc[0]

runyrse = range(startyre,endyre+1)
coefse  = pd.DataFrame(index=runyrse,columns=list(spatial_cycles.columns)+['Intercept','r2'])
coefsl  = pd.DataFrame(index=runyrse,columns=list(spatial_cycles.columns)+['Intercept','r2'])
coefsl5 = pd.DataFrame(index=pd.MultiIndex.from_product([lensS_anom5.ens.values,runyrse]),
                       columns=list(spatial_cycles.columns)+['Intercept','r2'])
formulae = 't2m ~ ghg + oecd + nonoecd'
formulal = 'TREFHT ~ ghg + oecd + nonoecd'
for yr in runyrse:
    print('ECMWF regression year: {}'.format(yr))
    Ttmp = ecmwf_anomdf[ecmwf_anomdf.year==yr]
    Ttmp = Ttmp.reset_index().set_index(['lat','lon','month'])
    annual = Ttmp[['t2m','weights']].merge(spatial_cycles,left_index=True,right_index=True)
    fit = smf.wls(formula=formulae,data=annual,weights=annual['weights']).fit()
    coefse.loc[yr,'ghg'] = fit.params.loc['ghg'] 
    coefse.loc[yr,'oecd'] = fit.params.loc['oecd'] 
    coefse.loc[yr,'nonoecd'] = fit.params.loc['nonoecd'] 
    coefse.loc[yr,'Intercept'] = fit.params.loc['Intercept'] 
    coefse.loc[yr,'r2'] = fit.rsquared_adj

    Ttmpl = lensS_anomdf[lensS_anomdf.year==yr]
    Ttmpl = Ttmpl.reset_index().set_index(['lat','lon','month'])
    annual = Ttmpl[['TREFHT','weights']].merge(spatial_cycles,left_index=True,right_index=True)
    fit = smf.wls(formula=formulal,data=annual,weights=annual['weights']).fit()
    coefsl.loc[yr,'ghg'] = fit.params.loc['ghg'] 
    coefsl.loc[yr,'oecd'] = fit.params.loc['oecd'] 
    coefsl.loc[yr,'nonoecd'] = fit.params.loc['nonoecd'] 
    coefsl.loc[yr,'Intercept'] = fit.params.loc['Intercept'] 
    coefsl.loc[yr,'r2'] = fit.rsquared_adj

    # Repeat one more time for each LENS ensemble member insetad of LENS mean or ECMWF
    for e in lensS_anom5.ens.values:
        Ttmpl5 = lensS_anomdf5[(lensS_anomdf5.year==yr) & 
                               (lensS_anomdf5.index.get_level_values('ens')==e)]
        Ttmpl5 = Ttmpl5.reset_index().set_index(['lat','lon','month'])
        annual = Ttmpl5[['TREFHT','weights']].merge(spatial_cycles,left_index=True,right_index=True)
        fit = smf.wls(formula=formulal,data=annual,weights=annual['weights']).fit()
        coefsl5.loc[(e,yr),'ghg'] = fit.params.loc['ghg'] 
        coefsl5.loc[(e,yr),'oecd'] = fit.params.loc['oecd'] 
        coefsl5.loc[(e,yr),'nonoecd'] = fit.params.loc['nonoecd'] 
        coefsl5.loc[(e,yr),'Intercept'] = fit.params.loc['Intercept'] 
        coefsl5.loc[(e,yr),'r2'] = fit.rsquared_adj

# LENS and ECMWF time series
lens_wmean_anom = lens_wmean['TREFHT_mean'] - \
                  lens_wmean['TREFHT_mean'].sel(time=slice('1940-01-01','1949-12-31')).mean()
lens_dT  = lens_wmean_anom.isel(time=range(12*90,12*(90+84))).rolling(time=12*5).mean()
lens_wmean_anom5 = lens_wmean5['TREFHT_mean'] - \
                   lens_wmean5['TREFHT_mean'].sel(time=slice('1940-01-01','1949-12-31')).mean()
lens_dT5 = lens_wmean_anom5.isel(time=range(12*90,12*(90+84))).rolling(time=12*5).mean()
ecmwf_wmean_anom = ecmwf_wmean['t2m'] - \
                   ecmwf_wmean['t2m'].sel(date=slice('1940-01-01','1949-12-31')).mean()
ecmwf_dT = ecmwf_wmean_anom.rolling(date=12*5).mean()

# Rename, calculate rolling means, and calculate totals
coefsl = coefsl.rename(columns={'ghg':'GHG','oecd':'OECD aerosol','nonoecd':'nonOECD aerosol'})
coefslr = coefsl.rolling(5).mean()
coefslr['Total, no int'] = coefslr[['GHG','OECD aerosol','nonOECD aerosol']].sum(1)
coefslr['Total']         = coefslr[['GHG','OECD aerosol','nonOECD aerosol','Intercept']].sum(1)
coefslr['LENS dT']       = lens_dT.groupby('time.year').mean()

coefsl5 = coefsl5.rename(columns={'ghg':'GHG','oecd':'OECD aerosol','nonoecd':'nonOECD aerosol'})
coefslr5 = []
for ens in lensS_anom5.ens.values:
    coefslr5tmp = coefsl5.loc[ens].rolling(5).mean()
    coefslr5tmp.index = pd.MultiIndex.from_product([[ens],coefslr5tmp.index])
    coefslr5.append(coefslr5tmp)
coefslr5 = pd.concat(coefslr5)

coefslr5['Total, no int'] = coefslr5[['GHG','OECD aerosol','nonOECD aerosol']].sum(1)
coefslr5['Total']         = coefslr5[['GHG','OECD aerosol','nonOECD aerosol','Intercept']].sum(1)
coefslr5['LENS dT']       = lens_dT5.groupby('time.year').mean().transpose('ens','year').to_dataframe()

# Repeat for ECMWF
coefse = coefse.rename(columns={'ghg':'GHG','oecd':'OECD aerosol','nonoecd':'nonOECD aerosol'})
coefser = coefse.rolling(5).mean()
coefser['Total, no int'] = coefser[['GHG','OECD aerosol','nonOECD aerosol']].sum(1)
coefser['Total']         = coefser[['GHG','OECD aerosol','nonOECD aerosol','Intercept']].sum(1)
coefser['ECMWF dT']      = ecmwf_dT.groupby('date.year').mean()

# Save
coefse.to_csv('../data/regres/verify/reg_coefs_ECMWF_0624.csv')
coefsl.to_csv('../data/regres/verify/reg_coefs_LENS_0624.csv')
coefsl5.to_csv('../data/regres/verify/reg_coefs_LENS_sub5_0624.csv')
coefser.to_csv('../data/regres/verify/reg_coefs_ECMWF_rolling_0624.csv')
coefslr.to_csv('../data/regres/verify/reg_coefs_LENS_rolling_0624.csv')
coefslr5.to_csv('../data/regres/verify/reg_coefs_LENS_sub5_rolling_0624.csv')




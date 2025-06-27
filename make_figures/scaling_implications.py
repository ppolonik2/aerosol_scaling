import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

var = 'TREFHT' # TREFHT or PRECT

if var=='TREFHT':
    lim = 3
elif var == 'PRECT':
    lim = 1e-8

ghg = xr.open_dataset('../data/regres/ghg_oecd_{}_mon_dT_results.nc'.format(var))
aer = xr.open_dataset('../data/regres/aer_oecd_{}_mon_dT_results.nc'.format(var))

mnames = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',
          7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}

def plotmaps(dat,vlim,neg,cmap='RdBu_r',lab='C/C',extend='both'):
    if type(vlim)!=list:
        vlim=[-vlim,vlim]
    fig = plt.figure(figsize=(8,7))
    for mon in dat.mon.values:
        ax = plt.subplot(4,3,mon,projection=ccrs.Robinson())
        im = dat.sel(mon=mon).plot(ax=ax,transform=ccrs.PlateCarree(),
            vmin=vlim[0],vmax=vlim[1],cmap=cmap,add_colorbar=False)
        ax.coastlines(lw=0.5)
        ax.set_title(mnames[mon])
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.025, 0.08, 0.95, 0.03])
    if lab=='C/C':
        if neg:
            lab =  r'- $\degree$C / $\degree$C'
        else:
            lab =  r'$\degree$C / $\degree$C'
    elif lab.lower()=='r2':
        lab='$R^2$'

    cb = fig.colorbar(im,cax=cbar_ax,orientation='horizontal',extend=extend)
    cb.ax.tick_params(labelsize=14) 
    cb.set_label(label=lab,size=14)


plotmaps( ghg['dT_ghg'],lim,False)
plotmaps(-aer['dT_aer_OECD'],lim,True)
plotmaps(-aer['dT_aer_nonOECD'],lim,True)
plotmaps( ghg['r2'],[0,1],False,plt.cm.plasma,lab='r2',extend='neither')
plotmaps( aer['r2'],[0,1],False,plt.cm.plasma,lab='r2',extend='neither')

# Pop weighting
oecd = ['USA','GBR','BEL','FRA','DEU','NLD','MEX','TUR','FIN','GRC',
        'ISL','ISR','ITA','LUX','NOR','PRT','ESP','KOR','DNK','NZL','SWE',
        'AUT','IRL','CHE','CHL','EST','HUN','SVK','SVN','AUS','JPN','POL',
        'CAN','LVA','LTU','COL','CRI','CZE']

cid  = xr.open_dataset('~/CERM/data/input/country_raster_05deg.nc')['country_id']
meta = pd.read_csv('~/CERM/data/input/country_id_mapping.csv',index_col=0).set_index('ISO_A3')
oecd_id = meta.loc[oecd]
nonoecd_id = meta.loc[~meta.index.isin(oecd)]

popall     = xr.open_dataset('~/CERM/data/input/GDP_pop_corrected.nc')['pop'].sel(ssp=2,year=2020)
popoecd    = popall.where(cid.isin(oecd_id))
popnonoecd = popall.where(cid.isin(nonoecd_id))

pops = {'all':popall,'oecd':popoecd,'nonoecd':popnonoecd}

dT_ghg_wt         = {}
dT_aer_OECD_wt    = {}
dT_aer_nonOECD_wt = {}
r2_ghg_wt         = {}
r2_aer_wt         = {}
for n,pop in pops.items():
    totpop         = np.sum(pop)
    
    dT_ghg_wt[n]          = (ghg['dT_ghg']*pop/totpop).sum(['lat','lon'])
    dT_aer_OECD_wt[n]     = (aer['dT_aer_OECD']*pop/totpop).sum(['lat','lon'])
    dT_aer_nonOECD_wt[n]  = (aer['dT_aer_nonOECD']*pop/totpop).sum(['lat','lon'])
    r2_ghg_wt[n]          = (ghg['r2']*pop/totpop).sum(['lat','lon'])
    r2_aer_wt[n]          = (aer['r2']*pop/totpop).sum(['lat','lon'])


cols   = {'all':'k','oecd':'chocolate','nonoecd':'steelblue'}
labels = {'all':'All population','oecd':'OECD population','nonoecd':'nonOECD population'}
plt.figure(figsize=(15,5))
ax1 = plt.subplot(1,len(pops),1)
for n, pop in pops.items():
    dT_ghg_wt[n].plot(label=labels[n],lw=2.5,color=cols[n])
    ax1.set_ylabel('dT (C/C)',fontsize=13)
    ax1.set_title('GHG',fontsize=14)
    plt.annotate('(a)',(0.02,0.95),xycoords='axes fraction',fontsize=12)
    plt.legend(fontsize=13)
    ax1.set_ylim([1-1.3/2,1+1.3/2])
ax2 = plt.subplot(1,len(pops),2)
for n, pop in pops.items():
    (-dT_aer_OECD_wt[n]).plot(label=labels[n],lw=2.5,color=cols[n])
    ax2.set_ylabel('-dT (-C/C)',fontsize=13)
    ax2.set_title('OECD aerosol',fontsize=14)
    ax2.set_ylim([-2-1.3/2,-2+1.3/2])
    plt.annotate('(b)',(0.02,0.95),xycoords='axes fraction',fontsize=12)
ax3 = plt.subplot(1,len(pops),3)
for n, pop in pops.items():
    (-dT_aer_nonOECD_wt[n]).plot(label=labels[n],lw=2.5,color=cols[n])
    ax3.set_ylabel('-dT (C/C)',fontsize=13)
    ax3.set_title('nonOECD aerosol',fontsize=14)
    ax3.set_ylim([-1.3-1.3/2,-1.3+1.3/2])
    plt.annotate('(c)',(0.02,0.95),xycoords='axes fraction',fontsize=12)

for ax in[ax1,ax2,ax3]:
    ax.set_xticks(range(1,13))
    ax.set_xlabel('Month',fontsize=13)

plt.tight_layout()


# Make figure of 2015 temperature attribution
yrs = [1980,2015,2050]
fairT = xr.open_dataset('../data/fair/globxr_ssp3.nc').mean('config').groupby('date.year').mean()

limy = {'ghg':[-4,4],'oecd':[-1,1],'nonoecd':[-1,1]}

plt.figure(figsize=(15,2.5*len(yrs)))
for yi,yr in enumerate(yrs):
    ghgy     = fairT.sel(year=yr)['dT_ghg']         * ghg.mean('mon')['dT_ghg']
    oecdy    = fairT.sel(year=yr)['dT_aer_OECD']    * aer.mean('mon')['dT_aer_OECD']
    nonoecdy = fairT.sel(year=yr)['dT_aer_nonOECD'] * aer.mean('mon')['dT_aer_nonOECD']
    

    ax1 = plt.subplot(len(yrs),3,yi*3+1,projection=ccrs.Robinson())
    im = ghgy.plot(ax=ax1,transform=ccrs.PlateCarree(),
                    vmin=limy['ghg'][0],vmax=limy['ghg'][1],
                    cmap='RdBu_r',add_colorbar=True,cbar_kwargs={'label':''})
    ax1.coastlines(lw=0.5)
    ax1.set_title('GHG')
    ax1.set_ylabel(yr,fontsize=13)
    ax1.text(-0.03, 0.5, yr, transform=ax1.transAxes,
         rotation='vertical', va='center', ha='right', fontsize=14)
    
    ax2 = plt.subplot(len(yrs),3,yi*3+2,projection=ccrs.Robinson())
    im = oecdy.plot(ax=ax2,transform=ccrs.PlateCarree(),
                    vmin=limy['oecd'][0],vmax=limy['oecd'][1],
                    cmap='coolwarm',add_colorbar=True,cbar_kwargs={'label':''})
    ax2.coastlines(lw=0.5)
    ax2.set_title('OECD aerosol')
    
    ax3 = plt.subplot(len(yrs),3,yi*3+3,projection=ccrs.Robinson())
    im = nonoecdy.plot(ax=ax3,transform=ccrs.PlateCarree(),
                    vmin=limy['nonoecd'][0],vmax=limy['nonoecd'][1],
                    cmap='coolwarm',add_colorbar=True,cbar_kwargs={'label':''})
    ax3.coastlines(lw=0.5)
    ax3.set_title('nonOECD aerosol')

plt.tight_layout()

# Repeat but instead divide by the GHG column
yrs = [1980,2015,2050]
fairT = xr.open_dataset('../data/fair/globxr_ssp3.nc').mean('config').groupby('date.year').mean()

limf = [-0.5,0.5]
lab = 'dT$_{aer}$/dT$_{GHG}$'
plt.figure(figsize=(10,2.5*len(yrs)))
for yi,yr in enumerate(yrs):
    ghgy     = fairT.sel(year=yr)['dT_ghg']         * ghg.mean('mon')['dT_ghg']
    oecdy    = fairT.sel(year=yr)['dT_aer_OECD']    * aer.mean('mon')['dT_aer_OECD']
    nonoecdy = fairT.sel(year=yr)['dT_aer_nonOECD'] * aer.mean('mon')['dT_aer_nonOECD']

    oecdf    = oecdy    / ghgy
    nonoecdf = nonoecdy / ghgy
    
    
    ax1 = plt.subplot(len(yrs),2,yi*2+1,projection=ccrs.Robinson())
    im = oecdf.plot(ax=ax1,transform=ccrs.PlateCarree(),
                    vmin=limf[0],vmax=limf[1],
                    cmap='coolwarm',add_colorbar=True,cbar_kwargs={'label':lab})
    ax1.coastlines(lw=0.5)
    ax1.set_title('OECD aerosol')
    ax1.text(-0.03, 0.5, yr, transform=ax1.transAxes,
         rotation='vertical', va='center', ha='right', fontsize=14)
    
    ax2 = plt.subplot(len(yrs),2,yi*2+2,projection=ccrs.Robinson())
    im = nonoecdf.plot(ax=ax2,transform=ccrs.PlateCarree(),
                    vmin=limf[0],vmax=limf[1],
                    cmap='coolwarm',add_colorbar=True,cbar_kwargs={'label':lab})
    ax2.coastlines(lw=0.5)
    ax2.set_title('nonOECD aerosol')

plt.tight_layout()

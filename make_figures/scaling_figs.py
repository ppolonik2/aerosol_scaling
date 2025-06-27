import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns

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
plotmaps( ghg['dT_ghg_pval'],[0,0.1],False,'viridis','p-value','neither')
plotmaps( aer['dT_aer_OECD_pval'],[0,0.1],False,'viridis','p-value','neither')
plotmaps( aer['dT_aer_nonOECD_pval'],[0,0.1],False,'viridis','p-value','neither')

# Seasonal version of maps
seasons = {'DJF':[12,1,2],'MAM':[3,4,5],'JJA':[6,7,8],'SON':[9,10,11]}
seasonal = {'GHG':{},'OECD aerosol':{},'nonOECD aerosol':{}}
for sname, smons in seasons.items():
    seasonal['GHG'][sname]             =  ghg['dT_ghg'].sel(mon=smons).mean('mon')
    seasonal['OECD aerosol'][sname]    = -aer['dT_aer_OECD'].sel(mon=smons).mean('mon')
    seasonal['nonOECD aerosol'][sname] = -aer['dT_aer_nonOECD'].sel(mon=smons).mean('mon')

fig = plt.figure(figsize=(8.5,7))
sct = 0 
labs = {'GHG':          r'$\degree$C / $\degree$C',
        'OECD aerosol':   r'- $\degree$C / $\degree$C',
        'nonOECD aerosol':r'- $\degree$C / $\degree$C'}
for sname, smons in seasons.items():
    for cat in ['GHG','OECD aerosol','nonOECD aerosol']:
        sct += 1
        ax = plt.subplot(4,3,sct,projection=ccrs.Robinson())
        im = seasonal[cat][sname].plot(ax=ax,transform=ccrs.PlateCarree(),
            vmin=-lim,vmax=lim,cmap='RdBu_r',label=labs[cat],add_colorbar=False)
        ax.coastlines(lw=0.5)
        if sct in [1,2,3]:
            ax.set_title(cat)
        if sct in [1,4,7,10]:
            ax.annotate(sname,(-0.03,0.5),xycoords='axes fraction',
                        ha='right',va='center',rotation=90,fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12,left=0.05)
cbar_ax1 = fig.add_axes([0.055, 0.08, 0.29, 0.03])
cbar_ax2 = fig.add_axes([0.370, 0.08, 0.60, 0.03])
cb1 = fig.colorbar(im,cax=cbar_ax1,orientation='horizontal',extend='both',ticks=np.arange(-3,4))
cb1.ax.tick_params(labelsize=14) 
cb1.set_label(label=labs['GHG'],size=14)
cb2 = fig.colorbar(im,cax=cbar_ax2,orientation='horizontal',extend='both')
cb2.ax.tick_params(labelsize=14) 
cb2.set_label(label=labs['OECD aerosol'],size=14)

# End seasonal figure
# Correlation coefficients between seasonal maps
cor_ghgoecd, cor_ghgnonoecd, cor_oecdnonoecd = {}, {}, {}
for sname,smon in seasons.items():
    cor_ghgoecd[sname]     = np.corrcoef(seasonal['GHG'][sname].values.flatten(),
                                         seasonal['OECD aerosol'][sname].values.flatten())[0,1]
    cor_ghgnonoecd[sname]  = np.corrcoef(seasonal['GHG'][sname].values.flatten(),
                                         seasonal['nonOECD aerosol'][sname].values.flatten())[0,1]
    cor_oecdnonoecd[sname] = np.corrcoef(seasonal['OECD aerosol'][sname].values.flatten(),
                                         seasonal['nonOECD aerosol'][sname].values.flatten())[0,1]

cor_ghgoecd_mean     = np.mean([v for sname,v in cor_ghgoecd.items()])
cor_ghgnonoecd_mean  = np.mean([v for sname,v in cor_ghgnonoecd.items()])
cor_oecdnonoecd_mean = np.mean([v for sname,v in cor_oecdnonoecd.items()])


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
poplab = {'all':'All','oecd':'OECD','nonoecd':'nonOECD'}
plt.figure(figsize=(15,8))
ax1 = plt.subplot(2,len(pops),1)
for n, pop in pops.items():
    dT_ghg_wt[n].plot(label=poplab[n],lw=2.5,color=cols[n])
    ax1.set_ylabel('dT / dT$_{annual}$ [$\degree$C/$\degree$C]',fontsize=13)
    ax1.set_title('From GHG')
    ax1.set_ylim([1-1.3/2,1+1.3/2])
    plt.annotate('(a)',(0.02,0.94),xycoords='axes fraction',fontsize=12)
    plt.legend(title='Population weight')
ax2 = plt.subplot(2,len(pops),2)
for n, pop in pops.items():
    (-dT_aer_OECD_wt[n]).plot(label='{} pop'.format(n),lw=2.5,color=cols[n])
    ax2.set_ylabel('dT / dT$_{annual}$ [$\degree$C/$\degree$C]',fontsize=13)
    ax2.set_title('From OECD Aerosol')
    ax2.set_ylim([-1.99-1.3/2,-1.99+1.3/2])
    plt.annotate('(b)',(0.02,0.94),xycoords='axes fraction',fontsize=12)
    #plt.legend()
ax3 = plt.subplot(2,len(pops),3)
for n, pop in pops.items():
    (-dT_aer_nonOECD_wt[n]).plot(label='{} pop'.format(n),lw=2.5,color=cols[n])
    ax3.set_ylabel('-dT / dT$_{annual}$ [$\degree$C/$\degree$C]',fontsize=13)
    ax3.set_title('From nonOECD Aerosol')
    ax3.set_ylim([-1.3-1.3/2,-1.3+1.3/2])
    plt.annotate('(c)',(0.02,0.94),xycoords='axes fraction',fontsize=12)
    #plt.legend()

for ax in[ax1,ax2,ax3]:
    ax.set_xticks(range(1,13))
    ax.set_xlabel('Month',fontsize=13)


# Add histograms of attributed temperature change in certain years
yrs = [1980,2015,2050]
yrcols = {1980:'#fe9929',2015:'#d95f0e',2050:'#993404'}
xlimv = [[0,4.5],[-1,0],[-1,0]]
xlabv = ['dT$_{GHG}$ [$\degree$C]',
         'dT$_{OECD}$ / dT$_{GHG}$ [$\degree$C/$\degree$C]',
         'dT$_{nonOECD}$ / dT$_{GHG}$ [$\degree$C/$\degree$C]']
fairT = xr.open_dataset('../data/fair/globxr_ssp3.nc').mean('config').groupby('date.year').mean()
ghgy     = fairT['dT_ghg']         * ghg.mean('mon')['dT_ghg']
oecdy    = fairT['dT_aer_OECD']    * aer.mean('mon')['dT_aer_OECD']
nonoecdy = fairT['dT_aer_nonOECD'] * aer.mean('mon')['dT_aer_nonOECD']
wait

axvs = []
popwt = pops['all'].fillna(0).values.flatten()
alph = ['d','e','f']
for vi,v  in enumerate([ghgy,oecdy,nonoecdy]):
    axv = plt.subplot(2,len(pops),len(pops)+1+vi)
    axvs.append(axv)
    for yi,yr in enumerate(yrs):
        if vi==0:
            r = v.sel(year=yr)
        else:
            r = v.sel(year=yr) / ghgy.sel(year=yr)
        r = r.values.flatten()
        rlim = np.percentile(r,[0.5,99.5])
        r[(r<rlim[0])|(r>rlim[1])] = np.nan

        sns.ecdfplot(x=r,weights=popwt,ax=axv,color=yrcols[yr],label=yr,lw=3) 
    if vi==0:
        plt.legend(loc=4)
    axv.set_ylabel('Cumulative Density',fontsize=13)
    axv.set_xlim(xlimv[vi])
    axv.set_xlabel(xlabv[vi],fontsize=13)
    plt.annotate('({})'.format(alph[vi]),(0.02,0.94),xycoords='axes fraction',fontsize=12)
    axv.grid('on')

plt.tight_layout()

# Make figure with maps of attributed temperature change in certain years
limy = {'ghg':[-4,4],'oecd':[-1,1],'nonoecd':[-1,1]}
plt.figure(figsize=(15,2.5*len(yrs)))
for yi,yr in enumerate(yrs):
    ghgy     = fairT.sel(year=yr)['dT_ghg']         * ghg.mean('mon')['dT_ghg']
    oecdy    = fairT.sel(year=yr)['dT_aer_OECD']    * aer.mean('mon')['dT_aer_OECD']
    nonoecdy = fairT.sel(year=yr)['dT_aer_nonOECD'] * aer.mean('mon')['dT_aer_nonOECD']

    ax1 = plt.subplot(len(yrs),3,(yi*3)+1,projection=ccrs.Robinson())
    ghgy.plot(ax=ax1,transform=ccrs.PlateCarree(),
              add_colorbar=True,cbar_kwargs={'label':''},
              vmin=limy['ghg'][0],vmax=limy['ghg'][1],
              cmap='RdBu_r')
    ax1.set_title('GHG')
    ax1.coastlines(lw=0.5)
    ax1.text(-0.03, 0.5, yr, transform=ax1.transAxes,
             rotation='vertical', va='center', ha='right', fontsize=13)

    ax2 = plt.subplot(len(yrs),3,(yi*3)+2,projection=ccrs.Robinson())
    oecdy.plot(ax=ax2,transform=ccrs.PlateCarree(),
               add_colorbar=True,cbar_kwargs={'label':''},
               vmin=limy['oecd'][0],vmax=limy['oecd'][1],
               cmap='coolwarm')
    ax2.set_title('OECD aerosol')
    ax2.coastlines(lw=0.5)

    ax3 = plt.subplot(len(yrs),3,(yi*3)+3,projection=ccrs.Robinson())
    nonoecdy.plot(ax=ax3,transform=ccrs.PlateCarree(),
                  add_colorbar=True,cbar_kwargs={'label':''},
                  vmin=limy['nonoecd'][0],vmax=limy['nonoecd'][1],
                  cmap='coolwarm')
    ax3.set_title('nonOECD aerosol')
    ax3.coastlines(lw=0.5)

plt.tight_layout()


# Make figure of ratios
limr = [-0.5,0.5]
plt.figure(figsize=(10,2.5*len(yrs)))
labelr = 'dT$_{aer}$ / dT$_{GHG}$'
for yi,yr in enumerate(yrs):
    ghgy     = fairT.sel(year=yr)['dT_ghg']         * ghg.mean('mon')['dT_ghg']
    oecdy    = fairT.sel(year=yr)['dT_aer_OECD']    * aer.mean('mon')['dT_aer_OECD']
    nonoecdy = fairT.sel(year=yr)['dT_aer_nonOECD'] * aer.mean('mon')['dT_aer_nonOECD']

    oecdr    = oecdy    / ghgy
    nonoecdr = nonoecdy / ghgy

    ax1 = plt.subplot(len(yrs),2,(yi*2)+1,projection=ccrs.Robinson())
    oecdr.plot(ax=ax1,transform=ccrs.PlateCarree(),
               add_colorbar=True,cbar_kwargs={'label':labelr},
               vmin=limr[0],vmax=limr[1],
               cmap='coolwarm')
    ax1.set_title('OECD aerosol')
    ax1.coastlines(lw=0.5)
    ax1.text(-0.03, 0.5, yr, transform=ax1.transAxes,
             rotation='vertical', va='center', ha='right', fontsize=13)

    ax2 = plt.subplot(len(yrs),2,(yi*2)+2,projection=ccrs.Robinson())
    nonoecdr.plot(ax=ax2,transform=ccrs.PlateCarree(),
                  add_colorbar=True,cbar_kwargs={'label':labelr},
                  vmin=limr[0],vmax=limr[1],
                  cmap='coolwarm')
    ax2.set_title('nonOECD aerosol')
    ax2.coastlines(lw=0.5)

plt.tight_layout()


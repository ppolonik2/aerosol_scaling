import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys

#   Make map of oecd countries and time series of global CO2, OECD/nonOECD SO2
oecd = ['USA','GBR','BEL','FRA','DEU','NLD','MEX','TUR','FIN','GRC',
        'ISL','ISR','ITA','LUX','NOR','PRT','ESP','KOR','DNK','NZL','SWE',
        'AUT','IRL','CHE','CHL','EST','HUN','SVK','SVN','AUS','JPN','POL',
        'CAN','LVA','LTU','COL','CRI','CZE']

shapes = gpd.read_file('~/tempprecip/gdp/geo/ne_10m_admin_0_countries.shp')
#         NE dataset doesn't have standard ISO_A3 for FRA and NOR
#         Instead use SU_A3 for those two countries as the ISO_A3
shapes.loc[shapes.SU_A3=='FRA','ISO_A3'] = 'FRA'
shapes.loc[shapes.SU_A3=='NOR','ISO_A3'] = 'NOR'
shapes.loc[shapes.SU_A3=='FRA','ISO_N3'] =  250
shapes.loc[shapes.SU_A3=='NOR','ISO_N3'] =  578

shapes['OECD'] = 'nonOECD'
shapes.loc[shapes['ISO_A3'].isin(oecd),'OECD'] = 'OECD'

oecdc    = 'chocolate'
nonoecdc = 'steelblue'
plt.figure(figsize=(12,3.2))
ax1 = plt.subplot(1,2,1,projection=ccrs.Robinson())
shapes.plot(ax=ax1,transform=ccrs.PlateCarree(),
            color=shapes['OECD'].map({'OECD':oecdc,'nonOECD':nonoecdc}))
shapes.boundary.plot(ax=ax1,transform=ccrs.PlateCarree(),color='k',lw=0.5)

#aeremis = xr.open_dataset('/home/ppolonik/CERM/output/ssp3_NSMonly/aeremis.nc')
#ghgemis = xr.open_dataset('/home/ppolonik/CERM/output/ssp3_NSMonly/ghgemis.nc')
aeremis = xr.open_dataset('../data/fair/aeremis_ssp3.nc')
ghgemis = xr.open_dataset('../data/fair/ghgemis_ssp3.nc')
ax2 = plt.subplot(1,2,2)
l1 = aeremis['so2_oecd'].mean('config').plot(ax=ax2,label='OECD SO$_2$',color=oecdc,lw=3)
l2 = aeremis['so2_nonoecd'].mean('config').plot(ax=ax2,label='non-OECD SO$_2$',color=nonoecdc,lw=3)
ax2.set_ylabel('SO$_2$ [MT]',fontsize=14)
ax2.set_xlabel('')
ax2.set_xlim(pd.Timestamp('1850-01-01'),pd.Timestamp('2100-01-01'))
ax2.set_ylim(0,90)
tickyrs = [pd.Timestamp('{}-01-01'.format(y)) for y in range(1850,2101,50)] 
ax2.set_xticks(tickyrs,labels=[t.year for t in tickyrs],rotation=0,ha='center')
ax3 = ax2.twinx()
ax3.set_ylim(0,75)
l3 = ghgemis['co2'].mean('config').plot(ax=ax3,label='CO$_2$',color=[0.5]*3,lw=3)
ax3.plot([pd.Timestamp('2020-01-01')]*2,[0,75],'--k',lw=1)
ax3.set_ylabel('CO$_2$ [GT]',fontsize=14,rotation=270,va='bottom')
ax3.set_xlabel('')

lines = l1+l2+l3
ax3.legend(lines,[l.get_label() for l in lines],loc=2)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1,top=0.9)
plt.annotate('(a)',(0.02,0.93),xycoords='figure fraction',fontsize=14)
plt.annotate('(b)',(0.51,0.93),xycoords='figure fraction',fontsize=14)



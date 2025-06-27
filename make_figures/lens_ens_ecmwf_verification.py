import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

coefser  = pd.read_csv('../data/regres/verify/reg_coefs_ECMWF_rolling_0624.csv',index_col=0)
coefslr  = pd.read_csv('../data/regres/verify/reg_coefs_LENS_rolling_0624.csv',index_col=0)
coefslr5 = pd.read_csv('../data/regres/verify/reg_coefs_LENS_sub5_rolling_0624.csv',index_col=[0,1])

ensalph = 0.5

# Make figure
# CESM LENS
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,height_ratios=[2,1],figsize=(13,6),
                                                     sharex='all',sharey='row')
coefslr[['GHG','OECD aerosol','nonOECD aerosol']].plot(ax=ax1)
coefslr[['Intercept']].plot(ax=ax1,color='gold')
coefslr[['Total, no int']].plot(ax=ax1,color=[0.5]*3)
coefslr[['Total']].plot(ax=ax1,color='k')
coefslr[['LENS dT']].plot(ax=ax1,color='k',linestyle='--')
ax1.set_ylabel('Coefficient ($\degree$C)',fontsize=13)
ax1.set_xlabel('')
ax1.set_title('CESM LENS2, Ensemble Mean')
ax1.grid(axis='y',alpha=0.6)
ax1.set_xlim([1940,2024])
ax1.annotate('(a)',(0.02,0.03),xycoords='axes fraction',ha='left',va='bottom',fontsize=14)

# CESM LENS individual ensembles
for ens in range(5):
    if ens==0:
        leg=True
    else:
        leg=False
    coefslru = coefslr5.loc[ens]
    coefslru[['GHG']].plot(ax=ax2,alpha=ensalph,color='C0',legend=leg)
    coefslru[['OECD aerosol']].plot(ax=ax2,color='C1',alpha=ensalph,legend=leg)
    coefslru[['nonOECD aerosol']].plot(ax=ax2,color='C2',alpha=ensalph,legend=leg)
    coefslru[['Intercept']].plot(ax=ax2,color='gold',alpha=ensalph,legend=leg)
    coefslru[['Total, no int']].plot(ax=ax2,color=[0.5]*3,alpha=ensalph,legend=leg)
    coefslru[['Total']].plot(ax=ax2,color='k',alpha=ensalph,legend=leg)
    coefslru[['LENS dT']].plot(ax=ax2,color='k',linestyle='--',alpha=ensalph,legend=leg)
ax2.set_ylabel('Coefficient ($\degree$C)',fontsize=13)
ax2.set_xlabel('')
ax2.set_title('CESM LENS2, 5 Ensembles Members')
ax2.grid(axis='y',alpha=0.6)
ax2.set_xlim([1940,2024])
ax2.annotate('(b)',(0.02,0.03),xycoords='axes fraction',ha='left',va='bottom',fontsize=14)

# ECMWF
coefser[['GHG','OECD aerosol','nonOECD aerosol']].plot(ax=ax3)
coefser[['Intercept']].plot(ax=ax3,color='gold')
coefser[['Total, no int']].plot(ax=ax3,color=[0.5]*3)
coefser[['Total']].plot(ax=ax3,color='k')
coefser[['ECMWF dT']].plot(ax=ax3,color='k',linestyle='--')
ax3.set_ylabel('Coefficient [$\degree$C]',fontsize=13)
ax3.set_xlabel('')
ax3.set_title('ECMWF')
ax3.grid(axis='y',alpha=0.6)
ax3.set_xlim([1940,2024])
ax3.set_ylim(np.array(ax3.get_ylim())*np.array([0.9,0.95]))
ax3.annotate('(c)',(0.02,0.03),xycoords='axes fraction',ha='left',va='bottom',fontsize=14)

# R^2
coefslr['r2'].loc[1941:2050].plot(ax=ax4)
ax4.set_ylabel('R$^2$',fontsize=13)
ax4.set_ylim(0,0.8)
ax4.set_xlim([1940,2024])
ax4.grid(axis='y',alpha=0.6)
ax4.annotate('(d)',(0.02,0.96),xycoords='axes fraction',ha='left',va='top',fontsize=14)

for ens in range(5):
    coefslru = coefslr5.loc[ens]
    coefslru['r2'].loc[1941:2050].plot(ax=ax5,alpha=ensalph,color='C0')
ax5.set_ylabel('R$^2$',fontsize=13)
ax5.set_ylim(0,0.8)
ax5.set_xlim([1940,2024])
ax5.grid(axis='y',alpha=0.6)
ax5.annotate('(e)',(0.02,0.96),xycoords='axes fraction',ha='left',va='top',fontsize=14)

coefser['r2'].plot(ax=ax6)
ax6.set_ylabel('R$^2$',fontsize=13)
#ax6.set_ylim(0,0.15)
ax6.set_ylim(0,0.8)
ax6.grid(axis='y',alpha=0.6)
ax6.set_xlim([1940,2024])
ax6.annotate('(f)',(0.02,0.96),xycoords='axes fraction',ha='left',va='top',fontsize=14)

for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
    ax.tick_params(labelbottom=True,labelleft=True)


plt.tight_layout()

corghg     = np.corrcoef(coefslr['GHG'].dropna(),            coefser['GHG'].dropna())[1,0]
coroecd    = np.corrcoef(coefslr['OECD aerosol'].dropna(),   coefser['OECD aerosol'].dropna())[1,0]
cornonoecd = np.corrcoef(coefslr['nonOECD aerosol'].dropna(),coefser['nonOECD aerosol'].dropna())[1,0]

slopeghg     = np.polyfit(coefslr['GHG'].dropna(),            coefser['GHG'].dropna(),1)[0]
slopeoecd    = np.polyfit(coefslr['OECD aerosol'].dropna(),   coefser['OECD aerosol'].dropna(),1)[0]
slopenonoecd = np.polyfit(coefslr['nonOECD aerosol'].dropna(),coefser['nonOECD aerosol'].dropna(),1)[0]



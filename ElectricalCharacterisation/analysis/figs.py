import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
exec(open('analysis/common.py').read())
plt.rcParams.update({'font.size':8,'font.family':'serif','mathtext.fontset':'cm','axes.linewidth':0.7,
 'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'legend.frameon':False,
 'lines.linewidth':1.1,'savefig.bbox':'tight','savefig.pad_inches':0.02})
OUT='figures/'
W=(2.45,1.95)
V,C,G=hf; Cox=C.max()
# overlay
fig,ax=plt.subplots(figsize=W)
ax.plot(hf[0],hf[1]*1e12,'k',label='1 MHz, dark')
ax.plot(lf[0],lf[1]*1e12,color='#1f6fb4',label='1 kHz, dark')
ax.plot(li[0],li[1]*1e12,color='#d1782a',label='1 MHz, light')
ax.axhline(24.23,ls=':',color='gray',lw=0.8); ax.text(-9.6,22.6,r'$C_{FB}$',color='gray',fontsize=7)
ax.set_xlabel(r'$V_g$ (V)'); ax.set_ylabel(r'$C$ (pF)'); ax.set_xlim(-10,10); ax.set_ylim(8,32)
ax.legend(fontsize=6.5,loc='center right',bbox_to_anchor=(1.0,0.5),handlelength=1.2)
fig.savefig(OUT+'mis_overlay.pdf'); plt.close()
# 1/C2
y=1/C**2*1e-21
mm=(V>=-3.7-1e-6)&(V<=-3.3+1e-6); p=np.polyfit(V[mm],1/C[mm]**2,1)
VFB2=(1/Cox**2-p[1])/p[0]
fig,ax=plt.subplots(figsize=W)
ax.plot(V,y,'k.',ms=2.2,label='1 MHz data')
vv=np.linspace(-4.6,-2.6,10); ax.plot(vv,(p[0]*vv+p[1])*1e-21,color='#c0392b',lw=1,label='linear fit')
ax.axhline(1/Cox**2*1e-21,ls='--',color='#1f6fb4',lw=0.8); ax.text(-9.6,1/Cox**2*1e-21+0.3,r'$1/C_{ox}^2$',color='#1f6fb4',fontsize=7)
ax.plot([VFB2],[1/Cox**2*1e-21],'o',mfc='none',mec='#c0392b',ms=5); ax.annotate(r'$V_{FB}=%.2f$ V'%VFB2,(VFB2,1/Cox**2*1e-21),(-9.5,3.0),fontsize=7,arrowprops=dict(arrowstyle='-',lw=0.5))
ax.set_xlabel(r'$V_g$ (V)'); ax.set_ylabel(r'$1/C^2$ ($10^{21}$ F$^{-2}$)'); ax.set_xlim(-10,2); ax.set_ylim(0,10)
ax.legend(fontsize=6.5,loc='upper left',bbox_to_anchor=(0,1.0))
fig.savefig(OUT+'mis_invC2.pdf'); plt.close()
# Maserjian
o=np.argsort(V); Vs=V[o]; Cs=savgol_filter(C[o],5,2); Y=np.gradient(Cs,Vs)/Cs**3
m=(Vs>-6)&(Vs<-2.5); i=np.argmin(Y[m]); Ymin=Y[m][i]; Vmin=Vs[m][i]
ma=(Vs<Vmin)&(Vs>-7); VFBm=np.interp(Ymin/3,Y[ma][::-1],Vs[ma][::-1])
fig,ax=plt.subplots(figsize=W)
ax.plot(Vs,Y*1e-21,'k',lw=1)
ax.axhline(Ymin*1e-21,ls='--',color='#c0392b',lw=0.8); ax.axhline(Ymin/3*1e-21,ls=':',color='#1f6fb4',lw=0.8)
ax.plot([VFBm],[Ymin/3*1e-21],'o',mfc='none',mec='#1f6fb4',ms=5)
ax.text(-9.6,Ymin*1e-21+0.12,r'$Y_{min}\Rightarrow N_A$',color='#c0392b',fontsize=7)
ax.text(-9.6,Ymin/3*1e-21+0.12,r'$Y_{min}/3\Rightarrow V_{FB}$',color='#1f6fb4',fontsize=7)
ax.set_xlabel(r'$V_g$ (V)'); ax.set_ylabel(r'$C^{-3}\,dC/dV_g$ ($10^{21}$ F$^{-2}$V$^{-1}$)'); ax.set_xlim(-10,2); ax.set_ylim(-3.4,0.3)
fig.savefig(OUT+'mis_maserjian.pdf'); plt.close()
# diode 1/C2 raw and corrected
Vd,Cd,_=pn; Cp=1.5e-12
fig,ax=plt.subplots(figsize=(2.9,2.0))
for cc,lab,col in [(0,'raw $C$','k'),(Cp,r'$C-C_p$, $C_p=1.5$ pF','#c0392b')]:
    yy=1/(Cd-cc)**2*1e-21; pp=np.polyfit(Vd[Vd>=2],yy[Vd>=2],1)
    ax.plot(Vd,yy,'.',color=col,ms=2,label=lab); x=np.linspace(-1.5,15,5); ax.plot(x,np.polyval(pp,x),'-',color=col,lw=0.6)
ax.axvline(0,color='gray',lw=0.5)
ax.set_xlabel(r'$V$ (V) (reverse bias)'); ax.set_ylabel(r'$1/C^2$ ($10^{21}$ F$^{-2}$)'); ax.set_xlim(-1.5,15.5); ax.set_ylim(0,24)
ax.legend(fontsize=6.5,loc='upper left')
fig.savefig(OUT+'diode_invC2.pdf'); plt.close()
print('VFB2',VFB2,'VFBm',VFBm)

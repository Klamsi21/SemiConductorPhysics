import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7.5,'font.family':'serif','mathtext.fontset':'cm','savefig.bbox':'tight','savefig.pad_inches':0.02})
phiF=0.36; Eg2=0.56
cases=[('Accumulation\n$V_g<V_{FB}$',-0.25,0.10,0.55),('Depletion\n$V_{FB}<V_g<V_T$',0.35,0.40,-0.55),('Strong inversion\n$V_g>V_T$',0.80,0.55,-1.15)]
fig,axs=plt.subplots(1,3,figsize=(3.6,1.9),sharey=True)
x=np.linspace(0,1,300)
for ax,(t,ps,w,EFm) in zip(axs,cases):
    psi=np.where(x<w,ps*(1-x/w)**2,0)
    Ei=phiF-psi; Ec=Ei+Eg2; Ev=Ei-Eg2
    ax.axvspan(-0.28,0,color='0.85',lw=0)          # oxide
    ax.plot([-0.55,-0.28],[EFm,EFm],'k',lw=1.2)        # metal EF
    ax.text(-0.42,EFm+0.08,r'$E_{F,m}$',ha='center',fontsize=6.5)
    ax.plot(x,Ec,'k',lw=1); ax.plot(x,Ev,'k',lw=1); ax.plot(x,Ei,'k--',lw=0.6)
    ax.plot([0,1],[0,0],color='#c0392b',lw=0.9)
    if ax is axs[0]:
        ax.text(1.02,Ec[-1],'$E_C$',va='center',fontsize=6.5); ax.text(1.02,Ev[-1]-0.12,'$E_V$',va='center',fontsize=6.5)
        ax.text(1.02,Ei[-1]+0.05,'$E_i$',va='center',fontsize=6.5); ax.text(1.02,0.1,'$E_F$',va='center',color='#c0392b',fontsize=6.5)
    if ps<0: ax.text(0.07,0.13,'$h^+$',color='#1f6fb4',fontsize=7)
    if ps>0: ax.annotate('',xy=(0,-1.45),xytext=(w,-1.45),arrowprops=dict(arrowstyle='<->',lw=0.6)); ax.text(w/2,-1.62,'$W$',ha='center',fontsize=6.5)
    if ps>0.6: ax.text(0.13,0.17,'$e^-$',color='#1f6fb4',fontsize=7)
    ax.set_title(t,fontsize=7,pad=2); ax.set_xlim(-0.58,1.12); ax.set_ylim(-1.75,1.4); ax.axis('off')
    ax.text(-0.42,1.25,'M',ha='center',fontsize=6.5); ax.text(-0.14,1.25,'O',ha='center',fontsize=6.5); ax.text(0.5,1.25,'p-Si',ha='center',fontsize=6.5)
fig.subplots_adjust(wspace=0.2)
fig.savefig('figures/mis_bands.pdf')

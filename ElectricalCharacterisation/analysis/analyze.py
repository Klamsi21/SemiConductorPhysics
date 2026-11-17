import numpy as np, sys
from scipy.optimize import brentq
from scipy.signal import savgol_filter
D='data/'
def load(f):
    V=[];C=[];G=[]
    for i,l in enumerate(open(D+f)):
        if i<2: continue
        if l.startswith('SUMMARY') or not l.strip(): break
        p=l.split(','); V.append(float(p[0]));C.append(float(p[1]));G.append(float(p[2]))
    return np.array(V),np.array(C),np.array(G)
q=1.602e-19; e0=8.854e-14; eSi=11.7*e0; eox=3.9*e0; kT=0.0259; ni=1.45e10
A=4.0e-4
hf=load('mis_1MHz.MCV'); lf=load('mis_1kHz.MCV'); li=load('mis_light.MCV'); hy=load('mis_hyst.MCV'); pn=load('diode_CV.MCV')
V,C,G=hf
print('expected Cox 45/50nm A=4e-4:',eox*A/45e-7*1e12,eox*A/50e-7*1e12,' A=6.4e-4:',eox*6.4e-4/45e-7*1e12,eox*6.4e-4/50e-7*1e12)
Cox=C.max(); print('Cmax HF',Cox, 'hyst Cmax',hy[1].max(),'LF',lf[1].max(),'light',li[1].max())
for c in [Cox, hy[1].max()]:
    print(' tox(A=4e-4) nm',eox*A/c*1e7,' tox(A=6.4e-4)',eox*6.4e-4/c*1e7)
Cmin=C.min(); print('Cmin HF',Cmin,'at',V[C.argmin()])
Csi=1/(1/Cmin-1/Cox); W=eSi*A/Csi; print('Csi,min',Csi,'Wmax um',W*1e4)
f=lambda N: np.sqrt(4*eSi*kT*np.log(N/ni)/(q*N))-W
NA=brentq(f,1e13,1e19); print('NA CminCmax',NA)
phiF=kT*np.log(NA/ni); print('phiF',phiF)
LD=np.sqrt(eSi*kT/(q*NA)); CFB=1/(1/Cox+LD/(eSi*A)); print('LD um',LD*1e4,'CFB',CFB,'CFB/Cox',CFB/Cox)
# VFB on HF curve: interpolate on the falling branch (accum->depletion), V in [-6,-2]
m=(V<-2)&(V>-6.5); idx=np.argsort(C[m]); VFB=np.interp(CFB,C[m][idx],V[m][idx]); print('VFB CminCmax',VFB)
phims_poly=-(0.56+phiF); phims_Al=4.10-(4.05+0.56+phiF); print('phims poly',phims_poly,'Al',phims_Al)
for pm in [phims_poly,phims_Al]:
    print('  Ns',Cox*(pm-VFB)/(q*A))
Qd=np.sqrt(4*eSi*q*NA*phiF); Vt=VFB+2*phiF+Qd/(Cox/A); print('Vt theory',Vt)
# 1/C^2 method: fit window
y=1/C**2
best=None
for a in np.arange(-4.6,-3.0,0.1):
  for b in np.arange(a+0.4,-2.4,0.1):
    mm=(V>=a-1e-6)&(V<=b+1e-6)
    if mm.sum()<5: continue
    p,res,*_=np.polyfit(V[mm],y[mm],1,full=True)
    r2=1-res[0]/np.sum((y[mm]-y[mm].mean())**2)
    # choose max slope region with r2>0.998
    if r2>0.998 and (best is None or p[0]>best[0][0]): best=(p,a,b,r2)
p,a,b,r2=best; print('1/C2 fit window',a,b,'r2',r2,'slope',p[0])
NA2=2/(q*eSi*A**2*p[0]); VFB2=(1/Cox**2-p[1])/p[0]; print('NA 1/C2',NA2,'VFB 1/C2',VFB2)
for pm in [phims_poly]:
    print('  Ns(1/C2)',Cox*(pm-VFB2)/(q*A))
# also fit fixed window -4.3..-3.3
for (a,b) in [(-4.3,-3.3),(-4.0,-3.0),(-4.2,-3.6)]:
    mm=(V>=a-1e-6)&(V<=b+1e-6); pp=np.polyfit(V[mm],y[mm],1); print(' window',a,b,'NA',2/(q*eSi*A**2*pp[0]),'VFB',(1/Cox**2-pp[1])/pp[0])
# Maserjian
o=np.argsort(V); Vs=V[o]; Cs=savgol_filter(C[o],7,2); dC=np.gradient(Cs,Vs); Y=np.abs(dC/Cs**3)
mm=(Vs>-4.8)&(Vs<-2.0)
i=np.argmin(Y[mm]); print('Maserjian min',Y[mm][i],'at',Vs[mm][i],' NA',1/(q*eSi*A**2*Y[mm][i]))
Ymax=Y[(Vs<-3)&(Vs>-7)].max(); 
print('Ymax',Ymax)
# G analysis
print('G accum',G[-1],'wC',2*np.pi*1e6*C[-1],'Rs approx',G[-1]/(2*np.pi*1e6*C[-1])**2)
print('G peak',G[(V<-2)].max(),'at',V[(V<-2)][G[(V<-2)].argmax()])
print('C HF at +10',C[0],'light +10',li[1][0],'Cmin light',li[1].min(),'at',li[0][li[1].argmin()],'Cmin LF',lf[1].min(),'at',lf[0][lf[1].argmin()],'LF at +10',lf[1][-1])
# hysteresis shift at CFB
Vh,Ch,_=hy; n=len(Vh)//2; f1=(Vh[:n],Ch[:n]); f2=(Vh[n:],Ch[n:])
def vat(v,c,cc):
    m=(v<-2.5)&(v>-7); idx=np.argsort(c[m]); return np.interp(cc,c[m][idx],v[m][idx])
v1=vat(*f1,CFB); v2=vat(*f2,CFB); print('hyst VFB fwd(+15->-15)',v1,'bwd(-15->+15)',v2,'dV',v2-v1,'dN',Cox*abs(v2-v1)/(q*A))
for cc in [0.5*(Cox+Cmin)]:
    print(' mid',vat(*f1,cc),vat(*f2,cc))
print('LF VFB-level crossing',vat(lf[0],lf[1],CFB),'light',vat(li[0],li[1],CFB))
# Diode
Vd,Cd,Gd=pn; yd=1/Cd**2
for (a,b) in [(0,15),(0,3),(3,15),(5,15),(0,1)]:
    mm=(Vd>=a-1e-6)&(Vd<=b+1e-6); pp=np.polyfit(Vd[mm],yd[mm],1)
    print('diode window',a,b,'N',2/(q*eSi*A**2*pp[0]),'Vbi',pp[1]/pp[0])
o=np.argsort(Vd); vv=Vd[o]; yy=savgol_filter(yd[o],9,2); dy=np.gradient(yy,vv)
Np=2/(q*eSi*A**2*dy); Wp=eSi*A/Cd[o]
for k in range(0,len(vv),10): print(' V %.1f W %.3f um N %.3e'%(vv[k],Wp[k]*1e4,Np[k]))

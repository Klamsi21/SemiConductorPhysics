import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
exec(open('analysis/common.py').read())
V,C,G=hf
o=np.argsort(V); Vs=V[o]; Cs=savgol_filter(C[o],5,2); Y=np.gradient(Cs,Vs)/Cs**3   # signed
m=(Vs>-6)&(Vs<-2.5)
i=np.argmin(Y[m]); Ymin=Y[m][i]; Vmin=Vs[m][i]; print('Y min',Ymin,'at',Vmin,'NA',1/(q*eSi*A**2*abs(Ymin)))
# VFB at Ymin/3 on accumulation side (V<Vmin)
ma=(Vs<Vmin)&(Vs>-7); vv=Vs[ma]; yy=Y[ma]
VFBm=np.interp(Ymin/3, yy[::-1], vv[::-1]); print('VFB Maserjian',VFBm)
print(np.c_[Vs[m],Y[m]][::2])
# diode pad cap
Vd,Cd,Gd=pn
def lin(cp):
    y=1/(Cd-cp)**2; p,res,*_=np.polyfit(Vd,y,1,full=True); return res[0]/np.sum((y-y.mean())**2)
for cp in np.arange(0,4.01e-12,0.5e-12):
    y=1/(Cd-cp)**2; p=np.polyfit(Vd,y,1); print('Cp %.1f pF  1-r2 %.2e  N %.3e Vbi %.3f'%(cp*1e12,lin(cp),2/(q*eSi*A**2*p[0]),p[1]/p[0]))
r=minimize_scalar(lin,bounds=(0,6e-12),method='bounded'); cp=r.x; y=1/(Cd-cp)**2; p=np.polyfit(Vd,y,1)
print('best Cp',cp,'N',2/(q*eSi*A**2*p[0]),'Vbi',p[1]/p[0])
print('Vbi theory N+ 1e20 / NA 1.3e16:',kT*np.log(1e20*1.3e16/ni**2))
# sidewall-area hypothesis for rake
for Aeff in [4e-4,4.03e-4+4072e-4*0.5e-4,4.03e-4+4072e-4*1e-4]:
    y=1/Cd**2; p=np.polyfit(Vd[Vd>3],y[Vd>3],1); print('Aeff',Aeff,'N',2/(q*eSi*Aeff**2*p[0]))

import numpy as np
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

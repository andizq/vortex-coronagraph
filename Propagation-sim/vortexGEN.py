"""
Created on 29-07-2015
by AFIC

Last modification: 05-11-2015
"""

#Ideal SPP's (N=256)

import numpy as np
import matplotlib.pyplot as plt
import funcionesMOD as fM
import cmath as cm
import math

def SPP(X,Y,m,r,n,B=False):
    
    #C = 0
    #if B == 0:
    #k = 0.    
    ang = 0
    #n = 10.
    C = np.zeros((len(X),len(Y)),'complex')
    theta = np.zeros((len(X),len(Y)))
    P = fM.circ(-(X[1]-X[0])*len(X)/2,(X[1]-X[0])*len(X)/2,X[1]-X[0],2*r,0)

    for i in np.arange(0,len(X)):
        for j in np.arange(0,len(Y)):                                                                          
                 
            ang = math.atan2(Y[j],X[i])+np.pi # 0<=ang<=2pi
            #print round(ang/(2.*np.pi/n))
            
            #para n grandes: esto pues cuando se multiplique theta
            #con un m distinto de 1, si n es pequegno, habran inconsistencias, ej.: n=3,m=4
            #theta1=-pi,theta2=0,theta3=pi; m*t1=-4pi=2pi, m*t2=0, m*t3=4pi=2pi, por tanto,
            #el ancho de cada region: 2pi/n debe ser tal que al multiplicarlos por m
            #no sobrepasen 2pi

            #theta[i,j]= 2.*np.pi/(n-1)*np.floor(ang/(2*np.pi/n/m))-np.pi
            #theta[i,j]= (2.*np.pi/(n-1)*np.floor(ang/(2*np.pi/n/m)))-np.pi*(1+n*m*np.floor(ang/(2*np.pi/m))) 
            #/n/m --> numero de divisiones en todo el circulo
            #2pi/(n-1) --> ancho entre angulos posibles. si n=3, ancho=pi: -pi,0,pi
            #np.floor da la parte entera mas baja de la entrada
            #cuando ang sea n veces (2pi/n/m), np.floor=n, theta>2pi-np.pi --> i.e comienza
            #una nueva ronda entre -pi y pi, el numero de rondas esta pues determinado por 
            #la carga topologica m

            theta[i,j]=ang+np.pi            
                    
            if P[i,j] != 0:                                                                                                                                                         
                C[i,j] = cm.exp(1j*m*theta[i][j])#/m)
    
    #C[len(X)/2.,len(X)/2.]=cm.exp(1j*0)
    C = C*P
    
    return C

#SPP's with discretized levels

import numpy as np
import matplotlib.pyplot as plt
import funcionesMOD as fM
import cmath as cm
import math
import coherImagLIB as CIL
import random

def SPP(X,Y,m,r,n,B=False):

    ang = 0
    C = np.zeros((len(X),len(Y)),'complex')
    theta = np.zeros((len(X),len(Y)))
    P = fM.circ(-(X[1]-X[0])*len(X)/2,(X[1]-X[0])*len(X)/2,X[1]-X[0],2*r,0) #X[1]-X[0]=dx, dx*len(X)/2=L/2
    
    #Lists with information of angles, taking into account the topological charge and phase levels

    LISTA = [np.linspace(-np.pi,np.pi-2*np.pi/n,n)]*m #dividing the intervals according to the number of levels
                                             #this is repeated m times
    LISTA = np.array(LISTA) #matrix
    LISTA = np.reshape(LISTA,n*m) #destroying the matrix to form an unique-row list of n*m elements

    ANGULOS = np.linspace(0,2*np.pi,n*m+1) #angles to be assigned to each pixel depending of its position
    ANGULOS = np.array(ANGULOS)

    for i in np.arange(0,len(X)):    
        for j in np.arange(0,len(Y)):           

            ang = math.atan2(Y[j],X[i])+np.pi
            for k in range(0,n*m):

                if ang>ANGULOS[k] and ang<=ANGULOS[k+1]: 
                    theta[j,i]=LISTA[k] #Si vario Y, deben variar las filas (j)
                    break
            
            if P[j,i] != 0:                          
                C[j,i] = cm.exp(1j*theta[j][i])
    
    C = C*P
    
    return C 



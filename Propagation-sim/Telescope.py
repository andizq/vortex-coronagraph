"""
Created on 28-07-2016
by AFIC

Last modification: 16-11-2016
"""

#MAIN SCRIPT. This script takes account of a single propagation
# throughout the complete optical system.

import numpy as np
import propagators as prop
import funcionesMOD as fM
import coherImagLIB as CIL
import matplotlib.pyplot as plt
import cmath as cm
import vortexGEN as vGEN #Many levels
import vortexGEN_fewlevels as vGEN_FL #Few levels
from matplotlib import gridspec

#-------------------------
#LENSES PARAMETERS (in mm)
#-------------------------

f1 = 1000. 
R1 = 50. 
f2 = 300.
R2 = 6. #Apperture stop radius
f3 = 250.

#-------------------------
#-------------------------

#---------
#SAMPLING
#---------

L = 100. #plane size in mm
M = 4096 #number of samples
dx = L/M

#R2 = 247*dx
#---------
#---------

#------------------
#OPTICAL PARAMETERS
#------------------
Factor = 1/L/dx

wl = 532.e-6 #wavelength in mm  (green)
k = 2.*np.pi/wl


Ray = 1.22*wl/(2*R2)
RayL = Ray*f2 #Approx. size of Airy disk at plane f2: Tan(Ray)=RayL/f2-->Ray=RayL/f2
RealS = 6/L*wl*f2 #At radius of Airy disk there are approx. 6 samples,df=1/L,scale=wl*f2  
                  # RayL is approx RealS
                  #  Computer Airy size: AiryC = 6samples*dx
                  #   RayL = 6*dx*Factor*wl*f2
                  #    thus, AiryC = RayL/(Factor*wl*f2)

AiryC = RayL/(Factor*wl*f2)


#Neptune: 4.3e9km from Earth (mV=8) 
# Triton: 3.6e5km from Neptune (mV=13) if Delta_mV=5 --> I1=2.5**5*I2=100*I2
#  Nep-Tri: alpha from Earth is approx. 2*Ray <--> Tan(alpha)=3.6e5/4.3e9 
#   Nereid: 5.5e6km from Neptune (mV=19 --> 1e5 times less brilliant than Nep.)
#    Nep-Ner: alpha from Earth is approx. 22*Ray 
#     Pluto: (mV=15) Delta_mV=7 --> I1=2.5**7*I3=610*I3

alpha = 0*2*Ray #1st angular separation in rad
phi = 0*4*Ray #2nd angular separation in rad

#------------------
#------------------

#------------
#PLANET OFFSET
#------------

Roffx = np.tan(alpha)*f2 #Real optical x-axis offset in mm
Roffy = np.tan(phi)*f2   #y-axis...
Roff = np.sqrt(Roffx**2+Roffy**2)

Coffx = Roffx/(Factor*wl*f2) #Computational x-axis offset in mm 
Coffy = Roffy/(Factor*wl*f2) #y-axis...
Coff = np.sqrt(Coffx**2+Coffy**2) #=np.sqrt((OffsetSamples*dx)**2+(OffsetSamples*dy)**2)


#If alpha=0.00012 --> the real offset is approx. alpha*f2=0.036, the same than
 #22*1/L*wl*f2=0.035 --> 22 offset samples in frequency(1/L) with the scale wl*f2,
  #the same than 22*dx*Factor*wl*f2, where Factor is the relation between df and dx
   #Factor=(1/L)/dx. The offset given by the computer is Coff=22*dx=Roff/(Factor*wl*f2)   


#To reach AiryC we need an AiryAngle such that AiryC=Coffx, Roffx=RayL !! we already have
# the condition to reach AiryC: we need to reach in the reality RayL, which is reached
#  with Ray, of this way due to the explanation of the last paragraph we get AiryC:
#   with Ray we get RayL=Roffx in the real life but in the computer we get AiryC=Coffx. 

#------------
#------------

#--------------
#APPERTURE STOP
#--------------

#x=np.arange(-L/2,L/2+L/(M-1),L/(M-1))
#y=np.arange(-L/2,L/2+L/(M-1),L/(M-1))

x=np.arange(-L/2,L/2,dx)
y=np.arange(-L/2,L/2,dx)



AS = np.zeros((M,M))
lim1=M/2-(int(R2/dx)+10) #Number of pixels needed to reach R2 plus a 
lim2=M/2+(int(R2/dx)+10) # little extra-range, centered in M/2 (This process is made to avoid big calculations with the whole MxM matrix)
AS[lim1:lim2,lim1:lim2] = fM.circ(x[lim1],x[lim2],dx,2*R2,0)    

                                     #2*R2: Function expects the diameter
                                      #does not return values of x and 
                                       #gives the circular shape of the App.Stop

#--------------
#--------------

#--------------
#INITIAL FIELD
#--------------

z = 0
X,Y = np.meshgrid(x,y)

E0 = np.ones((M,M),'complex')#/np.sqrt(610)

#Let's include the phase term due to the inclination of the input plane


E0 = E0*np.exp(1j*k*(z*np.cos(alpha)*np.cos(phi)+X*np.sin(alpha)+
                     Y*np.cos(alpha)*np.sin(phi)))


#If one have a list and not an array, one have to work with the notation L[][] to 
# call an element from it, this is the case of X (see above), which is a natural list


Fase = CIL.phase(E0)

#See matplotlib.rcParams for general plotting parameters

ax=plt.axes()
ax.set_title("$Initial$ $Field$ $phase$")
ax.set_xlabel("$N_{x}$",labelpad=15)
ax.set_ylabel("$N_{y}$") 
plt.imshow(Fase)
cb=plt.colorbar()
cb.set_label("$Phase$ $Value$")
plt.show()

#---------------
#---------------

#-------------------------------
#FIELD AFTER THE FIRST APPERTURE
#-------------------------------

E1 = 0
E0 = (E0+E1)*AS

ax=plt.axes()
ax.set_title("$Telescope$ $Apperture,$ $z=0$",fontsize=17,position=(0.5,1.0))
ax.set_xlabel("$N_{x}$",labelpad=15)
ax.set_ylabel("$N_{y}$") 
plt.imshow((abs(E0)**2)[M/2-800:M/2+800,M/2-800:M/2+800],cmap='gray')
plt.show()

print np.sum(abs(E0)**2*dx*dx)

#-------------------------------
#-------------------------------

#--------------------------
#FIRST FOURIER PLANE (z=f2)
#--------------------------

#At the focal plane we have the Fourier transform of the initial Field times the pupil function
 #Lens2 (HAVE TO BE SCALED)

#fx2 = np.arange(-1/(2.*(L/(M-1))),1/(2.*(L/(M-1))),1./L-0.005/4096)
fx2 = np.arange(-1/(2.*dx),1/(2.*dx),1./L)
fy2 = fx2
 
Fx2,Fy2 = np.meshgrid(fx2,fy2)

Ef2 = np.fft.fftshift(E0)
Ef2 = np.fft.fft2(Ef2)
Ef2 = np.fft.ifftshift(Ef2)
Ef2 = Ef2*dx*dx
Ef2 = Ef2*np.exp(1j*k*f2*wl*wl*(Fx2*Fx2+Fy2*Fy2))#/(1j*wl*f2)

ax=plt.axes()
ax.set_title("$First$ $Fourier$ $Plane,$ $z=f_{2}$",fontsize=16,position=(0.5,1.0))
ax.set_xlabel("$N_{x}$",labelpad=15)
ax.set_ylabel("$N_{y}$") 
plt.imshow((abs(Ef2)**2)[M/2-148:M/2+148,M/2-148:M/2+148],cmap='gray')
cb=plt.colorbar()
cb.set_label(r"$Intensity$ $units$",fontsize=12)
plt.show()

print "Incoming max Intensity = ",(abs(Ef2)**2).max() #Pot/m**2. In a plane wave I=c*epsilon*E**2-->
                                                      # See Hector_Alzate Pag.128 Fisica de las ondas
print np.sum(abs(Ef2)**2*1./L**2)         #Potencia=I*m**2 


#-------------------
#-------------------

#----------------------
#INTRODUCING SPP (z=f2)
#----------------------

#Now let's model the SPP using its characteristic transmitance times a circ function  

C = np.zeros((M,M),'complex')

RC = L/9# 2*R2 #Radius of the SPP
m = 2
N = 4
lim1C=M/2-(int(RC/dx)+10) #Number of pixels needed to reach Rf plus a                                                                                                               
lim2C=M/2+(int(RC/dx)+10) # little extra-range, centered in M/2.                                                                                                                  
                          #  This is to minimize the computational journey

#C[lim1C:lim2C,lim1C:lim2C] = vGEN.SPP(x[lim1C:lim2C],y[lim1C:lim2C],m,RC,256)  #RC: Function expects the radius
C[lim1C:lim2C,lim1C:lim2C] = vGEN_FL.SPP(x[lim1C:lim2C],y[lim1C:lim2C],m,RC,N)  #RC: Function expects the radius

#C = vGEN.SPP(x,y,8,R2,256) #Optimo N=256. m=4-->32, Siguen m=5-->48, m=20--70, m=2,3=6=7-->80,30-->400,15-->1000, 25-->2000, 8-->3000
#C = vGEN_FL.SPP(x,y,4,R2,7) #Optimo N=4. m=2 --> 1%

FaseC = CIL.phase(C)

ax=plt.axes()
ax.set_title("$SPP.$ $m=%d,$ $N=%d,$ $z=f_{2}$"%(m,N),fontsize=16,position=(0.5,1.0))
ax.set_xlabel("$N_{x}$",labelpad=15)
ax.set_ylabel("$N_{y}$") 
plt.imshow(FaseC,cmap='gray')
cb=plt.colorbar()
cb.set_label(r"$Phase$ $value$",fontsize=12)
plt.show()

#---------------------
#---------------------

#---------------------
#FIELD PASSES BY C
#---------------------

Ev = Ef2*C
FaseEv=CIL.phase(Ev)

fig=plt.figure(figsize=(14,8))

ax0=plt.subplot(1,2,1)
ax0.set_title("$Field$ $phase$ $after$ $SPP,$ $z=f_{2}$",fontsize=16,position=(0.5,1.0))
ax0.set_xlabel("$N_{x}$",labelpad=15)
ax0.set_ylabel("$N_{y}$") 
map0=ax0.imshow(FaseEv[M/2-120:M/2+120,M/2-120:M/2+120],cmap='gray')
cb=plt.colorbar(map0,orientation='horizontal')
cb.set_label(r"$Phase$ $value$",fontsize=12)

ax1=plt.subplot(1,2,2)
ax1.set_title("$Intensity$ $field$ $after$ $SPP,$ $z=f_{2}$",fontsize=16,position=(0.5,1.0))
ax1.set_xlabel("$N_{x}$",labelpad=15)
ax1.set_ylabel("$N_{y}$") 
map1=ax1.imshow((abs(Ev)**2)[M/2-120:M/2+120,M/2-120:M/2+120])
cb=plt.colorbar(map1,orientation='horizontal')
cb.set_label(r"$Intensity$ $units$",fontsize=12)

plt.show()

print np.sum(abs(Ev)**2*1./L**2)

#-------------
#-------------

#------------------
#FIELD AT z=f2+2*f3
#------------------

#EvP = prop.propSypek(Ev,dx,wl,f3*1)

EvP = np.fft.fftshift(Ev)
EvP = np.fft.fft2(EvP)
EvP = np.fft.ifftshift(EvP)
EvP = EvP*1./L**2#/M#*dx*dx
EvP = EvP#/(1j*wl*f3)

ax=plt.axes()
ax.set_title("$Field$ $at$ $z=f_{2}+2*f_{3}$",fontsize=16,position=(0.5,1.0))
ax.set_xlabel("$N_{x}$",labelpad=15)
ax.set_ylabel("$N_{y}$") 
plt.imshow((abs(EvP)**2)[M/2-800:M/2+800,M/2-800:M/2+800],cmap='gray')
cb=plt.colorbar()
cb.set_label(r"$Intensity$ $units$",fontsize=12)
plt.show()

print np.sum(abs(EvP)**2*dx*dx)#*1./L**2)

#------------------
#------------------

#----------
#FILTERING
#----------

Rf = R2#*4/8. #Filtering radius
Aff = np.zeros((M,M))
lim1f=M/2-(int(Rf/dx)+10) #Number of pixels needed to reach Rf plus a                                                                                                               
lim2f=M/2+(int(Rf/dx)+10) # little extra-range, centered in M/2      
                                                                                                               
Aff[lim1f:lim2f,lim1f:lim2f] = fM.circ(x[lim1f],x[lim2f],dx,2*Rf,0)
Ef = EvP*Aff

ax = plt.gca()
ax.set_title("$Field$ $*$ $App.Stop,$ $z=f_{2}+2*f_{3}$",fontsize=16,position=(0.5,1.0))
ax.set_xlabel("$N_{x}$",labelpad=15)
ax.set_ylabel("$N_{y}$") 
plt.imshow((abs(Ef)**2)[M/2-800:M/2+800,M/2-800:M/2+800],cmap='gray')
cb=plt.colorbar(orientation='vertical')
cb.set_label(r"$Intensity$ $units$",fontsize=12)
plt.show()

print np.sum(abs(Ef)**2*dx*dx)

#----------
#----------

#--------
#IMAGING
#--------

Eff  = np.fft.fftshift(Ef)
Eff = np.fft.fft2(Eff)
Eff = np.fft.ifftshift(Eff)
Eff = Eff*dx*dx
Eff = Eff*np.exp(1j*k*f2*wl*wl*(Fx2*Fx2+Fy2*Fy2))#/(1j*wl*f2)

ax = plt.gca()
ax.set_title("$Intensity$ $field$ $at$ $camera,$ $z=f_{2}+3*f_{3}$",fontsize=16,position=(0.5,1.0))
ax.set_xlabel("$N_{x}$",labelpad=15)
ax.set_ylabel("$N_{y}$") 
plt.imshow((abs(Eff)**2)[M/2-120:M/2+120,M/2-120:M/2+120])#,cmap='gray')
cb=plt.colorbar(orientation='vertical')
cb.set_label(r"$Intensity$ $units$",fontsize=12)
plt.show()

print np.sum(abs(Eff)**2*1./L**2),m,N,L,M,dx
print "Final max Intensity = ",(abs(Eff)**2).max()

#plt.close()

#--------
#--------

#f = open("writing.txt",'w')
#for i in np.arange(0,len(x)): f.write("%f\n"%x[i])





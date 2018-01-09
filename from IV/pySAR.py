# SAR Sandbox in Python
# Author: Ignacio Chechile
# ICEYE 2017
# Based in "Synthetic Aperture Radar Imaging Simulated in MATLAB" by Matthew Schlutz (Master thesis for CalPoly)
# Source: http://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=1100&context=theses

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D     
import sys

#Helper FFT and IFFT functions (for y dimension we swap the axes, which equals to transpose the array)
def fty(s):
	return np.transpose(np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.transpose(s)))))

def ifty(fs):
	return np.transpose(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(np.transpose(fs)))))

def ftx(s):
	return np.fft.fftshift(np.fft.fft(np.fft.fftshift(s)))

def iftx(fs):
	return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(fs)))

# INITIALIZATION (User can modify these values to visualize how they impact in SAR processing
# Main SAR Parameters
PRF=400; # Pulse Repetition Frequency (Hz)
dur=1; # Time of Flight (sec), PRF*dur = received echoes
vp=300; # Velocity of platform 
fo=9.5e9; # Carrier frequency 
La=2.0; # Antenna length actual [m]
Xc=20000; # Range distance to center of target area
X0=200; # Half Target Area Width (Target is located within [Xc-X0,Xc+X0])
Tp=.25e-5; # Chirp Pulse Duration
B0=60e6; # Baseband bandwidth is plus/minus B0\
target_name='scene7s.gif'; #Name of Target Profile Image (GIF Grayscale Image)
noise=0; # Set this flag to add noise to signal
std_dev=.08; # Standard Deviation of Noise



# ----------------------------------------
# Software variables (do not modify)
cj= np.sqrt(-1+0j)
c=3e8; # Propagation speed
ic=1/c; # Propagation frequency
Lambda=c/fo; # Wavelength (60cm for fo = 4.5e9)
eta=np.linspace(0,dur,PRF*dur) # Slow Time (Azimuth time) Array
# Range Parameters
Kr=B0/Tp; # Range Chirp Rate
dt=1/(2*B0); # Time Domain Sampling Interval
Ts=(2*(Xc-X0))/c; # Start time of sampling
Tf=(2*(Xc+X0))/c+Tp; # End time of sampling
# Azimuth Parameters
Ka=(2*np.power(vp,2))/(Lambda*(Xc)); # Linear Azimuth FM rate
# Measurement Parameters
rbins=2*np.ceil((.5*(Tf-Ts))/dt); # Number of time (Range) samples/bins

t = np.arange(int(rbins))

t = Ts + t*dt

s=np.zeros((PRF*dur,int(rbins)),dtype=complex) # We created the signal array

# Target Initialization
#target=imread(target_name,'gif'); #Select Input Target Profile
target = ndimage.imread("scene7s.gif")

M,N = target.shape[0],target.shape[1] 
ntarget = M*N
#Target Intialization Variables
tnum=0
xn=np.zeros(ntarget,dtype=float)
yn=np.zeros(ntarget,dtype=float)
Fn=np.zeros(ntarget,dtype=float)


# Here we get the reflectivity matrix from the file we loaded. File is a grayscale GIF, you can play with different gray levels to get lower reflectivities
for m in range(1,M):
	for n in range(1,N):
		xn[tnum]=(n-N/2)
		yn[tnum]=(M/2-m+1)
		Fn[tnum]=float(target[m][n][0])/255 #Fm is the reflectivity (0 to 1) from targets as calculated from image
		tnum=tnum+1

stretch=3.0
xn=xn*stretch
yn=yn*stretch #Stretch out Target Profile


#Amount of iterations here is PRF*dur*ntarget
#Generate echoes and store in array s
for j in range (1,PRF*dur):
	for i in range(1,ntarget):
		wa=np.power(np.sinc(La*(np.arctan(vp*(eta[j]-dur/2+yn[i]/vp)/Xc))/Lambda),2) #Azimuth amplitude modulation in azimuth time (Eq.6)
		R=np.sqrt(np.power(Xc+xn[i],2) + np.power(vp,2) * np.power((eta[j]-dur/2+yn[i]/vp),2)) #Range computation (Eq.7)

		td=t-(2*R/c) #delay for return pulses

		aux1 = td>=0
		aux2 = td<=Tp
		mask = aux1 & aux2
		mask = mask.astype(int)
		s[j]=s[j]+noise * std_dev * np.random.rand(s.shape[1]) + Fn[i]*wa*np.exp(-cj*(4*np.pi*fo*ic*R)+cj*np.pi*Kr*(np.power(td,2)-td*Tp)) * mask #Eq. 4
		#s[j]=s[j]+noise * std_dev * np.random.rand(s.shape[1]) + Fn[i]*wa*np.exp(-cj*(4*np.pi*fo*ic*R)+cj*np.pi*Kr*(np.power(td,2)-td*Tp))*(0.5*(1-np.cos((2*np.pi*td*rbins/Tp)/(int(rbins)-1)))) * mask

	if np.mod(j,50)==0:
		print('{0:.0f}'.format((j/(PRF*dur))*100),"%") #Echo Gen. Percent Complete
	

figure = plt.figure()
figure.suptitle("Echos in samples domain")
plt.contour(abs(s))

# RANGE DOPPLER ALGORITHM (RDA)
# Range Reference Signal
aux1 = td>=0
aux2 = td<=Tp
mask = aux1 & aux2
mask = mask.astype(int)

td0=t-(2*(Xc/c)) 
pha20=np.pi*Kr*((np.power(td0,2))-td0*Tp)
s0=np.exp(cj*pha20) * mask
fs0=fty(s0) # Reference Signal in frequency domain

figure = plt.figure()
figure.suptitle("Range reference signal in freq domain")
plt.plot(abs(fs0))


# Power equalization
amp_max=1/np.sqrt(2) # Maximum amplitude for equalization
afsb0=abs(fs0)
P_max=max(afsb0);
I=np.where(afsb0 >= amp_max*P_max)[0]
fs0[I]=((amp_max*(np.power(P_max,2))*np.ones(len(I)))/afsb0[I])*np.exp(cj*np.angle(fs0[I]))

# Range cell migration
deltaR=2*np.power(Lambda,2)*(Xc)*np.power(Ka*(dur*0.5-eta),2)/(8*np.power(vp,2)) 
cells=np.round(deltaR/.56); # .56 meters/cell in range direction

figure = plt.figure()
figure.suptitle("Range migration cells")
plt.plot(cells)

rcm_max=9 #maximum range cell migration

#We create all the arrays for SAR processing
fs=np.zeros((PRF*dur,int(rbins)),dtype=complex) 
fsm=np.zeros((PRF*dur,int(rbins)),dtype=complex)  
fsmb=np.zeros((PRF*dur,int(rbins)),dtype=complex)  
fsmb2=np.zeros((PRF*dur,int(rbins)),dtype=complex)  
smb=np.zeros((PRF*dur,int(rbins)),dtype=complex)  
fsac=np.zeros((PRF*dur,int(rbins)),dtype=complex)  
sac=np.zeros((PRF*dur,int(rbins)),dtype=complex) 

#Range Compression
for k in range(1,(PRF*dur)):
	fs[k]=fty(s[k])
	#Range FFT
	fsm[k]=fs[k]*np.conj(fs0) #Range Matched Filtering
	smb[k]=ifty(fsm[k])

figure = plt.figure()
figure.suptitle("Echoes after range compression")
plt.contour(abs(smb))

#Azimuth Reference Signal
smb0=np.exp(cj*np.pi*Ka*eta*(2*eta[int(PRF*dur/2+1)]-eta))
fsmb0=ftx(smb0) #Azimuth Matched Filter Spectrum

figure = plt.figure()
figure.suptitle("Azimuth reference signal, freq domain")
plt.plot(abs(fsmb0))

for l in range(1,int(rbins)):
	fsmb[:,l]=ftx(smb[:,l])# Azimuth FFT

#Range Cell Migration Correction (RCMC)
for k in range(1,(dur*PRF)):
	for m in range(1,int(rbins)-rcm_max):
		try:
			fsmb2[k][m]=fsmb[k][m+int(cells[k])]
		except IndexError:#This is to avoid the error because we start from 1 (Because of MATLAB)
			continue

for l in range(1,int(rbins)):
    fsac[:,l]=iftx(fsmb2[:,l]) #Azimuth IFFT

#Azimuth Compression
for l in range(1,int(rbins)):
    fsac[:,l]=fsmb2[:,l]*np.conj(fsmb0) # Azimuth Matched Filtering in frequency domain (instead of convolution in time)
    sac[:,l]=iftx(fsac[:,l]) # Azimuth IFFT / Final Target Image

figure = plt.figure()
figure.suptitle("Final target contour, freq domain")
plt.contour(abs(fsac))

figure = plt.figure()
figure.suptitle("Final target contour graph")
plt.contour(abs(sac))

figure = plt.figure()
figure.suptitle("Final target image")
plt.imshow(abs(sac))
plt.show()


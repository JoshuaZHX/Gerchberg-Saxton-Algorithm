""" 
Program : Create_Gaussian_Beam 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.restoration import unwrap_phase 
from Function_Find_Sigma import Find_Sigma 
from Function_Propagator import Propagator 
from Function_Phase_Generator import Phase_Generator 

########## ----- Parameters for Gaussian Beam ----- ########## 
c  = 3e8                  # The speed of light. 
f  = 15e-2                # The focal length of the lens. 
l  = 600e-9               # The wavelength of Gaussian beam. 
w  = 2*np.pi*c / l        # The temporal frequency of Gaussian beam. 
k  = w / c                # The spatial frequency of Gaussian beam. 
wi = 1e-2                 # The initial waist of Gaussian beam. 
w0 = (l*f) / (np.pi*wi)   # The theoretical waist at focal point. 
zR = np.pi*(w0**2)/l      # The Rayleigh range of Gaussian beam. 

########## ----- Parameters for Object & Image Planes ----- ########## 
N      = 512   # The number of points in xy-axis. 

S1     = 40e-2                                       # The size of object plane. 
ds1    = S1/N                                        # The size of one pixel. 
x1     = np.linspace(-(N/2)*ds1, (N/2 - 1)*ds1, N)   # The axis in object plane. 
X1, Y1 = np.meshgrid(x1, x1)                         # The xy-grid in object plane. 

F1     = 1/ds1                                                       # The size of image plane. 
df1    = F1/N                                                        # The size of one pixel. 
kx1    = np.linspace(-(N/2)*df1*2*np.pi, (N/2 - 1)*df1*2*np.pi, N)   # The k-axis in image plane. 
x2     = (f/k)*kx1                                                   # The axis in image plane. 
X2, Y2 = np.meshgrid(x2, x2)                                         # The xy-grid in image plane. 
ds2    = x2[1] - x2[0]                                               # The size of one pixel. 

F2       = 1/ds2                                                   # The size of diffraction plane. 
df2      = F2/N                                                    # The size of one pixel. 
kx2      = np.linspace(-(N/2)*df2*2*np.pi, (N/2)*df2*2*np.pi, N)   # The k-axis in diffraction plane. 
kX2, kY2 = np.meshgrid(kx2, kx2)                                   # The kxky-grid in diffraction plane. 
kz       = np.real(np.emath.sqrt(k**2 - kX2**2 - kY2**2))          # The kz-axis in diffraction plane. 

########## ----- Phase Generator ----- ########## 
radius0 = np.sqrt(X1**2 + Y1**2) / (2*wi) 
theta0  = np.arctan2(Y1, X1) 
phase0  = 1 * Phase_Generator(radius0, theta0, 0, 0, 0, 0, 0) 

for i in range(N): 
    for j in range(N): 
        if (radius0[i][j] > 1): 
            phase0[i][j] = 0 

########## ----- Gaussian Beam in Object Plane ----- ########## 
E0a = np.exp(-(X1**2 + Y1**2) / wi**2)*np.exp(1j*phase0) 
E0b = np.fft.fftshift(E0a) 
I0  = np.real(E0a*np.conj(E0a)) 

########## ----- Gaussian Beam in Image Plane ----- ########## 
E1a = np.fft.fft2(E0b) 
E1b = np.fft.ifftshift(E1a) 
I1  = np.real(E1b*np.conj(E1b)) 
I1  = I1 / np.max(I1) 

sigma_min = Find_Sigma(x2, I1)[0] 
waist_min = np.sqrt(2) * sigma_min 

########## ----- Phase in Image Plane ----- ########## 
radius1 = np.sqrt(X2**2 + Y2**2) / (2*w0) 
phase1  = np.angle(E1b) 

for i in range(N): 
    for j in range(N): 
        if (radius1[i][j] > 1): 
            phase1[i][j] = 0 

########## ----- Gaussian Beam in Object Plane ----- ########## 
E2a = np.fft.fftshift(E1b) 
E2b = np.fft.ifft2(E2a) 
E2c = np.fft.ifftshift(E2b) 
I2  = np.real(E2c*np.conj(E2c)) 

phase2 = np.angle(E2c) 
phase2 = unwrap_phase(phase2) 

for i in range(N): 
    for j in range(N): 
        if (radius0[i][j] > 1): 
            phase2[i][j] = 0 

########## ----- Plot Intensity & Phase ----- ########## 
fig = plt.figure(1, figsize=(20,11)) 

ax1 = fig.add_subplot(231) 
im1 = ax1.pcolormesh(X1*1e2, Y1*1e2, phase0, cmap=plt.cm.jet, shading='auto') 
ax1.set_title('Phase in Object Plane') 
ax1.set_xlabel('Spatial Distance $x$ ($cm$)') 
ax1.set_ylabel('Spatial Distance $y$ ($cm$)') 
ax1.set_xlim([-5, 5]) 
ax1.set_ylim([-5, 5]) 
plt.colorbar(im1, ax=ax1) 

ax2 = fig.add_subplot(232) 
im2 = ax2.pcolormesh(X2*1e6, Y2*1e6, phase1, cmap=plt.cm.jet, shading='auto') 
ax2.set_title('Phase in Image Plane') 
ax2.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax2.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax2.set_xlim([-10, 10]) 
ax2.set_ylim([-10, 10]) 
plt.colorbar(im2, ax=ax2) 

ax3 = fig.add_subplot(233) 
im3 = ax3.pcolormesh(X1*1e2, Y1*1e2, phase2, cmap=plt.cm.jet, shading='auto') 
ax3.set_title('Phase in Object Plane') 
ax3.set_xlabel('Spatial Distance $x$ ($cm$)') 
ax3.set_ylabel('Spatial Distance $y$ ($cm$)') 
ax3.set_xlim([-5, 5]) 
ax3.set_ylim([-5, 5]) 
plt.colorbar(im3, ax=ax3) 

ax4 = fig.add_subplot(234) 
im4 = ax4.pcolormesh(X1*1e2, Y1*1e2, I0, cmap=plt.cm.jet, shading='auto') 
ax4.set_title('Intensity in Object Plane') 
ax4.set_xlabel('Spatial Distance $x$ ($cm$)') 
ax4.set_ylabel('Spatial Distance $y$ ($cm$)') 
ax4.set_xlim([-5, 5]) 
ax4.set_ylim([-5, 5]) 
plt.colorbar(im4, ax=ax4) 

ax5 = fig.add_subplot(235) 
im5 = ax5.pcolormesh(X2*1e6, Y2*1e6, I1, cmap=plt.cm.jet, shading='auto') 
ax5.set_title('Intensity in Image Plane') 
ax5.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax5.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax5.set_xlim([-10, 10]) 
ax5.set_ylim([-10, 10]) 
plt.colorbar(im5, ax=ax5) 

ax6 = fig.add_subplot(236) 
im6 = ax6.pcolormesh(X1*1e2, Y1*1e2, I2, cmap=plt.cm.jet, shading='auto') 
ax6.set_title('Intensity in Object Plane') 
ax6.set_xlabel('Spatial Distance $x$ ($cm$)') 
ax6.set_ylabel('Spatial Distance $y$ ($cm$)') 
ax6.set_xlim([-5, 5]) 
ax6.set_ylim([-5, 5]) 
plt.colorbar(im6, ax=ax6) 

########## ----- Propagation in Far Field ----- ########## 
Distance = np.linspace(-3*zR, +3*zR, 101) 

Intensity_X = [] 
Sigma_X1    = [] 
Sigma_X2    = [] 
Waist_X1    = [] 
Waist_X2    = [] 
for distance in Distance: 
    Eout = Propagator(E1b, kz, distance) 
    Iout = np.real(Eout * np.conj(Eout)) 
    
    intensity_x = np.sum(Iout, axis=0) 
    Intensity_X.append(intensity_x) 
    
    sigma_x, sigma_y = Find_Sigma(x2, Iout) 
    waist = np.sqrt(2) * sigma_x
    Sigma_X1.append(waist) 
    Sigma_X2.append(-waist) 
    
    w = waist_min * np.sqrt(1 + (distance / zR)**2) 
    Waist_X1.append(w) 
    Waist_X2.append(-w) 

Intensity_X = np.transpose(Intensity_X) 
Intensity_X = Intensity_X / np.max(Intensity_X) 

########## ----- Plot Propagation in Far Field ----- ########## 
fg2 = plt.figure(2, figsize=(20,10)) 

ax1 = fg2.add_subplot(111) 
im1 = ax1.pcolormesh(Distance, x2, Intensity_X, cmap=plt.cm.jet, shading='auto') 
#ax1.plot(Distance, Waist_X1, color='white', label='Theoretical Beam Waist') 
#ax1.plot(Distance, Waist_X2, color='white') 
#ax1.plot(Distance, Sigma_X1, color='red', label='Numerical Beam Waist') 
#ax1.plot(Distance, Sigma_X2, color='red') 
ax1.set_title('Gaussian Beam', fontsize=16) 
ax1.set_xlabel('Spatial Distance $z$ ($m$)', fontsize=12) 
ax1.set_ylabel('Spatial Distance $x$ ($m$)', fontsize=12) 
ax1.set_ylim([-2*1e-5, 2*1e-5]) 
ax1.legend() 
plt.colorbar(im1, ax=ax1) 
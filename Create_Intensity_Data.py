""" 
Program : Create_Intensity_Data 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import os 
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from Function_Phase_Generator import Phase_Generator 
from Function_Propagator import Propagator

########## ----- Parameters for Gaussian Beam ----- ########## 
c  = 3e8                  # The speed of light. 
f  = 15e-2                # The focal length of the lens. 
l  = 600e-9               # The wavelength of Gaussian beam. 
w  = 2*np.pi*c / l        # The temporal frequency of Gaussian beam. 
k  = w / c                # The spatial frequency of Gaussian beam. 
wi = 1e-2                 # The initial waist of Gaussian beam. 
w0 = (l*f) / (np.pi*wi)   # The theoretical waist at focal point. 
zR = np.pi*(w0**2)/l      # The Rayleigh range of Gaussian beam. 

########## ----- Parameters for Object & Image & Diffraction Planes ----- ########## 
N = 512   # The number of points in xy-axis. 

S1     = 40e-2                                   # The size of object plane. 
ds1    = S1/N                                    # The size of one pixel. 
x1     = np.linspace(-(N/2)*ds1, (N/2)*ds1, N)   # The axis in object plane. 
X1, Y1 = np.meshgrid(x1, x1)                     # The xy-grid in object plane. 

F1     = 1/ds1                                                   # The size of image plane. 
df1    = F1/N                                                    # The size of one pixel. 
kx1    = np.linspace(-(N/2)*df1*2*np.pi, (N/2)*df1*2*np.pi, N)   # The k-axis in image plane. 
x2     = np.multiply(f/k, kx1)                                   # The axis in image plane. 
X2, Y2 = np.meshgrid(x2, x2)                                     # The xy-grid in image plane.
ds2    = x2[1] - x2[0]                                           # The size of one pixel. 

F2       = 1/ds2                                                   # The size of diffraction plane. 
df2      = F2/N                                                    # The size of one pixel. 
kx2      = np.linspace(-(N/2)*df2*2*np.pi, (N/2)*df2*2*np.pi, N)   # The k-axis in diffraction plane. 
kX2, kY2 = np.meshgrid(kx2, kx2)                                   # The kxky-grid in diffraction plane. 
kz       = np.real(np.emath.sqrt(k**2 - kX2**2 - kY2**2))          # The kz-axis in diffraction plane. 

########## ----- Phase Generator ----- ########## 
radius0 = np.sqrt(X1**2 + Y1**2) / (2*wi) 
theta0  = np.arctan2(Y1, X1) 
C1      = 0   # Spherical. 
C2      = 1   # Astigmatism Oblique. 
C3      = 1   # Astigmatism Vertical. 
C4      = 1   # Coma Vertical. 
C5      = 1   # Coma Horizontal. 
phase0  = Phase_Generator(radius0, theta0, C1, C2, C3, C4, C5) 
phase0[radius0 > 1] = 0 

########## ----- Gaussian Beam in Object Plane ----- ########## 
E0 = np.exp(-(X1**2 + Y1**2) / wi**2)*np.exp(1j*phase0) 
I0 = np.real(E0*np.conj(E0)) 

########## ----- Gaussian Beam in Image Plane ----- ########## 
E1 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(E0)))   # The Gaussian beam at z=0. 
I1 = np.real(E1*np.conj(E1))            # The intensity at z=0. 

########## ----- Gaussian Beam Propagation ----- ########## 
D  = 3*zR                      # The propagation distance. 
E2 = Propagator(E1, kz, -D)    # The Gaussian beam at z=-D. 
E3 = Propagator(E1, kz, +D)    # The Gaussian beam at z=+D. 
I2 = np.real(E2*np.conj(E2))   # The intensity at z=-D. 
I3 = np.real(E3*np.conj(E3))   # The intensity at z=+D. 

########## ----- Plot Intensity ----- ########## 
fig = plt.figure(1, figsize=(20,5)) 

ax1 = fig.add_subplot(131) 
im1 = ax1.pcolormesh(X2*1e6, Y2*1e6, I2, cmap=plt.cm.jet, shading='auto') 
ax1.set_title('Intensity at z=-D') 
ax1.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax1.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax1.set_xlim([-25, 25]) 
ax1.set_ylim([-25, 25]) 
plt.colorbar(im1, ax=ax1) 

ax2 = fig.add_subplot(132) 
im2 = ax2.pcolormesh(X2*1e6, Y2*1e6, I1, cmap=plt.cm.jet, shading='auto') 
ax2.set_title('Intensity at z=0') 
ax2.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax2.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax2.set_xlim([-25, 25]) 
ax2.set_ylim([-25, 25]) 
plt.colorbar(im2, ax=ax2) 

ax3 = fig.add_subplot(133) 
im3 = ax3.pcolormesh(X2*1e6, Y2*1e6, I3, cmap=plt.cm.jet, shading='auto') 
ax3.set_title('Intensity at z=+D') 
ax3.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax3.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax3.set_xlim([-25, 25]) 
ax3.set_ylim([-25, 25]) 
plt.colorbar(im3, ax=ax3) 

########## ----- Save Intensity Data ----- ########## 
folder_path = 'D:/Internships/Internship -- LOA 2022/Numerical Data' 
folder_name = f'phase_{C1}_{C2}_{C3}_{C4}_{C5}' 
folder      = os.path.join(folder_path, folder_name) 
if (os.path.isdir(folder) != True): 
    os.mkdir(folder) 

I_max   = np.max([I1, I2, I3])       # To find the maximum intensity. 
I1_norm = (2**16 - 1)*(I1 / I_max)   # The normalized intensity at z=0. 
I2_norm = (2**16 - 1)*(I2 / I_max)   # The normalized intensity at z=-D. 
I3_norm = (2**16 - 1)*(I3 / I_max)   # The normalized intensity at z=+D. 

image1 = f'{folder}/Intensity_z=0.tiff' 
cv2.imwrite(image1, np.uint16(I1_norm)) 

image2 = f'{folder}/Intensity_z=-{D}.tiff' 
cv2.imwrite(image2, np.uint16(I2_norm)) 

image3 = f'{folder}/Intensity_z=+{D}.tiff' 
cv2.imwrite(image3, np.uint16(I3_norm)) 
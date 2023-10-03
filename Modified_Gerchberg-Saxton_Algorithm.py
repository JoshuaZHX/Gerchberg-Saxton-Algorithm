""" 
Program : Modified_Gerchberg-Saxton_Algorithm 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import os 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from Function_GSA_Cycle import GSA_Cycle 
from Function_Phase_Generator import Phase_Generator 
from skimage.restoration import unwrap_phase 

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

########## ----- Read Intensity Data ----- ########## 
folder_path = 'D:/Internships/Internship -- LOA 2022/Numerical Data' 
folder_name = 'phase_0_1_1_1_1' 
folder      = os.path.join(folder_path, folder_name) 

D = 3*zR   # The propagation distance. 

image1 = f'{folder}/Intensity_z=0.tiff' 
I1     = cv2.imread(image1, -1) 
image2 = f'{folder}/Intensity_z=-{D}.tiff' 
I2     = cv2.imread(image2, -1) 
image3 = f'{folder}/Intensity_z=+{D}.tiff' 
I3     = cv2.imread(image3, -1) 

########## ----- The True Phase ----- ########## 
phase_c = [float(folder_name[-9]), 
           float(folder_name[-7]), 
           float(folder_name[-5]), 
           float(folder_name[-3]), 
           float(folder_name[-1])] 

radius0 = np.sqrt(X1**2 + Y1**2) / (2*wi) 
theta0  = np.arctan2(Y1, X1) 
phase0  = Phase_Generator(radius0, theta0, phase_c[0], phase_c[1], phase_c[2], phase_c[3], phase_c[4]) 
phase0[radius0 > 1] = 0 

########## ----- Initial Estimation of Phase ----- ########## 
radius1 = np.sqrt(X2**2 + Y2**2) / (2*w0) 
theta1  = np.arctan2(Y2, X2) 
phase1  = Phase_Generator(radius1, theta1, 0, 0, 0, 0, 0) 
phase1[radius1 > 1] = 0 

########## ----- Gerchberg-Saxton Algorithm ----- ########## 
I1     = I1 / np.max(I1) 
I2     = I2 / np.max(I2) 
I3     = I3 / np.max(I3) 
A1     = np.sqrt(I1) 
A2     = np.sqrt(I2) 
A3     = np.sqrt(I3) 
I1_sum = np.sum(I1) 
I2_sum = np.sum(I2) 
I3_sum = np.sum(I3) 

iteration = 30 
error1    = np.zeros(iteration) 
error2    = np.zeros(iteration) 
error3    = np.zeros(iteration) 

for i in range(iteration): 
    if (i == 0): 
        phase, I1_updated, I2_updated, I3_updated = GSA_Cycle(A1, A2, A3, phase1, kz, D) 
        I1_updated = I1_updated / np.max(I1_updated) 
        I2_updated = I2_updated / np.max(I2_updated) 
        I3_updated = I3_updated / np.max(I3_updated) 
        error1[i]  = np.sum(np.abs(I1 - I1_updated)) / I1_sum 
        error2[i]  = np.sum(np.abs(I2 - I2_updated)) / I2_sum 
        error3[i]  = np.sum(np.abs(I3 - I3_updated)) / I3_sum 
    else: 
        phase, I1_updated, I2_updated, I3_updated = GSA_Cycle(A1, A2, A3, phase, kz, D) 
        I1_updated = I1_updated / np.max(I1_updated) 
        I2_updated = I2_updated / np.max(I2_updated) 
        I3_updated = I3_updated / np.max(I3_updated) 
        error1[i]  = np.sum(np.abs(I1 - I1_updated)) / I1_sum 
        error2[i]  = np.sum(np.abs(I2 - I2_updated)) / I2_sum 
        error3[i]  = np.sum(np.abs(I3 - I3_updated)) / I3_sum 

print('The Gerchberg-Saxton Algorithm is finished.') 

########## ----- Plot Intensity Data ----- ########## 
fg1 = plt.figure(1, figsize=(20,10)) 

ax1 = fg1.add_subplot(231) 
im1 = ax1.pcolormesh(X2*1e6, Y2*1e6, I2, cmap=plt.cm.jet, shading='auto') 
ax1.set_title('Intensity at z=-3zR') 
ax1.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax1.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax1.set_xlim([-25, 25]) 
ax1.set_ylim([-25, 25]) 
plt.colorbar(im1, ax=ax1) 

ax2 = fg1.add_subplot(232) 
im2 = ax2.pcolormesh(X2*1e6, Y2*1e6, I1, cmap=plt.cm.jet, shading='auto') 
ax2.set_title('Intensity at z=0') 
ax2.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax2.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax2.set_xlim([-25, 25]) 
ax2.set_ylim([-25, 25]) 
plt.colorbar(im2, ax=ax2) 

ax3 = fg1.add_subplot(233) 
im3 = ax3.pcolormesh(X2*1e6, Y2*1e6, I3, cmap=plt.cm.jet, shading='auto') 
ax3.set_title('Intensity at z=+3zR') 
ax3.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax3.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax3.set_xlim([-25, 25]) 
ax3.set_ylim([-25, 25]) 
plt.colorbar(im3, ax=ax3) 

ax4 = fg1.add_subplot(234) 
im4 = ax4.pcolormesh(X2*1e6, Y2*1e6, I2_updated, cmap=plt.cm.jet, shading='auto') 
ax4.set_title('Intensity at z=-3zR at last iteration') 
ax4.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax4.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax4.set_xlim([-25, 25]) 
ax4.set_ylim([-25, 25]) 
plt.colorbar(im4, ax=ax4) 

ax5 = fg1.add_subplot(235) 
im5 = ax5.pcolormesh(X2*1e6, Y2*1e6, I1_updated, cmap=plt.cm.jet, shading='auto') 
ax5.set_title('Intensity at z=0 at last iteration') 
ax5.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax5.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax5.set_xlim([-25, 25]) 
ax5.set_ylim([-25, 25]) 
plt.colorbar(im5, ax=ax5) 

ax6 = fg1.add_subplot(236) 
im6 = ax6.pcolormesh(X2*1e6, Y2*1e6, I3_updated, cmap=plt.cm.jet, shading='auto') 
ax6.set_title('Intensity at z=+3zR at last iteration') 
ax6.set_xlabel('Spatial Distance $x$ ($\mu m$)') 
ax6.set_ylabel('Spatial Distance $y$ ($\mu m$)') 
ax6.set_xlim([-25, 25]) 
ax6.set_ylim([-25, 25]) 
plt.colorbar(im6, ax=ax6) 

########## ----- Plot Error History ----- ########## 
Color = ['red', 'green', 'blue'] 
Error = [error1, error2, error3] 
Label = ['Error at z=0', 'Error at z=-3zR', 'Error at z=+3zR'] 

fg2 = plt.figure(2, figsize=(10,5)) 

ax1 = fg2.add_subplot(111)
for i in range(len(Error)): 
    ax1.plot(Error[i], color=Color[i], label=Label[i]) 
ax1.set_title('Error History') 
ax1.set_xlabel('Iteration') 
ax1.set_ylabel('Error') 
ax1.set_xlim([0, iteration]) 
ax1.legend() 

########## ----- Phase Comparison in Object Plane ----- ########## 
E1 = np.sqrt(I1)*np.exp(1j*phase) 
E0 = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(E1))) 

phase = np.angle(E0) 
phase[radius0 > 1] = 0 
phase_unwrapped = unwrap_phase(phase) 

fg3 = plt.figure(3, figsize=(20,8)) 

ax1 = fg3.add_subplot(121) 
im1 = ax1.pcolormesh(X1*1e2, Y1*1e2, phase0, cmap=plt.cm.jet, shading='auto') 
ax1.set_title('Original Phase in Object Plane') 
ax1.set_xlabel('Spatial Distance $x$ ($cm$)') 
ax1.set_ylabel('Spatial Distance $y$ ($cm$)') 
ax1.set_xlim([-5, 5]) 
ax1.set_ylim([-5, 5]) 
plt.colorbar(im1, ax=ax1) 

ax2 = fg3.add_subplot(122) 
im2 = ax2.pcolormesh(X1*1e2, Y1*1e2, phase_unwrapped, cmap=plt.cm.jet, shading='auto') 
ax2.set_title('Recovered Phase in Object Plane') 
ax2.set_xlabel('Spatial Distance $x$ ($cm$)') 
ax2.set_ylabel('Spatial Distance $y$ ($cm$)') 
ax2.set_xlim([-5, 5]) 
ax2.set_ylim([-5, 5]) 
plt.colorbar(im2, ax=ax2) 

########## ----- Phase Decomposition ----- ########## 
P1 = np.sqrt(5)*(6*(radius0**4) - 6*(radius0**2) + 1)         # Spherical. 
P2 = np.sqrt(6)*(radius0**2)*np.sin(2*theta0)                 # Astigmatism Oblique. 
P3 = np.sqrt(6)*(radius0**2)*np.cos(2*theta0)                 # Astigmatism Vertical. 
P4 = np.sqrt(8)*(3*(radius0**3) - 2*radius0)*np.sin(theta0)   # Coma Vertical. 
P5 = np.sqrt(8)*(3*(radius0**3) - 2*radius0)*np.cos(theta0)   # Coma Horizontal. 

P1[radius0 > 1] = 0 
P2[radius0 > 1] = 0 
P3[radius0 > 1] = 0 
P4[radius0 > 1] = 0 
P5[radius0 > 1] = 0 

C1_unwrapped = np.sum(phase_unwrapped * P1) / np.sum(P1**2) 
C2_unwrapped = np.sum(phase_unwrapped * P2) / np.sum(P2**2) 
C3_unwrapped = np.sum(phase_unwrapped * P3) / np.sum(P3**2) 
C4_unwrapped = np.sum(phase_unwrapped * P4) / np.sum(P4**2) 
C5_unwrapped = np.sum(phase_unwrapped * P5) / np.sum(P5**2) 

print(f'C1 unwrapped is {C1_unwrapped}.') 
print(f'C2 unwrapped is {C2_unwrapped}.') 
print(f'C3 unwrapped is {C3_unwrapped}.') 
print(f'C4 unwrapped is {C4_unwrapped}.') 
print(f'C5 unwrapped is {C5_unwrapped}.') 
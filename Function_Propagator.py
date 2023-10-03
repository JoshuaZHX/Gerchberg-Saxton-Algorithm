""" 
Program : Function_Propagator 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Propagator ----- ########## 
# Input  : E0 = The Gaussian beam at z=0.            (N*N Grid) 
#          kz = The kz-axis.                         (N*N Grid) 
#          D  = The propagation distance from z=0.   (A Number) 
# Output : E7 = The Gaussian beam at z=D.            (N*N Grid) 

def Propagator(E0, kz, D): 
    E1 = np.fft.fftshift(E0) 
    E2 = np.fft.fft2(E1) 
    E3 = np.fft.ifftshift(E2) 
    
    E4 = E3 * np.exp(1j*kz*D) 
    
    E5 = np.fft.fftshift(E4) 
    E6 = np.fft.ifft2(E5) 
    E7 = np.fft.ifftshift(E6) 
    return E7 
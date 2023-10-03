""" 
Program : Function_GSA_Cycle  
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 
from Function_Propagator import Propagator 

########## ----- Function : GSA_Cycle ----- ########## 
# Input  : A1     = The amplitude at z=0.                           (N*N Grid) 
#          A2     = The amplitude at z=-D.                          (N*N Grid) 
#          A3     = The amplitude at z=+D.                          (N*N Grid) 
#          phase0 = The phase.                                      (N*N Grid) 
#          kz     = The kz-axis of diffraction plane.               (N*N Grid) 
#          D      = The propagation distance.                       (N*N Grid) 
# Output : phase  = The updated phase after one cycle.              (N*N Grid) 
#          I1     = The updated intensty at z=0 after one cycle.    (N*N Grid) 
#          I2     = The updated intensty at z=-D after one cycle.   (N*N Grid) 
#          I3     = The updated intensty at z=+D after one cycle.   (N*N Grid) 

def GSA_Cycle(A1, A2, A3, phase0, kz, D): 
    E0 = A1 * np.exp(1j*phase0) 
    
    E2_1 = Propagator(E0, kz, -D) 
    E2_2 = A2 * np.exp(1j*np.angle(E2_1)) 
    E2_3 = Propagator(E2_2, kz, +D) 
    I2   = np.real((E2_1)*np.conj(E2_1)) 

    E3_1 = Propagator(E0, kz, +D) 
    E3_2 = A3 * np.exp(1j*np.angle(E3_1)) 
    E3_3 = Propagator(E3_2, kz, -D) 
    I3   = np.real((E3_1)*np.conj(E3_1)) 

    Eout  = (E2_3 + E3_3) / 2 
    I1    = np.real(Eout*np.conj(Eout)) 
    phase = np.angle(Eout) 
    return phase, I1, I2, I3 
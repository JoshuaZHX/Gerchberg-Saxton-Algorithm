""" 
Program : Function_Phase_Generator  
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Phase_Generator ----- ########## 
# Input  : radius = The normalized radius.             (N*N Grid) 
#          theta  = The azimuth angle.                 (N*N Grid) 
#          C      = The coefficients of aberrations.   (A Number) 
# Output : phase  = The phase generated.               (N*N Grid) 

def Phase_Generator(radius, theta, C1, C2, C3, C4, C5): 
    P1 = C1 * np.sqrt(5)*(6*(radius**4) - 6*(radius**2) + 1)        # Spherical. 
    P2 = C2 * np.sqrt(6)*(radius**2)*np.sin(2*theta)                # Astigmatism Oblique. 
    P3 = C3 * np.sqrt(6)*(radius**2)*np.cos(2*theta)                # Astigmatism Vertical. 
    P4 = C4 * np.sqrt(8)*(3*(radius**3) - 2*radius)*np.sin(theta)   # Coma Vertical. 
    P5 = C5 * np.sqrt(8)*(3*(radius**3) - 2*radius)*np.cos(theta)   # Coma Horizontal. 
    
    phase = P1 + P2 + P3 + P4 + P5 
    return phase 
""" 
Program : Find_Sigma 
Author : Haoxuan ZHANG 
Ecole Polytechnique IP-Paris 
Laboratoire d'Optique Appliquée (APPLI) 
""" 

########## ----- Import ----- ########## 
import numpy as np 

########## ----- Function : Find_Sigma ----- ########## 
# Input  : x         = The xy-axis of intensity data. 
#          Intensity = The intensity data. 
# Output : sigma_x   = The standard deviation along x. 
#          sigma_y   = The standard deviation along y. 

def Find_Sigma(x, Intensity): 
    x_sum   = np.sum(Intensity, axis=0) 
    y_sum   = np.sum(Intensity, axis=1) 
    x0      = np.sum(x * x_sum) / np.sum(x_sum) 
    y0      = np.sum(x * y_sum) / np.sum(y_sum) 
    x0_2    = np.sum((x**2) * x_sum) / np.sum(x_sum) 
    y0_2    = np.sum((x**2) * y_sum) / np.sum(y_sum) 
    sigma_x = np.sqrt(x0_2 - (x0**2)) 
    sigma_y = np.sqrt(y0_2 - (y0**2)) 
    return sigma_x, sigma_y 
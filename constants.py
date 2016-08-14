from __future__ import division

import numpy as np

#==============================================================================
#                           ----- Inputs -----
#==============================================================================

#Also see p.677 Wetzel paper

# viscosity ratio droplet/matrix, unitless

lam = 10                                   

# matrix viscosity,  Pa s=(N s)/(m ^2)
mu_si = 1e-2                                  

#---Unit Conversion---
mu = mu_si * 1e-6                    # microNewton s / micrometer^2

# shear rate, 1/s
#Increasing shear rate doesn't seem to incease deformation 
#Amount of deformation for larger values of lam does not much depend on shear rate
gammadot = 0.1                          

# interfacial tension, N/m
Gamma = 1e-8                           


#Parameters for cell movement and proliferation

sim_step = 1 / gammadot # seconds

dt = sim_step / 10

# Mean cell cycle time in seconds
tau_p = 30*60

#Flow type for simulations
flow_type = 2



#Hetzian repulsion model, see p.419 of Liedekerke

young_mod = 5e6

pois_num = 0.5

E_hat = 0.5 * young_mod / (1 - pois_num**2) 

cell_rad = 0.5   #micrometers

r_cut = 3*cell_rad #micrometers

R_hat = cell_rad / 2
    
rep_const = 4/3*E_hat * np.sqrt( R_hat ) * 1e-6 #Nondimensionalized repulsive constant

pull_const = 1e-2

#Friction rate induced by viscousity of the ECM
ksi = 6*cell_rad*np.pi *mu_si*lam #kg/s

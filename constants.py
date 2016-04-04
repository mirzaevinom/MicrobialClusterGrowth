
# ----- Inputs -----
#Also see p.677 Wetzel paper

# viscosity ratio droplet/matrix, unitless

lam = 10                                   

# matrix viscosity,  Pa s=(N s)/(m ^2)
mu_si = 1e-2                                  

#---Unit Conversion---
mu = mu_si * 10**(-6)                    # microNewton s / micrometer^2

# shear rate, 1/s
#Increasing shear rate doesn't seem to incease deformation 
#Amount of deformation for larger values of lam does not much depend on shear rate
gammadot = 1                             

# interfacial tension, N/m
Gamma = 1e-7                            
                                 


# ----- List for Import -----
constants = [ lam , mu , gammadot , Gamma]

def import_constants():
  return constants


# ----- Inputs -----
#Also see p.677 Wetzel paper

# viscosity ratio droplet/matrix, unitless
lam = 50                                       

# matrix viscosity,  Pa s=(N s)/(m ^2)
mu_si = 0.01                                    

# shear rate, 1/s
# for shear rates greater than 50 make absolute tolerance in odeint more strict
 
gammadot_si = 50                              

# interfacial tension, N/m
# that is what Eric Wetzel uses in his simulations
Gamma_si = 1e-9                             

# stress, Pa = N / m^2
max_stress_si = 0.005                           

# pressure, Pa = N / m^2
p0_si = 0                                       

# ----- Unit Conversions ------
gammadot = gammadot_si                   # 1/s
mu = mu_si * 10**(-6)                    # microNewton s / micrometer^2
Gamma = Gamma_si                         # microNewton / micrometer
p0 = p0_si * 10**(-6)                    # microNewton / micrometer^2
max_stress = max_stress_si * 10**(-6)    # microNewton / micrometer^2

# ----- List for Import -----
constants = [ lam, mu, gammadot, Gamma, max_stress, p0]

def import_constants():
  return constants

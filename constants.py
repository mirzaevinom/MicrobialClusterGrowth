
# ----- Inputs -----
lam = 20                                    # viscosity ratio, unitless
mu_si = 0.01                                # matrix viscosity,  Pa s=(N s)/(m ^2)
gammadot_si = 1000.                         # shear rate, 1/s
Gamma_si = 4.1e-9                           # interfacial tension, N/m
max_stress_si = 5                           # stress, Pa = N / m^2
p0_si = 0                                   # pressure, Pa = N / m^2

# ----- Unit Conversions ------
gammadot = gammadot_si                   # 1/s
mu = mu_si * 10**(-6)                    # microNewton s / micrometer^2
Gamma = Gamma_si                         # microNewton / micrometer
p0 = p0_si * 10**(-6)                    # microNewton / micrometer^2
max_stress = max_stress_si * 10**(-6)    # microNewton / micrometer^2

# ----- List for Import -----
constants = [lam, mu, gammadot, Gamma, max_stress, p0]

def import_constants():
  return constants

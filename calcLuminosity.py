import numpy as np
import sys


sys.stdout.reconfigure(encoding='utf-8')

def CalcLuminosity(gama, nb, N,f_rev, beta, epsilon, R):
    num = nb* (N**2)*f_rev
    den = 4*(np.pi)*beta*epsilon

    return gama*(num/den)*R


gamma_LHC = gamma_HL_LHC = 7461
nb_LHC = 2808
N_LHC = 1.15*10**(11)
f_rev_LHC = f_rev_HL_LHC = 11200
beta_LHC = 0.55
epsilon_LHC = 3.75*10**(-6)
r_LHC = 0.836


nb_HL_LHC = 2736
N_HL_LHC = 2.2*10**(11)
beta_HL_LHC = 0.25
epsilon_HL_LHC = 2.5*10**(-6)
r_HL_LHC = 0.831


mu_LHC = CalcLuminosity(gamma_LHC, nb_LHC, N_LHC, f_rev_LHC, beta_LHC, epsilon_LHC, r_LHC)
mu_HL_LHC = CalcLuminosity(gamma_HL_LHC, nb_HL_LHC, N_HL_LHC, f_rev_HL_LHC, beta_HL_LHC, epsilon_HL_LHC, r_HL_LHC)

# Multiplicando por 1e-4 para converter de m^-2 para cm^-2
print(f'⟨μ⟩ LHC = {(mu_LHC/ 1e34):.2f}e+34 cm^-2 s^-1')
print(f'⟨μ⟩ HL-LHC = {(mu_HL_LHC/ 1e34):.2f}e+34 cm^-2 s^-1')




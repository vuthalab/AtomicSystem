import os
os.chdir("/home/labuser/googledrive/Calculations/Calcium/polarizability/example")

from atomic_sys import AtomicSys
import numpy as np


## Calcium

f_levels = 'calcium_levels.txt'
f_lines = 'calcium_oscillator_strengths.txt'

calcium = AtomicSys(f_levels,f_lines,subset=20)

# returns a list of all states with the label specified:
g_state_list = [x for x in calcium.basis if (x.label=='4s21S0')]
p_state_list = [x for x in calcium.basis if (x.label=='4s4p1P1')]

# pick out the mJ=0 sublevels:
g_state = [x for x in g_state_list if (x.mJ==0)][0]
p_state = [x for x in p_state_list if (x.mJ==0)][0]


calcium.calculateAlpha(g_state,omega=0.0001,Q=1.0,prnt=True)

print(calcium.E1Moment(g_state,p_state))

H0 = calcium.genFreeHamiltonian()
HE = calcium.genE1Hamiltonian(1.0)
HM = calcium.genM1Hamiltonian(1.0)

H = H0+HE+HM
eigvals,eigvecs = calcium.diagonalizeHamiltonian(H)

print(eigvals)
print(np.diag(eigvecs)) #should be all 1s


from __future__ import division

import matplotlib.pyplot as plt
import numpy as np



plt.close('all')

radius = np.linspace( 0.93 , 2 , 100)


force = 24  / radius * (  (1/ radius)**12 - (1/ radius)**6   )

plt.plot(radius , force , linewidth=2, color='b')

plt.grid(True, which='both')

myaxis = list( plt.axis() )

myaxis[-1] = 10

plt.axis(myaxis)




plt.xlabel(r'$r_{ij} / \sigma$' , fontsize=20)
plt.ylabel(r'$f_{ij}(t) / \varepsilon$' , fontsize=20)

plt.savefig('lennard_jones.png' , dpi=400)
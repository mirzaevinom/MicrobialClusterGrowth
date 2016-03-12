from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps

shape = 60

scale = 1 / shape

s1 = np.random.gamma(shape, scale, 10**4)


shape = 60
scale = 1.5 / shape
s2 = np.random.gamma(shape, scale, 10**4)

shape = 60
scale = 2 / shape
s3 = np.random.gamma(shape, scale, 10**4)

s = np.concatenate( (s1, s2, s3 ) )

np.random.shuffle( s )

plt.close('all')

count, bins, ignored = plt.hist(s, 50, normed=True, alpha=0.6, color='g')

#y = bins**(shape-1)*(np.exp(-bins/scale) /
#                     (sps.gamma(shape)*scale**shape))
#plt.plot( bins , y , linewidth=2 , color='k')
#plt.xlabel(r'$\tau$' , fontsize=20)
#plt.ylabel(r'$f(\tau)$' , fontsize=20)

plt.savefig('erlang.png' , dpi=400)
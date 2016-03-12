from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np



plt.close('all')

fig = plt.figure(0)

ax = fig.add_subplot( 111 ,  projection='3d')

x = np.random.normal( 0, 1 , size=2000)
y = np.random.normal( 0 , 1 , size=2000)
z = np.random.normal( 0 , 1 , size=2000)

mynorm = np.sqrt( x**2 + y**2 + z**2)

sphr_shift = ( np.array( [ x , y , z ] ) / mynorm ).T


distances = cdist( sphr_shift , sphr_shift ) 
 
distances = np.tril( distances ) + np.triu(np.ones( (len( sphr_shift ) , len( sphr_shift )) ) )

tbd = np.unique( np.nonzero(distances < 0.1)[0] )

sphr_shift = np.delete(sphr_shift , tbd , axis=0)

ax.scatter( sphr_shift[:, 0 ] , sphr_shift[ : , 1 ] , sphr_shift[:, 2 ])

plt.savefig( 'example_directions.png' , dpi=400 )

sphr_size = np.zeros(1000)


for nn in range( len(sphr_size) ):
    
    x = np.random.normal( 0 , 1 , size=2000)
    y = np.random.normal( 0 , 1 , size=2000)
    z = np.random.normal( 0 , 1 , size=2000)
    
    mynorm = np.sqrt( x**2 + y**2 + z**2)
    
    sphr_shift = ( np.array( [ x , y , z ] ) / mynorm ).T
    
    distances = cdist( sphr_shift , sphr_shift )      
    distances = np.tril( distances ) + np.triu(np.ones( (len( sphr_shift ) , len( sphr_shift )) ) )    
    tbd = np.unique( np.nonzero(distances < 0.1)[0] )
 
    sphr_size[nn] = len( x ) - len(tbd)



plt.figure(1)

plt.hist( sphr_size , bins=25 , normed=True , alpha=0.6 ,  color='green')

xmin, xmax = plt.xlim()

xx = np.linspace( xmin , xmax , 100)

mu , std = norm.fit( sphr_size )

pdf = norm.pdf(xx, mu, std)
mytitle = "$\mu$= %.2f,  $\sigma$ = %.2f" % (mu, std)
plt.title( mytitle )
plt.plot( xx , pdf , 'k', linewidth=2)

plt.savefig('direction_size_pdf.png' , dpi=400 )

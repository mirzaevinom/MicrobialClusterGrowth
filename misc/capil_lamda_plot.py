from __future__ import division
from scipy.interpolate import griddata
from constants import import_constants


import matplotlib.pyplot as plt
import deformation as dfm
import numpy as np
import time, os

# import the constants
lam, mu, gammadot, Gamma= import_constants()

t0 = 0
t1 = 20

dt = 1e-1 / gammadot
start = time.time()

# set up the matrix velocity gradient L defined by du/dy=gammadot
L = np.zeros( [3,3] )

flowtype = 0

if flowtype==0:
    #Simple planar flow
    L[0,1] = gammadot
elif flowtype==1:
    #Circulating flow
    L[0,1] = gammadot
    L[2, 0] = -gammadot
elif flowtype==2:
    #Elongational flow
    L[0,0] = 1*gammadot
    L[1, 1] = -gammadot
else:
    raise Exception("Please specify a valid flowtype")

a_length = np.linspace( 1 , 20, 20)
lambdas = np.linspace( 10 , 100, 20 )
capillary = mu * a_length * gammadot / Gamma

taylor_deform = np.zeros( (  len(a_length) , len(lambdas)) )

max_deform = np.zeros_like( taylor_deform )

for mm in range( len(a_length) ):
    
    for nn in range( len(lambdas) ):
        
        lam = lambdas[nn] 

        # set the initial axes
        a0 = a_length[mm] * np.ones(3)

        # set up the initial shape tensor
        G0 = np.diag( 1.0 / a0**2 )
        G0v = dfm.tens2vec( G0 )
        axes = dfm.evolve( t0 , t1 , dt , G0v , lam , mu , L , Gamma )
        
        taylor_deform[ mm , nn ]  = np.max( ( axes[:, 0] - axes[:, 2] ) / ( axes[:, 0] + axes[:, 2]) )
        max_deform[ mm , nn ] = 100*np.max( np.sum( np.abs(1 - axes / a0 ) , axis=1 ) / 2 )


plt.close('all' )

plt.figure(0)

aaa_ , lll_ = np.meshgrid( a_length , lambdas )
 
aaa = np.ravel( aaa_)
lll = np.ravel( lll_)
 
points = np.array( [ aaa, lll] ).T
values = np.ravel( taylor_deform )

finer_a = np.linspace( a_length[0] , a_length[-1] , 1000 )
finer_l = np.linspace( lambdas[0] , lambdas[-1] , 1000 )

grid_x, grid_y = np.meshgrid( finer_a , finer_l )
grid_z = griddata( points, values, (grid_x, grid_y) , method='cubic')


plt.imshow( grid_z.T, origin='lower' , cmap = 'jet' )

plt.xticks( np.linspace( 0,  len(finer_a)-1 , 5 ) , 
            np.round( np.linspace( capillary[0],  capillary[-1] , 5 ) , 1) )
            
plt.yticks( np.linspace( 0,  len(finer_l)-1 , 5 ) , 
            np.round( np.linspace( lambdas[0] ,  lambdas[-1] , 5) , 0) )

plt.xlabel( '$Ca$' , fontsize=20 )
plt.ylabel( '$\lambda$' , fontsize=20 )
plt.colorbar()

img_name = 'capillar_lambdas_'+str(flowtype)+'.png'

plt.savefig( os.path.join( 'images' , img_name ) , dpi=400, bbox_inches='tight')

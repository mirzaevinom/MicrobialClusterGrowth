from __future__ import division
from mayavi import mlab

import matplotlib.pyplot as plt

import deformation as dfm

from constants import lam, mu, gammadot, Gamma

import dla_3d
import numpy as np
import visual_functions as vf
import time, cPickle, os

""" Test the core functions of deformation.py
"""



# set the initial axes
a0 = np.array( [ 10.0, 9.0 , 8.0 ] )
#a0 = np.sort( 1 + 10*np.random.rand(3) )[::-1]

t0 = 0
sim_step = 0.25

dt = 1e-1 / gammadot

start = time.time()
# set up the matrix velocity gradient L defined by du/dy=gammadot


L = np.zeros([3,3])

flow_type = 2

if flow_type == 0:
    # Simple shear in one direction
    L[0,1] = gammadot
    
elif flow_type ==1:
    
    # Shear plus elongation flow
    L[0,1] = gammadot
    L[0,0] = gammadot
    L[1, 1] = -gammadot

elif flow_type == 2:

    #Elongational flow
    L[0,0] = gammadot
    L[1, 1] = -gammadot

else:
    raise Exception("Please specify a valid flow type")
    
    
# set up the initial shape tensor
G0 = np.diag( 1.0 / a0**2 )
G0v = dfm.tens2vec( G0 )

a1 = dfm.deform(t0, sim_step , dt , G0v , lam , mu , L , Gamma )[0]

print a1
print a1/a0
aaa     = np.prod( a1 ) / np.prod( a0 )
vol_err = round( 100 * np.abs( 1- aaa ) , 6 )
print 'Error in volume', vol_err, 'percent'

end = time.time()

print 'Time elapsed' , round( end - start, 2), 'seconds'


axes = dfm.evolve(t0, sim_step , dt , G0v , lam , mu , L , Gamma )[0]

taylor_deform  = np.max( ( axes[:, 0] - axes[:, 2] ) / ( axes[:, 0] + axes[:, 2]) )

mean_deform = np.mean( np.sum( np.abs(1 - axes / a0 ) , axis=1 ) / 2 )

print 'Mean deformation', round(mean_deform, 2)*100, 'percent'

max_deform = np.max( np.sum( np.abs(1 - axes / a0 ) , axis=1 ) / 2 )

print 'Max deformation', round(max_deform, 2)*100, 'percent'
print 'Max Taylor deformation', taylor_deform

plt.close('all')

plt.figure(0)

plt.plot(axes)

mlab.close(all=True)


#points = 10*np.random.rand(10**3, 3)
#points  = points - np.mean(points , axis=0)
#dists = np.sum( points**2 , axis=1)
#points = points[dists<25]
#floc = points * np.array([ 10 , 8 , 5 ]) 
floc = np.load( 'dla_floc.npy')

#floc = dla_3d.dla_generator( num_particles = 1000)    

fig = mlab.figure( size=( 800 , 800 ) ,  bgcolor=(1,1,1) )

vf.mayavi_ellipsoid( floc , fig )
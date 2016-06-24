from __future__ import division
from mayavi import mlab

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import  moviepy.editor as mpy

import deformation as dfm

from constants import import_constants

import numpy as np

import time, cPickle, os

""" Test the core functions of deformation.py
"""

# import the constants
lam, mu, gammadot, Gamma= import_constants()

# set the initial axes
a0 = np.array( [ 5.0, 5.0 , 5.0 ] )
#a0 = np.sort( 1 + 10*np.random.rand(3) )[::-1]

t0 = 0
sim_step = 2

dt = 1e-1 / gammadot

start = time.time()
# set up the matrix velocity gradient L defined by du/dy=gammadot


L = np.zeros([3,3])

flow_type = 0

if flow_type == 0:
    # Simple shear in one direction
    L[0,1] = gammadot
    
elif flow_type ==1:
    
    # flow in multiple directions
    L[0,1] = gammadot
    L[1, 2] = gammadot
    #L[0, 2] = gammadot/3

elif flow_type == 2:
    
    #Elongational flow
    L[0,0] = gammadot
    L[1, 1] = -gammadot
    #L[2, 2] = -gammadot
    #L *= 0.1
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


axes = dfm.evolve(t0, sim_step , dt , G0v , lam , mu , L , Gamma )

taylor_deform  = np.max( ( axes[:, 0] - axes[:, 2] ) / ( axes[:, 0] + axes[:, 2]) )

mean_deform = np.mean( np.sum( np.abs(1 - axes / a0 ) , axis=1 ) / 2 )

print 'Mean deformation', round(mean_deform, 2)*100, 'percent'

max_deform = np.max( np.sum( np.abs(1 - axes / a0 ) , axis=1 ) / 2 )

print 'Max deformation', round(max_deform, 2)*100, 'percent'
print 'Max Taylor deformation', taylor_deform

plt.close('all')

plt.figure(0)

plt.plot(axes)


"""
points = 10*np.random.rand(10**4, 3)
points  = points - np.mean(points , axis=0)
dists = np.sum( points**2 , axis=1)
points = points[dists<25]
points = points * np.array([ 10 , 10 , 1 ]) 

#import scipy.io as sio
#
#dla_mat = sio.loadmat( 'test.mat' )[ 'map' ]
#
#cells = np.nonzero( dla_mat )
#
#points = np.array(cells).T


fig = plt.figure(1)

pts , radii , A = dfm.set_initial_pars(points)
print radii


ax = fig.add_subplot(111, projection='3d')

# plot points
ax.scatter( pts[:, 0] , pts[:, 1] , pts[:, 2] , color='g' )

# plot ellipsoid
dfm.plotEllipsoid( radii ,  ax=ax, plotAxes=True )

#Change the view angle and elevation
ax.view_init( azim=-10, elev=30 )


fig = plt.figure(1)

(pts, radii , shape_tens) = dfm.get_body_ellipse(points)

print radii

ax = fig.add_subplot(111, projection='3d')

# plot points
ax.scatter( pts[:, 0] , pts[:, 1] , pts[:, 2] , color='g' )

# plot ellipsoid
dfm.plotEllipsoid( radii ,  ax=ax, plotAxes=True )

#Change the view angle and elevation
ax.view_init( azim=-10, elev=30 )



mlab.close(all=True)
mlab.figure( size=(600, 600) )

myaxes = axes2[::int( len(axes2) / 100) ]
num_fps = 24

N = len(myaxes) - 1

duration = N / num_fps 

def make_frame(t):
    
    [a, b, c] = myaxes[ int(t*num_fps) ]
    
    mlab.clf()
    p_axis = np.max( [a, b, c] )
    x, y, z = np.ogrid[-p_axis:p_axis:100j, -p_axis:p_axis:100j, -p_axis:p_axis:100j]
    F = x**2/a**2 + y**2/b**2 + z**2/c**2 - 1
    mlab.contour3d(F, contours = [0] , transparent=True, opacity=0.5)
    mlab.points3d( 0, 0, 0,  1 , scale_factor=2)

    return mlab.screenshot( antialiased=True )


animation = mpy.VideoClip(make_frame, duration=duration)

#vf_name = 'mlab_animation_' + time.strftime("%Y_%m_%d_%H_%M", time.gmtime()) + '.mp4'
vf_name = 'deformation.mp4'

animation.write_videofile( vf_name , fps = num_fps)
"""
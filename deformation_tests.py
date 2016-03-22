from __future__ import division
from mayavi import mlab

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import  moviepy.editor as mpy

import deformation as dfm

from constants import import_constants

import numpy as np

import time

""" Test the core functions of deformation.py
"""

# import the constants
lam, mu, gammadot, Gamma, max_stress, p0 = import_constants()

# set the initial axes
a0 = np.array( [ 10.0, 10.0 , 10.0 ] )

t0 = 0
t1 = 20

dt = 1e-1


#some random points

points = 10 * np.random.rand(10**3, 3)
points  = points - np.mean(points , axis=0)
dists = np.sum( points**2 , axis=1)
points = points[dists<25]
points = points * np.array([ 3 , 2 , 1 ]) 


points = np.load( 'sample_cluster.npy' )[:, 0:3]


#(points, radii , shape_tens) = dfm.get_body_ellipse( points )

#print radii

start = time.time()

# set up the initial shape tensor
G0 = np.diag( 1 / a0**2 )
G0v = dfm.tens2vec(G0)
  
a1 , G0v , V = dfm.deform(t0, t1 , dt , G0v , lam , mu , gammadot , Gamma )

print a1
print a1/a0

print np.prod(a1) / np.prod(a0)

axes = dfm.evolve(t0, t1 , dt , G0v , lam , mu , gammadot , Gamma )

plt.close('all')

plt.figure(3)

plt.plot(axes)
#radii , G0v = dfm.deform(t0, t1 , dt , G0v , lam , mu , gammadot , Gamma )
#
#print radii

end = time.time()

print 'Time elapsed' , round( end - start, 2), 'seconds'


"""
fig = plt.figure(0)

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

(points, radii , shape_tens) = dfm.get_body_ellipse(points)

print radii

ax = fig.add_subplot(111, projection='3d')

# plot points
ax.scatter( points[:, 0] , points[:, 1] , points[:, 2] , color='g' )

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
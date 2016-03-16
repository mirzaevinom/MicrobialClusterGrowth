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
a0 = np.array( [2.316 , 1.361 , 1.304] )

t0 = 0
t1 = 1

t0,t1,dt,tau,cap = dfm.set_tau_cap(a0, lam, mu, gammadot, Gamma)

#some random points
#points = 20*( np.random.rand(1000, 3) - 0.5 )

points = np.load( 'sample_cluster.npy' )[:, 0:3]
start = time.time()

(points, radii , shape_tens) = dfm.get_body_ellipse(points)
print radii


# set up the initial shape tensor
G0 = np.diag( 1 / a0**2 )
G0v = dfm.tens2vec(G0)
  
radii , G0v = dfm.deform(t0, t1 , dt , G0v , lam , mu , gammadot , Gamma )

print radii
#
#radii , G0v = dfm.deform(t0, t1 , dt , G0v , lam , mu , gammadot , Gamma )
#
#print radii

end = time.time()

print 'Time elapsed' , round( end - start, 2), 'seconds'


plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot points
ax.scatter( points[:, 0] , points[:, 1] , points[:, 2] , color='g' )

# plot ellipsoid
dfm.plotEllipsoid( radii ,  ax=ax, plotAxes=True )

#Change the view angle and elevation
ax.view_init( azim=-10, elev=30 )

"""

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
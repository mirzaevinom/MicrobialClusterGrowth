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
a0 = np.array( [20., 15., 10] )



start = time.time()

t0=0
t1 = 20

print dfm.deform(t0, t1 , 1e-4,  a0 , lam , mu , gammadot , Gamma )

end = time.time()

print 'Time elapsed' , round( end - start, 2), 'seconds'



#some random points
points = 20*( np.random.rand(1000, 3) - 0.5 )

start = time.time()

(center, radii, rotation) = dfm.getMinVolEllipse(points, 0.01)

end = time.time()

print 'Ellipsoid time', round(end - start , 2)
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot points
ax.scatter(points[:,0], points[:,1], points[:,2], color='g', s=100)

# plot ellipsoid
dfm.plotEllipsoid(center, radii, rotation, ax=ax, plotAxes=True)

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
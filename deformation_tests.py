from __future__ import division
from mayavi import mlab

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
# compute the time interval
t0,t1,dt,tau,cap = dfm.set_tau_cap(a0, lam, mu, gammadot, Gamma)

start = time.time()


t1 = 10
# run the deformation integral
#Y,T = dfm.integrate_dgdt(t0 , t1 , dt , a0 , lam , mu , gammadot , Gamma)
# get the rotations and the axes
#axes, R = dfm.shapetensors_to_axes_rots(Y)
# test angular velocity computations
#w = dfm.angular_velocity(R, dt)
# test the wrapper function deform

#axes2, R2, w2, T2 = dfm.deform( t0 , t1, dt, a0, lam, mu , gammadot , Gamma ) 

print dfm.deform_ode_solve(t0, t1 , 1e-4,  a0 , lam , mu , gammadot , Gamma )

end = time.time()

print 'Time elapsed' , round( end - start, 2), 'seconds'

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